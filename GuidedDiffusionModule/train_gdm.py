"""Train the unconditional SIC diffusion prior used by IceDiff-GDM.

Paper (Sec. 4.2): daily SIC maps resized/cropped to 256x256, AdamW (0.9, 0.999), fp16 with loss
scaling but fp32 master weights/EMA/optimizer state, EMA 0.9999, 1000 diffusion steps.
Unstated in the paper (defaults follow OpenAI's 256x256 unconditional model): lr, batch size,
iterations, UNet width/depth/attention.

Single GPU:   python train_gdm.py --log_dir runs/gdm
Multi GPU:    torchrun --nproc_per_node=8 train_gdm.py --log_dir runs/gdm
"""
import argparse
import copy
import glob
import os
import time

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from guided_diffusion import logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
from guided_diffusion.script_util_x0 import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from sic_dataset import SICCropDataset


def setup_dist():
    if "RANK" not in os.environ:
        return 0, 1, 0
    backend = "nccl" if th.cuda.is_available() else "gloo"
    dist.init_process_group(backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    if th.cuda.is_available():
        th.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


def device_for(local_rank):
    return th.device(f"cuda:{local_rank}") if th.cuda.is_available() else th.device("cpu")


def infinite(loader, sampler):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def latest_checkpoint(log_dir):
    ckpts = sorted(glob.glob(os.path.join(log_dir, "model[0-9]*.pt")))
    return ckpts[-1] if ckpts else None


def step_of(path):
    return int(os.path.basename(path)[len("model"):-len(".pt")])


@th.no_grad()
def validation_loss(model, diffusion, loader, device, num_batches):
    model.eval()
    losses, n = [], 0
    for x, _ in loader:
        x = x.to(device)
        t = th.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        losses.append(diffusion.training_losses(model, x, t)["loss"].mean().item())
        n += 1
        if n >= num_batches:
            break
    model.train()
    return float(np.mean(losses))


@th.no_grad()
def save_samples(ema_model, sample_diffusion, device, path, n, size):
    ema_model.eval()
    x = th.randn(n, 3, size, size, device=device)
    for i in reversed(range(sample_diffusion.num_timesteps)):
        t = th.full((n,), i, device=device, dtype=th.long)
        out = sample_diffusion.p_mean_variance(ema_model, x, t, clip_denoised=True)
        noise = th.randn_like(x) if i > 0 else th.zeros_like(x)
        x = out["mean"] + th.exp(0.5 * out["log_variance"]) * noise
    img = ((x[:, 0] + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu().numpy()
    Image.fromarray(np.concatenate(list(img), axis=1)).save(path)
    ema_model.train()


def main():
    args = create_argparser().parse_args()
    rank, world, local_rank = setup_dist()
    device = device_for(local_rank)
    th.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    logger.configure(dir=args.log_dir, format_strs=["stdout", "log", "csv"] if rank == 0 else [])
    logger.log(f"rank {rank}/{world} on {device}; args: {vars(args)}")

    model_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**model_kwargs)
    model.to(device)
    sample_diffusion = create_model_and_diffusion(
        **{**model_kwargs, "timestep_respacing": str(args.sample_steps)}
    )[1]
    logger.log(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    resume = latest_checkpoint(args.log_dir) if args.resume else None
    step = 0
    if resume:
        step = step_of(resume)
        model.load_state_dict(th.load(resume, map_location="cpu"))
        logger.log(f"resumed from {resume} at step {step}")

    mp_trainer = MixedPrecisionTrainer(model=model, use_fp16=args.use_fp16,
                                       fp16_scale_growth=args.fp16_scale_growth)
    opt = th.optim.AdamW(mp_trainer.master_params, lr=args.lr, betas=(0.9, 0.999),
                         weight_decay=args.weight_decay)
    ema_params = copy.deepcopy(mp_trainer.master_params)
    if resume:
        ema_path = os.path.join(args.log_dir, f"ema_{args.ema_rate}_{step:06d}.pt")
        ema_params = mp_trainer.state_dict_to_master_params(th.load(ema_path, map_location="cpu"))
        ema_params = [p.to(device) for p in ema_params]
        opt.load_state_dict(th.load(os.path.join(args.log_dir, f"opt{step:06d}.pt"), map_location="cpu"))

    ddp_model = DDP(model, device_ids=[local_rank] if th.cuda.is_available() else None,
                    find_unused_parameters=False) if world > 1 else model

    train_set = SICCropDataset(args.train_npz, args.ocean_mask, size=args.image_size,
                               upscale=args.upscale, min_ocean=args.min_ocean)
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank) if world > 1 else None
    loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.num_workers, drop_last=True, pin_memory=th.cuda.is_available(),
                        persistent_workers=args.num_workers > 0)
    val_loader = None
    if args.val_npz and os.path.exists(args.val_npz):
        val_set = SICCropDataset(args.val_npz, args.ocean_mask, size=args.image_size,
                                 upscale=args.upscale, min_ocean=args.min_ocean,
                                 samples_per_epoch=args.val_batches * args.batch_size)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)

    def ema_model():
        m = copy.deepcopy(model)
        m.load_state_dict(mp_trainer.master_params_to_state_dict(ema_params))
        return m.to(device)

    def save():
        if rank != 0:
            return
        th.save(mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
                os.path.join(args.log_dir, f"model{step:06d}.pt"))
        th.save(mp_trainer.master_params_to_state_dict(ema_params),
                os.path.join(args.log_dir, f"ema_{args.ema_rate}_{step:06d}.pt"))
        th.save(opt.state_dict(), os.path.join(args.log_dir, f"opt{step:06d}.pt"))
        logger.log(f"saved checkpoint at step {step}")

    data = infinite(loader, sampler)
    t0 = time.time()
    while args.max_steps <= 0 or step < args.max_steps:
        x, _ = next(data)
        x = x.to(device, non_blocking=True)
        mp_trainer.zero_grad()
        t = th.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        losses = diffusion.training_losses(ddp_model, x, t)
        loss = losses["loss"].mean()
        mp_trainer.backward(loss)
        if mp_trainer.optimize(opt):
            update_ema(ema_params, mp_trainer.master_params, rate=args.ema_rate)
        step += 1

        for k, v in losses.items():
            logger.logkv_mean(k, v.mean().item())
        if step % args.log_interval == 0:
            logger.logkv("step", step)
            logger.logkv("samples", step * args.batch_size * world)
            logger.logkv("sec_per_step", (time.time() - t0) / args.log_interval)
            if val_loader is not None and rank == 0:
                logger.logkv("val_loss", validation_loss(model, diffusion, val_loader, device, args.val_batches))
            logger.dumpkvs()
            t0 = time.time()
        if step % args.save_interval == 0:
            save()
            if rank == 0 and args.num_preview > 0:
                save_samples(ema_model(), sample_diffusion, device,
                             os.path.join(args.log_dir, f"preview_{step:06d}.png"),
                             args.num_preview, args.image_size)
    if step % args.save_interval != 0:
        save()
    if world > 1:
        dist.destroy_process_group()


def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        resblock_updown=True,
        use_scale_shift_norm=True,
        learn_sigma=True,
        noise_schedule="linear",
        use_fp16=True,
    )
    defaults.update(
        train_npz="/home/icediff/data/gdm/sic_train.npz",
        val_npz="/home/icediff/data/gdm/sic_val.npz",
        ocean_mask="/home/icediff/data/gdm/ocean_mask.npy",
        log_dir="runs/gdm",
        upscale=4,
        min_ocean=0.25,
        lr=1e-4,
        weight_decay=0.0,
        batch_size=8,
        ema_rate=0.9999,
        fp16_scale_growth=1e-3,
        max_steps=0,
        log_interval=100,
        save_interval=10000,
        val_batches=4,
        num_preview=4,
        sample_steps=100,
        num_workers=4,
        seed=0,
        resume=True,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
