"""Ground-truth downscaling evaluation for IceDiff-GDM (100 km -> 25 km, see METRICS.md).

Each selected 25 km map (448x304) is 4x4 average-pooled to 100 km (112x76) and used as y;
the guided sampler downscales it back to 448x304, so the 25 km map is the ground truth.
Patches (6 per map) are denoised in one batch; guidance is the same as sample_gdm.py.

Output (--out_dir):
  truth.npz            hr uint8 [N,448,304], lr float32 [N,112,76] in [0,1], dates, ocean
  <tag>__<cfg>.npz     sr uint8 [N,448,304], seconds per map
Then run eval_metrics.py on --out_dir.
"""
import argparse
import os
import time

import numpy as np
import torch as th
import torch.nn.functional as F

from guided_diffusion.script_util_x0 import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from sample_gdm import grid_starts, to_unit


class BatchedPatchSampler:
    def __init__(self, model, diffusion, a, device):
        self.model, self.diffusion, self.a, self.device = model, diffusion, a, device

    @th.no_grad()
    def sample(self, y_lr):
        """y_lr: [1, 3, h, w] in [0, 1]; returns [1, 3, h*s, w*s] in [-1, 1]."""
        a, s, p, dev = self.a, self.a.scale, self.a.patch_size, self.device
        H, W = y_lr.shape[2] * s, y_lr.shape[3] * s
        y_hr = F.interpolate(y_lr, scale_factor=s, mode="bicubic", align_corners=False).clamp(0, 1)
        corners = [(h, w) for h in grid_starts(H, p, a.stride) for w in grid_starts(W, p, a.stride)]
        n = len(corners)
        if a.guidance_space == "lr":
            yp = th.cat([y_lr[:, :, h // s:(h + p) // s, w // s:(w + p) // s] for h, w in corners])
        else:
            yp = th.cat([y_hr[:, :, h:h + p, w:w + p] for h, w in corners])

        k = th.zeros(3 * n, 1, 3, 3, device=dev)
        k[:, :, 1, 1] = 1.0

        T = self.diffusion.num_timesteps
        if a.start_t > 0:
            t0 = min(a.start_t, T) - 1
            x = self.diffusion.q_sample(y_hr * 2 - 1, th.full((1,), t0, device=dev, dtype=th.long))
        else:
            t0 = T - 1
            x = th.randn(1, 3, H, W, device=dev)

        for i in range(t0, -1, -1):
            xp = th.cat([x[:, :, h:h + p, w:w + p] for h, w in corners])
            out = self.diffusion.p_mean_variance(self.model, xp, th.full((n,), i, device=dev, dtype=th.long),
                                                 clip_denoised=True)
            with th.enable_grad():
                x0 = out["pred_xstart"].detach().float().requires_grad_(True)
                kk = k.detach().requires_grad_(True)
                d = F.conv2d(to_unit(x0).reshape(1, 3 * n, p, p), kk, padding=1, groups=3 * n).reshape(n, 3, p, p)
                if a.guidance_space == "lr":
                    d = F.avg_pool2d(d, s)
                gx, gk = th.autograd.grad(((d - yp) ** 2).sum(), [x0, kk])
            k = k - a.kernel_lr * gk
            mean = out["mean"] - a.guidance_scale * gx
            mean_acc, logvar_acc, count = th.zeros_like(x), th.zeros_like(x), th.zeros_like(x)
            for j, (h, w) in enumerate(corners):
                mean_acc[:, :, h:h + p, w:w + p] += mean[j:j + 1]
                logvar_acc[:, :, h:h + p, w:w + p] += out["log_variance"][j:j + 1]
                count[:, :, h:h + p, w:w + p] += 1
            noise = th.randn_like(x) if i > 0 else th.zeros_like(x)
            x = mean_acc / count + th.exp(0.5 * logvar_acc / count) * noise
        return x.clamp(-1, 1)


def select_days(path, every, limit):
    d = np.load(path)
    idx = np.arange(0, len(d["arr_0"]), every)[:limit or None]
    return d["arr_0"][idx], d["dates"][idx]


def main():
    a = create_argparser().parse_args()
    dev = th.device("cuda" if th.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    print("device:", th.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "torch", th.__version__)

    hr, dates = select_days(a.npz, a.every, a.limit)
    ocean = np.load(a.ocean_mask)
    lr = F.avg_pool2d(th.from_numpy(hr.astype(np.float32) / 255.0)[:, None], a.scale)[:, 0].numpy()
    np.savez(os.path.join(a.out_dir, "truth.npz"), hr=hr, lr=lr, dates=dates, ocean=ocean)
    print(f"{len(hr)} maps from {a.npz}: {dates[0]} .. {dates[-1]}")

    models = [m for m in a.model_paths.split(",") if m]
    tags = a.tags.split(",") if a.tags else [os.path.splitext(os.path.basename(m))[0] for m in models]
    configs = [c.split(":") for c in a.configs.split(",")]

    a.use_fp16 = a.use_fp16 and dev.type == "cuda"
    model_kwargs = args_to_dict(a, model_and_diffusion_defaults().keys())
    for mpath, tag in zip(models, tags):
        model, diffusion = create_model_and_diffusion(**model_kwargs)
        model.load_state_dict(th.load(mpath, map_location="cpu"))
        model.to(dev)
        if a.use_fp16:
            model.convert_to_fp16()
        model.eval()
        for space, scale in configs:
            a.guidance_space, a.guidance_scale = space, float(scale)
            name = f"{tag}__{space}{scale}"
            out_path = os.path.join(a.out_dir, name + ".npz")
            if os.path.exists(out_path):
                print("skip existing", out_path)
                continue
            sampler = BatchedPatchSampler(model, diffusion, a, dev)
            sr, secs = [], []
            for n in range(len(hr)):
                th.manual_seed(a.seed + n)
                y = th.from_numpy(lr[n])[None, None].repeat(1, 3, 1, 1).to(dev)
                t = time.time()
                x = sampler.sample(y)
                if dev.type == "cuda":
                    th.cuda.synchronize()
                secs.append(time.time() - t)
                out = ((x[0].mean(0) + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu().numpy()
                out[~ocean] = 0
                sr.append(out)
                rmse = np.sqrt(((out[ocean].astype(np.float32) - hr[n][ocean]) ** 2).mean()) / 2.55
                print(f"{name} [{n + 1}/{len(hr)}] {dates[n]} ocean RMSE={rmse:.2f}%  {secs[-1]:.1f}s", flush=True)
            np.savez(out_path, sr=np.stack(sr), secs=np.array(secs))
            print("saved", out_path)
        del model
        th.cuda.empty_cache()


def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(
        image_size=256,
        num_channels=128,
        num_res_blocks=2,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        resblock_updown=True,
        use_scale_shift_norm=True,
        learn_sigma=True,
        noise_schedule="linear",
        use_fp16=True,
        timestep_respacing="",
    )
    defaults.update(
        model_paths="",
        tags="",
        configs="lr:8",
        npz="../data/sic_val.npz",
        ocean_mask="../data/ocean_mask.npy",
        every=30,
        limit=0,
        out_dir="../eval_out",
        scale=4,
        patch_size=256,
        stride=128,
        kernel_lr=1e-8,
        start_t=0,
        seed=0,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
