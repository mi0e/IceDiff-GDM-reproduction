"""IceDiff-GDM guided down-scaling (paper Algorithm 1 + patch strategy of Fig. 3).

Given a low-resolution SIC map y (448x304, 25 km), sample a high-resolution map x (448s x 304s)
from the unconditional prior. At every reverse step each overlapping 256x256 patch:
  1. predicts x0_hat with the prior,
  2. degrades it with its own optimizable 3x3 conv kernel D,
  3. computes L = sum of squared errors between D(x0_hat) and y_patch, shifts the posterior mean by -s * dL/dx0_hat,
  4. updates D's kernel with dL/dkernel.
Overlapping patch means/variances are averaged.

guidance_space=hr (released downscale.py): y is bicubic-upsampled, D runs at target resolution.
guidance_space=lr: D is followed by s x s average pooling and compared with y at 25 km.

Input  --lr_npz: arr_0 uint8 [N,448,304] (prepare_sic.py output) or float SIC [N,448,304] in [0,1].
Output samples.npz: hr uint8 [N,448s,304s], lr uint8 [N,448,304], dates (if present).
"""
import argparse
import os

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image

from guided_diffusion.script_util_x0 import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def grid_starts(length, patch, stride):
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def load_lr(path, indices):
    data = np.load(path)
    arr = data["arr_0"]
    if arr.ndim == 4:
        arr = arr[..., 0]
    if arr.dtype != np.uint8:
        arr = np.round(np.clip(arr, 0, 1) * 255).astype(np.uint8)
    dates = data["dates"] if "dates" in data.files else np.arange(len(arr))
    if indices is not None:
        arr, dates = arr[indices], dates[indices]
    return arr, dates


def to_unit(x):
    return (x + 1) / 2


class GuidedPatchSampler:
    def __init__(self, model, diffusion, args, device):
        self.model = model
        self.diffusion = diffusion
        self.args = args
        self.device = device

    def guidance(self, pred_xstart, y_patch, kernel):
        """Return (grad wrt x0_hat, grad wrt kernel, mse) for one patch."""
        s = self.args.scale
        with th.enable_grad():
            x0 = pred_xstart.detach().requires_grad_(True)
            k = kernel.detach().requires_grad_(True)
            d = F.conv2d(to_unit(x0), k, padding=1, groups=3)
            if self.args.guidance_space == "lr":
                d = F.avg_pool2d(d, s)
            se = ((d - y_patch) ** 2).sum()
            gx, gk = th.autograd.grad(se, [x0, k])
        return gx, gk, se.item() / d.numel()

    @th.no_grad()
    def sample(self, y_lr):
        """y_lr: [1, 3, H, W] in [0, 1]."""
        a, s, p = self.args, self.args.scale, self.args.patch_size
        H, W = y_lr.shape[2] * s, y_lr.shape[3] * s
        y_hr = F.interpolate(y_lr, scale_factor=s, mode="bicubic", align_corners=False).clamp(0, 1)
        corners = [(h, w) for h in grid_starts(H, p, a.stride) for w in grid_starts(W, p, a.stride)]

        kernels = {}
        for c in corners:
            k = th.rand(3, 1, 3, 3, device=self.device) / 100 if a.kernel_init == "random" \
                else th.zeros(3, 1, 3, 3, device=self.device)
            if a.kernel_init == "identity":
                k[:, :, 1, 1] = 1.0
            kernels[c] = k

        T = self.diffusion.num_timesteps
        if a.start_t > 0:
            t0 = min(a.start_t, T) - 1
            x = self.diffusion.q_sample(y_hr * 2 - 1, th.full((1,), t0, device=self.device, dtype=th.long))
            steps = range(t0, -1, -1)
        else:
            x = th.randn(1, 3, H, W, device=self.device)
            steps = range(T - 1, -1, -1)

        for i in steps:
            t = th.full((1,), i, device=self.device, dtype=th.long)
            mean_acc = th.zeros_like(x)
            logvar_acc = th.zeros_like(x)
            count = th.zeros_like(x)
            mses = []
            for (h, w) in corners:
                xp = x[:, :, h:h + p, w:w + p]
                out = self.diffusion.p_mean_variance(self.model, xp, t, clip_denoised=True)
                if a.guidance_space == "lr":
                    yp = y_lr[:, :, h // s:(h + p) // s, w // s:(w + p) // s]
                else:
                    yp = y_hr[:, :, h:h + p, w:w + p]
                gx, gk, mse = self.guidance(out["pred_xstart"], yp, kernels[(h, w)])
                kernels[(h, w)] = kernels[(h, w)] - a.kernel_lr * gk
                mean = out["mean"] - a.guidance_scale * gx
                mean_acc[:, :, h:h + p, w:w + p] += mean
                logvar_acc[:, :, h:h + p, w:w + p] += out["log_variance"]
                count[:, :, h:h + p, w:w + p] += 1
                mses.append(mse)
            mean = mean_acc / count
            logvar = logvar_acc / count
            noise = th.randn_like(x) if i > 0 else th.zeros_like(x)
            x = mean + th.exp(0.5 * logvar) * noise
            if a.verbose and i % 50 == 0:
                print(f"t={i:4d}  guidance mse={np.mean(mses):.6f}")
        return x.clamp(-1, 1)


def consistency(hr_u8, lr_u8, scale):
    """GDP-style consistency: MSE (0-255 scale) between avg-pooled output and LR input."""
    hr = th.from_numpy(hr_u8.astype(np.float32))[:, None]
    down = F.avg_pool2d(hr, scale)[:, 0].numpy()
    return float(((down - lr_u8.astype(np.float32)) ** 2).mean())


def main():
    args = create_argparser().parse_args()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    th.manual_seed(args.seed)

    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    idx = None
    if args.indices:
        idx = [int(i) for i in args.indices.split(",")]
    elif args.num_samples > 0:
        idx = list(range(args.num_samples))
    lr_all, dates = load_lr(args.lr_npz, idx)
    ocean = np.load(args.ocean_mask) if args.ocean_mask else None

    os.makedirs(args.save_dir, exist_ok=True)
    sampler = GuidedPatchSampler(model, diffusion, args, device)
    hr_out, cons = [], []
    for n, lr in enumerate(lr_all):
        y = th.from_numpy(lr.astype(np.float32) / 255.0)[None, None].repeat(1, 3, 1, 1).to(device)
        x = sampler.sample(y)
        hr = ((x[0].mean(0) + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu().numpy()
        if ocean is not None and args.apply_land_mask:
            land_hr = np.kron(~ocean, np.ones((args.scale, args.scale), dtype=bool))
            hr[land_hr] = 0
        hr_out.append(hr)
        cons.append(consistency(hr[None], lr[None], args.scale))
        print(f"[{n + 1}/{len(lr_all)}] date={dates[n]} consistency={cons[-1]:.3f}")
        if args.save_png:
            Image.fromarray(hr).save(os.path.join(args.save_dir, f"{dates[n]}_hr.png"))
            Image.fromarray(lr).save(os.path.join(args.save_dir, f"{dates[n]}_lr.png"))

    np.savez(os.path.join(args.save_dir, "samples.npz"), hr=np.stack(hr_out), lr=lr_all, dates=dates)
    print(f"mean consistency: {np.mean(cons):.3f}  -> {args.save_dir}/samples.npz")


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
        use_fp16=False,
        timestep_respacing="",
    )
    defaults.update(
        model_path="",
        lr_npz="/home/icediff/data/gdm/sic_test.npz",
        ocean_mask="/home/icediff/data/gdm/ocean_mask.npy",
        save_dir="runs/gdm_samples",
        num_samples=4,
        indices="",
        scale=4,
        patch_size=256,
        stride=128,
        guidance_space="hr",
        guidance_scale=0.5,
        kernel_lr=1e-8,
        kernel_init="identity",
        start_t=0,
        apply_land_mask=True,
        save_png=True,
        verbose=True,
        seed=0,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
