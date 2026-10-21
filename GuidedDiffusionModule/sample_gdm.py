"""IceDiff-GDM guided down-scaling: paper Algorithm 1 with the patch strategy of Fig. 3.

Given a 25 km SIC map y (448x304), sample a 6.25 km map x (1792x1216) from the unconditional prior.
At every reverse step t, for each overlapping 256x256 patch (stride 128) with its own kernel k:
  x0_hat = prediction of the prior
  L      = || y_patch - D_k(x0_hat) ||^2,   D_k = 3x3 conv with k (normalised to sum 1), then s x s down-scaling
  x0_hat <- x0_hat - g * dL/dx0_hat
  mean, logvar = posterior q(x_{t-1} | x_t, x0_hat)
  k      <- k - l * dL/dk
Overlapping means / variances are averaged, then x_{t-1} is sampled.
--blend ramp (not the paper setting) weights each patch with a tent window instead of a plain average.
--dates paper selects the appendix examples (PAPER_DATES); --dates YYYYMMDD,... any dates.

--sampler ddim (with --timestep_respacing N) is a fast preview, not the paper setting: the posterior step
is replaced by the DDIM step q_sigma(x_{t-1} | x_t, x0_hat) with sigma set by --eta.

Not given in the paper (values follow the released downscale.py): L is the mean squared error,
g = 3800, k initialised to rand/100, l = 0.01. The down-scaling in D is s x s average pooling.
The kernel is kept non-negative and normalised to sum 1 inside D (not in the released code): otherwise
on uniform patches (all water / all pack ice) its scale is free and the guidance degenerates.

Input  --lr_npz: arr_0 uint8 [N,448,304] (0..255) or float [N,448,304] in [0,1], optional `dates`.
Output samples.npz: hr uint8 [N,448s,304s], lr uint8 [N,448,304], dates, final normalised kernels
[N,K,3,3,3] (patch, channel) and patch corners [K,2] (6.25 km pixels).
"""
import argparse
import os
import time

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


PAPER_DATES = [20210708, 20210301, 20180415]  # the down-scaling examples shown in the paper appendix


def load_lr(path, every, limit, dates_arg=""):
    data = np.load(path)
    arr = data["arr_0"]
    if arr.ndim == 4:
        arr = arr[..., 0]
    if arr.dtype != np.uint8:
        arr = np.round(np.clip(arr, 0, 1) * 255).astype(np.uint8)
    dates = data["dates"] if "dates" in data.files else np.arange(len(arr))
    if dates_arg:
        want = PAPER_DATES if dates_arg == "paper" else [int(d) for d in dates_arg.split(",")]
        idx = np.array([int(np.where(dates == d)[0][0]) for d in want])
    else:
        idx = np.arange(0, len(arr), every)[:limit or None]
    return arr[idx], dates[idx]


def blend_weight(p, mode, device):
    """Per-pixel weight of a patch when overlapping patches are combined: 'mean' (paper) or 'ramp' (tent window)."""
    if mode == "mean":
        return th.ones(p, p, device=device)
    r = th.minimum(th.arange(p, device=device), th.arange(p - 1, -1, -1, device=device)).float() + 1
    return r[:, None] * r[None, :]


def normalise(k):
    return k / k.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


class GuidedPatchSampler:
    def __init__(self, model, diffusion, a, device):
        self.model, self.diffusion, self.a, self.device = model, diffusion, a, device

    @th.no_grad()
    def sample(self, y):
        """y: [1, 3, h, w] in [0, 1]; returns [1, 3, h*s, w*s] in [-1, 1]."""
        a, s, p, dev = self.a, self.a.scale, self.a.patch_size, self.device
        H, W = y.shape[2] * s, y.shape[3] * s
        corners = [(h, w) for h in grid_starts(H, p, a.stride) for w in grid_starts(W, p, a.stride)]
        y_patches = th.cat([y[:, :, h // s:(h + p) // s, w // s:(w + p) // s] for h, w in corners])
        kernels = (th.rand(len(corners), 1, 1, 3, 3, device=dev) / 100).repeat(1, 3, 1, 1, 1)
        self.corners, self.kernels_init = corners, normalise(kernels)

        x = th.randn(1, 3, H, W, device=dev)
        wgt = blend_weight(p, a.blend, dev)
        ddim = a.sampler == "ddim"
        for i in range(self.diffusion.num_timesteps - 1, -1, -1):
            if ddim:
                ab, ab_prev = self.diffusion.alphas_cumprod[i], self.diffusion.alphas_cumprod_prev[i]
                sigma = a.eta * np.sqrt((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev))
            mean_acc, logvar_acc, count = th.zeros_like(x), th.zeros_like(x), th.zeros_like(x)
            for c0 in range(0, len(corners), a.patch_batch):
                cs = corners[c0:c0 + a.patch_batch]
                n = len(cs)
                xp = th.cat([x[:, :, h:h + p, w:w + p] for h, w in cs])
                t = th.full((n,), i, device=dev, dtype=th.long)
                out = self.diffusion.p_mean_variance(self.model, xp, t, clip_denoised=True)
                with th.enable_grad():
                    x0 = out["pred_xstart"].detach().float().requires_grad_(True)
                    k = kernels[c0:c0 + n].detach().requires_grad_(True)
                    kn = normalise(k)
                    d = F.conv2d(((x0 + 1) / 2).reshape(1, 3 * n, p, p), kn.reshape(3 * n, 1, 3, 3),
                                 padding=1, groups=3 * n).reshape(n, 3, p, p)
                    d = F.avg_pool2d(d, s)
                    loss = ((d - y_patches[c0:c0 + n]) ** 2).mean(dim=(1, 2, 3)).sum()
                    gx, gk = th.autograd.grad(loss, [x0, k])
                kernels[c0:c0 + n] = (k.detach() - a.kernel_lr * gk).clamp_min(0)
                x0 = x0.detach() - a.guidance_scale * gx
                if ddim:
                    eps = self.diffusion._predict_eps_from_xstart(xp, t, x0)
                    mean = np.sqrt(ab_prev) * x0 + np.sqrt(1 - ab_prev - sigma ** 2) * eps
                    logvar = th.full_like(mean, 2 * np.log(max(sigma, 1e-20)))
                else:
                    mean, _, logvar = self.diffusion.q_posterior_mean_variance(x0, xp, t)
                for j, (h, w) in enumerate(cs):
                    mean_acc[:, :, h:h + p, w:w + p] += wgt * mean[j:j + 1]
                    logvar_acc[:, :, h:h + p, w:w + p] += wgt * logvar[j:j + 1]
                    count[:, :, h:h + p, w:w + p] += wgt
            noise = th.randn_like(x) if i > 0 else th.zeros_like(x)
            x = mean_acc / count + th.exp(0.5 * logvar_acc / count) * noise
        self.kernels = normalise(kernels)
        return x.clamp(-1, 1)


def main():
    a = create_argparser().parse_args()
    dev = th.device("cuda" if th.cuda.is_available() else "cpu")
    a.use_fp16 = a.use_fp16 and dev.type == "cuda"

    model, diffusion = create_model_and_diffusion(**args_to_dict(a, model_and_diffusion_defaults().keys()))
    model.load_state_dict(th.load(a.model_path, map_location="cpu"))
    model.to(dev)
    if a.use_fp16:
        model.convert_to_fp16()
    model.eval()

    lr_all, dates = load_lr(a.lr_npz, a.every, a.limit, a.dates)
    os.makedirs(a.save_dir, exist_ok=True)
    sampler = GuidedPatchSampler(model, diffusion, a, dev)
    hr_out, kernels = [], []
    for n, lr in enumerate(lr_all):
        th.manual_seed(a.seed + n)
        y = th.from_numpy(lr.astype(np.float32) / 255.0)[None, None].repeat(1, 3, 1, 1).to(dev)
        t0 = time.time()
        x = sampler.sample(y)
        hr = ((x[0].mean(0) + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu().numpy()
        hr_out.append(hr)
        kernels.append(sampler.kernels[:, :, 0].cpu().numpy())
        print(f"[{n + 1}/{len(lr_all)}] date={dates[n]}  {time.time() - t0:.0f}s", flush=True)
        if a.save_png:
            Image.fromarray(hr).save(os.path.join(a.save_dir, f"{dates[n]}_hr.png"))
            Image.fromarray(lr).save(os.path.join(a.save_dir, f"{dates[n]}_lr.png"))
        np.savez(os.path.join(a.save_dir, "samples.npz"), hr=np.stack(hr_out), lr=lr_all[:n + 1], dates=dates[:n + 1],
                 kernels=np.stack(kernels), corners=np.array(sampler.corners))
    print(f"-> {a.save_dir}/samples.npz")


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
        model_path="",
        lr_npz="../data/sic_test.npz",
        save_dir="runs/gdm_samples",
        every=1,
        limit=0,
        dates="",
        blend="mean",
        scale=4,
        patch_size=256,
        stride=128,
        patch_batch=16,
        sampler="ddpm",
        eta=0.0,
        guidance_scale=3800.0,
        kernel_lr=0.01,
        save_png=True,
        seed=0,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
