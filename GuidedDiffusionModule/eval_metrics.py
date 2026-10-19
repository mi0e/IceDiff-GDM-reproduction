"""Table 2 metrics for sample_gdm.py output: Consistency and FID, for IceDiff-GDM and the
nearest / bilinear / bicubic baselines (all 25 km -> 6.25 km).

Not defined in the paper (choices here):
  Consistency: MSE between the s x s average-pooled output and the 25 km input, SIC on the 0..255 scale
               (GDP / SR3 definition).
  FID:         pytorch-fid Inception-v3 pool3 features of 256x256 patches (stride 128) of the outputs,
               against 256x256 patches of the 25 km input maps; SIC replicated to 3 channels.

Usage: python eval_metrics.py --samples runs/gdm_samples/samples.npz
"""
import argparse
import os

import numpy as np
import torch as th
import torch.nn.functional as F

from sample_gdm import grid_starts


def baselines(lr_u8, scale):
    y = th.from_numpy(lr_u8.astype(np.float32) / 255.0)[:, None]
    out = {}
    for mode in ("nearest", "bilinear", "bicubic"):
        kw = {} if mode == "nearest" else {"align_corners": False}
        up = F.interpolate(y, scale_factor=scale, mode=mode, **kw)[:, 0].clamp(0, 1)
        out[mode] = (up * 255).round().to(th.uint8).numpy()
    return out


def consistency(hr_u8, lr_u8, scale):
    down = F.avg_pool2d(th.from_numpy(hr_u8.astype(np.float32))[:, None], scale)[:, 0].numpy()
    return float(((down - lr_u8.astype(np.float32)) ** 2).mean())


def patches(maps, size=256, stride=128):
    H, W = maps.shape[1:]
    return np.concatenate([maps[:, h:h + size, w:w + size]
                           for h in grid_starts(H, size, stride) for w in grid_starts(W, size, stride)])


class FID:
    def __init__(self, device):
        from pytorch_fid.inception import InceptionV3
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
        self.device = device

    @th.no_grad()
    def stats(self, u8):
        fs = []
        for i in range(0, len(u8), 50):
            x = th.from_numpy(u8[i:i + 50].astype(np.float32) / 255.0)[:, None].repeat(1, 3, 1, 1).to(self.device)
            fs.append(self.net(x)[0][:, :, 0, 0].cpu().numpy())
        f = np.concatenate(fs).astype(np.float64)
        return f.mean(0), np.cov(f, rowvar=False)


def frechet(mu1, s1, mu2, s2):
    from scipy import linalg
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        eps = np.eye(len(s1)) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean.real))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="runs/gdm_samples/samples.npz")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--ref_npz", default="", help="25 km maps for the FID reference set (default: the inputs)")
    p.add_argument("--ref_every", type=int, default=1)
    a = p.parse_args()

    d = np.load(a.samples)
    lr = d["lr"]
    methods = {**baselines(lr, a.scale), "IceDiff-GDM": d["hr"]}
    fid = FID(th.device("cuda" if th.cuda.is_available() else "cpu"))
    ref = np.load(a.ref_npz)["arr_0"][::a.ref_every] if a.ref_npz else lr
    real = fid.stats(patches(ref))

    rows = []
    for name, hr in methods.items():
        rows.append((name, consistency(hr, lr, a.scale), frechet(*real, *fid.stats(patches(hr)))))
        print(f"{name:12s} Consistency={rows[-1][1]:.3f}  FID={rows[-1][2]:.2f}")

    out = os.path.join(os.path.dirname(a.samples), "metrics.md")
    with open(out, "w") as fh:
        fh.write(f"{len(lr)} maps ({d['dates'][0]}..{d['dates'][-1]}), 25 km -> 6.25 km; "
                 f"FID reference: {len(ref)} maps of 25 km patches\n\n")
        fh.write("| Method | Consistency | FID |\n|---|---|---|\n")
        for name, c, f in rows:
            fh.write(f"| {name} | {c:.2f} | {f:.2f} |\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
