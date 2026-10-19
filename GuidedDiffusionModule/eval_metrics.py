"""Metrics for eval_gdm.py outputs (definitions in METRICS.md).

For every <tag>__<cfg>.npz in --out_dir plus nearest / bilinear / bicubic baselines:
  RMSE, MAE (SIC %, ocean pixels), SSIM (ocean-averaged map), IIEE (15% edge, 10^3 km^2),
  Consistency (MSE of 4x4 avg-pooled output vs 100 km input; 0-255 and [0,1] scales, all / ocean-only
  blocks), FID (256x256 patches, needs `pip install pytorch-fid`).
Writes metrics.md, metrics.csv and compare.png into --out_dir.
"""
import argparse
import glob
import os

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image

PIX_KM2 = 25.0 * 25.0


def ssim_map(x, y, win=11, sigma=1.5):
    """x, y: [N,1,H,W] in [0,1]; Gaussian-window SSIM map (Wang et al. 2004)."""
    g = th.exp(-((th.arange(win) - win // 2) ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())[None, None]
    blur = lambda z: F.conv2d(F.conv2d(z, g[..., None, :], padding=(0, win // 2)), g[..., :, None], padding=(win // 2, 0))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
    return ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))


def baselines(lr, ocean, scale):
    y = th.from_numpy(lr)[:, None]
    out = {}
    for mode in ("nearest", "bilinear", "bicubic"):
        kw = {} if mode == "nearest" else {"align_corners": False}
        up = F.interpolate(y, scale_factor=scale, mode=mode, **kw)[:, 0].clamp(0, 1)
        up = (up * 255).round().to(th.uint8).numpy()
        up[:, ~ocean] = 0
        out[mode] = up
    return out


def patches(maps, size=256, stride=128):
    from sample_gdm import grid_starts
    H, W = maps.shape[1:]
    ps = [maps[:, h:h + size, w:w + size] for h in grid_starts(H, size, stride) for w in grid_starts(W, size, stride)]
    return np.concatenate(ps)


class FID:
    def __init__(self, device):
        from pytorch_fid.inception import InceptionV3
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
        self.device = device

    @th.no_grad()
    def feats(self, u8):
        fs = []
        for i in range(0, len(u8), 50):
            x = th.from_numpy(u8[i:i + 50].astype(np.float32) / 255.0)[:, None].repeat(1, 3, 1, 1).to(self.device)
            fs.append(self.net(x)[0][:, :, 0, 0].cpu().numpy())
        f = np.concatenate(fs).astype(np.float64)
        return f.mean(0), np.cov(f, rowvar=False)

    def __call__(self, real_stats, fake_u8):
        return frechet(*real_stats, *self.feats(fake_u8))


def frechet(mu1, s1, mu2, s2):
    from scipy import linalg
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        eps = np.eye(len(s1)) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean.real))


def evaluate(sr, hr, lr, ocean, scale):
    o = ocean[None].repeat(len(hr), 0)
    s, h = sr.astype(np.float32) / 255.0, hr.astype(np.float32) / 255.0
    err = (s - h)[o]
    sm = ssim_map(th.from_numpy(s)[:, None], th.from_numpy(h)[:, None])[:, 0].numpy()
    iiee = ((s >= 0.15) != (h >= 0.15)) & o
    down = F.avg_pool2d(th.from_numpy(s)[:, None], scale)[:, 0].numpy()
    lr_ocean = F.avg_pool2d(th.from_numpy(ocean.astype(np.float32))[None, None], scale)[0, 0].numpy() == 1
    cons = (down - lr) ** 2
    return {
        "RMSE%": 100 * np.sqrt((err ** 2).mean()),
        "MAE%": 100 * np.abs(err).mean(),
        "SSIM": sm[o].mean(),
        "IIEE_1e3km2": iiee.sum((1, 2)).mean() * PIX_KM2 / 1e3,
        "Cons255": (cons * 255 ** 2).mean(),
        "Cons255_ocean": (cons[:, lr_ocean] * 255 ** 2).mean(),
        "Cons01": cons.mean(),
    }


def compare_png(path, rows, dates, n_show=4):
    """Rows of [LR nearest | methods... | truth] for a few days, separated by 4 px gaps."""
    names = list(rows)
    idx = np.linspace(0, len(dates) - 1, min(n_show, len(dates))).astype(int)
    H, W = rows[names[0]].shape[1:]
    canvas = np.full((len(idx) * (H + 4) + 20, len(names) * (W + 4)), 128, np.uint8)
    for r, n in enumerate(idx):
        for c, k in enumerate(names):
            canvas[20 + r * (H + 4):20 + r * (H + 4) + H, c * (W + 4):c * (W + 4) + W] = rows[k][n]
    img = Image.fromarray(canvas).convert("RGB")
    try:
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        for c, k in enumerate(names):
            d.text((c * (W + 4) + 4, 4), k[:40], fill=(255, 0, 0))
    except Exception:
        pass
    img.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="../eval_out")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--no_fid", action="store_true")
    a = p.parse_args()

    t = np.load(os.path.join(a.out_dir, "truth.npz"))
    hr, lr, ocean, dates = t["hr"], t["lr"], t["ocean"], t["dates"]
    methods = baselines(lr, ocean, a.scale)
    for f in sorted(glob.glob(os.path.join(a.out_dir, "*__*.npz"))):
        sr = np.load(f)["sr"]
        if len(sr) == len(hr):
            methods[os.path.basename(f)[:-4]] = sr
        else:
            print("skip incomplete", f)

    fid = None
    if not a.no_fid:
        try:
            fid = FID(th.device("cuda" if th.cuda.is_available() else "cpu"))
            real_stats = fid.feats(patches(hr))
        except ImportError:
            print("pytorch-fid not installed; FID skipped (pip install pytorch-fid)")
            fid = None

    rows = []
    for name, sr in methods.items():
        m = evaluate(sr, hr, lr, ocean, a.scale)
        if fid is not None:
            m["FID"] = fid(real_stats, patches(sr))
        rows.append((name, m))
        print(name, {k: round(float(v), 4) for k, v in m.items()})

    keys = list(rows[0][1])
    n_patch = len(patches(hr[:1])) * len(hr)
    with open(os.path.join(a.out_dir, "metrics.csv"), "w") as fh:
        fh.write("method," + ",".join(keys) + "\n")
        for name, m in rows:
            fh.write(name + "," + ",".join(f"{float(m[k]):.6g}" for k in keys) + "\n")
    with open(os.path.join(a.out_dir, "metrics.md"), "w") as fh:
        fh.write(f"{len(hr)} maps ({dates[0]}..{dates[-1]}), 100 km -> 25 km, ocean pixels; "
                 f"FID on {n_patch} patches of 256x256 (few patches -> FID biased high, compare relatively)\n\n")
        fh.write("| method | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 1) + "\n")
        for name, m in rows:
            fh.write(f"| {name} | " + " | ".join(f"{float(m[k]):.4g}" for k in keys) + " |\n")
    vis = {"lr(nearest)": methods["nearest"], **{k: v for k, v in methods.items() if k != "nearest"}, "truth": hr}
    compare_png(os.path.join(a.out_dir, "compare.png"), vis, dates)
    print("wrote", os.path.join(a.out_dir, "metrics.md"))


if __name__ == "__main__":
    main()
