"""Figures for sample_gdm.py / eval_metrics.py output.

  compare_gray.png   single-channel SIC, 25 km input | nearest | bilinear | bicubic | IceDiff-GDM, with ice-edge zooms
  compare_color.png  same layout in the paper's style (blue-white SIC colormap, land in yellow, SIC colorbar)
  table.png          Consistency / FID table from metrics.md

Usage: python plot_results.py --samples runs/ddpm1000/samples.npz --ocean_mask ../data/ocean_mask.npy
"""
import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from eval_metrics import baselines

INK, MUTED, EDGE = "#1f2328", "#59636e", "#d1d9e0"
LAND = "#e3c35b"
ZOOM = 320


def edge_window(lr_map, size=ZOOM // 4):
    """Top-left (in 6.25 km pixels) of the window with the most 15-85 % SIC pixels in the 25 km map."""
    e = ((lr_map > 38) & (lr_map < 217)).astype(float)
    cs = np.pad(e.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    _, y, x = max((cs[y + size, x + size] - cs[y, x + size] - cs[y + size, x] + cs[y, x], y, x)
                  for y in range(0, e.shape[0] - size, 4) for x in range(0, e.shape[1] - size, 4))
    return y * 4, x * 4


def read_metrics(path):
    rows, head = [], None
    for line in open(path):
        if line.startswith("|") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if head is None:
                head = cells
            else:
                rows.append(cells)
    return head, rows


def fmt_date(d):
    d = str(d)
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def compare(path, columns, dates, idx, metrics, land=None, color=False):
    n = len(idx)
    fig, axes = plt.subplots(2 * n, len(columns), figsize=(2.9 * len(columns) + (0.6 if color else 0), 5.9 * n),
                             gridspec_kw={"height_ratios": [1.45, 1] * n})
    if color:
        cmap = plt.get_cmap("Blues_r")
    for r, k in enumerate(idx):
        y0, x0 = edge_window(columns[0][1][k])
        for c, (name, arr) in enumerate(columns):
            img = arr[k].astype(float) / 2.55
            f = 1 if c == 0 else 4
            crop = img[y0 // 4:(y0 + ZOOM) // 4, x0 // 4:(x0 + ZOOM) // 4] if c == 0 else img[y0:y0 + ZOOM, x0:x0 + ZOOM]
            for ax, im, is_zoom in ((axes[2 * r, c], img, False), (axes[2 * r + 1, c], crop, True)):
                if color:
                    m = land[0] if (c == 0) else land[1]
                    if is_zoom:
                        m = m[y0 // 4:(y0 + ZOOM) // 4, x0 // 4:(x0 + ZOOM) // 4] if c == 0 else m[y0:y0 + ZOOM, x0:x0 + ZOOM]
                    h = ax.imshow(np.ma.masked_where(m, im), cmap=cmap, vmin=0, vmax=100, interpolation="nearest")
                    ax.imshow(np.ma.masked_where(~m, np.ones_like(im)), cmap=ListedColormap([LAND]), interpolation="nearest")
                else:
                    h = ax.imshow(im, cmap="gray", vmin=0, vmax=100, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_edgecolor(EDGE)
            axes[2 * r, c].add_patch(Rectangle((x0 / 4 * f - .5, y0 / 4 * f - .5), ZOOM / 4 * f, ZOOM / 4 * f,
                                               fill=False, ec="#d1242f", lw=1.2))
            if r == 0:
                title = name
                if name in metrics:
                    title += f"\nConst. {metrics[name][0]}  |  FID {metrics[name][1]}"
                axes[0, c].set_title(title, fontsize=10.5, color=INK)
        axes[2 * r, 0].set_ylabel(fmt_date(dates[k]), fontsize=11, color=INK)
        axes[2 * r + 1, 0].set_ylabel("ice-edge zoom", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 0.93 if color else 1, 1))
    if color:
        cax = fig.add_axes([0.945, 0.3, 0.012, 0.4])
        cb = fig.colorbar(h, cax=cax)
        cb.set_label("SIC (%)", color=INK)
        cb.outline.set_edgecolor(EDGE)
        cb.ax.tick_params(colors=MUTED)
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def table(path, head, rows, caption):
    best = {j: min(float(r[j]) for r in rows) for j in (1, 2)}
    fig, ax = plt.subplots(figsize=(5.6, 0.36 * (len(rows) + 1)))
    ax.axis("off")
    cells = [[r[0].capitalize() if r[0] != "IceDiff-GDM" else r[0]] + r[1:] for r in rows]
    t = ax.table(cellText=cells, colLabels=["Method", "Consistency ↓", "FID ↓"], bbox=[0, 0, 1, 1], cellLoc="center",
                 colWidths=[0.42, 0.29, 0.29])
    t.auto_set_font_size(False); t.set_fontsize(11)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor(EDGE); cell.set_linewidth(0.6); cell.get_text().set_color(INK)
        if i == 0:
            cell.set_facecolor("#f6f8fa"); cell.get_text().set_weight("bold")
        elif j > 0 and float(rows[i - 1][j]) == best[j]:
            cell.get_text().set_weight("bold")
    ax.set_title("25 km → 6.25 km down-scaling", fontsize=12, color=INK, pad=8)
    ax.text(0, -0.08, caption, transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=MUTED, wrap=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="runs/ddpm1000/samples.npz")
    p.add_argument("--ocean_mask", default="../data/ocean_mask.npy")
    p.add_argument("--dates", default="20170101,20170703,20220105,20220707")
    p.add_argument("--scale", type=int, default=4)
    a = p.parse_args()

    d = np.load(a.samples)
    lr, hr, dates = d["lr"], d["hr"], d["dates"]
    want = [int(x) for x in a.dates.split(",")]
    idx = [int(np.where(dates == w)[0][0]) for w in want if w in dates] or list(range(min(4, len(dates))))
    b = baselines(lr, a.scale)
    columns = [("25 km input (guidance y)", lr), ("Nearest", b["nearest"]), ("Bilinear", b["bilinear"]),
               ("Bicubic", b["bicubic"]), ("IceDiff-GDM", hr)]
    out = os.path.dirname(a.samples)
    head, rows = read_metrics(os.path.join(out, "metrics.md"))
    names = {"nearest": "Nearest", "bilinear": "Bilinear", "bicubic": "Bicubic", "IceDiff-GDM": "IceDiff-GDM"}
    metrics = {names[r[0]]: (r[1], r[2]) for r in rows}

    compare(os.path.join(out, "compare_gray.png"), columns, dates, idx, metrics)
    land_lr = ~np.load(a.ocean_mask)
    land = (land_lr, np.kron(land_lr, np.ones((a.scale, a.scale), dtype=bool)))
    compare(os.path.join(out, "compare_color.png"), columns, dates, idx, metrics, land=land, color=True)
    caption = (f"{len(dates)} test maps ({', '.join(fmt_date(x) for x in sorted(dates))}). Consistency: MSE between the "
               "25 km input and the output passed through the convolution function D (per-patch IceDiff-GDM kernels, "
               "then 4×4 average pooling; 0–255). FID: 256×256 patches vs. 25 km test patches.")
    table(os.path.join(out, "table.png"), head, rows, caption)
    print("wrote", ", ".join(os.path.join(out, f) for f in ("compare_gray.png", "compare_color.png", "table.png")))


if __name__ == "__main__":
    main()
