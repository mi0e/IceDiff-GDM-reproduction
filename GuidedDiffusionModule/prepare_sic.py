"""Convert NSIDC G02202 V6 yearly aggregates into uint8 SIC stacks for IceDiff-GDM.

Output (per split): sic_<split>.npz with
  arr_0: uint8 [N, 448, 304]  SIC in 0..255 (0..100% mapped linearly; land/coast/lake -> 0)
  dates: int32 [N] YYYYMMDD
"""
import argparse
import glob
import os
from datetime import date, timedelta

import netCDF4 as nc
import numpy as np

SPLITS = {"train": (1978, 2013), "val": (2014, 2015), "test": (2016, 2023)}


def load_year(path, ocean, max_missing):
    ds = nc.Dataset(path)
    v = ds.variables["cdr_seaice_conc"]
    v.set_auto_maskandscale(False)
    raw = v[:]
    days = ds.variables["time"][:].astype(int)
    ds.close()

    missing = (raw > 100) & ocean
    frac = missing.reshape(len(raw), -1).sum(1) / ocean.sum()
    keep = frac <= max_missing

    sic = np.where(raw <= 100, raw, 0).astype(np.float32) / 100.0
    sic = np.round(sic * 255).astype(np.uint8)
    dates = np.array(
        [int((date(1970, 1, 1) + timedelta(days=int(d))).strftime("%Y%m%d")) for d in days],
        dtype=np.int32,
    )
    return sic[keep], dates[keep], int((~keep).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="/home/icediff/data/raw/aggregate")
    p.add_argument("--ancillary", default="/home/icediff/data/raw/G02202-ancillary-psn25-v06r00.nc")
    p.add_argument("--out_dir", default="/home/icediff/data/gdm")
    p.add_argument("--max_missing", type=float, default=0.01,
                   help="drop days whose fraction of missing ocean pixels exceeds this")
    args = p.parse_args()

    anc = nc.Dataset(args.ancillary)
    st = anc.variables["surface_type"]
    st.set_auto_maskandscale(False)
    ocean = st[:] == 50
    anc.close()

    files = sorted(glob.glob(os.path.join(args.raw_dir, "sic_psn25_*-*_v06r00.nc")))
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "ocean_mask.npy"), ocean)

    for split, (y0, y1) in SPLITS.items():
        sics, dates, dropped = [], [], 0
        for f in files:
            year = int(os.path.basename(f).split("_")[2][:4])
            if y0 <= year <= y1:
                s, d, n = load_year(f, ocean, args.max_missing)
                sics.append(s)
                dates.append(d)
                dropped += n
        if not sics:
            print(f"{split}: no files found, skipped")
            continue
        sic = np.concatenate(sics)
        dt = np.concatenate(dates)
        out = os.path.join(args.out_dir, f"sic_{split}.npz")
        np.savez(out, arr_0=sic, dates=dt)
        print(f"{split}: {len(sic)} days ({dt[0]}..{dt[-1]}), dropped {dropped} -> {out}")


if __name__ == "__main__":
    main()
