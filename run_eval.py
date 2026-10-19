"""One-command GDM evaluation. Usage:
  python run_eval.py tune    # val set, 12 maps, 4 guidance settings, newest model only
  python run_eval.py final --configs lr:8   # test set, ~98 maps, every model in models/
  python run_eval.py metrics --out eval_tune # recompute metrics only
Extra arguments after the mode are passed to eval_gdm.py (e.g. --timestep_respacing 250);
--no_fid skips FID.
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "GuidedDiffusionModule")
DATA = os.path.join(HERE, "data")


def run(script, *args):
    cmd = [sys.executable, os.path.join(CODE, script), *args]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=CODE, check=True)


def main():
    mode, extra = sys.argv[1], sys.argv[2:]
    fid = ["--no_fid"] if "--no_fid" in extra else []
    extra = [x for x in extra if x != "--no_fid"]
    if mode == "metrics":
        run("eval_metrics.py", "--out_dir", os.path.join(HERE, extra[extra.index("--out") + 1]), *fid)
        return
    models = sorted(glob.glob(os.path.join(HERE, "models", "*.pt")))
    assert models, "put EMA weights (*.pt) into models/"
    if mode == "tune":
        models = models[-1:]
        base = ["--npz", os.path.join(DATA, "sic_val.npz"), "--every", "61", "--configs", "hr:0.5,lr:2,lr:8,lr:32"]
        out = os.path.join(HERE, "eval_tune")
    elif mode == "final":
        base = ["--npz", os.path.join(DATA, "sic_test.npz"), "--every", "30", "--configs", "lr:8"]
        out = os.path.join(HERE, "eval_final")
    else:
        sys.exit(__doc__)
    tags = [os.path.splitext(os.path.basename(m))[0] for m in models]
    run("eval_gdm.py", *base, "--ocean_mask", os.path.join(DATA, "ocean_mask.npy"), "--out_dir", out,
        "--model_paths", ",".join(models), "--tags", ",".join(tags), *extra)
    run("eval_metrics.py", "--out_dir", out, *fid)


if __name__ == "__main__":
    main()
