#!/bin/bash
# One-shot GDM training for a Jupyter-only AMD box.
# Usage (from the folder this file is in):  nohup bash start_train.sh HOURS [BATCH] > train.log 2>&1 &
# HOURS = training time; leave ~45 min of the machine's remaining time for packing and download.
set -u
HOURS=${1:?usage: bash start_train.sh HOURS [BATCH]}
BS=${2:-32}
HERE=$(cd "$(dirname "$0")" && pwd)
CODE=$HERE/GuidedDiffusionModule
RUN=$HERE/runs/gdm_native
OUT=$HERE/download

PY=""
for p in /opt/venv/bin/python python3 python; do
  if command -v $p >/dev/null 2>&1 && $p -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then PY=$(command -v $p); break; fi
done
[ -z "$PY" ] && { echo "no python with a working GPU torch found"; exit 1; }
$PY -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1e9), 'GB')"

find_data() {
  for d in "$HERE" /workspace /root /home /data /mnt; do
    f=$(find "$d" -maxdepth 4 -name "$1" -size +1k 2>/dev/null | head -1)
    [ -n "$f" ] && { echo "$f"; return; }
  done
}
TRAIN=$(find_data sic_train.npz); MASK=$(find_data ocean_mask.npy); VAL=$(find_data sic_val.npz)
if [ -n "${DATA:-}" ]; then
  TRAIN=$DATA/sic_train.npz; MASK=$DATA/ocean_mask.npy; VAL=$(ls $DATA/sic_val.npz 2>/dev/null)
fi
[ -z "$TRAIN" ] && { echo "sic_train.npz not found"; exit 1; }
[ -z "$MASK" ] && { echo "ocean_mask.npy not found (required)"; exit 1; }
echo "train=$TRAIN mask=$MASK val=${VAL:-none}"

$PY -c "import numpy, PIL" 2>/dev/null || $PY -m pip install -q numpy pillow

DEADLINE=$(( $(date +%s) + ${HOURS%.*} * 3600 + $(echo "$HOURS" | awk '{printf "%d", ($1-int($1))*3600}') ))
echo "$(date -u) training until $(date -u -d @$DEADLINE), batch $BS"
mkdir -p $RUN $OUT
cd $CODE
$PY train_gdm.py \
  --num_channels 128 --num_res_blocks 2 --attention_resolutions 32,16,8 --num_head_channels 64 \
  --resblock_updown True --use_scale_shift_norm True --learn_sigma True --image_size 256 \
  --use_fp16 True --batch_size $BS --lr 1e-4 --ema_rate 0.9995 --upscale 1 \
  --train_npz $TRAIN --val_npz ${VAL:-none} --ocean_mask $MASK --log_dir $RUN \
  --deadline $DEADLINE --log_interval 200 --save_interval 2000 --num_preview 4 --sample_steps 50 \
  --num_workers 8

s=$($PY - "$RUN" <<'PYEOF'
import glob, sys, torch as th
for p in sorted(glob.glob(sys.argv[1] + "/model[0-9]*.pt"), reverse=True):
    try:
        th.load(p, map_location="cpu"); print(p[-9:-3]); break
    except Exception:
        pass
PYEOF
)
[ -z "$s" ] && { echo "no checkpoint saved"; exit 1; }
echo "$(date -u) training stopped at step $s, packing"
rm -f $OUT/resume_*.tar $OUT/fp16_*.pt $OUT/preview_*.png
$PY - <<EOF
import torch as th
for name in ("model$s.pt", "ema_0.9995_$s.pt"):
    sd = th.load("$RUN/" + name, map_location="cpu")
    th.save({k: v.half() if v.is_floating_point() else v for k, v in sd.items()}, "$OUT/fp16_" + name)
EOF
cp $RUN/log.txt $RUN/progress.csv $OUT/
cp $(ls -1 $RUN/preview_*.png | tail -2) $OUT/ 2>/dev/null
ls -1 $RUN/model[0-9]*.pt | head -n -3 | sed -E "s/.*model([0-9]+)\.pt/\1/" | while read k; do rm -f $RUN/model$k.pt $RUN/opt$k.pt $RUN/ema_0.9995_$k.pt; done
tar cf $OUT/resume_$s.tar -C $RUN model$s.pt ema_0.9995_$s.pt opt$s.pt log.txt progress.csv
(cd $OUT && md5sum * > MD5SUMS)
echo "$(date -u) DONE step=$s -> $OUT"
ls -la $OUT
