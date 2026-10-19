#!/bin/bash
# Boot-time entry point for a Jupyter-only AMD box with a persistent /mnt/workspace.
# Usage:  bash boot.sh REMAINING_HOURS [BATCH]
#   REMAINING_HOURS = time left on the machine (e.g. 7.1). Training stops 0.8 h before that.
# Re-running is safe: data is fetched only if missing/invalid, training resumes from the last checkpoint.
set -u
REMAIN=${1:?usage: bash boot.sh REMAINING_HOURS [BATCH]}
BS=${2:-32}
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${DATA:-/mnt/workspace/nsidc_ms}
MS_REPO=${MS_REPO:-kiiisara/nsidc_1978-2023}
RESERVE=${RESERVE:-0.8}

PY=""
for p in /opt/venv/bin/python python3 python; do
  command -v $p >/dev/null 2>&1 && $p -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && { PY=$(command -v $p); break; }
done
[ -z "$PY" ] && { echo "no python with a working GPU torch"; exit 1; }

if pgrep -f "python[0-9.]* train_gdm.py" >/dev/null; then
  echo "training already running:"; pgrep -af "python[0-9.]* train_gdm.py" | head -1; exit 0
fi

valid() {
  $PY - "$1" <<'EOF' 2>/dev/null
import sys, numpy as np
d = np.load(sys.argv[1])
assert d["arr_0"].shape[1:] == (448, 304) and len(d["arr_0"]) > 1000
EOF
}
mask_ok() { [ "$(stat -c%s "$1" 2>/dev/null)" = "136320" ]; }

fetch_modelscope() {
  echo "$(date -u) downloading $MS_REPO from ModelScope"
  $PY -c "import modelscope" 2>/dev/null || $PY -m pip install -q modelscope
  $PY - "$MS_REPO" "$DATA" <<'EOF'
import os, sys
from modelscope.hub.snapshot_download import snapshot_download
if os.environ.get("MODELSCOPE_TOKEN"):
    from modelscope.hub.api import HubApi
    HubApi().login(os.environ["MODELSCOPE_TOKEN"])
snapshot_download(sys.argv[1], repo_type="dataset", local_dir=sys.argv[2])
EOF
}

fetch_nsidc() {
  echo "$(date -u) building dataset from NSIDC G02202 V6"
  $PY -c "import netCDF4" 2>/dev/null || $PY -m pip install -q netCDF4
  RAW=${RAW:-/root/raw}; mkdir -p $RAW/agg
  U=https://noaadata.apps.nsidc.org/NOAA/G02202_V6/north
  curl -s $U/aggregate/ | grep -oE 'sic_psn25_(19[0-9]{2}|20(0[0-9]|1[0-5]))[0-9]{4}-[0-9]{8}_v06r00\.nc' | sort -u > $RAW/list.txt
  for attempt in 1 2 3; do
    (cd $RAW/agg && grep -vxFf <(ls) $RAW/list.txt | xargs -r -P 8 -I{} curl -sfLO $U/aggregate/{})
  done
  curl -sfL -o $RAW/anc.nc https://noaadata.apps.nsidc.org/NOAA/G02202_V6/ancillary/G02202-ancillary-psn25-v06r00.nc
  echo "$(ls $RAW/agg/*.nc | wc -l) / $(wc -l < $RAW/list.txt) files"
  (cd $HERE/GuidedDiffusionModule && $PY prepare_sic.py --raw_dir $RAW/agg --ancillary $RAW/anc.nc --out_dir $DATA)
}

mkdir -p $DATA
if ! valid $DATA/sic_train.npz || ! mask_ok $DATA/ocean_mask.npy; then
  fetch_modelscope || true
  if ! valid $DATA/sic_train.npz || ! mask_ok $DATA/ocean_mask.npy; then
    echo "ModelScope copy missing or LFS pointer only, falling back to NSIDC"
    fetch_nsidc
  fi
fi
valid $DATA/sic_train.npz && mask_ok $DATA/ocean_mask.npy || { echo "data still invalid in $DATA"; exit 1; }
echo "data ok: $DATA"

HOURS=$(awk -v r="$REMAIN" -v s="$RESERVE" 'BEGIN{h=r-s; if (h<0.2) h=0; printf "%.2f", h}')
[ "$HOURS" = "0.00" ] && { echo "not enough time left"; exit 1; }
rm -f $HERE/runs/gdm_native/*.tmp
last=$(ls -1 $HERE/runs/gdm_native/model[0-9]*.pt 2>/dev/null | tail -1)
echo "resume from: ${last:-scratch}; training $HOURS h, batch $BS"
DATA=$DATA nohup bash $HERE/start_train.sh $HOURS $BS > $HERE/train.log 2>&1 < /dev/null &
echo "started (pid $!). progress: tail -12 $HERE/runs/gdm_native/log.txt"
