#!/usr/bin/env bash
set -euo pipefail
experiment_arm="$1"
experiment_gpu="$2"
physical_cache="/data/2_readEEG/KBS_Surprisal_CAS_Submission_20260929/model_code/data_cache/char_epochs_physical_1200ms_20261006"
while [ ! -f "$physical_cache/CACHE_AUDIT.json" ]; do
  sleep 10
done
/home/zcg/.conda/envs/cogalign/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('data_cache/char_epochs_physical_1200ms_20261006')
r=json.loads((p/'CACHE_AUDIT.json').read_text())
assert r['status']=='complete' and r['n_epochs']==917096 and len(r['recordings'])==19
assert all(i['interior_drift']==0 and i['status']=='complete' for i in r['recordings'])
assert not list(p.glob('*.partial.npy'))
PY
experiment_entry="scripts/run_charbag_cv.py"
experiment_extra=()
case "$experiment_arm" in
  phys_control) ;;
  phys_valid) experiment_extra=(--valid-slot-norm) ;;
  phys_shared) experiment_entry="scripts/run_shared_reference_cv.py" ;;
  *) exit 2 ;;
esac
SURPRISAL_EEG_CACHE="$physical_cache" /home/zcg/.conda/envs/cogalign/bin/python "$experiment_entry" \
  --arch deconv --no-deconv --csd --seq-block modern --batch-size 64 \
  --epochs 25 --patience 6 --seed 2026 --matched-train-rng --eval-split val \
  --split-manifest results/response_mechanism_20261002/sentence_manifest.csv \
  --folds 0 3 --save-checkpoint --cache-dir "$physical_cache" \
  "${experiment_extra[@]}" --arm "$experiment_arm" --device "cuda:$experiment_gpu" \
  --out-dir "results/single_model_breakthrough_20261006/physical_screen/$experiment_arm"
