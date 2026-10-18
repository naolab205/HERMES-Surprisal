#!/usr/bin/env bash
set -euo pipefail
arm="$1"
gpu="$2"
cache="$PWD/data_cache/char_epochs_fixed_grid_350ms_20261006"
while [ ! -f "$cache/CACHE_AUDIT.json" ]; do sleep 10; done
entry_extra=()
case "$arm" in
  fixed_base) entry_extra=(--no-deconv) ;;
  fixed_aligned) ;;
  fixed_noise) entry_extra=(--fixed-clock-noise) ;;
  fixed_generic) entry_extra=(--generic-neighbor-control) ;;
  *) exit 2 ;;
esac
SURPRISAL_EEG_CACHE="$cache" /home/zcg/.conda/envs/cogalign/bin/python scripts/run_fixed_grid_cv.py \
  --arch deconv --csd --seq-block modern --batch-size 64 --epochs 25 --patience 6 \
  --seed 2026 --matched-train-rng --eval-split val --valid-slot-norm --save-checkpoint \
  --split-manifest results/response_mechanism_20261002/sentence_manifest.csv \
  --folds 0 3 --cache-dir "$cache" --arm "$arm" --device "cuda:$gpu" \
  --out-dir "results/single_model_breakthrough_20261006/physical_screen/$arm" "${entry_extra[@]}"
