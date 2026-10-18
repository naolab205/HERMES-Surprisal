#!/usr/bin/env bash
set -euo pipefail
physical_cache="/data/2_readEEG/KBS_Surprisal_CAS_Submission_20260929/model_code/data_cache/char_epochs_physical_1200ms_20261006"
while [ ! -f "$physical_cache/CACHE_AUDIT.json" ]; do sleep 10; done
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
SURPRISAL_EEG_CACHE="$physical_cache" \
SURPRISAL_EYE_CACHE=/data/2_readEEG/superisal_code/data_cache/eye_char \
/home/zcg/.conda/envs/cogalign/bin/python scripts/run_physical_encoding.py --permutations 199
