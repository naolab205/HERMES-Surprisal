#!/usr/bin/env bash
# Run from /data/2_readEEG/KBS_Surprisal_CAS_Submission_20260929/model_code.
# This file records exact matched diagnostic commands; archived results are separate.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
PYTHON=/home/zcg/.conda/envs/cogalign/bin/python
EPOCHS=/data/2_readEEG/superisal_code/data_cache/char_epochs_1200ms
OLD_EYE=/data/2_readEEG/superisal_code/data_cache/eye_char
NEW_EYE=/data/2_readEEG/KBS_Surprisal_CAS_Submission_20260929/model_code/data_cache/eye_char_1200ms_20261008
OUTPUT=/data/2_readEEG/KBS_Surprisal_CAS_Submission_20260929/model_code/results/window_alignment_20261008
"$PYTHON" scripts/build_eye_1200_window_probe.py --out-dir "$NEW_EYE" --workers 4
"$PYTHON" scripts/run_window_alignment_cv.py --eeg-post-ms 800 --eye-post-ms 800 --eye-cache "$OLD_EYE" --arm window_E800_G800 --arch mmhier --seq-block modern --csd --cache-dir "$EPOCHS" --eye-stats-scope train --folds 0 1 2 3 4 5 6 --eval-split val --matched-train-rng --device cuda:0 --out-dir "$OUTPUT/E800_G800"
"$PYTHON" scripts/run_window_alignment_cv.py --eeg-post-ms 1200 --eye-post-ms 800 --eye-cache "$OLD_EYE" --arm window_E1200_G800 --arch mmhier --seq-block modern --csd --cache-dir "$EPOCHS" --eye-stats-scope train --folds 0 1 2 3 4 5 6 --eval-split val --matched-train-rng --device cuda:1 --out-dir "$OUTPUT/E1200_G800"
"$PYTHON" scripts/run_window_alignment_cv.py --eeg-post-ms 1200 --eye-post-ms 1200 --eye-cache "$NEW_EYE" --arm window_E1200_G1200 --arch mmhier --seq-block modern --csd --cache-dir "$EPOCHS" --eye-stats-scope train --folds 0 1 2 3 4 5 6 --eval-split val --matched-train-rng --device cuda:2 --out-dir "$OUTPUT/E1200_G1200"
"$PYTHON" scripts/audit_eye_window_timestamps.py
"$PYTHON" scripts/summarize_window_alignment.py
"$PYTHON" scripts/plot_window_alignment.py
