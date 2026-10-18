# Reproducibility and protocol boundaries

## What you can verify now

The public package includes the primary 102,499-reading prediction table, seven-fold main/ablation/reference tables, numerical frozen labels, the spatial matrix, matched eye-control predictions, observation-window validation predictions, and small diagnostic records. You can recompute primary metrics and draw the README comparison without downloading EEG.

```bash
python scripts/evaluate_surprisal_predictions.py results/tables/charbag_MM_mid_predictions.csv --out-dir outputs/primary_metrics
python scripts/plot_release_overview.py
```

Tests can be run with `python -m pip install pytest` followed by `python -m pytest tests/test_combined_model.py`. The checks concern train-only eye statistics and the zero-initialized residual; passing them does not validate predictive generalization.

## What the main result means

The main `MM_mid` run used seven contiguous item blocks per book. Fold k evaluates block k, selects the checkpoint on block k+1 modulo 7, and trains on the remaining five blocks. Readers recur. The model family was explored using these same blocks. Targets are frozen mean token surprisal under `uer/gpt2-chinese-cluecorpussmall`, with up to 192 preceding tokens.

The main archived model used global unlabeled eye statistics. Training-only normalization is the current runner default and has a separate seven-fold result. Window positions were estimated from row timing, not observed character triggers. Slot masks expose count, and absolute gaze can expose screen layout. Neither fixed-width windows nor a lack of explicit text input eliminates these cues.

The exact multimodal launch options and trained weights were not fully retained. Recomputed table metrics are reproducible from predictions; an exact end-to-end retraining of those archived weights is not claimed.

## Signal caches

Training expects, for each available subject/book group:

- EEG: `X_<subject>_<book>.npy` and `meta_<subject>_<book>.csv`, with epochs shaped `[N,128,167]` for the nominal 1200-ms post-reference window.
- A fixed `csd_matrix.npy` in the selected EEG cache directory when `--csd` is enabled. The archived matrix is included under `data_cache/char_epochs_1200ms/`.
- Eye: corresponding arrays and metadata under `SURPRISAL_EYE_CACHE`, with four channels and 45 samples per epoch for the nominal 800-ms post-reference endpoint. Loading matches observation/slot keys, rather than assuming EEG and eye array indices are equal.
- Frozen target values from `data_cache/surprisal_1829a99ad4.csv`.

The underlying source is the authors' `filtered_0.5_30` derivative in ChineseEEG. `CHINESEEEG_ROOT` configures several source readers. Signal caches are large and must be obtained or rebuilt locally. Merely setting the environment variables does not create them.

### Important builder status

The old README once recommended `build_long_window_cache.py`. That script now deliberately stops because its concatenated row timeline cannot supply trustworthy long physical-time windows. Do not remove this guard to obtain a benchmark score.

`scripts/rebuild_physical_char_cache.py` rebuilds actual continuous-recording windows and records provenance. It expects the original epoch-order metadata, prior arrays for interior-match checks, and the continuous event cache. Its `DATA`, `OLD`, and `CONT` paths must be adapted to local source locations before use. These are research utilities with historical cache dependencies, not a one-command raw-data installer. `build_fixed_grid_cache.py` additionally changes the sampling convention and therefore defines a separate experiment. Their inputs and results must not be substituted for the archived cache while claiming exact reproduction of the main score.

The row/character/eye source builders are retained in `scripts/` for inspection. A new preprocessing installation should verify observation keys, physical intervals, boundary handling, and target matching before training. The package does not claim that a fresh raw-data preprocessing run reproduces the archived 0.405 score.

## Evidence map

| Question | Records |
| --- | --- |
| Main score and primary ablations | `results/tables/` |
| Additional ranking/agreement metrics | `results/diagnostics/` |
| Training-only fusion and eye regression | `results/eye_clean_decoder_20260930/` |
| Window endpoints | `results/window_alignment_20261008/` |
| Response-mechanism screening | `results/response_mechanism_20261002/` |
| Continuous-source, noise, and fixed-grid controls | `results/single_model_breakthrough_20261006/physical_screen/` |
| Scientific scalp plotting sources | `analysis/` |

Small fold tables and records are retained for unsuccessful candidates as well. Large experiment arrays and checkpoints are excluded, so this is a documented code/evidence release rather than a complete byte-for-byte archive of every experiment. Historical reports may refer to additional local files that are not distributed.
