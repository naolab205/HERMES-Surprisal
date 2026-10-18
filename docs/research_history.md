# Character-aligned surprisal decoder

This directory is the reorganized combined-model code from the foundation worktree now present at `/data/2_readEEG/superisal_code` (previously referred to as `JBHI_Foundation_v2`). The paper's primary arm is `MM_mid`: character-locked EEG plus eye epochs, surface Laplacian, separate temporal encoders, a zero-initialized eye residual, a cross-character sequence block, and one row-surprisal regression head. `src/foundation/hiernet.py` contains the architecture; `scripts/run_charbag_cv.py` contains training, validation selection, and metric computation. `src/cogalign/` contains only the event, configuration, and item-split helpers required by this code. The old row-level source remains in `/data/2_readEEG/JBHI_Surprisal_Package_original_20260929`.

## Archived evidence and its boundary

- `results/tables/charbag_MM_mid_{summary.json,per_fold.csv,predictions.csv}` and `results/logs/MM_mid.log` are copied from the foundation run. The seven-fold mean scores are single-reading Spearman 0.3542, item Spearman 0.4049, and high/low item AUC 0.7476. The matched EEG-only ablation is `ABL_noeye`.
- `surprisal_cv_full_*.csv` is the historical frozen row-level EEG comparator, not a run of the new architecture.
- The foundation family was developed against the same item-block folds. The test tables are exploratory evidence, not a fresh sealed outer evaluation. Readers overlap across train and test folds.
- The archived `MM_mid` eye encoder used *global unlabeled* valid-sample means and standard deviations. `--eye-stats-scope global` reproduces that behavior. The default `train` computes statistics separately on the training observations of each fold. The new seven-fold train-only result is in `results/eye_clean_decoder_20260930/` (item rho 0.3974 versus 0.3988 for EEG-only); it must not be merged with the archived global-statistics result.
- The valid-character mask exposes the number of highlighted characters. The architecture is not fully length-blind. Absolute gaze coordinates can also carry layout information.
- Large epoch caches and model checkpoints are absent. The exact original launch command was not saved; the command below is a reconstruction from source defaults, cache audits, and run logs. The EEG-only arm has now reproduced the archived table exactly, although the archived multimodal training configuration retains its separate global Eye-normalization policy.
- A fresh seven-fold EEG-only run using the command below reproduced the archived `ABL_noeye` metrics exactly in every fold; see `results/eye_clean_decoder_20260930/paired_summary.json`.

## Data and environment

Use the public ChineseEEG / OpenNeuro `ds004952` data and the supplied frozen label table `data_cache/surprisal_1829a99ad4.csv` (the label is text-derived). Install dependencies from `requirements.txt`; `bsdtar` is also needed to read the eye archives. Run commands from this `model_code/` directory. Set `CHINESEEEG_ROOT` to the local raw/derivative dataset. The code expects the dataset's `derivatives/preproc/filtered_0.5_30` EEG tier.

A direct preparation route is:

```bash
export CHINESEEEG_ROOT=/path/to/ds004952
python scripts/build/build_epoch_cache.py --data-root "$CHINESEEEG_ROOT"
python scripts/build_char_epoch_cache.py
python scripts/build_continuous_cache.py
python scripts/build_long_window_cache.py --post-ms 1200
python scripts/build_eye_char_cache.py
```

The first builder writes `data_cache/epochs`; the next builders write `data_cache/char_epochs`, `data_cache/continuous`, `data_cache/char_epochs_1200ms`, and `data_cache/eye_char`. The supplied `csd_matrix.npy` and cache audits document the archived spatial operator and 1200 ms window; they do not replace the missing signal arrays. If row epochs already exist elsewhere, set `SURPRISAL_ROW_CACHE` to their directory. To use existing caches from the source worktree without copying gigabytes:

```bash
export SURPRISAL_EEG_CACHE=/data/2_readEEG/superisal_code/data_cache/char_epochs_1200ms
export SURPRISAL_EYE_CACHE=/data/2_readEEG/superisal_code/data_cache/eye_char
```

## Run the combined model

```bash
python scripts/run_charbag_cv.py \
  --arm MM_mid_train_stats --arch mmhier --seq-block modern --csd \
  --cache-dir "$SURPRISAL_EEG_CACHE" --eye-stats-scope train \
  --folds 0 1 2 3 4 5 6 --device cuda:0
```

`--eye-stats-scope global` selects the archived normalization. Output arm names should be new so archived tables are not overwritten. `--no-eye` gives the matched EEG-only architecture. The saved scores use seven contiguous item blocks per book: test block `k`, validation block `k+1`, remaining blocks training. Checkpoint selection uses validation single-reading Spearman. The item score averages repeated test readings after the split. Train-only tercile edges define high/low AUC.

Regenerate the vector figures with `python scripts/make_submission_figures.py`. The small source checks in `tests/test_combined_model.py` cover train-only eye statistics and the zero-initialized eye residual; run them with `pytest` if available. They were also executed directly in the prepared CogAlign environment because that environment has no `pytest` module.

## Eye-guided ocular correction feasibility

`scripts/eye_guided_ica_pilot.py` and `scripts/eye_to_ica_probe.py` inspect
seven author-provided ICA decompositions using synchronized eye recordings.
They do not change the EEG cache or fit the surprisal decoder. The time-blocked
eye-to-ICA probe found that the most eye-predictable component had already
been excluded by the dataset authors in five of seven recordings; two of
seven recordings had any unexcluded component with test R² above 0.02.
See `results/eye_guided_ica_pilot_20260930/README.md` for exact records,
controls, and limitations. This does not establish a decoding improvement.

The separate downstream control in `results/eye_clean_decoder_20260930/`
fits an Eye-to-EEG ridge map from **training epochs only** and applies it to
the already ICA-cleaned EEG before the same EEG-only decoder. Its seven-fold
item rho is 0.3958 versus 0.3988 for unmodified EEG (3/7 positive folds),
so this particular extra regression correction has no stable gain. A fresh
Eye+EEG fusion rerun with train-only Eye statistics reached 0.3974 (2/7
positive folds versus EEG-only). These exploratory controls do not test
eye-guided ICA on the pre-ICA EEG tier.

## Extended endpoints and new architecture screens

`scripts/evaluate_surprisal_predictions.py` calculates item Pearson,
Kendall, MAE, RMSE, R², concordance, average precision, and book/count
conditional rank endpoints from saved predictions. The archived combined arm
has item Spearman 0.4049 ± 0.0148, but its book/count weighted within-stratum
Spearman is 0.1776 ± 0.0258. Full seven-fold values and provenance are in
`results/diagnostics/`.

Two original task-specific modules, `OverlapDemixer` and
`CenteredEvidencePool`, are implemented as optional flags in
`src/foundation/hiernet.py`. A count-matched ranking objective is optional in
the runner. The matched validation-only experiments on folds 0 and 3 are
documented in `results/architecture_search_20260930/README.md`. Neither
module nor the objective passed the two-fold item-rank improvement gate. They
are research candidates, not part of the paper's validated main model.

## October 2026 code-only screens

`results/cross_reader_probe_20261002/` found no clear same-item EEG signal in
its fixed cross-reader matching probe, and reader-quality weighting worsened
validation item ranking. `results/spectral_probe_20261002/` found that a
simple induced band-power regressor did not beat the time-domain decoder.

`src/foundation/deconvnet.py` implements a scheduled, time-resolved neighbour
subtraction before character pooling. Its shared layers now initialize
identically to the `--no-deconv` control. A complete paired **validation-only**
seven-fold run improved single-reading Spearman in all seven folds (+0.0098
mean), but item Spearman gained only +0.0016 and book/count weighted item
Spearman +0.0092; the predeclared seven-fold item gate failed. See
`results/schedule_deconv_validation_20261002/README.md` and its complete
prediction/paired-metric files. No manuscript value or default model changed.

`FrozenChineseLanguagePrior` is another optional experiment selected with
`--llm-prior pretrained|random`. It inserts two frozen Chinese GPT-2 blocks
between the EEG character encoder and the existing row integration block,
with trainable adapters and no text or token input at inference. A paired
validation screen found no item or book/count gain from the pretrained arm
on fold 0. A random-block control gained on fold 0 but lost on fold 3. Exact
weights, fold metrics, and the stop decision are in
`results/llm_prior_validation_20260930/README.md`. This experiment is not
included in the main paper model or its archived seven-fold results.

## Repeated-reader training and checkpoint selection audits

The optional `--item-pair-batches` sampler and `--item-mean-weight` loss
directly test the mismatch between single-reading training and item-mean
reporting. On validation fold 0, the paired sampler alone gave item rho
0.3535 versus matched baseline 0.3484; the added item-mean loss gave 0.3460.
Neither passed the predeclared item and book/count gate, so the comparison
stopped before more folds. See
`results/item_pair_training_validation_20260930/README.md`.

`--selection-metric item_spearman` instead selects the checkpoint using
validation item rho while keeping the original training unchanged. Its fixed
seven-fold exploratory evaluation gave mean item rho 0.3985 versus 0.3988
for the reproduced trial-selected EEG-only baseline, and book/count weighted
rho 0.1508 versus 0.1557. The complete fold table is in
`results/checkpoint_selection_20260930/README.md`. The default remains
`trial_spearman` because the alternative did not improve the held-out
endpoint.

## Retrospective sentence split and response diagnosis

`results/response_mechanism_20261002/README.md` records the new audited
sentence-disjoint, context-purged seven-block manifest, two-fold forward EEG
encoding test, matched baseline and count oracle, and the target/event audit.
The predeclared response gate failed in both tested development folds
(familywise permutation p=0.345 and 0.850), so the conditional response
dictionary module was not trained or promoted. The new-protocol EEG baseline
item rho was 0.3527 and 0.4021 on the two validation folds; the count oracle
was 0.3133 and 0.3924. These are retrospective checks, not fresh outer-test
results or values to substitute into the manuscript.

## Window-length diagnostic (2026-10-08)

`results/window_alignment_20261008/REPORT_CN.md` records21 matched validation runs for800/800,1200/800 and1200/1200ms windows. Item correlations are0.3993,0.4011,0.4001; matching endpoints provides no stable gain. Primary archived0.405 artifacts and manuscript remain unchanged. `scripts/run_window_alignment_cv.py` is the training entry; `commands.sh`, snapshots and cache audits are in the result directory.
