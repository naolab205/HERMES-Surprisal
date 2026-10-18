<div align="center">
  <img src="docs/assets/hero_animated.gif" width="100%" alt="HERMES: hierarchical EEG and eye representations for row-level surprisal" />

  <p><strong>Hierarchical EEG–Eye Representations for Model-Estimated Surprisal</strong></p>

  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&amp;logoColor=white" /></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-model-EE4C2C?logo=pytorch&amp;logoColor=white" /></a>
  <a href="https://openneuro.org/datasets/ds004952"><img alt="Dataset" src="https://img.shields.io/badge/Data-ChineseEEG-168B88" /></a>
  <a href="docs/REPRODUCIBILITY.md"><img alt="Reproducibility" src="https://img.shields.io/badge/Code-reproducibility%20guide-6750A4" /></a>

  <p><a href="#the-model">Model</a> · <a href="#results-and-interpretation">Results</a> · <a href="#quick-start">Quick start</a> · <a href="docs/REPRODUCIBILITY.md">Reproducibility</a> · <a href="docs/PROVENANCE.md">Provenance</a></p>
</div>

## What HERMES predicts

HERMES predicts the mean language-model surprisal of a displayed Chinese text row from EEG and eye tracking. A frozen Chinese GPT-2 model computes the labels using up to 192 preceding tokens. The physiological decoder receives signal windows and their slot mask; it receives no text or token embeddings at inference.

The model name refers to the existing `MM_mid` implementation. It does not define a new training run. The repository accompanies research on **hierarchical EEG and eye modeling of row-level surprisal in Chinese reading**; no publication acceptance is claimed.

## The model

<p align="center"><img src="docs/assets/architecture_animated.gif" width="100%" alt="Separate EEG and eye window encoders, residual fusion, cross-slot integration, masked pooling, and row regression" /></p>

| Stage | Operation |
| :-- | :-- |
| EEG window | 128 channels × 167 samples; nominal −100 to +1200 ms |
| Eye window | Gaze x/y, pupil, validity; 4 channels × 45 samples; nominal −100 to +800 ms |
| Local encoding | Separate temporal encoders; concatenated temporal mean/max summaries project to 96 dimensions |
| Fusion | A zero-initialized linear eye residual is added to the EEG embedding |
| Row integration | A cross-slot sequence block, masked mean pooling, and a regression head predict the row score |

The animations illustrate the computation flow; moving elements are schematic and do not show recorded waveforms or measured activations. [Static architecture](docs/assets/architecture_animated_still.png). Regenerate both GIFs with `python scripts/build_readme_animations.py`.

The primary architecture is in [`src/foundation/hiernet.py`](src/foundation/hiernet.py). The training entry is [`scripts/run_charbag_cv.py`](scripts/run_charbag_cv.py). Optional research modules are included for traceability and are **not** components of the primary model unless their flags are enabled.

## Results and interpretation

<p align="center"><img src="docs/assets/archived_results.png" width="900" alt="Seven observed item-block scores for the main hierarchy and matched component controls" /></p>

| Archived seven-block model | Single-reading Spearman ρ | Item Spearman ρ | High/low item AUC |
| :-- | --: | --: | --: |
| **HERMES (`MM_mid`)** | **0.354 ± 0.019** | **0.405 ± 0.015** | **0.748 ± 0.010** |
| Matched EEG only | 0.354 ± 0.021 | 0.399 ± 0.020 | 0.743 ± 0.013 |
| No cross-character block | 0.261 ± 0.027 | 0.374 ± 0.027 | 0.730 ± 0.020 |
| Historical resampled-row EEG | 0.192 ± 0.031 | 0.310 ± 0.039 | 0.696 ± 0.024 |

Values are mean ± fold standard deviation. Item scores average readers within each evaluated row. AUC uses only the outer target terciles with training-derived boundaries; it is not a separate classification experiment. Exact records are in [`results/tables/`](results/tables/).

> **Read the scores with their protocol.** These item blocks were reused during model development, and readers recur across splits. The archived eye normalization used pooled unlabeled samples. Estimated slot centers, overlap geometry, valid-slot count, and absolute gaze can retain presentation information. The primary score does not establish timing-independent decoding or generalization to new readers.

The matched controls provide context:

- Fusion with **training-only eye statistics** reaches item ρ ≈ **0.397**, compared with **0.399** for the matched EEG-only arm. Eye regression correction reaches ≈ **0.396**. Neither control shows a stable eye-related gain.
- Item correlation falls to ≈ **0.178** after conditioning on book and valid-character count.
- Equal EEG/Eye endpoints do not consistently improve the seven-setting validation comparison.
- Continuous-source noise and fixed-grid experiments use different inputs or cohorts. Their results remain separate from the archived main result.

See the [reproducibility guide](docs/REPRODUCIBILITY.md) and [research history](docs/research_history.md) for the evidence behind these statements, including failed candidates.

## Scalp dynamics appendix

Two data-based animations extend the manuscript's scalp and reader analyses: the group high-minus-low voltage over four row-onset intervals, and the between-reader dispersion of the same contrasts. They use actual saved EEG summaries with fixed color scales and no temporal interpolation.

<p align="center">
<a href="docs/appendix/README.md"><img src="docs/appendix/topography/group_contrast.gif" width="48%" alt="Animation S1: group row-surprisal scalp contrast" /></a>
<a href="docs/appendix/README.md"><img src="docs/appendix/topography/reader_dispersion.gif" width="48%" alt="Animation S2: between-reader SD of the same scalp contrast" /></a>
</p>

[Read the appendix, definitions, and interpretation](docs/appendix/README.md) · [Source arrays and provenance](docs/appendix/topography/provenance.json)

## Quick start

### 1. Install

```bash
git clone https://github.com/naolab205/HERMES-Surprisal.git
cd HERMES-Surprisal
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.10+ is required by the source syntax. The [tested environment](docs/environment_tested.txt) records the packages used for the release checks. CUDA is recommended for full training. `bsdtar` is needed when reading the dataset's eye-tracking archives.

### 2. Check the archived predictions without downloading signal data

```bash
python scripts/evaluate_surprisal_predictions.py \
  results/tables/charbag_MM_mid_predictions.csv \
  --out-dir outputs/primary_metrics

python scripts/plot_release_overview.py
```

The first command recomputes metrics from actual saved predictions. The second regenerates the README figure from the seven-fold tables. These commands do not train a new model. Conditional book/count metrics require additional cache metadata; the archived diagnostic records are included in [`results/diagnostics/`](results/diagnostics/).

### 3. Train when the required caches are available

Obtain [ChineseEEG / OpenNeuro ds004952](https://openneuro.org/datasets/ds004952). Follow the [cache and protocol notes](docs/REPRODUCIBILITY.md#signal-caches) before training. The repository does not contain the large EEG/Eye caches or archived weights.

```bash
export CHINESEEEG_ROOT=/path/to/ds004952
export SURPRISAL_EEG_CACHE=/path/to/eeg_cache
export SURPRISAL_EYE_CACHE=/path/to/eye_cache

python scripts/run_charbag_cv.py \
  --arm HERMES_train_stats \
  --arch mmhier --seq-block modern --csd \
  --cache-dir "$SURPRISAL_EEG_CACHE" \
  --eye-stats-scope train \
  --folds 0 1 2 3 4 5 6 --seed 2026 \
  --device cuda:0 --out-dir outputs/hermes_train_stats
```

New runs use training-only eye statistics. The archived `MM_mid` result used `--eye-stats-scope global`; changing this policy or the sampling protocol changes the experiment. The command above does not promise to reproduce 0.405. Add `--no-eye` for the matched EEG-only architecture. Use a fresh arm/output name to preserve existing evidence.

## Repository map

```text
HERMES-Surprisal/
├── src/                 # Encoders, fusion, datasets, and split helpers
├── scripts/             # Training, cache utilities, controls, and metric analysis
├── tests/               # Model and data-policy checks
├── data_cache/          # Frozen numeric targets, fixed CSD matrix, cache audits
├── results/             # Archived predictions, fold tables, and control records
├── analysis/            # Scientific plotting code and small source arrays
└── docs/                # Reproduction notes, provenance, and README graphics
```

Historical source snapshots and forensic reports can contain the original machine paths. The release's quick-start commands use paths relative to the clone. Some original preprocessing utilities still require adapting local dataset/cache paths; see the reproduction guide.

## Citation and credits

Use [`CITATION.cff`](CITATION.cff) to cite this software. The research authors are Chenggong Zhong, Shiqi Li, Weijie Luo, Jinke Luo, and Yao Miao. Correspondence: **miaoyao@sztu.edu.cn**.

The ChineseEEG dataset and the frozen language model retain their own attribution and usage terms. A repository-wide code license has not yet been designated. Publication details will be added when available.
