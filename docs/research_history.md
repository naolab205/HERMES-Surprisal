# Research history and evidence map

This page indexes archived experiments. Use the [reproducibility guide](REPRODUCIBILITY.md) for current setup and cache requirements. Historical logs and source snapshots document the runs that produced the records; their machine paths are not portable setup commands.

## Main result and controls

| Evidence | Archived interpretation | Records |
| :-- | :-- | :-- |
| Primary `MM_mid` hierarchy | Seven retrospective item blocks; item ρ ≈ 0.4049 and high/low item AUC ≈ 0.7476 | [Fold tables and predictions](../results/tables/), [original run log](../results/logs/MM_mid.log) |
| Historical resampled-row EEG | Item ρ ≈ 0.310 under the same block protocol; the representation and network differ | [`surprisal_cv_full_per_fold.csv`](../results/tables/surprisal_cv_full_per_fold.csv) |
| Matched EEG-only rerun | Reproduced all seven archived EEG-only fold scores | [Paired summary](../results/eye_clean_decoder_20260930/paired_summary.json) |
| Training-only Eye statistics | Fusion item ρ ≈ 0.3974 versus EEG-only 0.3988 | [Eye-control records](../results/eye_clean_decoder_20260930/) |
| Eye-to-EEG regression correction | Item ρ ≈ 0.3958; 3/7 positive item differences against EEG only | [Eye-control records](../results/eye_clean_decoder_20260930/) |
| Conditional metrics | Archived item ρ falls to ≈ 0.1776 within book/count groups | [Diagnostics](../results/diagnostics/) |
| Matched endpoint comparison | 800/800, 1200/800 and 1200/1200 ms give validation item ρ ≈ 0.3993, 0.4011 and 0.4001 | [Window report](../results/window_alignment_20261008/REPORT_CN.md) |

The primary run used pooled unlabeled Eye statistics. The current runner defaults to training-only statistics. Readers recur, and development used the same item blocks. These results do not supply a new-reader or timing-independent test.

## Architecture and training screens

These candidates remain separate from the primary `MM_mid` architecture.

| Candidate | Outcome | Evidence |
| :-- | :-- | :-- |
| Overlap demixing and centered evidence pooling | Did not pass the paired item/conditional-rank improvement gate | [Architecture screen](../results/architecture_search_20260930/README.md), [demixing](../results/overlap_demix_validation_20260930/), [pooling](../results/centered_evidence_validation_20260930/) |
| Count-matched ranking objective | Did not pass the development improvement gate | [Count-ranking records](../results/count_rank_validation_20260930/) |
| Scheduled temporal deconvolution | Seven-fold validation improved item ρ by only ≈ 0.0016 and conditional ρ by ≈ 0.0092; the gate failed | [Paired report](../results/schedule_deconv_validation_20261002/README.md) |
| Frozen language-model blocks | Pretrained blocks gave no item/conditional gain in the matched fold-0 screen; random-block gains did not persist across the examined folds | [Language-prior records](../results/llm_prior_validation_20260930/README.md) |
| Repeated-reader item-pair batches and item-mean loss | Neither passed the fold-0 item/conditional gate | [Item-pair report](../results/item_pair_training_validation_20260930/README.md) |
| Item-based checkpoint selection | Seven-fold item ρ ≈ 0.3985 versus trial-selected 0.3988; the default remains trial-based selection | [Checkpoint-selection report](../results/checkpoint_selection_20260930/README.md) |
| Spectral representation probe | Did not outperform the time-domain decoder | [Spectral report](../results/spectral_probe_20261002/README.md) |
| Reader-quality weighting | Worsened validation item ranking in the examined probe | [Cross-reader report](../results/cross_reader_probe_20261002/README.md) |

Optional modules and their source remain available for inspecting these comparisons. Their presence in `src/` does not make them parts of the primary model.

## Signal, target and sampling audits

| Analysis | Evidence and scope |
| :-- | :-- |
| Eye-guided ICA feasibility | [Pilot report](../results/eye_guided_ica_pilot_20260930/README.md); inspection of author-provided decompositions, without a demonstrated decoding gain |
| Sentence grouping and response mechanism | [Response report](../results/response_mechanism_20261002/README.md); the predeclared response gate failed in both examined blocks, so the response dictionary was not promoted |
| Character and reader-teacher probes | [Exploration records](../results/exploration_20261003/); separate local-supervision and cross-reader diagnostics |
| Continuous-source, noise and fixed-grid checks | [Physical-screen records](../results/single_model_breakthrough_20261006/); different sampling geometry and cohorts from the archived primary result |
| Physical-support and unique-trajectory candidates | [Support-consensus report](../results/support_consensus_20261006/REPORT.md); separate retrospective screens, not components of the main EEG–Eye hierarchy |
| Recorded scalp dynamics | [Supplementary S1/S2](appendix/extended/README.md); descriptive training-data summaries linked to Appendix A |

Estimated character positions are not observed character triggers. Slot count, window overlap and absolute gaze may retain presentation information. The audited physical-window utilities and their cache checks address different conditions from the historical concatenated-row pipeline. The legacy `build_long_window_cache.py` entry remains a guard that stops before reading or writing caches; its superseded implementation is preserved in the experiment snapshots.

## What the archive retains

- Original fold metrics, predictions, protocol records and unsuccessful candidates.
- Source snapshots whose exact versions are used by audits or recorded checksums.
- Frozen numerical labels, the spatial operator and scientific source arrays.

Large signal caches and trained checkpoints are not included. Some reports refer to local inputs that must be supplied separately. Model-selection rules and protocol differences should be retained when interpreting any of these files.
