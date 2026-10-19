# Experiment records

Start with the archived primary predictions and fold tables. The control and exploration directories retain their own protocols and cohorts.

## Primary evidence

| Records | Contents |
| :-- | :-- |
| [`tables/`](tables/) | Seven-fold main, ablation and reference-model tables; primary predictions; historical row-model comparator |
| [`diagnostics/`](diagnostics/) | Additional rank, agreement, error and conditional metrics |
| [`logs/MM_mid.log`](logs/MM_mid.log) | Original primary training log |

## Matched controls and sampling checks

| Records | Analysis |
| :-- | :-- |
| [`eye_clean_decoder_20260930/`](eye_clean_decoder_20260930/) | Training-only Eye normalization, matched EEG-only rerun and Eye-to-EEG correction |
| [`window_alignment_20261008/`](window_alignment_20261008/) | Matched EEG/Eye endpoints with a separate validation protocol |
| [`response_mechanism_20261002/`](response_mechanism_20261002/) | Sentence grouping, context purging and response-mechanism screening |
| [`single_model_breakthrough_20261006/`](single_model_breakthrough_20261006/) | Continuous-source, matched-noise, fixed-grid and reference controls |

<details>
<summary><strong>Earlier architecture and training screens</strong></summary>

- [Architecture screen](architecture_search_20260930/), [overlap demixing](overlap_demix_validation_20260930/) and [centered evidence pooling](centered_evidence_validation_20260930/)
- [Count-matched ranking](count_rank_validation_20260930/)
- [Item-pair training](item_pair_training_validation_20260930/) and [checkpoint selection](checkpoint_selection_20260930/)
- [Language-model prior](llm_prior_validation_20260930/)
- [Scheduled deconvolution](schedule_deconv_validation_20261002/)
- [Cross-reader probe](cross_reader_probe_20261002/), [spectral probe](spectral_probe_20261002/) and [local-supervision exploration](exploration_20261003/)
- [Eye-guided ICA pilot](eye_guided_ica_pilot_20260930/)
- [Support and trajectory candidates](support_consensus_20261006/)

These records include unsuccessful candidates. Their modules and scores should not be substituted for the primary `MM_mid` model. See the [research history](../docs/research_history.md) for outcomes.

</details>

Recompute the primary metrics from the repository root:

```bash
python scripts/evaluate_surprisal_predictions.py \
  results/tables/charbag_MM_mid_predictions.csv \
  --out-dir outputs/primary_metrics
```

The main scores are retrospective, repeated-reader item-block estimates. Changes to normalization, sampling, grouping or cohort define a different experiment. [Protocol and reproduction guide](../docs/REPRODUCIBILITY.md).
