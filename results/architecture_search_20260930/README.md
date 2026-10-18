# Validation-only architecture and objective screen (30 September 2026)

## Question and boundary

The requested development goal is item-level Spearman correlation above 0.50.
The archived combined EEG + Eye `MM_mid` result is 0.4049 ± 0.0148 across
seven **development** item-block folds. Its book-by-character-count weighted
within-stratum correlation is 0.1776 ± 0.0258. The valid-slot mask gives the
model the number of highlighted characters; character count itself correlates
with the row-surprisal target (0.2963 ± 0.0193 across folds). This prompted two
task-specific architectural hypotheses and one training-objective control.

The new screens below use only the preassigned validation block of folds 0 and
3. Neither historical test block was predicted or scored for candidate
selection. All arms use the same EEG-only `mmhier` backbone, fixed cache,
surface Laplacian, seed 2026, maximum 25 epochs, patience 6, checkpoint rule
(validation single-reading Spearman), and reset training RNG. The baseline
is a fresh matched run; it is **not** the seven-fold archived combined arm.
These two development folds cannot support a generalization claim or a
confirmatory significance test.

## Candidate mechanisms

1. `OverlapDemixer` in `src/foundation/hiernet.py` contrasts the current
   character's middle and late EEG response bins against physically aligned
   early bins from the next one or two character windows. It adds a small,
   zero-initialized masked residual to the EEG character embedding. The
   hypothesis was that overlapping responses from later highlighted characters
   dilute the row-surprisal signal.
2. `CenteredEvidencePool` computes the centered second and third moments of
   valid character embeddings after cross-character integration and adds a
   zero-initialized residual to the row representation. This is invariant to
   a row-constant embedding shift and normalizes by valid count. The hypothesis
   was that variation *within* the row carries signal beyond its mean and
   length.
3. The `--within-count-rank-weight 0.2` objective adds a training-only pairwise
   logistic ordering loss for rows with the same book and character count. Book
   and count identify pairs; neither is passed to the predictor. The
   hypothesis was that supervised rank pressure among count-matched rows
   would increase the conditional endpoint.

All architecture modules are off by default. Their zero-initialized
projections preserve baseline output at initialization. Source-level tests
verify masking, invariance, gradients, and baseline equivalence. These are
implementation checks, not scientific evidence of improvement.

## Measured validation results

The table shows fold-mean metrics. `Item rho` first averages repeated readings
of each item. `Book/count rho` is the item-weighted mean of within-book and
within-character-count Spearman correlations. Every evaluated item belongs to
a qualifying stratum. Full per-fold predictions and diagnostic CSVs are in the
linked result directories.

| EEG-only arm | Item rho (fold 0 / 3) | Mean item rho | Mean book/count rho | Mean item AUC |
| --- | ---: | ---: | ---: | ---: |
| Matched baseline | 0.3484 / 0.3972 | 0.3728 | 0.1235 | 0.7246 |
| OverlapDemixer | 0.3580 / 0.3740 | 0.3660 | 0.1167 | 0.7207 |
| CenteredEvidencePool | 0.3454 / 0.3833 | 0.3644 | 0.1042 | 0.7189 |
| Count-matched rank loss | 0.3525 / 0.3928 | 0.3727 | 0.1160 | 0.7235 |

The first architecture improves fold 0 and worsens fold 3. The second worsens
both folds. The count-matched rank objective leaves mean pooled item rho nearly
unchanged and also lowers the conditional mean. None meets the predeclared
two-fold gate of improvement in both folds without a conditional collapse.
**No candidate is promoted to the paper's main architecture, and no >0.50
result is claimed.** Further nearby hyperparameter search on the same folds
would increase selection bias. The next decisive experiment needs a frozen
architecture and untouched reader/text evaluation, alongside count-controlled
metrics.

## Artifacts

- Matched baseline and OverlapDemixer:
  `../overlap_demix_validation_20260930/`.
- CenteredEvidencePool: `../centered_evidence_validation_20260930/`.
- Count-matched rank objective: `../count_rank_validation_20260930/`.
- Archived combined-model full seven-fold metrics:
  `../diagnostics/MM_mid_archived/`.
- Recompute all additional metrics with
  `scripts/evaluate_surprisal_predictions.py` and the original prediction CSVs.

The local caches used for these runs are in
`/data/2_readEEG/superisal_code/data_cache/`; they are not inside the small
Overleaf upload archive.
