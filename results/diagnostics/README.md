# Archived row-surprisal decoder: expanded held-out diagnostics

`scripts/evaluate_surprisal_predictions.py` recomputes metrics separately in
each held-out item-block fold from the archived `MM_mid` prediction CSV.
The values below are means across seven development folds, not independent
reader replications or a new outer test.

| Item-level endpoint | Mean | Fold SD |
|---|---:|---:|
| Spearman rho, pooled books/counts | 0.4049 | 0.0148 |
| Rank rho within book | 0.3467 | 0.0221 |
| Rank rho within book and character-window count | 0.1794 | 0.0255 |
| Item-weighted Spearman across book/count strata | 0.1776 | 0.0258 |
| Pearson r | 0.4393 | 0.0193 |
| Kendall tau | 0.2814 | 0.0108 |
| R² | 0.1903 | 0.0171 |
| Concordance correlation coefficient | 0.3066 | 0.0319 |
| MAE (training-target SD units) | 0.7016 | 0.0258 |
| RMSE (training-target SD units) | 0.8970 | 0.0363 |
| High-versus-low item ROC AUC | 0.7476 | 0.0101 |
| High-versus-low item average precision | 0.7472 | 0.0199 |

The count-only oracle's item Spearman against the label is 0.2963 ± 0.0193.
This is a descriptive association, not a fitted count-only model. In the
largest fixed-count subset (`n_slots=10`), the combined archive's item rho is
0.209 in GarnettDream (6,137 items) and 0.199 in LittlePrince (1,726 items).

The item-weighted endpoint averages ordinary within-stratum Spearman
coefficients, weighting each book/count stratum by its number of items.
The within-stratum rank endpoint maps each target and prediction to its rank
fraction inside a book/count stratum, centers it at 0.5, then correlates the
concatenated centered ranks. It removes between-stratum ordering while
weighting strata by their item counts. It is a diagnostic measure; it does
not establish the causal contribution of EEG without matched no-signal and
count-only trained controls.

Files under `MM_mid_archived/` give fold, book, and reader metrics. The
model's current character mask exposes window count, so a higher pooled score
alone is insufficient evidence of stronger neural surprisal decoding.
