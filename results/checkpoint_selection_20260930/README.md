# Frozen seven-fold checkpoint-rule comparison

The source runner's checkpoint rule maximizes validation **single-reading**
Spearman, whereas the primary reported endpoint averages repeated readers of
an item first. Late epochs in the paired-reader screen sometimes increased
validation item correlation while decreasing single-reading correlation. The
only intervention here is to maximize validation **item** Spearman instead.
Architecture, original random-row training sampler, optimizer, data, seed,
folds, and evaluation code remain at the archived EEG-only defaults.

## Decision rule recorded before new evaluation

Run all seven item-block rotations once with `--selection-metric
item_spearman`. The earlier EEG-only seven-fold rerun selected by trial
Spearman is fixed as comparator; its two prediction files have been combined
into `baseline_trial_selected_predictions.csv` without changing any value.
The comparator has seven-fold item rho 0.3988 ± 0.0198 and book/count weighted
item rho 0.1557 ± 0.0269. An interesting development improvement requires
paired mean item rho at least +0.01, nonnegative mean change in book/count
weighted rho, and no single-reading collapse. Per-fold signs and metrics must
be reported, including failures. No fold result may trigger hyperparameter,
seed, or loss changes in this screen.

The historical item-block evaluation rotations were already inspected during
model development; these are exploratory comparisons, not fresh sealed outer
tests. Checkpoint selection uses the designated validation block only.

## Measured outcome

All seven rotations completed. The item-selected run has the same 102,499
readings and 11,037 held-out items as the trial-selected comparator. The
mean paired differences (item-selected minus trial-selected) are:

| Endpoint | Trial-selected mean | Item-selected mean | Paired mean change | Positive folds |
| --- | ---: | ---: | ---: | ---: |
| Single-reading Spearman | 0.3544 | 0.3468 | -0.0076 | 1/7 |
| Item Spearman | 0.3988 | 0.3985 | -0.0003 | 3/7 |
| Book/count weighted item Spearman | 0.1557 | 0.1508 | -0.0049 | 3/7 |
| Item high/low AUC | 0.7434 | 0.7437 | +0.0003 | 4/7 |

Fold-5 item rho gained 0.0103, but folds 0 and 2 lost 0.0088 and 0.0080;
the full distribution is in `paired_fold_metrics.csv`. The declared gate
fails, so validation item selection is rejected as a default. Selecting on
the validation item endpoint does not by itself improve held-out item
ranking. The paper's original checkpoint rule and archived table remain in
place.

`a_fixed.log`, `b_fixed.log`, and `c_fixed.log` are the completed run logs.
Earlier unsuffixed logs were stopped before any fold completed because a
logging label displayed the selected metric under the name `val_trial`;
the corrected runs report both validation metrics separately. Predictions,
expanded metrics and paired summaries are in this directory. No further
checkpoint-rule tuning was performed.
