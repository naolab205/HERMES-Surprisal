# Repeated-reader item objective: validation-only screen

## Diagnosed mismatch

Each text row has 7–10 training readings in the selected folds and one shared
surprisal target. The archived model is optimized with randomly mixed
single-reading MSE plus correlation, whereas the primary reported item metric
averages readers first. In folds 0 and 3 there are respectively 7,882 and
7,884 training items; none is a singleton. A paired-reader training batch can
test whether directly fitting a partial item mean improves item ordering.

## Fixed comparison before results

Use the original EEG-only `mmhier` baseline and validation folds 0, then 3
only if a candidate passes fold 0. The paired sampler draws 32 training items
per batch and two **different readers** of each item, with no validation/test
item in the sampler. It makes 64-reading batches and the same number of
optimization steps as the baseline. One arm uses the original loss on these
batches (sampler control); the second adds weight 0.5 times the MSE of the
pair-mean prediction (item-objective arm). Both have identical model weights,
modalities, train RNG, input cache, seed 2026, 25-epoch cap, six-epoch
patience, and validation single-reading Spearman checkpoint rule. Pair item
IDs determine only sampling and the training loss; they are not model inputs.

The fold-0 gate for expansion is item Spearman at least 0.01 above the matched
baseline and book/count weighted item Spearman at least 0.01 above baseline,
without single-reading Spearman falling by more than 0.01. An item-loss gain
must also exceed the paired-sampler control to attribute it to the new loss.
The existing seven-fold test predictions are never used to choose this
candidate. Validation scores are exploratory and share their block with
checkpoint selection.

## Fold-0 outcome and stop decision

| Arm | Trial rho | Item rho | Book/count weighted item rho | Item AUC |
| --- | ---: | ---: | ---: | ---: |
| Matched original sampler | 0.3250 | 0.3484 | 0.0968 | 0.6996 |
| Paired-item sampler, original loss | 0.3283 | 0.3535 | 0.1022 | 0.7018 |
| Paired-item sampler, extra item-mean loss | 0.3257 | 0.3460 | 0.0775 | 0.6972 |

The extra item-mean loss is lower than its paired-sampler control on pooled
and count-conditioned item ranking. The paired sampler alone gains only 0.0051
pooled item rho and 0.0054 conditional rho, below the registered 0.01/0.01
gate. Neither arm advances to fold 3 or to the paper. The result shows that
directly emphasizing two-reader means at this weight does not resolve the
item-score limitation; it does not rule out other training objectives.

Full fold-0 predictions and diagnostics are in `sampler/` and `item_loss/`.
