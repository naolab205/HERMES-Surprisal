# Scheduled deconvolution validation screen

## Decision fixed before training

The current EEG-only item-block baseline has validation fold-0 item Spearman
0.3484 and book/count weighted Spearman 0.0968. A cheap spectral probe in
`../spectral_probe_20261002/` found no useful induced-power signal, so this
screen tests a different representation mechanism: keeping the physical time
axis across character slots and subtracting temporally shifted neighbouring
responses before cross-character integration. This is implemented already in
`src/foundation/deconv.py` and `src/foundation/deconvnet.py`, but has no
completed result in this package.

Run `DeconvNet` with `--no-deconv` as the exact-topology control and with its
default two-step, two-neighbour deconvolution as candidate. Both use the same
1200 ms EEG cache, no Eye input, train-fold CSD, seed 2026, modern sequence
blocks, optimizer, batch size, 25-epoch cap, patience 6, and the original
single-reading validation checkpoint rule. This is a fold-0 validation screen;
the historical test block stays inaccessible. No parameter will be changed
after seeing fold 0.

The gate to run a second validation fold is candidate minus control item
Spearman at least +0.01 **and** book/count weighted Spearman at least +0.01,
with no single-reading decrease larger than 0.01. A failed gate stops this
architecture. The 0.50 goal is not a selection rule for this one-fold screen.

## Outcome

Fold 0 passed the predeclared expansion gate on exactly 1,577 validation
items and 14,791 readings:

| Fold 0 | Control | Deconvolution | Difference |
| --- | ---: | ---: | ---: |
| Single-reading Spearman | 0.3288 | 0.3403 | +0.0115 |
| Item Spearman | 0.3576 | 0.3726 | +0.0151 |
| Book/count weighted item Spearman | 0.0972 | 0.1458 | +0.0486 |
| High/low item AUC | 0.7063 | 0.7187 | +0.0124 |

The two zero-initialized gates acquired nonzero norms (7.57 and 6.61) in the
selected checkpoint. Shared parameters and initial outputs were verified
identical in a deterministic model smoke check before training. Fold 3 is now
the fixed replication screen. These are development validation results, not a
new sealed outer test and not evidence of item rho above 0.50.

Fold 3 independently passed the same gate:

| Fold 3 | Control | Deconvolution | Difference |
| --- | ---: | ---: | ---: |
| Single-reading Spearman | 0.3520 | 0.3598 | +0.0078 |
| Item Spearman | 0.3862 | 0.3985 | +0.0124 |
| Book/count weighted item Spearman | 0.1497 | 0.1839 | +0.0342 |
| High/low item AUC | 0.7444 | 0.7481 | +0.0037 |

The candidate's fold-3 first epoch emitted one mixed-precision skipped-step
learning-rate warning; the run completed with finite predictions, and the
warning is retained in `candidate_f3.log`. The selected gates had norms
6.55 and 5.85. This is promising but still one seed and two development
validation blocks.

**Expansion rule recorded before the remaining runs:** freeze every model,
training, and checkpoint parameter above; run folds 1, 2, 4, 5, and 6 once on
the validation partition only. For a seven-fold exploratory gain, require
positive item and book/count differences in at least five of seven folds,
mean paired item gain at least +0.01, mean paired book/count gain at least
+0.01, and no mean single-reading decline. Report all folds, even if the
rule fails. This is not a fresh outer test: readers and source books overlap,
and historical item-block results influenced earlier model work.

## Completed seven-fold validation comparison

The remaining five folds completed once under that fixed configuration. The
paired prediction audit verified the same 102,499 readings, 11,037 unique
items, reader/item keys, targets, and validation split for control and
candidate. `paired_validation_folds.csv` contains every fold and
`paired_validation_books.csv` contains the book slices.

| Endpoint | Control mean | Deconvolution mean | Paired mean change | Positive folds |
| --- | ---: | ---: | ---: | ---: |
| Single-reading Spearman | 0.3534 | 0.3632 | +0.0098 | 7/7 |
| Item Spearman | 0.3995 | 0.4011 | +0.0016 | 5/7 |
| Book/count weighted item Spearman | 0.1575 | 0.1666 | +0.0092 | 4/7 |
| High/low item AUC | 0.7456 | 0.7454 | −0.0003 | 4/7 |

The full item and conditional differences by fold are:

| Fold | Δ item ρ | Δ book/count ρ |
| ---: | ---: | ---: |
| 0 | +0.0151 | +0.0486 |
| 1 | +0.0013 | −0.0120 |
| 2 | +0.0055 | +0.0321 |
| 3 | +0.0124 | +0.0342 |
| 4 | +0.0054 | +0.0087 |
| 5 | −0.0142 | −0.0425 |
| 6 | −0.0142 | −0.0050 |

**The predeclared seven-fold gate fails.** The module reliably improves the
single-reading endpoint, but item-mean and count-controlled improvements are
small and heterogeneous. It is not adopted as the default surprisal model or
represented as a >0.50 result. The existing manuscript and its result tables
were not changed.

Post hoc book audit: the mean item gain is +0.0360 in LittlePrince and +0.0025
in GarnettDream; mean within-book/count gain is +0.0501 and −0.0022,
respectively. This is a lead for future domain-shift work, not an accepted
book-specific model. The code's 420 ms nominal shift is quantized to 13
stride-4 tokens at 128 Hz, or 406.25 ms. The actual character-onset interval
medians are 424.6 ms for LittlePrince and 412.9 ms for GarnettDream (from the
cache metadata). The book that benefits more is actually *farther* from the
implemented shift, so simple shift error does not explain this slice. Changing
the shift or gating the module by
book after seeing this audit would be a new selected experiment, so neither
was done here.

The fold-3 candidate logged one mixed-precision skipped-step scheduler warning
at the first epoch. This is a training-stability caveat for that run, not an
explanation of the overall gate failure. All final predictions were finite.

## Reproduction

Run from `model_code/` with the CogAlign Python environment and the
`SURPRISAL_EEG_CACHE` path set to the 1200 ms character cache. The full
commands and logs are in `control*.log` and `candidate*.log`; each arm used
`scripts/run_charbag_cv.py --arch deconv --seq-block modern --csd
--batch-size 64 --epochs 25 --patience 6 --seed 2026 --matched-train-rng
--eval-split val`. The control additionally used `--no-deconv`.
`scripts/summarize_schedule_deconv.py` regenerates the paired tables and gate
result from the six saved prediction files.
