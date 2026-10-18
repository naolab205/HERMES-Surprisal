# Two low-cost checks before another model search (2 October 2026)

The goal remains a defensible gain in **row-surprisal decoding**, especially
within book and character-count groups. The archived combined model's pooled
item Spearman is 0.4049; the corresponding book/count weighted Spearman is
0.1776. These screens do not modify the manuscript or promote a new model.

## Cross-reader EEG item signal

`scripts/probe_cross_reader_eeg.py` uses only the training partition of the
existing fold-0 item-block split. It selects rows with exactly ten characters
and ten EEG slots, three fixed readers, and 1,210 items across both books.
The 847 items selected within book train a fixed ridge cross-reader
linear map; 363 other training-partition items are the probe set. Four fixed
character slots contribute five post-stimulus EEG bins each. Same-item
cross-reader pairs are compared with different items in the same book and run,
also with exactly ten characters, preferably 5–60 rows away. Pre-stimulus
features and shuffled cross-reader training pairs are controls. The target
surprisal values are used only by the existing fold-assignment function; no
target value is used to fit or score the probe.

| Reader pair | Raw post AUC | Pre AUC | Mapped post AUC | Shuffled-fit AUC |
| --- | ---: | ---: | ---: | ---: |
| 04 → 05 | 0.505 | 0.509 | 0.480 | 0.491 |
| 04 → 06 | 0.519 | 0.522 | 0.527 | 0.531 |
| 05 → 06 | 0.494 | 0.509 | 0.513 | 0.508 |

The corresponding same-item-versus-matched-other-item win rates after mapping
are 0.490, 0.532, and 0.507; all three item-bootstrap 95% intervals contain
0.5 (see `probe.json`). The single 0.527 AUC is also below its shuffled-fit
control. This particular four-slot, binned-EEG probe provides **no positive
evidence** for a generic cross-reader item-contrastive pretraining stage. It
does not prove that all neural features or all readers lack a shared signal.

## Reader-quality weighting

`scripts/screen_reader_weighting.py` uses the previously saved *validation*
predictions of the matched EEG-only baseline on folds 0 and 3. A reader's
weight is its within-book/count Spearman on the **other** validation block;
weights are frozen before scoring the destination block. No test prediction or
label is used in this screen.

| Destination validation fold | Equal item ρ | Weighted item ρ | Equal book/count ρ | Weighted book/count ρ |
| --- | ---: | ---: | ---: | ---: |
| 0 (weights from 3) | 0.3484 | 0.3469 | 0.0968 | 0.0874 |
| 3 (weights from 0) | 0.3972 | 0.3940 | 0.1503 | 0.1401 |

Reader weighting loses 0.0023 pooled item ρ and 0.0098 conditional ρ on
average. It is not a candidate for the paper.

Both checks are development analyses; the historical seven-fold item-block
rotations have already informed prior model choices and are not an untouched
outer evaluation. The present results support stopping further nearby module,
pretraining, and reader-weight tuning on these validation blocks. A materially
different next experiment would have to audit stimulus/EEG alignment and the
row-surprisal target first, predeclare a single signal-processing and model
change, then evaluate it with count-controlled endpoints on new held-out
readers or texts.

An additional cache check ruled out a suspected input truncation: 46.4% of
row observations have `n_char > 10`, but `n_char` counts punctuation. The
presentation has at most ten non-punctuation highlighted slots, all of which
are present in the cache. Punctuation surprisal is assigned to adjacent
highlighted slots by the existing target builder. Therefore, increasing
`max_slots` alone would add padding rather than recover missing characters.

## Reproduction

From `model_code/`:

```bash
SURPRISAL_EEG_CACHE=/data/2_readEEG/superisal_code/data_cache/char_epochs_1200ms \
OPENBLAS_NUM_THREADS=4 /home/zcg/.conda/envs/cogalign/bin/python scripts/probe_cross_reader_eeg.py
/home/zcg/.conda/envs/cogalign/bin/python scripts/screen_reader_weighting.py
```
