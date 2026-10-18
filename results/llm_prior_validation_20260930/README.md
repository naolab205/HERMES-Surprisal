# Frozen Chinese GPT-2 prior over EEG character embeddings

## Hypothesis and boundary

The archived seven-fold EEG + Eye model has pooled item Spearman 0.4049, but
book-by-character-count weighted item Spearman 0.1776. A language-pretrained
causal sequence transformation might help the EEG encoder represent contextual
changes among displayed characters. This is a **validation-only** screen, not
a paper result or a sealed outer evaluation.

The candidate in `src/foundation/hiernet.py` maps each EEG-derived character
embedding into the hidden width of the locally cached
`uer/gpt2-chinese-cluecorpussmall`, passes it through two **frozen** causal
GPT-2 blocks, and maps it back through a zero-initialized residual adapter.
The existing EEG and cross-character network remains intact. The input is EEG
only: no token IDs, text, language-model score, book, reader, or target enters
the candidate at inference. The fact that this is the same language model used
to define the target score is disclosed; without a pretrained-versus-random
control, a gain would not show that language pretraining mattered.
The EEG-only backbone has 338,593 trainable parameters. Both language-prior
arms have 487,585 trainable parameters and 15,449,761 total parameters; the
rest of the added weights are frozen. The cached GPT-2 weight file has SHA-256
`5478b950f53650c82f7f7e91010787c870813a76adeabe23c8571b01af1379b4`.

## Registered screen before reading outcomes

- Fold 0: compare the pretrained frozen blocks with equal-size frozen random
  blocks and the already measured matched EEG-only baseline. Identical seed,
  EEG cache, spatial operator, backbone, train RNG reset, maximum 25 epochs,
  patience 6, and validation single-reading Spearman checkpoint rule.
- Expand to fold 3 only if pretrained item rho improves the baseline by at
  least 0.01, book/count weighted rho improves by at least 0.01, and pretrained
  item rho exceeds the random-block control by at least 0.005 without a
  single-reading collapse. These are screening gates, not significance tests.
- If both folds pass, freeze the design and seek an untouched reader/text
  evaluation before any paper claim. If fold 0 fails, stop this expensive
  pretrained-block hypothesis rather than tune layer count or adapters on the
  same validation observations.

The historical fold-0 test block is removed from the validation runner's
training/evaluation frame. The validation block is used both for checkpoint
selection and the screen, so the score is inherently optimistic.

## Fold-0 outcome and one bounded control follow-up

| Arm | Trial rho | Item rho | Book/count weighted item rho | Item AUC |
| --- | ---: | ---: | ---: | ---: |
| Matched EEG-only baseline | 0.3250 | 0.3484 | 0.0968 | 0.6996 |
| Pretrained GPT-2 blocks | 0.3177 | 0.3485 | 0.0851 | 0.6993 |
| Random frozen GPT-2 blocks | 0.3265 | 0.3620 | 0.1066 | 0.7079 |

The pretrained arm fails the declared gate: item rho is essentially unchanged
from baseline, conditional rho decreases, and the random control is better.
Do not run pretrained fold 3 or describe a benefit from language pretraining.

The random-block control has a narrow, unanticipated fold-0 improvement. As a
single bounded reconciliation test, run **only the random arm** on preassigned
validation fold 3 with exactly the same configuration and paired baseline.
Require both item rho and book/count weighted item rho to improve without a
single-reading collapse before considering this random-transform mechanism.
No layer-depth, adapter, epoch, or seed sweep is permitted after seeing fold 3.

Fold 3 completed with the validation-selected checkpoint. Its matched baseline
has trial rho 0.3565, item rho 0.3972, book/count weighted item rho 0.1503,
and item AUC 0.7496. The random-block arm reached 0.3413, 0.3867, 0.1248,
and 0.7455, respectively. Thus the fold-0 improvement does **not** repeat:
item rho changes by +0.0136 on fold 0 but -0.0106 on fold 3; the conditional
metric changes by +0.0098 and -0.0255. The two-fold mean item rho is 0.3743
for random blocks versus 0.3728 for the paired baseline, while the two-fold
mean book/count weighted rho is 0.1157 versus 0.1235. This mixed development
screen does not support promoting either the random blocks or pretrained
blocks. No test block was scored for these candidates.

## Source and artifacts

- Model: `src/foundation/hiernet.py`, `FrozenChineseLanguagePrior`.
- Runner flag: `--llm-prior pretrained|random|none`.
- Baseline fold-0 validation predictions:
  `../overlap_demix_validation_20260930/tables/charbag_baseline_val_predictions.csv`.
- Matched candidate logs: `pretrained_matched.log` and
  `random_matched.log` in this directory. The unsuffixed logs are aborted
  setup runs: they were stopped after detecting unequal adapter
  initialization and are excluded from analysis. A source smoke check now
  verifies equal shared weights, equal trainable-adapter weights, and equal
  initial predictions across baseline, pretrained, and random arms.
- The LM checkpoint is loaded from the local Hugging Face cache with
  `local_files_only=True`; no download or online model update occurs.
- Per-fold predictions and expanded endpoints are in `pretrained_matched/`,
  `random_matched/`, and `random_fold3/`. The pretrained arm was intentionally
  stopped after fold 0 because it failed the registered gate.

From `model_code/`, the paired screens used this command form (replace the
mode, fold, device, arm, and output directory for the other arms):

```bash
python scripts/run_charbag_cv.py \
  --folds 0 --eval-split val --arch mmhier --seq-block modern --csd --no-eye \
  --matched-train-rng --epochs 25 --patience 6 \
  --cache-dir /data/2_readEEG/superisal_code/data_cache/char_epochs_1200ms \
  --llm-prior pretrained --arm llm_pretrained_matched_val_f0 \
  --device cuda:0 --out-dir results/llm_prior_validation_20260930/pretrained_matched
```

Expanded endpoints were computed with
`scripts/evaluate_surprisal_predictions.py` using the saved prediction CSV
and the same character-cache metadata. No checkpoint from these screens was
used for the archived paper table.
