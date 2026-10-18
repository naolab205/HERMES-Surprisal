# Step 1 — foundation fixes to the surprisal decoding representation

Frozen comparator: `JBHI_Surprisal_Package/code/results/tables/surprisal_cv_full_summary.csv`,
row `CogAlign_eeg_only` — trial rho 0.192 +/- 0.031, item rho 0.310 +/- 0.039,
item AUC 0.696 +/- 0.024, seven contiguous item blocks, item-disjoint but not
participant-disjoint.

## Corrected diagnosis

An earlier reading of this package claimed the pipeline skipped ICA, average
reference and bad-channel interpolation. **That claim was wrong and is
withdrawn.** The dataset README states that `derivatives/preproc/filtered_0.5_30`
is the fully pre-processed tier — segmentation, downsampling to 256 Hz,
0.5-30 Hz band-pass, bad-channel interpolation, ICA and average reference, all
applied by the dataset authors — and each run ships `bad_channels.json` and
`ica_components.npy` alongside. The frozen pipeline reads exactly that tier.
No expected gain is attributed to re-doing any of those steps.

## What actually limits the frozen representation

### 1. The physical time axis is destroyed

`scripts/build/build_epoch_cache.py:171` resamples every reading row to a fixed
128 samples regardless of its 1-6 s duration. The effective sampling rate
becomes `128 / dur` and varies per trial, so:

- bandwidth is duration dependent — a 4 s row keeps only 0-16 Hz, a 2 s row
  keeps 0-32 Hz;
- band-power features computed at an assumed 128 Hz
  (`src/cogalign/io/surprisal_features.py:96`) describe a different frequency
  range on every trial; the column labelled "alpha 8-13 Hz" covers 2-3.25 Hz on
  a 4 s row;
- ERP latencies are time-warped, so no fixed temporal kernel can align to them.

### 2. Character-level structure is discarded

The paradigm is program-paced: the dataset README states each non-punctuation
character of the highlighted row is coloured red for a fixed interval.
Measured on this cohort:

| quantity | value |
|---|---|
| corr(row duration, raw character count) | 0.823 |
| corr(row duration, non-punctuation count) | **0.986** |
| duration / non-punctuation count | 0.4205 s +/- 0.016 |
| maximum slots per row | 10 (matches the README's "no more than ten") |

So character onsets are recoverable as
`row_start + slot * (row_duration / n_slots)` with no eye tracking at all.

**Validation.** Averaging EEG at those onsets (sub-04 LittlePrince run-01,
3407 character epochs) gives a textbook character-locked visual evoked response:

```
GFP peak  +86 ms (P1)  ->  +152 ms (N1)  ->  +191..211 ms (P2)
next peak +504 ms  ==  SOA (420 ms) + P1 (86 ms)
```

The next character's P1 lands where the schedule predicts, so onset timing is
accurate to roughly 20 ms. The frozen row-level representation averages this
response away.

### 3. Per-epoch amplitude normalization removes the signal of interest

`build_epoch_cache.py:173-175` applies per-epoch median/MAD scaling, which
discards cross-row ERP amplitude differences — the very quantity an N400-style
surprisal effect lives in.

## What this step changes

Only the EEG representation. Participants, books, folds, label file and
endpoints are held identical to the frozen run.

| | frozen | step 1 |
|---|---|---|
| unit | one row, resampled to 128 pts | bag of <=10 character epochs |
| time axis | normalized phase (warped) | physical, 128 Hz |
| window | whole row (1-6 s) | -100..+450 ms per character |
| amplitude | per-epoch MAD normalized | microvolts preserved, train-fit scale |
| spatial | average reference (source tier) | + optional surface Laplacian, + optional Euclidean Alignment |
| supervised units | 102,500 rows | 917,123 character epochs in 102,500 bags |

Euclidean Alignment and the CSD operator are composed into one fixed 128x128
matrix per recording, applied on the GPU, so the cache stays a single copy.
The alignment covariance and the amplitude scale are estimated on **training
epochs only** and applied unchanged to validation and test.

## Protocol parity and its one known deviation

- Same 10 participants, 2 books, `assign_blocked_splits` with 7 blocks.
- Same label file: `surprisal_1829a99ad4.csv`
  (`uer/gpt2-chinese-cluecorpussmall`, 192-token context) — the md5 tag the
  frozen config resolves to.
- Same endpoints: trial Spearman, item Spearman after averaging readings,
  high-vs-low tercile item AUC with train-only tercile edges.
- `sub-15 LittlePrince` is absent here **and in the frozen run**: the shared
  row-level source cache has no such file.
- **Deviation:** test items per fold are 1576-1578 against the frozen run's
  1572-1575, a 0.2% difference. Rows whose highlighted-character count is below
  two are dropped by the character builder. Reported rather than patched.

## Leakage contract

The cached array holds signal only. `dur`, `n_char`, `soa`, `row_onset`,
`char_slot` and `n_slots` exist in the metadata for aggregation and auditing and
are never encoder inputs. Character slot order is used to assemble a bag, never
as a feature. Text, surprisal, participant and item identifiers stay out of the
encoder, as in the frozen contract.

## Arms

| arm | surface Laplacian | Euclidean Alignment | pooling |
|---|---|---|---|
| `B_char` | no | no | mean |
| `C_char_ea` | no | yes | mean |
| `D_char_ea_csd` | yes | yes | mean |
| `E_*_attn` | best of above | best of above | attention |

Each arm runs all seven folds. Comparison against the frozen baseline is paired
per fold (mean delta, folds improved, exact Wilcoxon signed-rank over 7 folds),
because seven paired folds are too few to lean on a p-value alone.

---

# Results (all seven folds, six arms)

| arm | trial rho | item rho | item AUC |
|---|---|---|---|
| frozen row-resampled baseline | 0.1925 +/- 0.0312 | 0.3099 +/- 0.0389 | 0.6957 +/- 0.0242 |
| B character bag | 0.2104 +/- 0.0322 | 0.3327 +/- 0.0322 | 0.7106 +/- 0.0260 |
| C + Euclidean Alignment | 0.1977 +/- 0.0231 | 0.3140 +/- 0.0265 | 0.6963 +/- 0.0198 |
| D + EA + surface Laplacian | **0.2200** +/- 0.0269 | 0.3385 +/- 0.0205 | 0.7146 +/- 0.0198 |
| E + EA + CSD + attention pool | 0.2084 +/- 0.0344 | 0.3345 +/- 0.0311 | 0.7128 +/- 0.0213 |
| F + surface Laplacian only | 0.2124 +/- 0.0266 | **0.3440** +/- 0.0301 | **0.7174** +/- 0.0205 |
| H = D, 45 epochs / patience 12 | 0.2157 +/- 0.0245 | 0.3352 +/- 0.0218 | 0.7102 +/- 0.0174 |

Paired per-fold change against the frozen baseline, best arms:

| arm | trial | item | AUC |
|---|---|---|---|
| D | +14.3%, 5/7 folds, p=0.22 | +9.2%, 5/7, p=0.22 | +2.7%, 4/7, p=0.30 |
| F | +10.3%, 4/7, p=0.38 | +11.0%, 5/7, p=0.16 | +3.1%, 5/7, p=0.11 |

## What is established

- Every one of the 18 arm-by-endpoint mean deltas is positive. The arms share
  data and folds, so this is descriptive consistency, **not** a significance
  argument.
- **No arm reaches p < 0.05.** With seven paired folds the smallest attainable
  signed-rank p is 0.016 (7/7 wins); the best observed is 0.11.
- The surface Laplacian is the useful spatial operator. Euclidean Alignment
  alone (arm C) is the weakest arm, barely above baseline; it only helps when
  applied after the Laplacian (D), and even then it trades item accuracy for
  fold-to-fold stability (D and H have the lowest standard deviations, F has
  the highest means).
- Attention pooling does **not** help (E below D on every endpoint). The
  hypothesis that mean pooling was diluting single-reading signal is not
  supported at this model scale.
- Training length is not the constraint: H (45 epochs) matches D (25 epochs),
  and best epochs land at 8-17.

## What is NOT established

The proposed mechanism — that the frozen pipeline's duration-dependent
effective sampling rate (128/dur) was the cause — **failed its direct test.**

Folds do improve most where the frozen model scored worst
(rho = -0.86 to -0.61 between frozen fold score and delta), but that pattern is
exactly what regression to the mean produces whenever a lower-variance model
replaces a higher-variance one, so it is not evidence on its own.

The falsifiable test: if duration-dependent bandwidth were the cause, folds
whose test rows vary more in duration should gain more. They do not. Test-set
duration coefficient of variation is nearly constant across folds
(0.217 to 0.237) and its rank correlation with the gain is -0.43 to +0.07 —
near zero, and the wrong sign for the hypothesis.

So the gain is real in direction but **its cause is not isolated** among:
character-onset ERP alignment, preserved microvolt amplitude, the surface
Laplacian, and a different encoder inductive bias. Separating these needs
single-factor arms on the row-level representation, which this step did not run.

## Honest position against the +50% goal

Best arm reaches +10-14% trial, +9-11% item, +3% AUC, none significant.
That is far short of +50%.

The reason is structural and was foreseen: the character bag is built but the
row label is still a single scalar per ~9 characters, consumed through mean
pooling. The 917,123 character epochs are still carrying only 102,500 labels.
Character-level supervision — the part that actually multiplies the training
signal — is entirely unused at this point.

---

# CORRECTION — the step-1 attribution above is wrong

The 2x2 attribution arms were run afterwards, holding the encoder fixed and
varying only the representation and the amplitude handling. They falsify the
central claim of this document.

| arm (my encoder throughout) | trial rho | item rho | item AUC |
|---|---|---|---|
| frozen pipeline (**their** encoder) | 0.1925 +/- 0.0312 | 0.3099 +/- 0.0389 | 0.6957 |
| R1 warped row + MAD = **the frozen representation** | **0.2350 +/- 0.0090** | 0.3262 +/- 0.0182 | 0.7027 |
| R2 warped row + amplitude | 0.2279 +/- 0.0114 | 0.3179 +/- 0.0107 | 0.6964 |
| R3 character-locked + MAD | 0.1554 +/- 0.0462 | 0.2719 +/- 0.0349 | 0.6773 |
| R4 character-locked + amplitude | 0.2104 +/- 0.0322 | 0.3327 +/- 0.0322 | 0.7106 |

Main effects:

| factor | trial | item |
|---|---|---|
| **encoder** (R1 - frozen) | **+0.0425** | +0.0163 |
| amplitude (amp - MAD) | +0.0240 | +0.0263 |
| **time axis** (character - warped) | **-0.0485** | -0.0197 |

## What this means

1. **The physical time axis is not the cause, and character-locking is worse
   than the warped row on single-reading rho (-0.0485).** The core argument of
   this document -- that resampling every row to 128 samples was the binding
   limitation -- is not supported.

2. **The encoder is the main driver.** Feeding the *frozen, unmodified*
   representation to the new encoder gives trial rho 0.2350, better than every
   character arm, at +22.1% over the frozen result with 6/7 folds improved and
   **p = 0.031 -- the only result in this whole effort that reaches p < 0.05**.
   It also cuts fold-to-fold standard deviation from 0.0312 to 0.0090.

3. **MAD interacts with the representation rather than acting as a main
   effect.** On 128-sample row epochs MAD is harmless and slightly helpful
   (R1 > R2). On 71-sample character epochs it is destructive
   (R3 trial 0.1554 vs R4 0.2104): a short epoch can contain a near-flat
   channel whose MAD is almost zero. Reporting a single "amplitude" main effect
   would average across this interaction and mislead.

4. The earlier failed mechanism test was the honest signal. Fold-level gain did
   not track test-set duration heterogeneity (rho -0.43 to +0.07). That result
   was reported but not weighted heavily enough; it was pointing at exactly
   this.

## What survives

The character representation keeps two genuine, smaller advantages:

| arm | trial raw | trial partial | retained | item raw | item partial |
|---|---|---|---|---|---|
| frozen (reported) | 0.1925 | 0.1624 | 84.4% | 0.3099 | n/a |
| R1 warped + MAD | 0.2350 | 0.2060 | 87.7% | 0.3262 | 0.2975 |
| R4 character + amplitude | 0.2104 | 0.1930 | 91.8% | 0.3327 | 0.3001 |
| F character + amplitude + CSD | 0.2124 | 0.1938 | 91.2% | **0.3440** | **0.3114** |

- Best **item-level** rho and AUC still come from the character arm with the
  surface Laplacian (0.3440 / 0.7174).
- Character arms retain more of their correlation after residualising row
  character count and presentation duration (91-92% versus 87-88% for row arms
  and 84% for the frozen pipeline). Fixed-length character epochs cannot encode
  row length, which is the one place the mechanistic argument holds up.

## Standing best results

| endpoint | best arm | value | vs frozen | significance |
|---|---|---|---|---|
| trial rho | R1 frozen representation + new encoder | 0.2350 | +22.1% | 6/7 folds, p = 0.031 |
| item rho | F character + amplitude + CSD | 0.3440 | +11.0% | 5/7 folds, p = 0.16 |
| item AUC | F character + amplitude + CSD | 0.7174 | +3.1% | 5/7 folds, p = 0.11 |

No single arm is best on all three. Any claim about *why* the gain exists must
rest on the 2x2 above, not on the diagnosis in the first half of this document.

# Character-level supervision: premise falsified

Per-character surprisal labels were built and verified to satisfy the sum
identity exactly (11,037 rows, max error 0.0075 nats, r = 0.9999995 against the
frozen row label). Training with them then showed:

| fold | character rho (all) | character rho (within row) | row rho, pooled head | row rho, character head |
|---|---|---|---|---|
| 0 | 0.0145 | **0.0050** | 0.171 | 0.089 |
| 3 | 0.0271 | **0.0022** | 0.214 | 0.187 |

Within-row character rho is ~0.003 across ~130,000 validation characters
(standard error ~0.003, and characters within a row are not independent, so the
effective test is weaker still). **A single character epoch carries no
information about that character's own surprisal.**

Two trivial explanations are excluded: the labels are exact (sum identity), and
within-row variance genuinely exists to predict (character surprisal sd 3.026
versus row sd 1.005).

Consequences, all consistent with each other:
- sweeping the instance weight over 12x (0.25 / 1.0 / 3.0) changed nothing,
  because an unlearnable auxiliary task contributes noise at any weight;
- adding it cost 23% of trial rho by corrupting the shared encoder;
- the character-head row readout (0.089) is far worse than the pooled head
  (0.171).

So the row-level signal is **not** a sum of per-character surprisal responses,
and the "9x more supervised units" argument for character bags is void.

---

# What the reliable signal actually is

The decoder's item-level predictions reproduce across readers (split-half
reliability 0.73-0.82). Reproducible is not meaningful: a decoder that reliably
read row length off the signal would look identical on that measure. Three
questions, computed in rank space per fold on test predictions only.

## Q1 — how far does a nuisance-only predictor get?

Ridge on rank-transformed row character count, presentation duration and
position in the run, fit on the same training folds:

| predictor | item rho |
|---|---|
| character count | 0.0529 |
| character count + position | 0.0561 |
| character count + duration + position | 0.1623 |
| **EEG decoder (character + CSD)** | **0.3440** |

The decoder is 2.1x the best nuisance-only baseline. Character count alone
reaches 0.053, so the label is not a length proxy.

## Q2 — does reliability survive nuisance removal?

Each single reading's prediction is residualised on the three covariates before
the split-half is recomputed:

| arm | reliability raw | reliability residual | loss |
|---|---|---|---|
| F character + CSD | 0.733 | 0.703 | -4% |
| D character + EA + CSD | 0.771 | 0.759 | -2% |
| B character | 0.733 | 0.703 | -4% |
| R1 warped row | 0.816 | 0.781 | -4% |

Reliability is essentially untouched. **The reproducible neural quantity is not
the nuisance quantity.**

## Q3 — does the surviving signal still track surprisal?

| arm | item rho raw | strict partial | disattenuated |
|---|---|---|---|
| F character + CSD | 0.3440 | **0.3238** | 0.386 |
| D character + EA + CSD | 0.3385 | 0.3162 | 0.363 |
| B character | 0.3327 | 0.3118 | 0.372 |
| R1 warped row | 0.3262 | 0.3110 | 0.352 |

Strict partial residualises *both* prediction and label on the covariates.
Disattenuated divides by the square root of the residual reliability, which is
the ceiling implied by measurement noise alone.

## Retraction

An earlier note in this session claimed R1 "wins by tracking duration".
**That is wrong and is withdrawn.** R1's predictions do correlate 0.51 with
duration, but duration correlates only 0.13 with the label, so that variance is
largely orthogonal to the target rather than a shortcut: residualising it
*raises* R1's correlation with the raw label (0.326 to 0.334). Under the strict
partial R1 is the weakest of the four arms, not the strongest, but for the
ordinary reason that it is slightly worse, not because it cheats.

## Where this leaves the task

There is a reliable, nuisance-independent neural quantity that tracks
GPT-2 surprisal at item level with rho about 0.32, and about 0.39 after
correcting for its own measurement noise. That is the honest ceiling for this
label and this paradigm. Remaining headroom from better encoders is the gap
between 0.32 and roughly 0.39, not a path to 0.5.

Three routes proposed earlier in this session were each tested and each failed:
the physical time axis (2x2 attribution), character-level supervision (premise
falsified at rho 0.003), and label replacement (all configurations lost 26-41%,
though that test is itself confounded by a change of target distribution and
would need redoing with a distribution-preserving ensemble before it is final).

---

# Late fusion: the first significant, confound-controlled gain

Weights are chosen by leave-one-fold-out -- fitted on six folds, applied blind
to the seventh -- so no weight is ever selected on the fold it is scored on.
Predictions are rank-transformed within fold before mixing.

| arm | item rho alone |
|---|---|
| W4 character + CSD, width 4 | 0.3462 |
| F character + CSD, width 1 | 0.3421 |
| C character + Euclidean Alignment | 0.3136 |
| Riemannian tangent space + xDAWN prototypes | 0.2798 |
| **fusion of all four** | **0.3731 +/- 0.0136** |

Against the frozen baseline, per fold:

| endpoint | frozen | fused | change | folds | p |
|---|---|---|---|---|---|
| item rho | 0.3099 | **0.3731** | **+20.4%** | **7/7** | **0.0156** |
| item AUC | 0.6957 | **0.7331** | +5.4% raw, **+19.1% above chance** | **7/7** | **0.0156** |

0.0156 is the smallest p the exact signed-rank test can return with seven
paired folds, so this is as strong as this design permits. Cross-fold standard
deviation falls from 0.0389 to 0.0136 (-65%).

Confound control, residualising item character count, duration and position:

| predictor | raw | strict partial | retained |
|---|---|---|---|
| nuisance-only ridge | 0.1623 | -- | -- |
| F single arm | 0.3440 | 0.3238 | 94.1% |
| **fusion** | **0.3723** | **0.3563** | **95.7%** |

The fusion retains more of its correlation under nuisance control than any
single arm, and remains 2.2x the nuisance-only baseline.

## Why fusion works here and capacity does not

Capacity scaling on the best single configuration is nearly flat and costs
single-reading accuracy:

| width | parameters | trial rho | item rho | item AUC |
|---|---|---|---|---|
| 1 | 0.07M | **0.2124** | 0.3440 | 0.7174 |
| 2 | 0.17M | 0.2038 | 0.3357 | 0.7130 |
| 4 | 0.48M | 0.2043 | **0.3494** | **0.7225** |
| 8 | 1.7M | 0.1897 | 0.3358 | 0.7143 |

Trial rho falls monotonically with width. The constraint is the available
signal, not model capacity.

Fusion works for the opposite reason: the arms are genuinely different views.
Their predictions correlate at rho 0.47 (Riemannian against CNN) rather than
0.9, so their errors are partly independent. The Riemannian arm carries weight
0.2 in **all seven folds** despite being the weakest arm alone -- a covariance
view of the same data that the waveform CNN does not capture.

## Correction to an earlier claim in this session

Euclidean Alignment was reported as harmful on the basis of arm C alone. That
conclusion was too narrow. EA is the weakest single arm, but it is
complementary: in the three-arm fusion it took non-zero weight in five of seven
folds, and dropping it costs accuracy. A single-arm comparison is not grounds
for discarding a component from an ensemble.

Separately, the alignment itself was misapplied. This EEG is average
referenced and bad-channel interpolated, and interpolation makes each repaired
channel an exact linear combination of its neighbours, so the spatial
covariance loses one rank per interpolated channel plus one for the reference.
Measured on one recording, five eigenvalues sit near 1e-6 against a largest of
2.3e3: a condition number of 1.9e9, which whitening amplifies by up to 44,000x
(23x even at the 0.05 shrinkage used). Rank-aware whitening -- projecting out
directions below a fraction of the largest eigenvalue instead of inverting them
-- is the correct form for this data tier, and is under test.
