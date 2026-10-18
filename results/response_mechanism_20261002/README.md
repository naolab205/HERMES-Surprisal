# Retrospective surprisal response diagnosis (2026-10-02)

This is a development analysis. The ChineseEEG books, readers, and older item
blocks have already been used for model exploration. The new partitions are
stricter robustness checks, **not a newly sealed outer test**. The target is
the existing row-average negative log likelihood from
`uer/gpt2-chinese-cluecorpussmall` with a 192-token context. The primary
decoder, if a candidate qualifies, has EEG-only inference.

## Prior mechanisms and the proposed distinction

The following comparison is based on the publisher abstracts/method previews
and available author materials; it is not a claim that every full text was
available or that an untested module is already novel.

| Source | Mechanism already studied | Consequence for this task |
| --- | --- | --- |
| [Guo et al., KBS 2024, FC-FAN](https://doi.org/10.1016/j.knosys.2023.111199) | Functional connectivity plus grouped attention for EEG emotion recognition | Channel connectivity or attention alone is an inadequate novelty claim. |
| [Ren et al., KBS 2024, PLMSDANet](https://doi.org/10.1016/j.knosys.2024.112669) | Multi-source domain adaptation and pairwise learning using a small labeled target domain | A no-calibration new-reader test cannot be compared with labeled target adaptation without saying so. |
| [Guo et al., KBS 2025, TSSAN](https://doi.org/10.1016/j.knosys.2025.113762) | Synchronous temporal, spectral and spatial attention with masks and gates | More attention or band fusion is not, by itself, a task-specific advance. |
| [Wang et al., KBS 2025, DBPM](https://doi.org/10.1016/j.knosys.2025.114650) | Density prototypes and sequential cross-subject alignment | Prototype matching already has a close EEG precedent. |
| [Li et al., KBS 2026, MRFCP-Net](https://doi.org/10.1016/j.knosys.2026.115551) | Shared/private EEG–Eye representations for continuous affect estimation | Shared/private EEG–Eye fusion alone is not a new mechanism. |
| [Natural sentence reading, iScience 2025](https://doi.org/10.1016/j.isci.2025.112798) | Forward EEG encoding of linguistic features with low-level controls | Motivates testing whether surprisal has incremental EEG explanation before inverse decoding. |
| [Rank-N-Contrast, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html) | Target-ordered contrast for generic continuous regression | A rank loss alone would not establish a new algorithm here. |

The candidate mechanism in the plan is a response-constrained iterative
evidence extractor after the current EEG encoder. Its status is conditional:
only a reproducible, nuisance-adjusted target response could justify building
and training it. Existing encoding/decoding and unrolled sparse optimization
literature would still need direct novelty comparison.

## Split and validity audit

`scripts/build_retrospective_manifest.py` produces `sentence_manifest.csv` and
`manifest_audit.json`. Within each book/run, the text is divided once at a
sentence boundary. Each resulting chunk is assigned to one of seven
book-stratified blocks. Twenty-four rows on each side are excluded; the
measured minimum gap is **451 Chinese character tokens**, exceeding the frozen
LM's 192-token context. Repeated and near-duplicate sentences assigned to
different blocks are excluded across all readers. The audit finds zero
remaining cross-block sentence overlaps. A broader four-gram and sequence
similarity audit also finds zero cross-block near-duplicate sentence pairs.

Of 11,037 distinct source rows, 9,835 are retained, 1,200 fall in context
gaps, and two form a cross-block repeated sentence. Fold indices 0 and 3 are
fixed diagnostic development choices; blocks 1 and 4 are their corresponding
validation partitions. A fold's test-designated block is removed before
encoding-model fitting or permutation. Every repeated reader shares an item
assignment and, in the permutation test, one permuted item label.

## Encoding-model gate

`scripts/diagnose_surprisal_response.py` averages all valid character epochs
within each reading in four fixed latency windows (0–200, 200–400, 400–800,
800–1200 ms). The EEG response has 128 channels times four windows. A
train-only nuisance model includes reader, book/run, character count, row
position and available eye-channel summaries. The full model adds the frozen
row surprisal target. Held-out validation `delta R^2` measures its incremental
explanation. Eye features are nuisance covariates in this forward analysis,
not inference inputs and not a causal measurement of ocular artifact.

The null shuffles item labels within book and rounded character-count strata,
preserving repeated-reader groups. The statistic is the maximum positive
`delta R^2` across every channel and latency window. The registered expansion
gate requires familywise permutation `p <= 0.05` in **both** folds and a
positive slice in at least seven of ten readers in each fold.

| Development fold | Max validation delta R² | Familywise p | Positive reader slices |
| ---: | ---: | ---: | ---: |
| 0 | 0.000285 | 0.345 | 6/10 |
| 3 | 0.000193 | 0.850 | 7/10 |

The gate **fails**. These four coarse, row-averaged windows do not establish a
stable surprisal-specific response after the selected controls. This does not
prove that no more localized neural response exists. Under the stated plan, a
response dictionary trained from this diagnostic would have no validated
target pattern to encode, so no such module is promoted or run. The next
bottleneck investigation is label granularity, event alignment, and signal
quality; changing model depth or tuning a response basis to these validation
maps would invalidate the gate.

## Baseline and external data status

The new-protocol baseline uses the existing `DeconvNet --no-deconv` control,
train-only CSD/scales, modern sequence blocks, seed 2026 and single-reading
validation checkpoint selection. It is run on folds 0 and 3; output is in
`baseline_f0/` and `baseline_f3/`. A train-only book/observed-slot-count
oracle is reported as a nuisance comparator, not as a valid EEG decoder.
All metrics are in `retrospective_baseline_summary.json`.

| Validation fold | EEG item rho | Count oracle item rho | EEG minus oracle | Sentence-cluster bootstrap 95% CI | EEG within-book/count rho |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.3527 | 0.3133 | +0.0394 | [0.0142, 0.0663] | 0.1099 |
| 3 | 0.4021 | 0.3924 | +0.0097 | [−0.0175, 0.0424] | 0.1243 |

The fold-3 gain over the count oracle is unresolved by this bootstrap. The
observed slot count can differ slightly among readers of an item; averaging
that oracle across readers can therefore create a small nonzero conditional
ranking. No candidate model is compared with the baseline because the
preceding gate failed. The retrospective split must never be compared
numerically as though it were the old item-block protocol.

`target_event_audit.json` provides the prespecified pivot checks. The frozen
label averages all tokens in a row, whereas only nonpunctuation characters
receive a highlighted event. Punctuation accounts for **10.6%** of row
tokens; 7,964 of 11,037 items contain it. Surprisal has Spearman 0.296 with
highlighted slot count and −0.297 with punctuation count. Only 19 of 102,499
readings have missing display slots; the median stimulus interval is 0.413 s.
These correlations make target granularity and layout a stronger next
investigation than adding capacity. They do not establish that punctuation
causes the decoder's residual error.

The locally available ZuCo 2.0 tree currently contains only 23 files under a
task-specific reading subset, insufficient to assert complete normal-reading
external replication. ChineseEEG-2 uses a different speech/listening task and
overlapping text, so it is not a new-text silent-reading test.

## Reproduction

From `model_code/` with the CogAlign Python environment:

```bash
export SURPRISAL_EEG_CACHE=/data/2_readEEG/superisal_code/data_cache/char_epochs_1200ms
export SURPRISAL_EYE_CACHE=/data/2_readEEG/superisal_code/data_cache/eye_char
python scripts/build_retrospective_manifest.py
python scripts/diagnose_surprisal_response.py --permutations 199
```

The baseline uses `scripts/run_charbag_cv.py --arch deconv --no-deconv --csd
--seq-block modern --batch-size 64 --epochs 25 --patience 6 --seed 2026
--matched-train-rng --eval-split val --split-manifest
results/response_mechanism_20261002/sentence_manifest.csv`, with separate
`--folds 0` and `--folds 3` runs. `encoding_diagnostic.json` and
`encoding_fold{0,3}.npz` contain the observed effect, coefficient maps and
null maxima for audit.
