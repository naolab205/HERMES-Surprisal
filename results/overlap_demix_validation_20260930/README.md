# Adjacent-window overlap demixing: validation screen

**Question.** The archived EEG-only hierarchical model pools a 1.2-s EEG
window for each highlighted character, although the next character typically
appears about 420 ms later. Does an explicit, small correction for the
resulting response overlap improve row-surprisal ordering?

**Candidate.** `OverlapDemixer` in `src/foundation/hiernet.py` reads local EEG
features in fixed early (0–420 ms), middle (420–840 ms), and late (840–1200 ms)
intervals. At character slot `i`, it contrasts middle features with the next
slot's early features and late features with the next two slots' aligned
features. Two zero-initialized linear projections add a correction to the
existing character embedding. Missing later slots are masked. It receives no
SOA, duration, text, row score, book or reader identifier. The shared model
weights and initial predictions are identical to baseline under the same seed.

**Current screen.** Folds 0 and 3 only, using their preassigned validation
blocks as the evaluation split. The historical test blocks are loaded by the
shared runner but are not predicted, scored, or used for selection in this
screen. EEG-only baseline and candidate share input cache, Laplacian,
architecture except for this one module, seed 2026, 25-epoch maximum,
six-epoch early-stop patience, training RNG reset after model construction,
and validation single-reading Spearman checkpoint selection. They use
separate output arms and GPUs only for throughput. Item Spearman and
within-book item Spearman are the primary diagnostic; pooled item Spearman
alone could reward between-book differences. The selected checkpoint's
validation metric is a development screen, not an independent test estimate.

**Gate.** Expand to more validation rotations only if both screening folds
improve item correlation without a book-specific or single-reading collapse.
If the signature fails, reject the overlap mechanism as implemented instead
of tuning many nearby lags or model sizes. A final manuscript claim would
require a frozen architecture and a fresh evaluation boundary.
