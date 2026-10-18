Registered before re-extracted-cache training.

The spliced-cache trials and latency probe remain historical diagnostics.
After full source verification, rerun three isolated single-model arms on the
physical cache: default local reference; local reference with valid-only BN;
shared-row reference with default BN. No averaging models, checkpoints, or
readers differently from the fixed item endpoint; no bundled changes.

Preserve exact metadata/epoch order, frozen row labels and context192,
sentence-purged manifest SHA7cb6c95f..., CSD and training-only scale estimate,
seed2026, folds0/3, batch64, modern/no-deconv, max25epochs/patience6,
trial-rho validation selection. Save checkpoints. No local label loss.
The shared-frame and valid-normalization arms compare against the physically
repaired baseline, not the old cached baseline. Dataset repair effect reported
separately as a protocol/source change, not an algorithm contribution.

Both folds item and conditional gains >0, no trial decline, are needed for
expansion; subsequent >=.02 seven-fold/three-seed and null controls remain.
Regardless of gain, do not infer causality or semantic specificity from merely
restoring correct timestamps. Old forward/teacher failures do not exclude
mechanisms under correctly extracted physical EEG. Public source onset markers
still yield estimated within-row timings; no true per-character markers invented.
