# Surprisal performance and mechanism exploration

Created 2026-10-03. The associated thread goal is active. This document
registers the next exploration sequence; it does not claim a performance gain.
The manuscript is outside this work's edit scope.

## Goal and fixed scientific boundary

Improve EEG-only prediction of the existing frozen row-level surprisal target,
identify a mechanism that explains any gain, and evaluate its limits on new
readers. Item Spearman >=0.50 is a stretch target. The language model,
192-token context, row target, and audited sentence manifest remain fixed.
No target-side text, true label, identity embedding, or duration enters
inference. Text and target labels may be used for supervision on training
rows only. Old experiments and the newly purged splits are retrospective;
they cannot be relabeled as previously untouched outer tests.

Reference manifest:
`../response_mechanism_20261002/sentence_manifest.csv`, SHA-256
`7cb6c95f3aef8cb7e82ba92f2c87027abba9c5d25e3d701ff41c3b76c0a54fe8`.
Reference fold indices: 0 and 3, validation blocks 1 and 4, respectively.
The respective test-designated blocks do not enter fitting or selection.

## Evidence motivating the pivot

The coarse row-average forward-encoding screen failed its familywise gate
(p=0.345 and 0.850). That specific response-dictionary route stays rejected.
The corresponding baseline item rho is 0.3527/0.4021, versus a train-only
book/count oracle at 0.3133/0.3924. Row labels include punctuation while
highlighted EEG events exclude it. Punctuation represents 10.6% of tokens.

A static audit finds a correctness issue before any new neural experiment:
the stored character labels are sums with punctuation folded into a nearby
highlighted character; the frozen row label is their sum divided by token
count. `load_char_targets` currently assumes their average over highlighted
slots equals the frozen row target. On 7,939/11,037 rows these denominators
differ. The resulting raw slot-mean MAE is 0.3665 nats, versus 0.000762 nats
when using the recorded token denominator. See `character_identity_audit.json`.
Additionally, 19 readings lack one display slot, so labels must join actual
`char_slot` keys rather than taking a prefix of the label vector.
This is a correctness repair, not an innovation claim or a demonstrated gain.

## Five execution steps

1. **Repair and verify local supervision.** Bind row/character labels by
   hashes, preserve the frozen scalar target, convert stored token sums to
   slot contribution units so their masked mean matches the row target, and
   join actual physical slot keys. For incomplete bags, retain row supervision
   and explicitly mask the local/conservation term; do not invent a missing
   EEG event. Reconcile the small fp16 label discrepancy against the frozen
   row scalar explicitly. Verify sum identity, punctuation cases, missing
   first-slot cases, train-only scaling, and unchanged inference inputs.

2. **Test fine-grained supervision.** Compare the corrected local/conserved
   head against the same EEG architecture using only the existing row target.
   Fix seed 2026, modern blocks, CSD, batch 64, 25 epochs, patience 6, existing
   optimizer, and single-reading checkpoint selection. Match sampling RNG,
   observation sets, and compute. Compare real local targets with a
   within-row shuffled contribution control while preserving each row sum.
   This tests whether temporal localization contributes information beyond
   the scalar row label. It does not change the primary target.

3. **Test an original temporal-evidence mechanism only if localization
   helps.** First compare ordered versus order-shuffled character evidence
   and early/late neural features with all counts and row targets matched.
   Only a reproducible ordering/latency signature justifies designing a native
   module that delays temporal pooling and assigns local contributions to
   cross-character integration. Require a matched generic-capacity control,
   fixed projection, and removal of the new mechanism. The architecture and
   its equations are frozen after this diagnostic, before candidate training.
   Verify the closest prior methods before describing algorithmic novelty.

4. **Test training-only cross-reader supervision as a separate hypothesis.**
   The already-failed raw reader matching and paired-item mean loss are not
   repeated. Probe whether out-of-reader training predictions carry
   conditional target information for book/count-matched items. A teacher
   must exclude the student reader and use training texts only; it may never
   use validation or test readings of the same item. Only a positive probe
   permits residual-evidence distillation against shuffled/mismatched teacher
   controls. The deployed student receives one EEG reading.

5. **Freeze, expand, and report.** A route qualifies for expansion only if
   both fixed development folds improve item rho and book/count conditional
   rho with no single-reading decline. Expand qualifying candidates to seven
   folds and seeds 2026/2027/2028, then complete targeted ablations and null
   inputs. Freeze the winning recipe before leave-one-reader-out stress
   testing. External replication uses complete, audited natural-reading data;
   local ZuCo's incomplete task-specific subset does not qualify. If every
   route fails, finish with the failure ledger and precise supported claims,
   without presenting rho>=0.50 as achieved.

## Promotion and outputs

For a promoted development candidate require mean item rho and mean book/count
rho gains each >=0.02, improvement on at least five of seven folds for both,
positive average directions in all three seeds, no mean single-reading
decline, and paired confidence intervals supporting gains. Real EEG must beat
book/count, shuffled, mismatched, empty/noise and pre-onset controls. Existing
feature masks expose slot count; reporting their normalized mean does not
remove this confound. Null/control models must preserve those masks and
padding distributions.

Report trial/item Spearman, Pearson, MAE, RMSE, R², reader slices, training cost,
and clustered confidence intervals. Use paired text blocks/readers for
inference, preserving repeat readings. Never use outer-reader or external-test
results to choose hyperparameters. Stop an individual route after its
predeclared gate fails, and pivot to the next hypothesis instead of adding
epochs or neighboring settings.

Each run writes a unique directory with source/config hashes, exact command,
fold-level predictions, selected checkpoint/epoch, metrics, control results,
and accept/reject decision. The final deliverable is a verified module with
attributable gains, or a complete negative exploration report with the
remaining bottleneck; neither outcome is inferred from code tests alone.
