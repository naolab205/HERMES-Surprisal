# Training-only teacher feasibility gate
Registered before execution, 2026-10-03.
For each outer development fold 0/3, use its training texts only. Sample
complete sentences by metadata, with caps 1200/600 items per book, seed2026.
Cross-fit over all five remaining text blocks. Students sub04/05/06; teacher
excludes each student. Fit standardized 512-dimensional EEG window features
with Ridge alpha1000. Canonical item slot count is label-free median across
readings. Fit book/count residual oracle on each inner fitting block only.
Compare averaged other-reader EEG predictions with single-reader predictions
and item-mismatched teacher predictions within held-block book/count strata.
Pass only if mean teacher conditional rho exceeds .05 and both controls in
both outer development folds. This is a cheap raw-feature feasibility probe,
not evidence of distillation or a deployable multi-reader predictor. No outer
validation/test readings enter teacher fit or selection. If negative, stop
this registered teacher route instead of changing probe settings.
