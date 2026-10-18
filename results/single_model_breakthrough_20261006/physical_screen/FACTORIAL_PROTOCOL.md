Registered before the additional valid-normalization temporal runs.

Purpose: avoid building the final candidate on padding-contaminated stem
statistics. Complete the four-cell factorial: baseline/temporal module ×
all-slot/valid-slot normalization. Three cells already registered independently;
add only valid-slot schedule-deconv, plus its exact-parameter unshifted control.
No new modalities, branch ensembles, local language supervision, depths, losses,
or lag tuning. All raw physical inputs, seeds2026, folds0/3, budget25ep/patience6,
CSD, batch64, fixed checkpoint criterion and source manifests identical.

For the scientifically preferred valid-normalization model require schedule
candidate > phys_valid and > valid-normalization unshifted control on item and
conditional rho in BOTH folds, no trial decline against phys_valid. This isolates
the one temporal mechanism in a corrected pipeline. Do not report the combined
change against a different-input/different-normalization baseline as an isolated
module effect. Retain all cells including failures. Only an attributable single
candidate expands; generic capacity and two reference corrections are controls,
not models to ensemble at inference.
