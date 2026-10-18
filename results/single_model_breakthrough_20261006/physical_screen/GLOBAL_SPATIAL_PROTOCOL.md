Registered before global-spatial training. Close the strict inference boundary.

The original backbone passes recording IDs to a buffer lookup selecting a
per-recording amplitude scale, fitted on training data only. This is legitimate
seen-reader calibration but does not meet a literal identity-free inference
contract. Do not call it new-reader/no-calibration evidence.

New route uses one CSD matrix and one shared training-only scalar for all
recordings. Estimate scalar as the weighted RMS of the previously fitted
training-group scale estimates, weighted by available training epochs. Group
IDs may organize training normalization estimates but are ignored in the
forward function; model outputs invariant to arbitrary group values. No target
reader statistics, EEG from validation/test, label, text, or duration fit scaling.

Run matched global baseline(no-deconv), true13token schedule module, and zero
shift capacity control. All use valid-slot normalization, physical source,
seed2026, folds0/3, batch64, modern blocks,25ep/patience6, trial-rho selection.
Compare architectures ONLY within the global-normalization cells. Benefits of
dataset repair/normalization are reported separately. Require true schedule
beats both controls on item/conditional rho in both folds, no trial decline vs
baseline. If passed, freeze the identity-independent architecture and run seven
folds × seeds2026/27/28 before any reader or external evaluation.

Execution amendment after point scores, before expansion: variable-SOA overlap can expose duration through duplicated sample spacing. Add stationary-overlap noise falsifier; do not promote the variable-clock route without this validity gate. Freeze a separate fixed .35s sampling protocol before any fixed-grid scores. No architecture gains may be computed across input protocols.
