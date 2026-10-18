# Single-model breakthrough exploration — 2026-10-06

No ensembles, no new linguistic labels, no manuscript edits. Preserve frozen
GPT2 labels/context192, purged sentence manifest, EEG-only inference and
retrospective status. No text, eye, target label, or reader embedding at inference.

## Diagnosed next layer: normalization

Previous fine-grained labels, coarse response dictionary and raw-window reader
teacher screens failed. Length/book controls remain strong. The EEG stem applies
BatchNorm BEFORE the character mask, so padding fits its running statistics.
An isolated same-signal probe shows valid features change when adding zero slots;
valid-only statistics remove this dependence without adding parameters.
This is a correctness/inductive-bias intervention, not a novelty claim.

## Registered experiment (before training)

Two arms: default all-slot BatchNorm vs valid-slot BatchNorm only.
Same DeconvNet with deconvolution disabled, modern blocks, CSD, row loss,
seed2026, folds0/3 (validation blocks1/4), batch64, max25epochs/patience6,
matched initialization and training RNG, fixed trial-rho checkpoint selection.
No local-label auxiliary, extra capacity, or architecture bundles.
Save selected checkpoints and predictions to permit mechanism audits.
Rerun baseline to verify pipeline drift against archived row-only result.
Both folds must improve item and book×slot-count conditional rho; trial rho
cannot decline. If either fold fails, no seven-fold expansion of this candidate.

## Separate representation hypothesis

The stem's time block is initialized close to identity, then mean+max pooling
removes anchored latency. Longer windows previously helped while deconvolution
and local labels did not. Hypothesis: row-level distributed evidence requires
latency-specific integration without unreliable local targets. First examine
fitted time-block/pooling sensitivity and training-text-only signed-window
features. A latency module may be considered only after a diagnostic supplies
new evidence, with matched temporal-scramble control. Do not revive the rejected
label-estimated response dictionary or use validation to fit response templates.

## Scope and stopping

These are cheap retrospective development screens, not fresh independent tests.
Only a qualifying single model expands to seven folds/three seeds and null,
layout, reader and external checks under the prior >=.02 mean-gain rules.
Archive all failures and source hashes. Do not report selecting a favorable
checkpoint or averaging models as progress. Missing local-label mapping is a
separate data limitation, not assumed to explain every failure.
