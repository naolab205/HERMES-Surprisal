# Eye-guided ICA feasibility probe (2026-09-30)

Run from `model_code/` with the prepared environment:

```bash
/home/zcg/.conda/envs/cogalign/bin/python scripts/eye_guided_ica_pilot.py
/home/zcg/.conda/envs/cogalign/bin/python scripts/eye_to_ica_probe.py
```

Both commands use seven named recordings, the dataset's existing ICA component
activations, author exclusion metadata, and synchronized raw eye archives.
`component_event_variance.csv` reports saccade/fixation and invalid-gap
variance ratios. `eye_to_ica_test_r2.csv` reports time-blocked ridge prediction
of every ICA component from eye data and a 10-second eye-time mismatch control.
The JSON files fix thresholds, counts, time split and source recording paths.
The eye-event speed thresholds were selected during this exploratory pilot
after inspecting the first recording's gaze-speed distribution; they are not
preregistered or independently validated.

| Recording | Best aligned test R² | Best component author-excluded? | Best unexcluded test R² |
|---|---:|:---:|---:|
| sub-04 / LittlePrince | 0.292 | yes | 0.017 |
| sub-04 / GarnettDream | 0.237 | yes | 0.012 |
| sub-05 / GarnettDream | 0.089 | yes | 0.037 |
| sub-05 / LittlePrince | 0.000 | no | 0.000 |
| sub-06 / LittlePrince | 0.002 | no | 0.002 |
| sub-07 / LittlePrince | 0.438 | yes | 0.028 |
| sub-08 / GarnettDream | 0.027 | yes | 0.003 |

The largest correctly aligned components have near-zero R² under the
mismatched-eye control. The probe did **not** clean EEG, train a surprisal
decoder, or estimate item Spearman/AUC. The authors' ICA exclusion labels
include non-ocular noise; agreement is only an approximate sanity check.
The apparent headroom for *additional* eye-guided cleaning of the existing
ICA-cleaned cache is limited and inconsistent in this sample.
