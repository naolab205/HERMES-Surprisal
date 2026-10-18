# Window-length experiment — 2026-10-08

Completed21 matched validation runs (seven settings × three configurations), seed2026. Test blocks were not scored. Selected-epoch validation scores are exploratory and are not directly interchangeable with archived0.405 test-block scores.

| EEG/Eye post-window(ms) | Item rho | Reading rho | Item AUC |
|---|---:|---:|---:|
| E800_G800 | 0.3993 ± 0.0226 | 0.3594 ± 0.0198 | 0.7451 ± 0.0180 |
| E1200_G800 | 0.4011 ± 0.0289 | 0.3582 ± 0.0211 | 0.7451 ± 0.0216 |
| E1200_G1200 | 0.4001 ± 0.0219 | 0.3566 ± 0.0189 | 0.7448 ± 0.0156 |

Paired differences and descriptive block-bootstrap intervals are in `paired_differences.json`; intervals based on seven reused development blocks are not confirmatory p-values.
Long Eye windows were rebuilt from real raw records; all19 recordings matched old keys and had exact800ms prefixes. EEG operators/scales were matched using training data on the full1200ms cache. Eye normalization was training-only in allarms. Archived duration-derived geometry remains; this study isolates window endpoints within that representation.

Raw Eye timestamp audit: 19 recordings, 221 runs; the median20-sample extension is 400.725ms. 1024 windows end at recording boundaries and remain validity-marked zero-filled; cohort membership is unchanged. This audit does not establish precise EEG/Eye clock synchronization.
