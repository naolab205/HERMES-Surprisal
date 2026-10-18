# Induced band-power feasibility probe (2 October 2026)

`scripts/probe_spectral_surprisal.py` uses fold-0 training items to fit a fixed
ridge regressor on theta (4–8 Hz), alpha (8–13 Hz), and beta (13–30 Hz) power
from the early and late 500 ms parts of four character-locked EEG epochs per
reading. Reader/recording normalization uses training observations only.
No text, language-model score, item identity, book, duration, or Eye feature
is a predictor. The validation set contains 600 unique items and 1,800
readings from readers 04–06. The original EEG-only model is scored on exactly
these same rows. A label permutation within book and slot count is a control.

| Validation fold 0, same 600 items | Pooled item ρ | Book/count weighted ρ |
| --- | ---: | ---: |
| Existing time-domain model | 0.3638 | 0.0701 |
| Spectral ridge probe | −0.0754 | 0.0104 |
| Within-stratum shuffled-label control | −0.0947 | −0.0766 |

The probe does not justify training an induced-power branch: it has no useful
conditional advantage over the existing model. It does not rule out all
time-frequency methods, because it uses a fixed four-slot summary, three
readers, one regularization value, and one validation block. No test block or
manuscript file was used to choose a model.

Reproduce from `model_code/` with:

```bash
OPENBLAS_NUM_THREADS=4 /home/zcg/.conda/envs/cogalign/bin/python scripts/probe_spectral_surprisal.py
```
