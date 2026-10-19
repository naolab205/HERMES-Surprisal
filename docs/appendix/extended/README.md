# Supplementary animations: row-level surprisal

The two animations below accompany manuscript Appendix A. They use recorded
continuous EEG from ten healthy readers and the same frozen row-surprisal
labels as the decoding task. They describe the temporal and reader-level
context of the prediction results.

## S1 · Time course of the row-level EEG contrast

![Supplementary Animation S1: group row-surprisal contrast](group_extended.gif)

S1 follows the high-minus-low row-surprisal voltage over 0–2000 ms after an
observed row start. Group scalp RMS is 0.186 µV at 0–100 ms, 0.434 µV at
700–800 ms, 0.473 µV at 1100–1200 ms, and 0.557 µV at 1900–2000 ms.
The sequence places the main 0–800-ms summary in the context of continued
reading. Later activity can include responses to subsequent highlighted
characters. Its larger contrast does not by itself establish a better decoder
window.

## S2 · Between-reader variation

![Supplementary Animation S2: individual reader contrasts](readers_extended.gif)

S2 follows all ten signed reader contrasts using one voltage scale across
readers and time. At 700–800 ms, reader-level scalp RMS ranges from 0.441 to
2.559 µV, with median 1.091 µV; the group RMS is 0.434 µV. Differences in
voltage sign and spatial pattern can cancel in the group average.

This display adds spatial and temporal context to the main reader clustering,
which summarizes each reader with four RMS magnitudes. A shared text-row
label can coexist with different recorded responses. Single-reading results
therefore remain needed alongside predictions averaged across readers.

### Analysis scope

The analysis uses setting-0 training blocks 2–6 and complete-ten-slot readings.
Within-book training terciles define the high and low surprisal groups.
Nine readers contribute both books; sub-15 contributes GarnettDream only.
Each reader averages the available books before readers receive equal weight.
This cohort differs from the balanced nine-reader analysis in the main text.

Each animation contains 39 measured 100-ms means advanced by 50 ms. Playback
holds each interval for 100 ms. Overlapping windows are not independent
observations. The maps retain fixed voltage scales without temporal morphing.
These unadjusted voltage contrasts provide descriptive context; they do not
identify the model's predictive sensors or supply another generalization test.

<details>
<summary><strong>Data and reproduction</strong></summary>

The directory retains the files needed to reproduce and check these two materials:

- [Numerical source arrays](extended_source_arrays.npz)
- [Continuous-source extraction checks](extended_extraction_provenance.json)
- [Rendering settings and input hash](rendering_provenance.json)
- [Numerical precision check](numerical_precision_check.json)

Rebuild the two GIFs from the saved arrays:

```bash
python scripts/build_extended_scalp_animations.py
```

The script produces `group_extended.gif` (S1), `readers_extended.gif` (S2), and
`rendering_provenance.json`. It does not generate static preview files.
To render into a separate directory:

```bash
python scripts/build_extended_scalp_animations.py --output-dir /path/to/output
```

Re-extracting the underlying EEG summaries requires the public ChineseEEG
recordings, the eligibility cache, and the frozen split manifest:

```bash
python scripts/extract_extended_scalp.py \
  --data-root /path/to/ds004952 \
  --cache-dir /path/to/char_epochs_fixed_grid_350ms_20261006 \
  --manifest /path/to/sentence_manifest.csv \
  --workers 6
```

The extraction checks physical-cache prefixes, row coverage, label thresholds,
condition counts, and numerical agreement with the main nine-reader summaries.
The large signal caches are obtained locally and are not distributed here.

</details>
