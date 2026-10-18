# Extended scalp animations: 2 seconds and ten readers

This extension re-extracts actual continuous EEG around observed row starts. It covers **0–2000 ms**, using **100-ms window means advanced by 50 ms**, for **39 measured frames**. Playback holds each frame for **100 ms**. Adjacent windows overlap; they are not independent samples or additional hypothesis tests.

## S3 · Group contrast over a longer interval

![Extended group high-minus-low scalp contrast](group_extended.gif)

The signed high-minus-low contrast is averaged within each book and reader, then across the ten available healthy readers. The video uses one fixed symmetric voltage scale for all frames. The RMS trace consists of actual window summaries. No temporal morphing or intermediate EEG sample generation is used.

## S4 · All ten readers shown together

![Ten-reader row-onset scalp contrast mosaic](readers_extended.gif)

Each head shows one reader's signed high-minus-low voltage in the same time window. Every reader and frame shares one fixed voltage scale, so differences are not hidden by independent color rescaling. RMS labels summarize the 128 sensor values for that reader and interval. A pale map can reflect smaller amplitude under the shared scale, rather than missing data.

The participants are **healthy readers, not patients**. ChineseEEG has ten participants in this analysis. Nine have both books; `sub-15` contributes the available GarnettDream recording only. The tenth participant is real and is not a duplicated or generated case. The ten-reader available-book mean differs from the balanced nine-reader mean in manuscript Figures 5 and 7. The original nine-reader results and S1/S2 remain unchanged.

## Data and controls

The source is the dataset authors' `filtered_0.5_30` continuous EEG derivative. Rows belong to the same setting-0 training blocks 2–6 and the same ten-complete-slot eligibility rule. Thresholds are frozen training-derived within-book surprisal terciles. The script applies the existing 256-to-128 Hz resampling, amplitude clipping, original cache precision, and 13-sample pre-row reference. It retains only complete windows that stay within the target row and the continuous recording.

The extraction checks source prefixes against the existing physical first-slot cache and compares the original nine-reader 0–800-ms contrasts with the manuscript arrays. Exact checks, source hashes, condition counts, exclusions, and book availability are saved in [extraction provenance](extended_extraction_provenance.json). Display scales, playback timings, and source-array hash are in [rendering provenance](rendering_provenance.json).

These are descriptive scalp-voltage displays. They are not cortical localization, model attribution, or a validated diagnosis. Adding a reader or refining the windows does not supply a new independent model evaluation.

## Reproduce the graphics

```bash
python scripts/build_extended_scalp_animations.py
```

This command uses the included `extended_source_arrays.npz` and sensor coordinates, so it does not need a new EEG download. Re-extracting the source waveforms requires the dataset, eligibility cache, and frozen split manifest:

```bash
python scripts/extract_extended_scalp.py \
  --data-root /path/to/ds004952 \
  --cache-dir /path/to/char_epochs_fixed_grid_350ms_20261006 \
  --manifest /path/to/sentence_manifest.csv \
  --workers 6
```

The cache supplies original observation order, metadata, and prefix validation. Its large EEG arrays are not distributed in this repository. Do not use raw data or caches from a different split while calling the resulting animation an exact reproduction.
