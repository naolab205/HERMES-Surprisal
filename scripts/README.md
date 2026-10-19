# Script guide

Run commands from the repository root. The active model and reproduction entries are listed below; the remaining utilities retain the research controls and their audit paths.

## Start here

| Script | Purpose | Requires signal caches? |
| :-- | :-- | :-- |
| [`evaluate_surprisal_predictions.py`](evaluate_surprisal_predictions.py) | Recompute metrics from archived predictions | No; conditional book/count metrics additionally need metadata |
| [`plot_release_overview.py`](plot_release_overview.py) | Regenerate the README result figure | No |
| [`run_charbag_cv.py`](run_charbag_cv.py) | Train the hierarchical model or its matched controls | Yes |
| [`build_readme_animations.py`](build_readme_animations.py) | Render the conceptual header and architecture | No |
| [`build_extended_scalp_animations.py`](build_extended_scalp_animations.py) | Reproduce supplementary S1/S2 from saved arrays | No |

[Quick start](../README.md#quick-start) · [Cache requirements and training protocol](../docs/REPRODUCIBILITY.md).

## Physical recording inputs

- [`rebuild_physical_char_cache.py`](rebuild_physical_char_cache.py) reconstructs windows from continuous recordings and checks them against existing metadata and cache interiors. Its local source paths must be adapted before use.
- [`build_fixed_grid_cache.py`](build_fixed_grid_cache.py) defines the separate fixed-grid sampling experiment.
- [`extract_extended_scalp.py`](extract_extended_scalp.py) re-extracts the supplementary scalp summaries with row-boundary and source checks.
- [`build_long_window_cache.py`](build_long_window_cache.py) is a compatibility guard for a retired concatenated-row builder. It stops without reading or writing caches.

The repository does not provide a one-command installation of raw EEG preprocessing or exact retraining of the archived multimodal weights.

<details>
<summary><strong>Research controls and supporting utilities</strong></summary>

The `run_*`, `probe_*`, `audit_*`, `summarize_*` and `bootstrap_*` utilities correspond to the [archived experiment records](../results/README.md). They include optional models, diagnostic controls and local-path research scripts. Their inclusion does not make them part of the main model.

Historical source snapshots remain alongside the results when an audit imports that version or a recorded checksum identifies it. See [research history](../docs/research_history.md) and [provenance](../docs/PROVENANCE.md).

</details>
