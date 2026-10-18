# Source and evidence provenance

HERMES is the publication name for the hierarchical `MM_mid` EEG and eye model. The implementation and archived results come from the local foundation project, originally named `JBHI_Foundation_v2` and later `superisal_code`. Renaming does not define a new experiment or change its results.

The public package preserves model and training source, frozen numerical labels, primary out-of-fold predictions, fold-level comparison records, matched eye-control predictions, observation-window validation predictions, and scientific analysis arrays. Result files are copied from the research package, not regenerated model scores. Some historical reports and source snapshots retain the original local paths; they are provenance records.

The full original launch command and archived multimodal weights were not retained. The archived model used pooled unlabeled eye statistics. New runs use training-only statistics unless explicitly changed. Main data are repeated-reader, retrospectively explored item blocks. The sampled character centers were estimated from row timing, not verified character triggers. Physical-cache and fixed-grid experiments are separate protocols.

Raw or processed signal caches, trained checkpoints, manuscript source, private keys, and personal machine configuration are not part of this repository. The public ChineseEEG dataset must be obtained from its provider. No repository-wide redistribution license has yet been designated; upstream data and model terms still apply.

README graphics were prepared with code. The architecture illustration contains author-arranged explanatory components prepared with AI assistance, as disclosed in the manuscript; data plots use saved numerical results. The release does not claim publication acceptance.

README animations are rendered with Pillow drawing primitives by `scripts/build_readme_animations.py`. They show conceptual message flow, without encoding recorded signals, fitted weights, or experimental scores. Static alternatives are included.
