#!/usr/bin/env python
"""Compatibility guard for the retired concatenated-row cache builder.

Use rebuild_physical_char_cache.py for windows from continuous recordings.
The archived builder is retained in the experiment source snapshots.
"""
from __future__ import annotations

MESSAGE = (
    "This legacy builder extracts across spliced row intervals and cannot "
    "produce valid long physical-time windows. Use "
    "scripts/rebuild_physical_char_cache.py for the audited 1200ms cache. "
    "Existing caches are preserved; no files were changed."
)


def build_one(stem, out_dir, pre_ms, post_ms, overwrite):
    """Reject calls through the historical Python API as well as the CLI."""
    raise RuntimeError(
        "Spliced row intervals cannot supply complete physical EEG windows; "
        "use rebuild_physical_char_cache.py."
    )


def main() -> None:
    raise SystemExit(MESSAGE)


if __name__ == '__main__':
    main()
