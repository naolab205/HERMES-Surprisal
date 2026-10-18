#!/usr/bin/env python
"""Build longer Eye windows from original gaze streams and archived row metadata."""
import inspect
from pathlib import Path
import build_eye_char_cache as builder
builder.POST_MS=1200.
builder.N_SAMPLES=65
# Row epoch signals are unnecessary: recover the identical displayed rows from
# the archived character-cache metadata. Read original eye stream per run.
source=inspect.getsource(builder.build_one)
source=source.replace('if not source.exists():', 'if False:')
source=source.replace('row_meta = pd.DataFrame(np.load(source, allow_pickle=True)["meta"])', '''row_meta = pd.read_csv(Path("/data/2_readEEG/superisal_code/data_cache/char_epochs_1200ms") / f"meta_{subject}_{book}.csv").drop_duplicates(["run", "row_idx"])''')
exec(source,builder.__dict__)
builder.main()
