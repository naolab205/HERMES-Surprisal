#!/usr/bin/env python3
"""One bounded reconciliation of inconsistent duplicate-window features."""
import run_support_consensus_cv as experiment
import json,sys
from pathlib import Path
from foundation.unique_trajectory import UniqueTrajectoryDecoder
experiment.runner.DeconvNet=UniqueTrajectoryDecoder
if __name__=='__main__':
    cache=Path(sys.argv[sys.argv.index('--cache-dir')+1])
    audit=json.loads((cache/'CACHE_AUDIT.json').read_text())
    assert audit['status']=='complete' and audit['period_s']==.35 and audit['duration_independent_geometry']
    assert not list(cache.glob('*.partial.npy'))
    experiment.runner.main()
