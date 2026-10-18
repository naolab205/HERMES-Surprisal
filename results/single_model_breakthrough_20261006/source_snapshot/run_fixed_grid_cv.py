#!/usr/bin/env python3
"""Identity-independent single-model fixed-grid decoding and noise control."""
import json
import sys
from pathlib import Path
noise = '--fixed-clock-noise' in sys.argv
if noise: sys.argv.remove('--fixed-clock-noise')
generic = '--generic-neighbor-control' in sys.argv
if generic: sys.argv.remove('--generic-neighbor-control')
import run_global_spatial_cv as global_runner
runner=global_runner.runner
from foundation.deconvnet import DeconvNet

class FixedGridDecoder(DeconvNet):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.use_deconv:
            shift=0 if generic else round(.35*128/kwargs.get('stride',4))
            self.deconv.shift=shift
            for step in self.deconv.steps:step.shift=shift

runner.DeconvNet=FixedGridDecoder
if noise:
    from run_overlap_noise_cv import StationaryOverlapStore
    StationaryOverlapStore.period_s=.35
    runner.GpuEpochStore=StationaryOverlapStore

if __name__=='__main__':
    cache=Path(sys.argv[sys.argv.index('--cache-dir')+1])
    audit=json.loads((cache/'CACHE_AUDIT.json').read_text())
    assert audit['status']=='complete' and audit['duration_independent_geometry']
    assert audit['period_s']==.35 and audit['n_epochs']==917096
    assert not list(cache.glob('*.partial.npy'))
    runner.main()
