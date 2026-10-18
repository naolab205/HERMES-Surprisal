#!/usr/bin/env python3
"""Matched-parameter generic-neighbor control for temporal deconvolution."""
import run_charbag_cv as runner
OriginalDecoder=runner.DeconvNet
class UnshiftedDecoder(OriginalDecoder):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not self.use_deconv:raise ValueError('Generic neighbor control requires deconv capacity')
        self.deconv.shift=0
        for step in self.deconv.steps:step.shift=0
runner.DeconvNet=UnshiftedDecoder
if __name__=='__main__':runner.main()
