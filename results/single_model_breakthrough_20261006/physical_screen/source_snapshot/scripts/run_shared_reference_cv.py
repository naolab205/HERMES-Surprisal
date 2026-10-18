#!/usr/bin/env python3
"""Single-model row-reference experiment, isolated from active standard runs.

One shared pre-onset reference per row preserves inter-character offsets while
leaving the mask-weighted row-average waveform identical to local baselining.
The fixed spatial operator/scales are still fitted by the same locked default
training-only procedure. No labels or context outside the row enter inference.
"""
import torch
import run_charbag_cv as runner
OriginalDecoder=runner.DeconvNet

def shared_reference(eeg,mask):
    weight=mask.to(eeg.dtype)
    pre=eeg[...,:13].mean(-1,keepdim=True)
    reference=(pre*weight[:,:,None,None]).sum(1,keepdim=True)/weight.sum(1).clamp_min(1.)[:,None,None,None]
    return (eeg-reference)*weight[:,:,None,None]

class SharedReferenceDecoder(OriginalDecoder):
    def forward(self,eeg,mask,group,eye=None):
        return super().forward(shared_reference(eeg,mask),mask,group,eye)

def prepare_raw(eeg,args):
    if args.representation!='char' or args.per_epoch_mad or args.arch!='deconv':
        raise ValueError('Shared-reference screen requires deconv, char, no MAD')
    return eeg

runner.DeconvNet=SharedReferenceDecoder
runner.prepare=prepare_raw
if __name__=='__main__':runner.main()
