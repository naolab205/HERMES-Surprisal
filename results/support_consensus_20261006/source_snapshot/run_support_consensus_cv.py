#!/usr/bin/env python3
"""Native support-consensus screen on equal-shape EEG, matched controls."""
import sys
from pathlib import Path
from dataclasses import replace
import torch
armmode='baseline'
for name in ['support','generic']:
    flag='--consensus-'+name
    if flag in sys.argv:
        sys.argv.remove(flag);armmode=name
import run_global_spatial_cv as global_runner
runner=global_runner.runner
from foundation.deconvnet import DeconvNet
from foundation.support_consensus import SupportConsensus
OriginalLoad=runner.load_char_bags

def equal_cohort(**kwargs):
    bags=OriginalLoad(**kwargs)
    keep=bags.obs.n_slots.eq(10).to_numpy().nonzero()[0]
    return replace(bags,obs=bags.obs.iloc[keep].reset_index(drop=True),
                   slot_index=[bags.slot_index[int(i)] for i in keep])

class ConsensusDecoder(DeconvNet):
    def __init__(self,*args,**kwargs):
        kwargs['use_deconv']=False
        super().__init__(*args,**kwargs)
        if armmode!='baseline':
            self.deconv=SupportConsensus(kwargs.get('d_model',96),
                stride=kwargs.get('stride',4),generic=armmode=='generic')
            self.use_deconv=True
    def forward(self,eeg,mask,group,eye=None):
        if not bool((mask==1).all()):raise ValueError('Unequal support in primary cohort')
        # All windows reference the same physical pre-row samples. This keeps
        # duplicate source samples identical; both controls use this reference.
        eeg=eeg-eeg[:,0,:,:13].mean(-1)[:,None,:,None]
        return super().forward(eeg,mask,group,eye)

def raw_prepare(eeg,args):
    if args.arch!='deconv' or args.per_epoch_mad or args.eye_clean or args.deconv_eye:
        raise ValueError('Frozen EEG-only support screen')
    return eeg

runner.load_char_bags=equal_cohort
runner.DeconvNet=ConsensusDecoder
runner.prepare=raw_prepare

def stable_correlation(pred,target):
    with torch.amp.autocast(pred.device.type,enabled=False):
        p=pred.float()-pred.float().mean()
        t=target.float()-target.float().mean()
        return 1-(p@t)/(p.norm()*t.norm()).clamp_min(1e-8)

runner.correlation_loss=stable_correlation
torch.set_autocast_dtype('cuda',torch.bfloat16)
if __name__=='__main__':
    cache=Path(sys.argv[sys.argv.index('--cache-dir')+1])
    if cache.name!='char_epochs_fixed_grid_350ms_20261006':raise ValueError('Fixed grid required')
    print('[precision] BF16; correlation loss float32; matched in all three arms',flush=True)
    runner.main()
