#!/usr/bin/env python3
"""Identity-independent CSD/scale for EEG-only temporal-module experiments."""
from pathlib import Path
import sys
import numpy as np
import torch
import run_charbag_cv as runner
from foundation.model import SpatialTransform
OriginalSpatial=runner.build_spatial
OriginalDecoder=runner.DeconvNet
# Capture the optional control flag outside the unchanged base CLI.
generic='--generic-neighbor-control' in sys.argv
if generic:sys.argv.remove('--generic-neighbor-control')
if generic:
    from run_unshifted_deconv_cv import UnshiftedDecoder
    runner.DeconvNet=UnshiftedDecoder
else:runner.DeconvNet=OriginalDecoder

class IdentityIndependentSpatial(SpatialTransform):
    def forward(self,eeg,group=None):
        # Recording/reader identifiers are deliberately never read.
        with torch.amp.autocast(eeg.device.type,enabled=False):
            return torch.einsum('cd,bsdt->bsct',self.matrices[0].float(),eeg.float())

def shared_spatial(bags,frame,shards,**kwargs):
    if kwargs.get('use_ea') or not kwargs.get('use_csd'):
        raise ValueError('Registered global-spatial route requires CSD without EA')
    local=OriginalSpatial(bags,frame,shards,**kwargs)
    csd=np.load(Path(kwargs['csd_dir'])/'csd_matrix.npy').astype(np.float32)
    matrices=local.matrices.cpu().numpy()
    scale=np.sum(matrices*csd[None],axis=(1,2))/np.sum(csd**2)
    train=frame[frame.split.eq('train')]
    weight=np.array([sum(len(bags.slot_index[int(i)]) for i in train.loc[train.group_id.eq(g),'obs_index']) for g in range(len(bags.groups))],dtype=float)
    if (scale<=0).any() or not np.isfinite(scale).all():raise ValueError('Invalid training scales')
    global_scale=1./np.sqrt(np.average(1./scale**2,weights=weight))
    shared=np.broadcast_to(csd[None],matrices.shape).copy()
    result=IdentityIndependentSpatial(shared,np.full(len(bags.groups),global_scale,dtype=np.float32))
    assert torch.equal(result.matrices,result.matrices[0].expand_as(result.matrices))
    print(f'[global-spatial] shared training-only scale={global_scale:.9g}; reader/group ignored at inference',flush=True)
    return result
runner.build_spatial=shared_spatial
if __name__=='__main__':runner.main()
