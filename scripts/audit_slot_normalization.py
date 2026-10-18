#!/usr/bin/env python3
"""Test whether zero padding changes valid EEG normalization statistics."""
import copy,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foundation.deconvnet import TimeResolvedStem
from foundation.model import SpatialTransform
from foundation.deconvnet import DeconvNet

torch.set_num_threads(4);torch.manual_seed(2026)
x=torch.randn(12,128,167)
padded=torch.cat([x,torch.zeros(18,128,167)])
valid=torch.cat([torch.ones(12),torch.zeros(18)])>0
results={}
for mode in (False,True):
    torch.manual_seed(2026)
    a=TimeResolvedStem(dropout=0,valid_slot_norm=mode).train();b=copy.deepcopy(a)
    ya=a(x,torch.ones(12,dtype=torch.bool));yb=b(padded,valid)[:12]
    gap=float((ya-yb).abs().max())
    results[str(mode)]={'valid_feature_max_padding_difference':gap,
                       'running_var_max_padding_difference':float((a.norm1.running_var-b.norm1.running_var).abs().max())}
    if mode: assert gap<2e-5,results
# When every slot is real, native and masked BatchNorm must agree.
torch.manual_seed(2026)
a=TimeResolvedStem(dropout=0).train();b=copy.deepcopy(a);b.valid_slot_norm=True
ya=a(x);yb=b(x,torch.ones(len(x),dtype=torch.bool))
assert torch.allclose(ya,yb,atol=2e-5,rtol=1e-4)
yb.square().mean().backward()
assert torch.isfinite(b.spatial.weight.grad).all() and b.spatial.weight.grad.abs().sum()>0
# Eval output may not depend on added batch padding.
a.eval();b.eval()
assert torch.allclose(b(x),b(padded)[:12],atol=2e-5,rtol=1e-4)
# Existing default matches the pre-edit baseline exactly with shared seed.
import importlib.util
spec=importlib.util.spec_from_file_location('foundation.deconvnet_old',ROOT/'results/single_model_breakthrough_20261006/source_snapshot/deconvnet_before.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sp=SpatialTransform(np.eye(128,dtype=np.float32)[None],np.ones(1))
torch.manual_seed(2026);m0=old.DeconvNet(sp,use_deconv=False,dropout=0).eval()
torch.manual_seed(2026);m1=DeconvNet(sp,use_deconv=False,dropout=0).eval()
eeg=torch.randn(2,10,128,167);mask=torch.ones(2,10);mask[0,7:]=0
with torch.no_grad():
    diff=float((m0(eeg,mask,torch.zeros(2,dtype=torch.long))['row']-m1(eeg,mask,torch.zeros(2,dtype=torch.long))['row']).abs().max())
assert diff==0,diff
results['default_baseline_max_difference']=diff
out=ROOT/'results/single_model_breakthrough_20261006/normalization_audit.json'
out.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
