import sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_shared_reference_cv import shared_reference
from foundation.data import baseline_correct

torch.manual_seed(2026)
x=torch.randn(3,10,128,167);mask=torch.ones(3,10);mask[0,5:]=0;mask[1,7:]=0
x=x*mask[:,:,None,None];x.requires_grad_(True)
local=baseline_correct(x);shared=shared_reference(x,mask)
weight=mask/mask.sum(1,keepdim=True)
a=torch.einsum('bs,bsct->bct',weight,local)
b=torch.einsum('bs,bsct->bct',weight,shared)
assert torch.allclose(a,b,atol=1e-6)
assert shared[mask==0].abs().max()==0
# Constant channel offsets are removed; padding values must not enter reference.
offset=torch.randn(3,1,128,1)
assert torch.allclose(shared_reference(x+offset,mask),shared,atol=1e-6)
shared.square().mean().backward();assert torch.isfinite(x.grad).all()
# Physical overlap: two shifted epochs share raw samples. Local baselining
# creates a different zero level per epoch; shared reference retains equality.
stream=torch.randn(128,220)
epochs=torch.stack([stream[:,:167],stream[:,53:220]])[None]
cmask=torch.ones(1,2)
l=baseline_correct(epochs);r=shared_reference(epochs,cmask)
local_error=float((l[0,0,:,53:]-l[0,1,:,:114]).abs().mean())
shared_error=float((r[0,0,:,53:]-r[0,1,:,:114]).abs().max())
assert shared_error<1e-6
result={'row_average_max_difference':float((a-b).detach().abs().max()),'synthetic_same_physical_samples_local_error':local_error,'synthetic_same_physical_samples_shared_max_error':shared_error,'scope':'algebra/gradient diagnostic; no performance claim'}
(ROOT/'results/single_model_breakthrough_20261006/shared_reference_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
