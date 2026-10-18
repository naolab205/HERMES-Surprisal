"""Represent each real EEG sample once before nonlinear temporal encoding."""
import torch
from .deconvnet import DeconvNet
from .hiernet import _apply
from .modern import ModernStack

OFFSETS=tuple(round(i*.35*128) for i in range(10))

def unique_trajectory(eeg,offsets=OFFSETS):
    if eeg.ndim!=4 or eeg.shape[1]!=10 or eeg.shape[-1]!=167:
        raise ValueError('Unique trajectory requires full fixed-support ten-slot input')
    if offsets[0]!=0 or any(b<=a or b-a>167 for a,b in zip(offsets,offsets[1:])):
        raise ValueError('Invalid support geometry')
    # The source cache audit proves duplicates are exact. Retain the first
    # support and then only newly covered tails; no averaging or new samples.
    parts=[eeg[:,0]]
    for slot in range(1,10):
        added=offsets[slot]-offsets[slot-1]
        parts.append(eeg[:,slot,:,-added:])
    return torch.cat(parts,dim=-1)

class UniqueTrajectoryDecoder(DeconvNet):
    def __init__(self,*args,**kwargs):
        kwargs['use_deconv']=False
        super().__init__(*args,**kwargs)

    def forward(self,eeg,mask,group,eye=None):
        if eye is not None or not bool((mask==1).all()):
            raise ValueError('Frozen equal-shape EEG-only trajectory input required')
        wave=unique_trajectory(eeg)
        return self.forward_trajectory(wave,group)

    def forward_trajectory(self,wave,group):
        if wave.shape[-1]!=570:raise ValueError('Fixed physical support required')
        wave=wave-wave[...,:13].mean(-1,keepdim=True)
        x=self.spatial_op(wave[:,None],group).squeeze(1)
        h=self.stem(x,torch.ones(len(x),device=x.device,dtype=torch.bool))
        if not isinstance(self.time_block,ModernStack):h=self.time_position(h)
        h=_apply(self.time_block,h,None)
        # Ten equal chronological bins of unique EEG, not claimed characters.
        z=torch.stack([chunk.mean(1)+chunk.amax(1) for chunk in torch.tensor_split(h,10,dim=1)],dim=1)
        bins=torch.arange(10,device=x.device)[None]
        z=_apply(self.char_block,z+self.slot_position(bins),None)
        pooled=z.mean(1);row=self.head(pooled).squeeze(-1)
        weights=torch.full((len(x),10),.1,device=x.device,dtype=z.dtype)
        return {'row':row,'char':torch.zeros_like(weights),'row_from_chars':row,
                'weights':weights,'embedding':pooled}
