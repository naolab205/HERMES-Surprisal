"""Physical-support consensus on overlapping EEG feature windows.

Each token covers four real samples. W transports token supports to a common
sample grid. P = W diag(W.T 1)^-1 W.T is symmetric and doubly stochastic.
It preserves feature sums while reconciling overlapping representations.
"""
import torch
from torch import nn

def support_projection(slots=10,tokens=41,stride=4,period=.35,sfreq=128):
    offsets=torch.round(torch.arange(slots,dtype=torch.float64)*period*sfreq).long()
    starts=(offsets[:,None]+torch.arange(tokens)[None]*stride).reshape(-1)
    cells=torch.arange(int((starts.max()+stride+stride-1)//stride))*stride
    overlap=(torch.minimum(starts[:,None]+stride,cells[None]+stride)-
             torch.maximum(starts[:,None],cells[None])).clamp_min(0).float()/stride
    assert torch.allclose(overlap.sum(1),torch.ones(len(starts)))
    mass=overlap.sum(0)
    # Fractional cells require their actual positive mass.
    p=(overlap/mass.clamp_min(1e-12)[None])@overlap.T
    assert torch.allclose(p.sum(1),torch.ones(len(starts)),atol=1e-6)
    return p

class SupportConsensus(nn.Module):
    def __init__(self,d_model,stride=4,steps=3,generic=False):
        super().__init__()
        if stride!=4:raise ValueError('Frozen support consensus requires stride4')
        self.tokens=167//stride
        if generic:
            kernel=torch.zeros(self.tokens,self.tokens)
            for i in range(self.tokens):
                kernel[i,i]=.5
                kernel[i,max(i-1,0)]+=.25
                kernel[i,min(i+1,self.tokens-1)]+=.25
            p=torch.block_diag(*([kernel]*10))
        else:p=support_projection(tokens=self.tokens,stride=stride)
        self.register_buffer('projection',p)
        self.gates=nn.Parameter(torch.zeros(steps,d_model))

    def forward(self,h,mask):
        b,s,t,d=h.shape
        if s!=10 or t!=self.tokens or not bool((mask==1).all()):
            raise ValueError('Consensus requires the frozen equal10slot cohort')
        x=h.reshape(b,s*t,d)
        for gate in self.gates:
            consensus=torch.matmul(self.projection,x)
            x=x+gate.tanh()[None,None]*(consensus-x)
        return x.reshape_as(h)

    def gate_magnitudes(self):
        return [float(g.detach().tanh().norm()) for g in self.gates]
