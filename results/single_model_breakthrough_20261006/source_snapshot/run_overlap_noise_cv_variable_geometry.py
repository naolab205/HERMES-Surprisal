#!/usr/bin/env python3
"""Falsification control: stationary noise with the cached sampling geometry.

Contains no EEG. Timing metadata only creates the diagnostic input geometry;
it is never supplied as a model feature. Streams are independent per reading.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_global_spatial_cv as global_runner
runner = global_runner.runner

class StationaryOverlapStore:
    def __init__(self, shards, slot_index, *, device, max_slots=10, eye=None):
        if eye is not None:
            raise ValueError('This control is EEG-only')
        self.device = device
        self.max_slots = max_slots
        self.eye = self.eye_clean = None
        cache = Path(sys.argv[sys.argv.index('--cache-dir') + 1])
        bags = runner.load_char_bags(cache_dir=cache)
        meta = pd.concat([pd.read_csv(cache / ('meta_' + g.replace('|','_') + '.csv'))
                          for g in bags.groups], ignore_index=True)
        if len(meta) != len(shards) or len(slot_index) != len(bags.obs):
            raise ValueError('Control ordering mismatch')
        offsets = np.zeros((len(slot_index),max_slots),dtype=np.int64)
        masks = np.zeros(offsets.shape,dtype=np.float32)
        for i, indices in enumerate(slot_index):
            if not np.array_equal(indices,bags.slot_index[i]):
                raise ValueError('Control reading order mismatch')
            m = meta.iloc[indices[:max_slots]]
            center = np.rint(m.char_slot.to_numpy()*m.soa.to_numpy()*128).astype(np.int64)
            offsets[i,:len(m)] = center - center[0]
            masks[i,:len(m)] = 1.
        length = max(1024, int(offsets.max()) + 167)
        length = ((length + 255)//256)*256
        self.offsets = torch.as_tensor(offsets,device=device)
        self.mask = torch.as_tensor(masks,device=device)
        self.data = torch.empty((len(slot_index),128,length),dtype=torch.float16,device=device)
        gen = torch.Generator(device=device).manual_seed(782026)
        freq = torch.fft.rfftfreq(length,d=1/128,device=device)
        band = (freq >= .5) & (freq <= 30)
        # One common population variance: no per-reading fitted amplitude.
        scale = 20. / np.sqrt(2*int(band.sum())/length)
        for lo in range(0,len(slot_index),128):
            hi = min(lo+128,len(slot_index))
            x = torch.randn((hi-lo,128,length),generator=gen,device=device)
            x = torch.fft.irfft(torch.fft.rfft(x)*band,n=length)*scale
            self.data[lo:hi] = x.half()
        out = Path(sys.argv[sys.argv.index('--out-dir')+1])
        out.mkdir(parents=True,exist_ok=True)
        report = {'control':'independent stationary band-limited Gaussian streams per reading',
                  'noise_seed':782026,'sampling_hz':128,'band_hz':[.5,30],
                  'population_sd_microvolt':20,'stream_length':length,
                  'n_readings':len(slot_index),'eeg_loaded_into_store':False,
                  'geometry':'same rounded char_slot * row SOA differences; same mask',
                  'scale':'shared real-training-only CSD normalization, identical to real arm'}
        (out/'noise_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
        # Verify exact duplicate samples for one overlapping pair.
        valid = np.flatnonzero((masks[:,1] == 1) & (offsets[:,1] < 167))
        i = int(valid[0]); batch,_ = self.batch(torch.tensor([i],device=device))
        delta = int(offsets[i,1])
        assert torch.equal(batch[0,0,:,delta:],batch[0,1,:,:167-delta])
        print('[noise] stationary streams generated; exact overlap verified',flush=True)

    @property
    def bytes_on_device(self):
        return self.data.numel()*self.data.element_size()

    def batch(self,obs_indices):
        wave = self.data.index_select(0,obs_indices)
        mask = self.mask.index_select(0,obs_indices)
        offsets = self.offsets.index_select(0,obs_indices)
        index = offsets[:,:,None] + torch.arange(167,device=self.device)
        wave = wave[:,None].expand(-1,self.max_slots,-1,-1)
        eeg = torch.gather(wave,3,index[:,:,None].expand(-1,-1,128,-1)).float()
        return eeg*mask[:,:,None,None],mask

runner.GpuEpochStore = StationaryOverlapStore
if __name__ == '__main__':
    runner.main()
