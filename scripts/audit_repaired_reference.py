#!/usr/bin/env python3
"""Check whether old first-slot baseline is actually future EEG of that slot."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cogalign.io.events import pair_rows_rowe,preproc_events_path
OLD=Path('/data/2_readEEG/superisal_code/data_cache/char_epochs_1200ms')
NEW=ROOT/'data_cache/char_epochs_physical_1200ms_20261006'
DATA=Path('/data/my/ds004952');results=[]
for stem in ('sub-04_LittlePrince','sub-04_GarnettDream'):
    assert (NEW/f'{stem}_provenance.json').exists()
    meta=pd.read_csv(OLD/f'meta_{stem}.csv');a=np.load(OLD/f'X_{stem}.npy',mmap_mode='r');b=np.load(NEW/f'X_{stem}.npy',mmap_mode='r')
    first=meta[meta.char_slot.eq(0)];n=0;matched=0;times=[];errors=[]
    for run,frame in meta.drop_duplicates(['run','row_idx']).groupby('run'):
        event=preproc_events_path(DATA,frame.subject.iloc[0],frame.book.iloc[0],int(run))
        pairs=pair_rows_rowe(pd.read_csv(event,sep='\t',encoding='utf-8-sig')).set_index('row_idx')
        prev=None
        for r in frame.itertuples():
            pair=pairs.loc[r.row_idx]
            center=int(round(float(pair.start)*128))
            idx=first.index[first.run.eq(run)&first.row_idx.eq(r.row_idx)]
            if prev is not None and len(idx):
                i=int(idx[0]);n+=1
                offset=prev-center
                if 13<=offset and offset+13<=167:
                    # Old baseline samples from prev segment's last13:
                    # in the correct epoch this is [offset,offset+13].
                    matched+=int(np.array_equal(a[i,:,:13],b[i,:,offset:offset+13]))
                    times.append((offset-13)/128.)
                errors.append(float(np.sqrt(np.mean((a[i,:,:13].astype(np.float32)-b[i,:,:13].astype(np.float32))**2))))
            prev=int(round((float(pair.end)+float(pair.end-pair.start)/r.n_slots)*128))
    results.append({'recording':stem,'first_slots_examined':n,'baseline_exactly_matches_future_13_samples':matched,'future_interval_starts_seconds_median':float(np.median(times)),'old_vs_true_baseline_sample_rmse_uv_median':float(np.median(errors))})
out=ROOT/'results/single_model_breakthrough_20261006/repaired_reference_audit.json';out.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
