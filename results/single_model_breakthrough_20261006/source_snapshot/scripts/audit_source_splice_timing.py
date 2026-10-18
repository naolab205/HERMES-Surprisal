#!/usr/bin/env python3
"""Quantify clock resets/overlaps introduced by concatenated row blocks."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cogalign.io.events import pair_rows_rowe,preproc_events_path
OLD=Path('/data/2_readEEG/superisal_code/data_cache/char_epochs_1200ms')
DATA=Path('/data/my/ds004952')
rows=[]
for path in sorted(OLD.glob('meta_*.csv')):
    meta=pd.read_csv(path);items=meta.drop_duplicates(['run','row_idx'])
    for run,frame in items.groupby('run'):
        p=preproc_events_path(DATA,frame.subject.iloc[0],frame.book.iloc[0],int(run))
        pairs=pair_rows_rowe(pd.read_csv(p,sep='\t',encoding='utf-8-sig')).set_index('row_idx')
        # The old builder stores rows in their metadata order within each run.
        previous=None
        for r in frame.itertuples():
            pair=pairs.loc[r.row_idx]
            if previous is not None:
                delta=float(pair.start-previous)
                rows.append({'recording':path.stem.removeprefix('meta_'),'run':int(run),'row_idx':int(r.row_idx),'new_start_minus_previous_retained_end_seconds':delta})
            previous=float(pair.end)+float(pair.end-pair.start)/r.n_slots
f=pd.DataFrame(rows);delta=f.new_start_minus_previous_retained_end_seconds.to_numpy()
out=ROOT/'results/single_model_breakthrough_20261006'
f.to_csv(out/'source_splice_timing.csv',index=False)
r={'within_run_row_splices':len(f),'backward_clock_jumps_fraction':float(np.mean(delta<0)),'signed_gap_seconds_quantiles':dict(zip(['p1','p50','p99'],np.quantile(delta,[.01,.5,.99]).tolist())),'interpretation':'Negative gap means the retained lead-out already contains EEG after the next row starts, then concatenation repeats those times. Run-boundary splices are additional and not included in this statistic.'}
(out/'source_splice_timing.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
