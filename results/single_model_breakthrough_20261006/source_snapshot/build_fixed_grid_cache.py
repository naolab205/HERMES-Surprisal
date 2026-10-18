#!/usr/bin/env python3
"""Fixed-clock EEG tiles anchored to ROWS, independent of recorded row duration.

The .35s grid is a sampling convention, not verified character trigger timing.
All raw source, labels, population and window width stay fixed.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import rebuild_physical_char_cache as physical

PERIOD = .35
NEW = physical.ROOT/'data_cache/char_epochs_fixed_grid_350ms_20261006'

def build(path):
    import mne
    mne.set_log_level('ERROR')
    stem=path.stem.removeprefix('meta_')
    dest=NEW/f'X_{stem}.npy'; report_path=NEW/f'{stem}_provenance.json'
    if dest.exists() and report_path.exists():
        r=json.loads(report_path.read_text())
        if r['period_s'] != PERIOD or physical.sha(dest) != r['cache_sha256']:
            raise ValueError('Existing fixed-grid cache mismatch')
        return r
    meta=pd.read_csv(path)
    source=[];checks=[]
    output=np.lib.format.open_memmap(NEW/f'X_{stem}.partial.npy',mode='w+',
                                    dtype=np.float16,shape=(len(meta),128,167))
    for run,frame in meta.groupby('run',sort=True):
        subject=frame.subject.iloc[0]; book=frame.book.iloc[0];session='ses-'+book
        prefix=f'{subject}_{session}_task-reading_run-{int(run):02d}'
        vhdr=physical.DATA/'derivatives/preproc/filtered_0.5_30'/subject/session/'eeg'/f'{prefix}_eeg.vhdr'
        events=physical.preproc_events_path(physical.DATA,subject,book,int(run))
        pairs=physical.pair_rows_rowe(pd.read_csv(events,sep='\t',encoding='utf-8-sig')).set_index('row_idx')
        raw=physical._read_raw_brainvision_robust(vhdr);raw.pick('eeg')
        if len(raw.ch_names)!=128 or raw.info['sfreq']!=256:
            raise ValueError('Unexpected physical source')
        signal=resample_poly(raw.get_data()*1e6,1,2,axis=1).astype(np.float32)
        del raw
        np.clip(signal,-300,300,out=signal)
        for row in frame.itertuples():
            # No duration, SOA, row-end marker, target or text enters this formula.
            center=round(float(pairs.loc[int(row.row_idx)].start)*128)+round(int(row.char_slot)*PERIOD*128)
            lo,hi=center-13,center+154
            if lo<0 or hi>signal.shape[1]:
                raise ValueError('Fixed tile exceeds physical recording')
            output[row.Index]=signal[:,lo:hi].astype(np.float16)
        # Deterministic first/middle/last direct-source verification each run.
        for index in frame.index[[0,len(frame)//2,-1]]:
            row=meta.loc[index]
            center=round(float(pairs.loc[int(row.row_idx)].start)*128)+round(int(row.char_slot)*PERIOD*128)
            assert np.array_equal(output[index],signal[:,center-13:center+154].astype(np.float16))
            checks.append({'epoch_index':int(index),'physical_center_sample':int(center)})
        source.append({'run':int(run),'vhdr_sha256':physical.sha(vhdr),
                       'eeg_sha256':physical.sha(vhdr.with_suffix('.eeg')),
                       'events_sha256':physical.sha(events)})
        print(f'[fixed-grid] {stem} run{run} complete',flush=True)
    output.flush();del output
    (NEW/f'X_{stem}.partial.npy').replace(dest)
    (NEW/path.name).write_bytes(path.read_bytes())
    r={'status':'complete','recording':stem,'epochs':len(meta),'period_s':PERIOD,
       'source_files':source,'direct_source_checks':checks,
       'metadata_sha256':physical.sha(path),'cache_sha256':physical.sha(dest)}
    report_path.write_text(json.dumps(r,indent=2)+'\n')
    return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6)
    args=ap.parse_args(); NEW.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(build,p) for p in sorted(physical.OLD.glob('meta_*.csv'))]
        reports=[job.result() for job in as_completed(jobs)]
    assert len(reports)==19 and sum(r['epochs'] for r in reports)==917096
    (NEW/'csd_matrix.npy').write_bytes((physical.OLD/'csd_matrix.npy').read_bytes())
    r={'status':'complete','sampling_convention':'ROWS + char_slot * .35s; not verified character triggers',
       'period_s':PERIOD,'sfreq_hz':128,'window_samples':[-13,154],
       'n_epochs':917096,'preserved_metadata':True,
       'duration_independent_geometry':True,'recordings':reports}
    (NEW/'CACHE_AUDIT.json').write_text(json.dumps(r,indent=2)+'\n')
    print('ALL FIXED-GRID TILES VERIFIED',flush=True)

if __name__=='__main__':main()
