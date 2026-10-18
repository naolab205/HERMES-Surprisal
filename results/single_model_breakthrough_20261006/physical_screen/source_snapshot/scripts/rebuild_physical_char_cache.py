#!/usr/bin/env python3
"""Re-extract locked epochs from actual continuous recordings, never row splices.

Preserves old metadata, epoch order and target population. Uses the same .5–30
Hz preprocessed source, 256->128Hz polyphase resampling and amplitude clip.
Only physical EEG provenance changes. No target labels fit any transformation.
"""
import argparse,hashlib,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from cogalign.io.events import pair_rows_rowe,preproc_events_path
from scripts.build.build_epoch_cache import _read_raw_brainvision_robust
DATA=Path('/data/my/ds004952')
OLD=Path('/data/2_readEEG/superisal_code/data_cache/char_epochs_1200ms')
CONT=OLD.parent/'continuous'
NEW=ROOT/'data_cache/char_epochs_physical_1200ms_20261006'
OUT=ROOT/'results/single_model_breakthrough_20261006'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def audit():
    rows=[]
    for path in sorted(OLD.glob('meta_*.csv')):
        stem=path.stem.removeprefix('meta_');meta=pd.read_csv(path)
        events=pd.read_csv(CONT/f'events_{stem}.csv')
        keys=['subject','book','run','row_idx','char_slot']
        joined=meta.merge(events[keys+['sample','block_start','block_end']],on=keys,validate='one_to_one')
        pre=joined['sample']-13<joined.block_start
        post=joined['sample']+154>joined.block_end
        rows.append({'recording':stem,'epochs':len(joined),'pre_crosses_splice':int(pre.sum()),'post_crosses_splice':int(post.sum()),'either_crosses_splice':int((pre|post).sum())})
    summary={'recordings':rows,'epochs':sum(r['epochs'] for r in rows),'pre_crosses_splice':sum(r['pre_crosses_splice'] for r in rows),'post_crosses_splice':sum(r['post_crosses_splice'] for r in rows),'either_crosses_splice':sum(r['either_crosses_splice'] for r in rows),'finding':'Old long-window builder checks concatenated recording limits but ignores row block_start/block_end. Source timeline may jump or repeat across those splices. This is not a measured performance explanation.'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'physical_boundary_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='recordings'},indent=2),flush=True)

def build(path):
    import mne
    mne.set_log_level('ERROR')
    stem=path.stem.removeprefix('meta_');dest=NEW/f'X_{stem}.npy';report_path=NEW/f'{stem}_provenance.json'
    if dest.exists() and report_path.exists():return json.loads(report_path.read_text())
    meta=pd.read_csv(path);old=np.load(OLD/f'X_{stem}.npy',mmap_mode='r')
    assert len(meta)==len(old) and old.shape[1:]==(128,167)
    temp=NEW/f'X_{stem}.partial.npy'
    output=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float16,shape=old.shape)
    source=[];interior_errors=[];changed=0
    for run,frame in meta.groupby('run',sort=True):
        subject=frame.subject.iloc[0];book=frame.book.iloc[0];session='ses-'+book
        prefix=f'{subject}_{session}_task-reading_run-{int(run):02d}'
        vhdr=DATA/'derivatives/preproc/filtered_0.5_30'/subject/session/'eeg'/f'{prefix}_eeg.vhdr'
        events_path=preproc_events_path(DATA,subject,book,int(run))
        pairs=pair_rows_rowe(pd.read_csv(events_path,sep='\t',encoding='utf-8-sig')).set_index('row_idx')
        raw=_read_raw_brainvision_robust(vhdr);raw.pick('eeg')
        if len(raw.ch_names)!=128 or raw.info['sfreq']!=256.:raise ValueError('Source shape/rate mismatch')
        signal=resample_poly(raw.get_data()*1e6,up=1,down=2,axis=1).astype(np.float32);del raw
        np.clip(signal,-300.,300.,out=signal)
        eeg_path=vhdr.with_suffix('.eeg')
        source.append({'run':int(run),'vhdr_sha256':sha(vhdr),'events_sha256':sha(events_path),'eeg_sha256':sha(eeg_path),'eeg_bytes':eeg_path.stat().st_size})
        cont=pd.read_csv(CONT/f'events_{stem}.csv');cont=cont[cont.run.eq(run)].set_index(['row_idx','char_slot'])
        for row in frame.itertuples():
            pair=pairs.loc[int(row.row_idx)]
            duration=float(pair.end-pair.start);soa=duration/int(row.n_slots)
            if abs(soa-row.soa)>1e-8:raise ValueError('Timing metadata mismatch')
            center=int(round(float(pair.start)*128))+int(round(int(row.char_slot)*soa*128))
            lo,hi=center-13,center+154
            if lo<0 or hi>signal.shape[1]:raise ValueError('Incomplete physical window; fail closed')
            output[row.Index]=signal[:,lo:hi].astype(np.float16)
            event=cont.loc[(row.row_idx,row.char_slot)]
            interior=(event['sample']-13>=event.block_start and event['sample']+154<=event.block_end)
            # Same-source interior epochs must be bit-identical. Check all,
            # not only selected windows or favorable examples.
            same=np.array_equal(output[row.Index],old[row.Index])
            changed+=not same
            if interior and not same:
                interior_errors.append({'index':int(row.Index),'max_error':float(np.abs(output[row.Index].astype(np.float32)-old[row.Index].astype(np.float32)).max())})
        print(f'[cache] {stem} run{run} done',flush=True)
    output.flush();del output
    if interior_errors:raise ValueError(f'Unexpected interior epoch drift {interior_errors[:3]}')
    temp.replace(dest);(NEW/path.name).write_bytes(path.read_bytes())
    report={'recording':stem,'epochs':len(meta),'changed_epochs':int(changed),'interior_drift':len(interior_errors),'metadata_sha256':sha(path),'cache_sha256':sha(dest),'source_files':source,'status':'complete'}
    report_path.write_text(json.dumps(report,indent=2)+'\n');return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--audit-only',action='store_true');args=ap.parse_args()
    audit()
    if args.audit_only:return
    NEW.mkdir(parents=True,exist_ok=True)
    paths=sorted(OLD.glob('meta_*.csv'));reports=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(build,path) for path in paths]
        for job in as_completed(jobs):
            r=job.result();reports.append(r);print(f'[complete] {r["recording"]} {r["epochs"]} epochs',flush=True)
    if len(reports)!=19 or sum(r['epochs'] for r in reports)!=917096:raise ValueError('Incomplete full cache')
    (NEW/'csd_matrix.npy').write_bytes((OLD/'csd_matrix.npy').read_bytes())
    result={'status':'complete','window_samples':[-13,154],'sfreq_hz':128,'preserved_metadata':True,'old_cache':str(OLD),'source':'actual .5–30Hz recording time, same per-run resampling','n_epochs':sum(r['epochs'] for r in reports),'changed_epochs':sum(r['changed_epochs'] for r in reports),'recordings':sorted(reports,key=lambda x:x['recording'])}
    (NEW/'CACHE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'physical_cache_rebuild.json').write_text(json.dumps({k:v for k,v in result.items() if k!='recordings'},indent=2)+'\n');print('ALL PHYSICAL EPOCHS VERIFIED',flush=True)
if __name__=='__main__':main()
