"""Extract 2-s row-onset contrasts from actual continuous ChineseEEG recordings.

Preserves setting-0 training membership and ten-slot eligibility. Includes all ten
available healthy readers, with sub-15 contributing GarnettDream only. No target
labels enter EEG transformations. Raw signal caches are never uploaded.
"""
from pathlib import Path
import os,sys,argparse,json,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from cogalign.io.events import pair_rows_rowe,preproc_events_path
from scripts.build.build_epoch_cache import _read_raw_brainvision_robust
PRE=13;POST=256;RATE=128

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def run_task(task):
 import mne
 mne.set_log_level('ERROR')
 subject,book,run,frame,data_root,cache,thresholds,parts_dir=task
 data_root=Path(data_root);cache=Path(cache)
 stem=f'{subject}_ses-{book}_task-reading_run-{int(run):02d}'
 vhdr=data_root/'derivatives/preproc/filtered_0.5_30'/subject/f'ses-{book}'/'eeg'/f'{stem}_eeg.vhdr'
 ep=preproc_events_path(data_root,subject,book,int(run))
 parts_dir=Path(parts_dir);parts_dir.mkdir(parents=True,exist_ok=True)
 part=parts_dir/(subject+'_'+book+'_'+str(run))
 identity=hashlib.sha256((frame.to_csv(index=False)+str(thresholds.tolist())+str(vhdr.stat().st_size)+str(vhdr.stat().st_mtime_ns)+sha(vhdr)+sha(ep)).encode()).hexdigest()
 if part.with_suffix('.json').exists() and part.with_suffix('.npz').exists():
  saved=json.loads(part.with_suffix('.json').read_text())
  if saved.pop('task_identity',None)==identity:
   arrays=np.load(part.with_suffix('.npz'));saved['low_sum']=arrays['low_sum'];saved['high_sum']=arrays['high_sum'];return saved
 pairs=pair_rows_rowe(pd.read_csv(ep,sep='\t',encoding='utf-8-sig')).set_index('row_idx')
 raw=_read_raw_brainvision_robust(vhdr);raw.pick('eeg')
 if len(raw.ch_names)!=128 or raw.info['sfreq']!=256:raise ValueError('Unexpected source shape/rate')
 signal=resample_poly(raw.get_data()*1e6,1,2,axis=1).astype(np.float32);del raw
 np.clip(signal,-300,300,out=signal)
 old=np.load(cache/f'X_{subject}_{book}.npy',mmap_mode='r')
 low_sum=np.zeros((128,PRE+POST),dtype=np.float64);high_sum=low_sum.copy();nlow=nhigh=0
 exclusions=[];prefix_checks=[];eligible=[]
 for row in frame.itertuples():
  event=pairs.loc[int(row.row_idx)];center=round(float(event.start)*RATE);lo,hi=center-PRE,center+POST
  if float(event.end-event.start)<2.0:
   exclusions.append([int(row.row_idx),'target_row_shorter_than_2000ms']);continue
  if lo<0 or hi>signal.shape[1]:
   exclusions.append([int(row.row_idx),'incomplete_recording_window']);continue
  epoch=signal[:,lo:hi].astype(np.float16)
  eligible.append((int(row.row_idx),int(row.epoch_index)))
  if len(prefix_checks)<3:
   if not np.array_equal(epoch[:,:167],np.asarray(old[int(row.epoch_index)])):
    raise ValueError(f'First-slot source mismatch: {subject} {book} {run} {row.row_idx}')
   prefix_checks.append(int(row.row_idx))
  epoch=epoch.astype(np.float32);epoch-=epoch[:,:PRE].mean(-1,keepdims=True)
  if row.surprisal<=thresholds[0]:low_sum+=epoch;nlow+=1
  elif row.surprisal>=thresholds[1]:high_sum+=epoch;nhigh+=1
 result={'subject':subject,'book':book,'run':int(run),'low_sum':low_sum,'high_sum':high_sum,'n_low':nlow,'n_high':nhigh,
         'source':{'run':int(run),'source_file':str(vhdr),'vhdr_sha256':sha(vhdr),'eeg_sha256':sha(vhdr.with_suffix('.eeg')),
                   'events_sha256':sha(ep),'eligible_rows':len(eligible),'prefix_checked_row_indices':prefix_checks,'exclusions':exclusions}}
 np.savez_compressed(part.with_suffix('.npz'),low_sum=low_sum,high_sum=high_sum)
 part.with_suffix('.json').write_text(json.dumps({k:v for k,v in result.items() if k not in ['low_sum','high_sum']}|{'task_identity':identity},indent=2)+'\n')
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,default=Path(os.environ.get('CHINESEEEG_ROOT','/data/my/ds004952')))
 ap.add_argument('--cache-dir',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True)
 ap.add_argument('--out-dir',type=Path,default=ROOT/'docs/appendix/extended');ap.add_argument('--workers',type=int,default=6)
 ap.add_argument('--parts-dir',type=Path,default=ROOT/'outputs/extended_run_parts');ap.add_argument('--pilot',action='store_true');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 labels_path=ROOT/'data_cache/surprisal_1829a99ad4.csv';labels=pd.read_csv(labels_path)
 mf=pd.read_csv(a.manifest);mf=mf[mf.included]
 joined=mf[['book','run','row_idx','block']].merge(labels[['book','run','row_idx','surprisal']],on=['book','run','row_idx'],validate='one_to_one')
 allrows=[]
 for mp in sorted(a.cache_dir.glob('meta_*.csv')):
  meta=pd.read_csv(mp,usecols=['subject','book','run','row_idx','char_slot','n_slots'])
  f=meta.groupby(['subject','book','run','row_idx'],sort=False).agg(n_epochs=('char_slot','size'),n_slots=('n_slots','first'),first=('char_slot','min'))
  f=f[(f.n_epochs==10)&(f.n_slots==10)&(f['first']==0)].reset_index()
  first=meta.loc[meta.char_slot.eq(0),['subject','book','run','row_idx']].copy();first['epoch_index']=first.index
  f=f.merge(first,on=['subject','book','run','row_idx'],validate='one_to_one').merge(joined,on=['book','run','row_idx'],validate='many_to_one')
  allrows.append(f[~f.block.isin([0,1])])
 train=pd.concat(allrows,ignore_index=True)
 thresholds={b:q.drop_duplicates(['run','row_idx']).surprisal.quantile([1/3,2/3]).to_numpy() for b,q in train.groupby('book')}
 tasks=[]
 for (sub,book,run),f in train.groupby(['subject','book','run'],sort=True):
  # Only outer-tercile rows are needed for the contrast, using frozen training thresholds.
  lo,hi=thresholds[book];f=f[(f.surprisal<=lo)|(f.surprisal>=hi)]
  if len(f):tasks.append((sub,book,int(run),f,a.data_root,a.cache_dir,thresholds[book],a.parts_dir))
 if a.pilot:tasks=tasks[:1]
 print(f'Continuous-source run tasks: {len(tasks)}; training readings: {len(train)}; thresholds: '+str({k:v.tolist() for k,v in thresholds.items()}),flush=True)
 aggregate={};records=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  jobs=[pool.submit(run_task,t) for t in tasks]
  for j,job in enumerate(as_completed(jobs),1):
   v=job.result();key=(v['subject'],v['book'])
   if key not in aggregate:aggregate[key]={'low_sum':np.zeros_like(v['low_sum']),'high_sum':np.zeros_like(v['high_sum']),'n_low':0,'n_high':0}
   for k in ['low_sum','high_sum','n_low','n_high']:aggregate[key][k]+=v[k]
   records.append({'subject':v['subject'],'book':v['book'],**v['source']})
   print(f'[{j}/{len(tasks)}] {key} run{v["run"]} low={v["n_low"]} high={v["n_high"]}',flush=True)
 if a.pilot:
  (a.out_dir/'pilot.json').write_text(json.dumps(records,indent=2)+'\n');print('PILOT VERIFIED',flush=True);return
 subjects=sorted(set(k[0] for k in aggregate));reader=[];book_contrasts=[];book_names=[];counts=[]
 for sub in subjects:
  parts=[]
  for (s,b),v in sorted(aggregate.items()):
   if s!=sub:continue
   if v['n_low']<1 or v['n_high']<1:raise ValueError('Empty condition')
   contrast=v['high_sum']/v['n_high']-v['low_sum']/v['n_low']
   parts.append(contrast);book_contrasts.append(contrast);book_names.append(s+'|'+b)
   counts.append({'subject':s,'book':b,'n_low':v['n_low'],'n_high':v['n_high']})
  reader.append(np.mean(parts,axis=0))
 reader=np.stack(reader);time=(np.arange(PRE+POST)-PRE)*1000/RATE
 windows=np.array([(lo,lo+100) for lo in range(0,1901,50)])
 binned=np.stack([reader[:,:,(time>=lo)&(time<hi)].mean(-1) for lo,hi in windows],axis=1)
 # Independently verify the nine-reader 0–800-ms summaries against the exact manuscript arrays.
 old=np.load(ROOT/'analysis/verified_revision_20261007/row_onset_topography.npz')
 common=[subjects.index(str(s)) for s in old['subjects']]
 old_bins=np.stack([reader[common][:,:,(time>=lo)&(time<hi)].mean(-1) for lo,hi in old['windows_ms']],axis=1)
 error=float(np.max(abs(old_bins-old['reader_uv'])))
 # Original reduction was float32; extended waveform sums are float64.
 # Require sub-0.0001µV agreement and identical published 3-decimal group RMS.
 if error>1e-4:raise ValueError(f'Manuscript contrast mismatch {error}')
 if [f'{x:.3f}' for x in np.sqrt((old_bins.mean(0)**2).mean(-1))] != [f'{x:.3f}' for x in np.sqrt((old['group_uv']**2).mean(-1))]:
  raise ValueError('Published group RMS rounding differs')
 old_audit=json.loads((ROOT/'analysis/verified_revision_20261007/topography_audit.json').read_text())
 lookup={(x['subject'],x['book']):x for x in counts}
 for x in old_audit['records']:
  c=lookup[(x['subject'],x['book'])]
  assert c['n_low']==x['n_low'] and c['n_high']==x['n_high'],('Changed old condition counts',x,c)
 if len(subjects)!=10:raise ValueError('Expected the ten available readers')
 np.savez_compressed(a.out_dir/'extended_source_arrays.npz',reader_uv=binned,group_uv=binned.mean(0),reader_sd_uv=binned.std(0,ddof=1),
                      reader_timecourse_uv=reader,time_ms=time,subjects=subjects,windows_ms=windows,
                      recording_timecourse_uv=np.stack(book_contrasts),recordings=book_names)
 provenance={'data':'ChineseEEG ds004952 author-filtered 0.5–30Hz continuous recordings','subjects':subjects,'n_readers':len(subjects),
             'n_recordings':len(aggregate),'counts':counts,'windows_ms':windows.tolist(),'sample_rate_hz':RATE,'source_epoch_samples':[-PRE,POST],
             'window_width_ms':100,'window_step_ms':50,'temporal_interpolation':False,'reader_sd_ddof':1,
             'scope':'Setting0 training blocks2–6; ten complete slots; observed ROWS onset; only rows and recordings with full -100..2000ms coverage; no crossing target row end.',
             'book_averaging':'Equal available books within reader, then equal ten readers. sub-15 contributes GarnettDream only; this ten-reader extension differs from the balanced nine-reader main analysis.',
             'thresholds':{k:v.tolist() for k,v in thresholds.items()},'baseline':'13 pre-row samples; quantize to original float16 cache precision before float32 baseline subtraction',
             'nine_reader_main_contrast_max_abs_difference_uv':error,'numeric_reduction_note':'Original summary used float32 joint reductions; extended sample-wise accumulation uses float64. Condition counts and published three-decimal group RMS are identical; max absolute reader-map difference must remain below 0.0001 µV.','manifest_sha256':sha(a.manifest),'labels_sha256':sha(labels_path),
             'sources':sorted(records,key=lambda v:(v['subject'],v['book'],v['run']))}
 (a.out_dir/'extended_extraction_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 print(f'COMPLETE: {len(subjects)} readers, {len(windows)} measured windows; main contrast max error={error:.8g} µV',flush=True)
if __name__=='__main__':main()
