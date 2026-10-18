#!/usr/bin/env python
"""Check real timestamp spacing for the archived and extended Eye samplers."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import numpy as np
import pandas as pd
import build_eye_char_cache as b
R=Path(__file__).resolve().parents[1];C=Path('/data/2_readEEG/superisal_code/data_cache/char_epochs_1200ms')
O=R/'results/window_alignment_20261008'

def audit(path):
 subject,book=path.stem.removeprefix('meta_').split('_',1)
 meta=pd.read_csv(path);out=[]
 for run,chunk in meta.groupby('run'):
  pp=b.preproc_events_path(b.DATA_ROOT,subject,book,int(run));rp=b.raw_events_path(b.DATA_ROOT,subject,book,int(run))
  if not pp.exists() or not rp.exists():continue
  raw=pd.read_csv(rp,sep='\t',encoding='utf-8-sig');eyes=raw.loc[raw.trial_type.eq('EYES'),'onset'].to_numpy(float)
  if len(eyes)==0:continue
  pre=b.pair_rows_rowe(pd.read_csv(pp,sep='\t',encoding='utf-8-sig'));rawp=b.pair_rows_rowe(raw)
  if pre.empty or rawp.empty:continue
  shift=float(rawp.start.iloc[0]-pre.start.iloc[0]-eyes[0]);pairs=pre.set_index('row_idx')
  data=b.read_gaze(b.rar_path(subject,book,int(run)))
  if data is None:continue
  times,_=data
  assert np.all(np.diff(times)>=0)
  rows=chunk.merge(pairs[['start','end']],left_on='row_idx',right_index=True)
  centers=(rows.start+rows.char_slot*(rows.end-rows.start)/rows.n_slots+shift).to_numpy()
  first=np.searchsorted(times,centers-.1)
  keep=first+64<len(times);first=first[keep];centers=centers[keep]
  if not len(first):continue
  extension=1000*(times[first+64]-times[first+44]);span=1000*(times[first+64]-times[first]);onseterr=1000*(times[first]-centers+.1)
  out.append({'subject':subject,'book':book,'run':int(run),'n_windows':len(first),'n_truncated_65samples':int((~keep).sum()),'median_sample_interval_ms':float(1000*np.median(np.diff(times))),'extension_median_ms':float(np.median(extension)),'extension_p01_ms':float(np.quantile(extension,.01)),'extension_p99_ms':float(np.quantile(extension,.99)),'extension_max_ms':float(extension.max()),'extended_first_to_last_median_ms':float(np.median(span)),'start_rounding_p99_ms':float(np.quantile(onseterr,.99)),'extension_outside_380_420_ms':int(((extension<380)|(extension>420)).sum())})
 return out

if __name__=='__main__':
 rows=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  tasks=[pool.submit(audit,p) for p in sorted(C.glob('meta_*.csv'))]
  for task in as_completed(tasks):rows.extend(task.result())
 frame=pd.DataFrame(rows).sort_values(['subject','book','run']);frame.to_csv(O/'eye_timestamp_audit.csv',index=False)
 report={'n_recordings':int(frame[['subject','book']].drop_duplicates().shape[0]),'n_runs':len(frame),'n_windows':int(frame.n_windows.sum()),'nominal_extension_ms':400,'run_median_extension_range_ms':[float(frame.extension_median_ms.min()),float(frame.extension_median_ms.max())],'max_run_p99_extension_ms':float(frame.extension_p99_ms.max()),'n_extension_outside_380_420_ms':int(frame.extension_outside_380_420_ms.sum()),'n_truncated_at_run_end':int(frame.n_truncated_65samples.sum()),'scope':'Raw Eye sampling intervals and window endpoints; does not validate EEG/Eye clock synchronization or character-event accuracy.'}
 (O/'eye_timestamp_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
