#!/usr/bin/env python
"""Persist live progress and finalize only after every matched arm finishes."""
from pathlib import Path
import json,re,time,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/window_alignment_20261008'
names=['E800_G800','E1200_G800','E1200_G1200']
while True:
 rows=[];status={}
 for name in names:
  log=(R/(name+'.log')).read_text(errors='replace')
  found=re.findall(r'^\[fold \d+\] (\{.*\})$',log,re.M)
  values=[dict(window=name,**json.loads(s)) for s in found];rows+=values
  current=re.findall(r'fold(\d+) ep(\d+)',log)
  status[name]={'completed_folds':len(values),'latest_epoch':current[-1] if current else None,'complete_summary_exists':(R/name/'tables'/f'charbag_window_{name}_summary.json').exists(),'traceback':bool(re.search(r'^Traceback',log,re.M))}
 (R/'LIVE_STATUS.json').write_text(json.dumps({'complete':False,'arms':status,'completed_fold_metrics':rows},indent=2))
 if any(x['traceback'] for x in status.values()):raise RuntimeError('A training arm failed; see logs')
 if all(x['complete_summary_exists'] for x in status.values()):break
 time.sleep(30)
subprocess.run([sys.executable,str(ROOT/'scripts/summarize_window_alignment.py')],check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/plot_window_alignment.py')],check=True)
(R/'LIVE_STATUS.json').write_text(json.dumps({'complete':True,'arms':status,'completed_fold_metrics':rows},indent=2))
print('COMPLETE:21folds,summary,paired-differences,report,andplots')
