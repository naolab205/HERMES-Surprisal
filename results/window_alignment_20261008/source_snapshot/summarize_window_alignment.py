#!/usr/bin/env python
"""Require all21folds and compare matched validation-only window experiments."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
R=Path(__file__).resolve().parents[1]/'results/window_alignment_20261008'
metrics=['item_spearman','trial_spearman','item_auc_hi_vs_lo']
folds=[];preds={};summaries=[]
for name in ['E800_G800','E1200_G800','E1200_G1200']:
 prefix=R/name/'tables'/f'charbag_window_{name}'
 t=pd.read_csv(str(prefix)+'_per_fold.csv');p=pd.read_csv(str(prefix)+'_predictions.csv')
 assert sorted(t.fold.tolist())==list(range(7)) and (t.eval_split=='val').all()
 assert p.eval_split.eq('val').all() and set(p.fold)==set(range(7))
 assert not p.duplicated(['fold','subject','book','run','row_idx']).any()
 preds[name]=p.sort_values(['fold','subject','book','run','row_idx']).reset_index(drop=True)
 extra=[]
 for f,q in p.groupby('fold'):
  item=q.groupby('item_id')[['y','pred']].mean()
  extra.append({'fold':f,'item_pearson':pearsonr(item.y,item.pred).statistic,'trial_pearson':pearsonr(q.y,q.pred).statistic,'trial_MAE_z':mean_absolute_error(q.y,q.pred),'trial_RMSE_z':np.sqrt(mean_squared_error(q.y,q.pred)),'trial_R2':r2_score(q.y,q.pred)})
 t=t.merge(pd.DataFrame(extra),on='fold');t['window']=name;folds.append(t)
 summaries.append({'window':name,**{f'{m}_{s}':float(getattr(t[m],s)()) for m in metrics+['item_pearson','trial_pearson','trial_MAE_z','trial_RMSE_z','trial_R2'] for s in ['mean','std']}})
reference=preds['E1200_G800']
for name,p in preds.items():
 assert p[['fold','subject','book','run','row_idx','item_id','y','tercile']].equals(reference[['fold','subject','book','run','row_idx','item_id','y','tercile']])
all_folds=pd.concat(folds,ignore_index=True);summary=pd.DataFrame(summaries)
all_folds.to_csv(R/'complete_metrics.csv',index=False);summary.to_csv(R/'summary.csv',index=False)
base=all_folds[all_folds.window=='E1200_G800'].set_index('fold')
rng=np.random.default_rng(2026);paired=[]
for name in ['E800_G800','E1200_G1200']:
 t=all_folds[all_folds.window==name].set_index('fold')
 for m in metrics:
  d=t[m]-base[m];samples=rng.choice(d.to_numpy(),size=(20000,7),replace=True).mean(axis=1)
  paired.append({'window':name,'metric':m,'mean_difference':float(d.mean()),'folds_improved':int((d>0).sum()),'fold_differences':d.tolist(),'descriptive_block_bootstrap_95':np.quantile(samples,[.025,.975]).tolist()})
(R/'paired_differences.json').write_text(json.dumps(paired,indent=2))
lines=['# Window-length experiment — 2026-10-08','','Completed21 matched validation runs (seven settings × three configurations), seed2026. Test blocks were not scored. Selected-epoch validation scores are exploratory and are not directly interchangeable with archived0.405 test-block scores.','','| EEG/Eye post-window(ms) | Item rho | Reading rho | Item AUC |','|---|---:|---:|---:|']
for row in summaries:
 lines.append('| '+row['window']+' | '+' | '.join(f"{row[m+'_mean']:.4f} ± {row[m+'_std']:.4f}" for m in metrics)+' |')
lines+=['','Paired differences and descriptive block-bootstrap intervals are in `paired_differences.json`; intervals based on seven reused development blocks are not confirmatory p-values.','Long Eye windows were rebuilt from real raw records; all19 recordings matched old keys and had exact800ms prefixes. EEG operators/scales were matched using training data on the full1200ms cache. Eye normalization was training-only in allarms. Archived duration-derived geometry remains; this study isolates window endpoints within that representation.']
audit=json.loads((R/'eye_timestamp_audit.json').read_text()) if (R/'eye_timestamp_audit.json').exists() else None
if audit:
 lines+=['',f"Raw Eye timestamp audit: {audit['n_recordings']} recordings, {audit['n_runs']} runs; the median20-sample extension is {audit['run_median_extension_range_ms'][0]:.3f}ms. {audit['n_truncated_at_run_end']} windows end at recording boundaries and remain validity-marked zero-filled; cohort membership is unchanged. This audit does not establish precise EEG/Eye clock synchronization."]
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(summary[['window']+[m+'_'+s for m in metrics for s in ['mean','std']]].to_string(index=False))
