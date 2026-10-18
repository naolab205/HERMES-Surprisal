#!/usr/bin/env python3
"""Paired fixed-development evaluation, no model/prediction averaging."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foundation.data import load_char_bags,CACHE_DIR
from summarize_retrospective_baseline import metrics,item_frame,conditional_rho

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--control',required=True);ap.add_argument('--candidates',nargs='+',required=True);ap.add_argument('--cache',default=str(CACHE_DIR));ap.add_argument('--suffix',default='');args=ap.parse_args()
    base=Path(args.base);counts=load_char_bags(cache_dir=Path(args.cache)).obs[['subject','book','run','row_idx','n_slots']]
    keys=['fold','subject','book','run','row_idx','item_id','y','tercile']
    arms={};results=[];deltas=[]
    for arm in [args.control]+args.candidates:
        frame=pd.read_csv(base/arm/'tables'/f'charbag_{arm}_predictions.csv').sort_values(keys[:5]).reset_index(drop=True)
        assert sorted(frame.fold.unique())==[0,3] and set(frame.eval_split)=={'val'}
        assert not frame.duplicated(keys[:5]).any() and np.isfinite(frame[['y','pred']]).all().all()
        if arms:assert frame[keys].equals(next(iter(arms.values()))[keys]),f'Population/target mismatch: {arm}'
        arms[arm]=frame
    for arm,frame in arms.items():
        f=frame.merge(counts,on=keys[1:5],validate='many_to_one')
        for fold,g in f.groupby('fold'):
            item=item_frame(g);results.append({'arm':arm,'fold':int(fold),'trial':metrics(g),'item':metrics(item),'conditional_rho':conditional_rho(item)})
    for arm in args.candidates:
        for fold in [0,3]:
            c=next(r for r in results if r['arm']==args.control and r['fold']==fold)
            v=next(r for r in results if r['arm']==arm and r['fold']==fold)
            deltas.append({'arm':arm,'fold':fold,'delta_trial_rho':v['trial']['spearman']-c['trial']['spearman'],'delta_item_rho':v['item']['spearman']-c['item']['spearman'],'delta_conditional_rho':v['conditional_rho']-c['conditional_rho']})
    gates={arm:all(r['delta_trial_rho']>=0 and r['delta_item_rho']>0 and r['delta_conditional_rho']>0 for r in deltas if r['arm']==arm) for arm in args.candidates}
    report={'scope':'fixed retrospective development folds0/3 seed2026; no ensemble','control':args.control,'cache':args.cache,'scores':results,'differences':deltas,'development_gate':gates}
    (base/f'screen_summary{args.suffix}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    pd.DataFrame(deltas).to_csv(base/f'paired_differences{args.suffix}.csv',index=False)
    print(json.dumps({'differences':deltas,'development_gate':gates},indent=2))
if __name__=='__main__':main()
