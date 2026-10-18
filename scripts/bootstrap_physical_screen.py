#!/usr/bin/env python3
"""Paired reader × contiguous-text-block bootstrap for repaired-source screens."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
BASE=ROOT/'results/single_model_breakthrough_20261006'

def paired(a,b,manifest,fold):
    keys=['subject','book','run','row_idx','item_id','y']
    a=a[a.fold.eq(fold)].sort_values(keys[:4]).reset_index(drop=True)
    b=b[b.fold.eq(fold)].sort_values(keys[:4]).reset_index(drop=True)
    assert a[keys].equals(b[keys])
    f=a.merge(b[keys+['pred']],on=keys,validate='one_to_one',suffixes=('_a','_b'))
    f=f.merge(manifest[['item_id','half']],on='item_id',validate='many_to_one')
    f['cluster']=f.book+'|'+f.run.astype(str)+'|'+f.half.astype(str)
    items=f.groupby('item_id').agg(y=('y','first'),book=('book','first'),cluster=('cluster','first'))
    readers=sorted(f.subject.unique())
    pa=f.pivot(index='item_id',columns='subject',values='pred_a').reindex(index=items.index,columns=readers).to_numpy()
    pb=f.pivot(index='item_id',columns='subject',values='pred_b').reindex(index=items.index,columns=readers).to_numpy()
    strata={book:items[items.book.eq(book)].cluster.unique() for book in items.book.unique()}
    lookup={cl:np.flatnonzero(items.cluster.eq(cl).to_numpy()) for cls in strata.values() for cl in cls}
    valid=np.isfinite(pa);y=items.y.to_numpy(float)
    def average(p,w):return np.nansum(p*w[None],axis=1)/np.sum(valid*w[None],axis=1)
    observed=float(spearmanr(y,average(pa,np.ones(len(readers)))).statistic-spearmanr(y,average(pb,np.ones(len(readers)))).statistic)
    rng=np.random.default_rng(2026+fold);diff=[]
    for _ in range(500):
        w=rng.multinomial(len(readers),np.full(len(readers),1/len(readers)))
        if np.any(np.sum(valid*w[None],axis=1)==0):continue
        ix=np.concatenate([np.concatenate([lookup[cl] for cl in rng.choice(cls,len(cls),replace=True)]) for cls in strata.values()])
        aa=average(pa,w)[ix];bb=average(pb,w)[ix]
        diff.append(float(spearmanr(y[ix],aa).statistic-spearmanr(y[ix],bb).statistic))
    return {'fold':fold,'observed_delta_item_rho':observed,'reader_textblock_bootstrap_95pct':np.quantile(diff,[.025,.975]).tolist(),'bootstrap_repetitions':len(diff),'n_reader_clusters':len(readers),'n_contiguous_text_clusters':len(lookup),'interval_scope':'development variability; not new independent evidence'}

def main():
    manifest=pd.read_csv(ROOT/'results/response_mechanism_20261002/sentence_manifest.csv')
    frames={arm:pd.read_csv(BASE/f'physical_screen/{arm}/tables/charbag_{arm}_predictions.csv') for arm in ('phys_control','phys_valid','phys_shared')}
    frames['norm_control']=pd.read_csv(BASE/'norm_control/tables/charbag_norm_control_predictions.csv')
    report=[]
    for candidate,control in [('phys_control','norm_control'),('phys_valid','phys_control'),('phys_shared','phys_control')]:
        for fold in (0,3):report.append({'candidate':candidate,'control':control,**paired(frames[candidate],frames[control],manifest,fold)})
    (BASE/'physical_screen/cluster_bootstrap.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
