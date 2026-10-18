#!/usr/bin/env python3
"""Training-block-only gate for signed, latency-anchored row EEG evidence."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foundation.data import load_char_bags,ShardedEpochs,CACHE_DIR,SURPRISAL_CSV
from cogalign.io.retrospective_splits import attach_retrospective_split
from probe_training_reader_teacher import subset_train,residual_labels,score
OUT=ROOT/'results/single_model_breakthrough_20261006/latency_probe'
WINDOWS=((13,39),(39,64),(64,115),(115,167))

def extract(bags,index):
    shards=ShardedEpochs(bags.epochs);csd=np.load(CACHE_DIR/'csd_matrix.npy').astype(np.float32)
    features=[]
    for lo in range(0,len(index),32):
        ix=[bags.slot_index[int(i)] for i in index[lo:lo+32]]
        counts=np.array([len(i) for i in ix]);x=shards.take(np.concatenate(ix))
        x-=x[...,:13].mean(-1,keepdims=True)
        # Spatial filtering is identical for all arms. Linear, so apply it to
        # window averages rather than every time sample.
        f=np.stack([x[...,a:b].mean(-1) for a,b in WINDOWS],axis=-1)
        f=np.einsum('cd,ndw->ncw',csd,f)
        avg=np.add.reduceat(f,np.r_[0,np.cumsum(counts)[:-1]],axis=0)/counts[:,None,None]
        features.append(avg)
    return np.concatenate(features).astype(np.float32)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    bags=load_char_bags(cache_dir=CACHE_DIR)
    manifest=pd.read_csv(ROOT/'results/response_mechanism_20261002/sentence_manifest.csv')
    frames={}
    for fold in (0,3):
        frame=attach_retrospective_split(bags.obs,manifest,fold,label_csv=str(SURPRISAL_CSV))
        frame=frame[frame.split.eq('train')].copy()
        frame['slot_stratum']=frame.groupby('item_id').n_slots.transform('median')
        frames[fold]=subset_train(frame,manifest,fold)
    ids=np.unique(np.concatenate([f.obs_index.to_numpy(int) for f in frames.values()]))
    raw=extract(bags,ids);lookup=dict(zip(ids,np.arange(len(ids))))
    # Average vs signed contrasts at equal dimensionality: no architecture
    # tuning or validation features/labels participate.
    contrast=raw-raw.mean(axis=2,keepdims=True)
    pooled=np.broadcast_to(raw.mean(axis=2,keepdims=True),raw.shape).copy()
    rng=np.random.default_rng(2026)
    # Each observation has an independent fixed permutation, shared over channels.
    perm=np.stack([rng.permutation(4) for _ in ids])
    scrambled=np.take_along_axis(contrast,perm[:,None,:],axis=2)
    features={'signed_latency':contrast.reshape(len(ids),-1),
              'pooled_amplitude':pooled.reshape(len(ids),-1),
              'scrambled_latency':scrambled.reshape(len(ids),-1)}
    rows=[];predictions=[]
    for fold,frame in frames.items():
        frame['feature_index']=frame.obs_index.map(lookup)
        for block in sorted(frame.block.unique()):
            fit=frame[frame.block.ne(block)];held=frame[frame.block.eq(block)]
            qfit,qheld=residual_labels(fit,held)
            for name,feat in features.items():
                xt=feat[fit.feature_index.to_numpy(int)];xv=feat[held.feature_index.to_numpy(int)]
                scaler=StandardScaler().fit(xt)
                model=Ridge(alpha=1000).fit(scaler.transform(xt),qfit)
                result=held.assign(q=qheld,pred=model.predict(scaler.transform(xv)))
                value=score(result)
                rows.append({'fold':fold,'held_training_block':int(block),'arm':name,**value})
                predictions.append(result[['subject','book','run','row_idx','item_id','slot_stratum','q','pred']].assign(fold=fold,held_training_block=int(block),arm=name))
            print(f'[probe] fold{fold} block{block}',flush=True)
    scores=pd.DataFrame(rows);mean=scores.groupby(['fold','arm']).conditional_rho.mean().unstack()
    passed=bool(((mean.signed_latency>.05)&(mean.signed_latency>mean.pooled_amplitude)&(mean.signed_latency>mean.scrambled_latency)).all())
    scores.to_csv(OUT/'block_metrics.csv',index=False)
    pd.concat(predictions).to_csv(OUT/'predictions.csv',index=False)
    report={'scope':'training texts only, full held-training-block cross-fitting; all observed readers',
            'mean_conditional_rho':mean.reset_index().to_dict(orient='records'),
            'latency_gate_passed':passed,'limitation':'linear CSD signed-window row-mean probe; does not exclude nonlinear latency or spatial mechanisms'}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
