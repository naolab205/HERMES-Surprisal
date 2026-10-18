#!/usr/bin/env python3
"""Development-only metadata controls for timing encoded in overlapping epochs."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foundation.data import load_char_bags,CACHE_DIR,SURPRISAL_CSV
from cogalign.io.retrospective_splits import attach_retrospective_split
from summarize_retrospective_baseline import metrics,item_frame,conditional_rho
OUT=ROOT/'results/single_model_breakthrough_20261006/clock_controls'
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'PROTOCOL.md').write_text('Registered before scoring: fixed metadata models book+actual slots; book+slots+n_char; book+slots+SOA; book+slots+n_char+SOA. HistGradientBoosting depth2,100iterations,min_leaf50,learning_rate.1,L2=1,seed2026,fit train only. Fixed val folds0/3. No participant/run/row identity features. These are diagnostic nuisance oracles, never candidate inputs. SOA is used for epoch placement by the old/physical builders, so overlapping copies can reveal it even without a supplied duration scalar. Assess book/count conditional correlations and preserve raw predictions.\n')
bags=load_char_bags(cache_dir=CACHE_DIR)
meta=pd.concat([pd.read_csv(CACHE_DIR/f'meta_{g.split("|")[0]}_{g.split("|")[1]}.csv').drop_duplicates(['subject','book','run','row_idx']) for g in bags.groups])
meta=meta[['subject','book','run','row_idx','soa','n_char']].rename(columns={'n_char':'source_n_char'})
manifest=pd.read_csv(ROOT/'results/response_mechanism_20261002/sentence_manifest.csv')
results=[];frames=[];associations=[]
for fold in (0,3):
    f=attach_retrospective_split(bags.obs,manifest,fold,label_csv=str(SURPRISAL_CSV))
    f=f[f.split.ne('test')].merge(meta,on=['subject','book','run','row_idx'],validate='one_to_one')
    f['book_code']=f.book.eq('LittlePrince').astype(int)
    train=f[f.split.eq('train')];val=f[f.split.eq('val')]
    for arm,features in [('book_slots',['book_code','n_slots']),('book_slots_chars',['book_code','n_slots','source_n_char']),('book_slots_clock',['book_code','n_slots','soa']),('book_slots_chars_clock',['book_code','n_slots','source_n_char','soa'])]:
        model=HistGradientBoostingRegressor(max_depth=2,max_iter=100,min_samples_leaf=50,learning_rate=.1,l2_regularization=1.,random_state=2026).fit(train[features],train.y)
        p=val.copy();p['pred']=model.predict(val[features]);items=item_frame(p)
        results.append({'fold':fold,'arm':arm,'trial':metrics(p),'item':metrics(items),'conditional_rho':conditional_rho(items)})
        frames.append(p[['subject','book','run','row_idx','item_id','n_slots','source_n_char','soa','y','pred']].assign(fold=fold,arm=arm))
    # Fit-independent association, training texts only; multiple readers averaged.
    item=train.groupby('item_id').agg(y=('y','first'),book=('book','first'),n_slots=('n_slots','median'),soa=('soa','median'),n_char=('source_n_char','first'))
    for cov in ['soa','n_char']:
        values=[(len(g),float(spearmanr(g.y,g[cov]).statistic)) for _,g in item.groupby(['book','n_slots']) if len(g)>=5 and g[cov].nunique()>1]
        values=[(n,r) for n,r in values if np.isfinite(r)]
        associations.append({'fold':fold,'covariate':cov,'training_book_slot_weighted_rho':float(np.average([r for n,r in values],weights=[n for n,r in values]))})
report={'scope':'diagnostic metadata controls; no EEG and no inference model change','scores':results,'training_associations':associations}
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');pd.concat(frames).to_csv(OUT/'predictions.csv',index=False)
print(json.dumps(report,indent=2))
