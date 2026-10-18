"""Restore archived scores and compute auxiliary AUC from frozen predictions.

No model retraining, threshold optimization, or outer-block evaluation.
"""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/support_consensus_20261006'
OUT=ROOT/'analysis/verified_revision_20261007'
sys.path.insert(0,str(ROOT/'model_code/scripts'))
from summarize_retrospective_baseline import item_frame

def main():
    rows=[];hashes={}
    for arm in ['consensus_base','consensus_source','consensus_generic','unique_trajectory']:
        p=RUN/arm/'tables'/f'charbag_{arm}_predictions.csv'
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        df=pd.read_csv(p);assert set(df.eval_split)=={'val'}
        df['n_slots']=10
        for f,d in df.groupby('fold'):
            assert d.groupby('item_id').tercile.nunique().max()==1
            it=item_frame(d).merge(d[['item_id','tercile']].drop_duplicates(),on='item_id',validate='one_to_one')
            row={'arm':arm,'setting':int(f)}
            for level,frame in [('reading',d),('item',it)]:
                q=frame[frame.tercile.ne(1)]
                y=q.tercile.eq(2).astype(int)
                assert y.nunique()==2
                row.update({level+'_auc':float(roc_auc_score(y,q.pred)),
                    level+'_ap':float(average_precision_score(y,q.pred)),
                    level+'_n_low':int((y==0).sum()),level+'_n_high':int((y==1).sum())})
            rows.append(row)
    scores=pd.DataFrame(rows);scores.to_csv(OUT/'auxiliary_auc_20261008.csv',index=False)
    (OUT/'auxiliary_auc_provenance_20261008.json').write_text(json.dumps({
        'definition':'Stored train-derived tercile0 vs tercile2; middle excluded; continuous prediction as score',
        'item_aggregation':'stable mean over readers, one item once',
        'threshold_selection':'No threshold search or validation-derived class boundaries',
        'scope':'pooled auxiliary separation, book differences remain; retrospective validation only',
        'prediction_sha256':hashes},indent=2))
    names={'consensus_base':'Baseline','consensus_source':'Physical support',
           'consensus_generic':'Local residual','unique_trajectory':'Unique trajectory'}
    tex=[r'\begin{table*}[!t]',
        r'\caption{Auxiliary high--low surprisal separation from the same strict-cohort validation predictions. Low and high classes use the stored training-derived terciles; middle-tercile observations are excluded. AUC is area under the receiver operating characteristic curve and AP is average precision. Both use the continuous regression prediction without a decision-threshold search.}',
        r'\label{tab:auc}',r'\centering\small',r'\begin{tabular}{lrrrrrr}',r'\toprule',
        r'Model & \multicolumn{2}{c}{Item AUC} & \multicolumn{2}{c}{Reading AUC} & \multicolumn{2}{c}{Item AP}\\',
        r' & s0 & s3 & s0 & s3 & s0 & s3\\\midrule']
    for a,n in names.items():
        q=scores[scores.arm.eq(a)].set_index('setting')
        values=[q.loc[f,m] for m in ['item_auc','reading_auc','item_ap'] for f in [0,3]]
        tex.append(n+' & '+' & '.join(f'{v:.4f}' for v in values)+r'\\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (OUT/'auc_table_20261008.tex').write_text('\n'.join(tex)+'\n')
    old=[]
    for a in ['MM_mid','ABL_noeye','ABL_nochar']:
        p=ROOT/'results/tables'/f'charbag_{a}_per_fold.csv'
        q=pd.read_csv(p).sort_values('fold');assert q.fold.tolist()==list(range(7))
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        row={'arm':a,'n_folds':7}
        for m in ['trial_spearman','item_spearman','item_auc_hi_vs_lo']:
            row[m+'_mean']=float(q[m].mean());row[m+'_sd']=float(q[m].std(ddof=1))
        old.append(row)
    pd.DataFrame(old).to_csv(OUT/'historical_metrics_20261008.csv',index=False)
    (OUT/'auc_and_history_source_hashes_20261008.json').write_text(json.dumps(hashes,indent=2))
    print(scores.to_string(index=False))
    print(pd.DataFrame(old).to_string(index=False))

if __name__=='__main__':main()
