"""Redraw archived 0.405 primary-model plots without refitting models."""
from pathlib import Path
import json, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
TABLES=ROOT/'results/tables'
OUT=ROOT/'outputs/paper_figures'
AUDIT=ROOT/'analysis/primary_fusion_revision_20261008'
AUDIT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})
COLORS=['#8593a8','#5380aa','#c28b4b','#167e83','#8572a8']
hashes={}

def folds(a):
    p=TABLES/f'charbag_{a}_per_fold.csv'
    hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    q=pd.read_csv(p).sort_values('fold');assert q.fold.tolist()==list(range(7))
    return q

def save(fig,n):
    for ext in ['pdf','png']:fig.savefig(OUT/f'{n}.{ext}',dpi=280,bbox_inches='tight')
    plt.close(fig)

def main():
    arms=['BL_atcnet','BL_conformer','ABL_nochar','ABL_noeye','MM_mid']
    names=['ATCNet port','Conformer port','No cross-slot','EEG only','EEG + Eye']
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.7),layout='constrained')
    for ax,m,label in zip(axes,['item_spearman','item_auc_hi_vs_lo'],['Item Spearman ρ','High–low item AUC']):
        for i,(a,c) in enumerate(zip(arms,COLORS)):
            vals=folds(a)[m].to_numpy()
            bp=ax.boxplot(vals,positions=[i+1],patch_artist=True,widths=.5,whis=(0,100),
                medianprops={'color':'#23344a','linewidth':1.5})
            bp['boxes'][0].set(facecolor=c,alpha=.4)
            ax.scatter(i+1+np.linspace(-.12,.12,7),vals,s=22,color=c,zorder=3)
        ax.set_xticks(range(1,6),['ATCNet\nport','Conformer\nport','No cross-\nslot','EEG\nonly','EEG +\nEye'])
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.15)
    save(fig,'fig03_primary_boxplots_20261008')
    mainfold=folds('MM_mid')
    variants=['ABL_noeye','ABL_nochar','ABL_nocsd','ABL_norope','ABL_win900','ABL_win450','MM_late']
    labels=['No Eye','No cross-slot block','No CSD','No rotary position','EEG +900 ms','EEG +450 ms','Late fusion']
    fig,ax=plt.subplots(figsize=(10.5,3.8),layout='constrained')
    rows=[]
    for i,(a,label) in enumerate(zip(variants,labels)):
        delta=mainfold.item_spearman.to_numpy()-folds(a).item_spearman.to_numpy()
        ax.scatter(delta,i+np.linspace(-.14,.14,7),s=27,color='#7795a6',alpha=.85)
        ax.scatter(delta.mean(),i,s=65,color='#167e83',marker='D',zorder=4)
        rows.extend({'arm':a,'fold':f,'delta_item_rho':float(v)} for f,v in enumerate(delta))
    ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.axvline(0,c='#9aa8b5',ls='--',lw=1)
    ax.set_xlabel('Combined main − ablation: item Spearman Δρ')
    ax.grid(axis='x',alpha=.15);ax.set_title('Seven paired blocks; diamonds show the mean')
    save(fig,'fig04_primary_ablation_dots_20261008')
    pd.DataFrame(rows).to_csv(AUDIT/'paired_ablation_points.csv',index=False)
    p=TABLES/'charbag_MM_mid_predictions.csv';hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    d=pd.read_csv(p);assert len(d)==102499
    # Group only within a fold: standardized targets differ by training fold.
    def mean(v):x=v.to_numpy(dtype=float);return x[0]+np.mean(x-x[0])
    it=d.groupby(['book','fold','item_id'],as_index=False).agg(y=('y','first'),pred=('pred',mean),tercile=('tercile','first'))
    assert len(it)==11037
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),sharey=True,layout='constrained')
    summaries=[]
    for ax,(book,q) in zip(axes,it.groupby('book')):
        vals=[q.loc[q.tercile.eq(i),'pred'].to_numpy() for i in range(3)]
        vp=ax.violinplot(vals,showmedians=False,showextrema=False)
        for body,c in zip(vp['bodies'],['#5380aa','#8593a8','#c28b4b']):body.set(facecolor=c,alpha=.6)
        for j,v in enumerate(vals):
            q1,md,q3=np.quantile(v,[.25,.5,.75]);ax.plot([j+1,j+1],[q1,q3],lw=3,c='#23344a')
            ax.plot([j+.9,j+1.1],[md,md],lw=1.5,c='white')
            ax.text(j+1,.02,f'n={len(v):,}',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
            summaries.append({'book':book,'tercile':j,'n_items':len(v),'median':float(md)})
        ax.set_title(book);ax.set_xticks([1,2,3],['Low','Middle','High']);ax.set_ylim(-1.85,2.1)
        ax.set_xlabel('Training-derived row-surprisal tercile');ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Item-mean predicted surprisal (training SD)')
    save(fig,'fig06_primary_prediction_violin_20261008')
    pd.DataFrame(summaries).to_csv(AUDIT/'prediction_tercile_summary.csv',index=False)
    (AUDIT/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=='__main__':main()
