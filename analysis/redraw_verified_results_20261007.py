"""Regenerate manuscript figures from recorded EEG and frozen predictions.

No image generation, model fitting, checkpoint reselection, or outer testing.
The two displayed settings score their validation blocks, not their outer blocks.
"""
from pathlib import Path
import hashlib
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'model_code'
RUN = CODE / 'results/support_consensus_20261006'
OUT = ROOT / 'outputs/paper_figures'
AUDIT = ROOT / 'analysis/verified_revision_20261007'
OUT.mkdir(exist_ok=True)
AUDIT.mkdir(exist_ok=True)
sys.path.insert(0, str(CODE / 'scripts'))
from summarize_retrospective_baseline import item_frame
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
    'axes.labelcolor':'#22334a','text.color':'#22334a', 'axes.titleweight':'bold'})
COLORS = ['#8593a8','#167e83','#c28b4b','#8572a8']
ARMS = ['consensus_base','consensus_source','consensus_generic','unique_trajectory']
NAMES = ['Baseline','Physical support','Local residual','Unique trajectory']
MET = pd.read_csv(RUN / 'complete_metrics.csv')
PROVENANCE = {'scope':'Retrospective validation settings 0 and 3, seed 2026',
              'inputs':{},'figures':{}}

def register(p):
    PROVENANCE['inputs'][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()

def save(fig, name, note):
    for ext in ['pdf','png']:
        fig.savefig(OUT / (name+'.'+ext),dpi=260,bbox_inches='tight')
    plt.close(fig)
    PROVENANCE['figures'][name]=note

def predictions(arm):
    p=RUN/arm/'tables'/('charbag_'+arm+'_predictions.csv')
    register(p)
    df=pd.read_csv(p)
    assert set(df.eval_split)=={'val'}
    df['n_slots']=10  # Guaranteed by the frozen complete-ten-slot cohort.
    return df

def reader_boxes():
    rows=[]
    for arm in ARMS:
        df=predictions(arm)
        for (setting,subject,book),d in df.groupby(['fold','subject','book']):
            rows.append(dict(arm=arm,setting=setting,subject=subject,book=book,n=len(d),
                             rho=float(spearmanr(d.y,d.pred).statistic)))
    slices=pd.DataFrame(rows)
    slices.to_csv(AUDIT/'reader_book_metrics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),sharey=True,layout='constrained')
    for ax,f in zip(axes,[0,3]):
        vals=[slices.loc[(slices.arm==a)&(slices.setting==f),'rho'].to_numpy() for a in ARMS]
        bp=ax.boxplot(vals,patch_artist=True,widths=.48,showfliers=False,
                      medianprops=dict(color='#22334a',linewidth=1.5))
        for box,c in zip(bp['boxes'],COLORS):box.set(facecolor=c,alpha=.32)
        for j,(a,v,c) in enumerate(zip(ARMS,vals,COLORS)):
            d=slices[(slices.arm==a)&(slices.setting==f)].reset_index(drop=True)
            for k,book in enumerate(['GarnettDream','LittlePrince']):
                vv=d.loc[d.book==book,'rho'].to_numpy()
                jitter=np.linspace(-.14,.14,len(vv))
                ax.scatter(j+1+jitter,vv,s=22,marker=['o','^'][k],color=c,
                           edgecolor='white',linewidth=.4,zorder=3)
        ax.set_xticks(range(1,5),['Baseline','Physical\nsupport','Local\nresidual','Unique\ntrajectory'])
        ax.axhline(0,color='#acb4bd',linewidth=.8)
        ax.set_title(f'Setting {f}: validation block {(f+1)%7}')
        ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Single-reading Spearman ρ, reader × book')
    axes[1].scatter([],[],marker='o',c='#506274',label='GarnettDream')
    axes[1].scatter([],[],marker='^',c='#506274',label='LittlePrince')
    axes[1].legend(loc='lower right',frameon=False,fontsize=8)
    save(fig,'fig03_reader_boxplots_20261007','19 reader-book observations per arm and setting; dependent descriptive boxes, not independent replicates.')

def geometry_controls():
    p=CODE/'results/single_model_breakthrough_20261006/complete_experiment_ledger.csv'
    register(p);d=pd.read_csv(p)
    fig,axes=plt.subplots(1,2,figsize=(10.6,3.7),sharey=True,layout='constrained')
    for ax,arms,title in zip(axes,
        [['global_aligned','noise_aligned'],['fixed_base','fixed_noise']],
        ['Variable clock + aligned module','Fixed grid + baseline']):
        for j,a in enumerate(arms):
            q=d[d.arm==a].sort_values('fold');assert len(q)==2
            ax.scatter([j-.06,j+.06],q.item_spearman,s=65,c=[COLORS[1] if j==0 else COLORS[2]],
                       marker='o',zorder=3)
            for x,(_,r) in zip([j-.06,j+.06],q.iterrows()):
                ax.annotate(f'{r.item_spearman:.3f}\ns{int(r.fold)}',(x,r.item_spearman),
                            xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks([0,1],['Recorded EEG','Matched synthetic noise'])
        ax.set_xlim(-.4,1.4);ax.set_ylim(.19,.48);ax.set_title(title)
        ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Item Spearman ρ')
    save(fig,'fig04_geometry_controls_20261007','Earlier full-cohort audit; each panel compares paired real/noise input geometry. Neither panel is a gain comparison against the strict ten-slot cohort.')

def scalp_maps():
    import mne
    cache=CODE/'data_cache/char_epochs_fixed_grid_350ms_20261006'
    manifest_path=CODE/'results/response_mechanism_20261002/sentence_manifest.csv'
    labels_path=CODE/'data_cache/surprisal_1829a99ad4.csv'
    register(cache/'CACHE_AUDIT.json');register(manifest_path);register(labels_path)
    labels=pd.read_csv(labels_path)
    mf=pd.read_csv(manifest_path)
    mf=mf[mf.included]
    joined=mf[['book','run','row_idx','block']].merge(labels[['book','run','row_idx','surprisal']],
        on=['book','run','row_idx'],validate='one_to_one')
    frames=[];indexes={};cache_population=[]
    for mp in sorted(cache.glob('meta*.csv')):
        meta=pd.read_csv(mp)
        r=meta.groupby(['subject','book','run','row_idx'],sort=False).agg(
            n_epochs=('char_slot','size'),n_slots=('n_slots','first'),first_slot=('char_slot','min'))
        r=r[(r.n_epochs==10)&(r.n_slots==10)&(r.first_slot==0)].reset_index()
        cache_population.append(r)
        loc=np.flatnonzero(meta.char_slot.eq(0))
        first=meta.iloc[loc][['subject','book','run','row_idx']].copy()
        first['epoch_index']=loc
        r=r.merge(first,on=['subject','book','run','row_idx'],validate='one_to_one')
        r=r.merge(joined,on=['book','run','row_idx'],validate='many_to_one')
        frames.append(r)
        indexes[(r.subject.iloc[0],r.book.iloc[0])]=mp
    pop=pd.concat(cache_population,ignore_index=True)
    eligible=pd.concat(frames,ignore_index=True)
    train=eligible[~eligible.block.isin([0,1])].copy()
    thresholds={b:g.drop_duplicates(['run','row_idx']).surprisal.quantile([1/3,2/3]).to_numpy()
                for b,g in train.groupby('book')}
    counts=[]
    for b,g in pop.groupby('book'):
        q=eligible[eligible.book==b]
        counts.append({'book':b,'cached_equal10_readings':len(g),
            'cached_equal10_items':len(g.drop_duplicates(['run','row_idx'])),
            'manifest_included_readings':len(q),
            'manifest_included_items':len(q.drop_duplicates(['run','row_idx']))})
    pd.DataFrame(counts).to_csv(AUDIT/'cohort_counts.csv',index=False)
    common=sorted(set(train[train.book=='GarnettDream'].subject)&set(train[train.book=='LittlePrince'].subject))
    wins=[(0,200),(200,400),(400,600),(600,800)]
    sample_ms=(np.arange(167)-13)*1000/128
    ii=[np.flatnonzero((sample_ms>=lo)&(sample_ms<hi)) for lo,hi in wins]
    contrasts=[];reports=[]
    for sub in common:
        pair=[]
        for book in ['GarnettDream','LittlePrince']:
            r=train[(train.subject==sub)&(train.book==book)]
            mp=indexes[(sub,book)]
            x=np.load(mp.with_name(mp.name.replace('meta_','X_').replace('.csv','.npy')),mmap_mode='r')
            vals=np.asarray(x[r.epoch_index.to_numpy()],dtype=np.float32)
            vals-=vals[:,:,:13].mean(-1,keepdims=True)
            lo,hi=thresholds[book];low=r.surprisal.to_numpy()<=lo;high=r.surprisal.to_numpy()>=hi
            diff=np.stack([vals[high][:,:,w].mean((0,2))-vals[low][:,:,w].mean((0,2)) for w in ii])
            pair.append(diff)
            reports.append(dict(subject=sub,book=book,n_low=int(low.sum()),n_high=int(high.sum())))
        contrasts.append(np.mean(pair,axis=0))
    by_reader=np.stack(contrasts);group=by_reader.mean(0)
    np.savez_compressed(AUDIT/'row_onset_topography.npz',group_uv=group,reader_uv=by_reader,
                        subjects=common,windows_ms=wins)
    electrode=Path('/data/my/ds004952/sub-04/ses-LittlePrince/eeg/sub-04_ses-LittlePrince_space-CapTrak_electrodes.tsv')
    e=pd.read_csv(electrode,sep='\t',encoding='utf-8-sig')
    info=mne.create_info(e['name'].tolist(),128,'eeg')
    montage=mne.channels.make_dig_montage(ch_pos={r['name']:np.array([r.x,r.y,r.z]) for _,r in e.iterrows()},coord_frame='head')
    info.set_montage(montage)
    lim=float(np.ceil(np.max(np.abs(group))*100)/100)
    fig,axes=plt.subplots(1,4,figsize=(11,3.15),layout='constrained')
    for ax,(lo,hi),v in zip(axes,wins,group):
        im,_=mne.viz.plot_topomap(v,info,axes=ax,show=False,cmap='RdBu_r',
            vlim=(-lim,lim),contours=0,sensors=True,sphere=(0,0,0,.12),extrapolate='head')
        ax.set_title(f'{lo}–{hi} ms',fontsize=11)
    cb=fig.colorbar(im,ax=axes,shrink=.7,pad=.01);cb.set_label('High − low row surprisal (µV)')
    save(fig,'fig05_row_onset_topography_20261007','Actual ROWS onset, first window only, fixed equal10 cohort; setting0 training blocks 2–6 only; nine readers with both books equal-weighted; baseline13 samples; unadjusted voltage, no CSD or model attribution.')
    (AUDIT/'topography_audit.json').write_text(json.dumps({'windows_ms':wins,'subjects':common,
        'thresholds':{k:v.tolist() for k,v in thresholds.items()},'records':reports,
        'group_rms_uv':np.sqrt((group**2).mean(1)).tolist(),'color_limit_uv':lim},indent=2))
    # Reader clustering of recorded row-onset contrasts: no learned model embeddings.
    rms=np.sqrt((by_reader**2).mean(-1))
    standardized=(rms-rms.mean(0))/rms.std(0)
    Z=linkage(standardized,method='ward')
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.5,4.2),gridspec_kw={'width_ratios':[1,1.45]},layout='constrained')
    dg=dendrogram(Z,labels=common,orientation='left',ax=ax1,color_threshold=0,
                  above_threshold_color='#718398')
    order=dg['leaves'];positions=np.arange(len(order))*10+5
    for j,(lo,hi) in enumerate(wins):
        ax2.scatter(rms[order,j],positions,s=50,c=[plt.cm.viridis(j/3)],label=f'{lo}–{hi} ms')
    ax1.set_xlabel('Ward merge distance');ax1.set_title('Descriptive reader clustering')
    ax2.set_yticks(positions,['']*len(order));ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlabel('Scalp RMS of high − low contrast (µV)')
    ax2.set_title('Recorded contrast magnitude');ax2.grid(axis='x',alpha=.15)
    ax2.legend(frameon=False,fontsize=8,loc='upper center',
               bbox_to_anchor=(.5,-.20),ncol=4)
    save(fig,'fig07_reader_clustering_20261007','Ward clustering from four standardized row-onset RMS features in setting0 training data, nine readers; post hoc, no statistical subtype or prediction claim.')

def distributions():
    df=predictions('consensus_source')
    fig,axes=plt.subplots(2,2,figsize=(10.5,6.3),layout='constrained')
    stats=[]
    for col,f in enumerate([0,3]):
        d=item_frame(df[df.fold==f].copy())
        # item_frame uses stable averaging and carries book/count descriptors.
        ax=axes[0,col]
        hb=ax.hexbin(d.y,d.pred,gridsize=26,mincnt=1,cmap='YlGnBu',linewidths=.2)
        ax.set_xlabel('Observed surprisal (training SD)');ax.set_ylabel('Predicted surprisal (training SD)')
        ax.set_title(f'Setting {f}: {len(d)} distinct rows')
        fig.colorbar(hb,ax=ax,label='Rows per hexagon',shrink=.8)
        ax=axes[1,col]
        for j,book in enumerate(['GarnettDream','LittlePrince']):
            q=d[d.book==book];bounds=q.y.quantile([1/3,2/3]).to_numpy()
            cat=np.digitize(q.y,bounds)
            vals=[q.pred[cat==i].to_numpy() for i in range(3)]
            pos=np.arange(3)+j*4
            parts=ax.violinplot(vals,positions=pos,showextrema=False,showmedians=True)
            for b in parts['bodies']:b.set_facecolor(COLORS[j+1]);b.set_alpha(.6)
            parts['cmedians'].set_color('#22334a')
            for x,v in zip(pos,vals):
                ax.text(x,.02,str(len(v)),fontsize=8,ha='center',va='bottom',
                        transform=ax.get_xaxis_transform())
            stats.append({'setting':f,'book':book,'item_rho':float(spearmanr(q.y,q.pred).statistic),'n_items':len(q)})
        ax.set_xticks([0,1,2,4,5,6],['Low','Mid','High']*2)
        ax.set_title('GarnettDream              LittlePrince',fontsize=10)
        ax.set_ylabel('Predicted score (training SD)')
        ax.grid(axis='y',alpha=.15)
    pd.DataFrame(stats).to_csv(AUDIT/'within_book_item_metrics.csv',index=False)
    save(fig,'fig06_prediction_distributions_20261007','Validation item means only. Hexagon densities and within-book/setting empirical observed-tercile violin groups describe prediction overlap; not thresholds selected for classification.')

def gate_figure():
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.5),layout='constrained')
    vals=[]
    for f in [0,3]:
        p=RUN/'consensus_source/checkpoints'/f'consensus_source_f{f}.pt'
        register(p);ck=torch.load(p,map_location='cpu',weights_only=False)
        vals.append(ck['state_dict']['deconv.gates'].tanh().numpy().reshape(-1))
    parts=axes[0].violinplot(vals,showmedians=True,showextrema=False)
    for b in parts['bodies']:b.set_facecolor(COLORS[1]);b.set_alpha(.5)
    axes[0].set_ylim(min(v.min() for v in vals)-.025,max(v.max() for v in vals)+.09)
    for j,v in enumerate(vals):
        axes[0].text(j+1,.94,f'{(v<0).mean():.1%} negative',ha='center',fontsize=9,
                     transform=axes[0].get_xaxis_transform())
    axes[0].set_xticks([1,2],['Setting 0','Setting 3']);axes[0].axhline(0,c='#9ba6b2',ls='--',lw=.8)
    axes[0].set_ylabel('Signed residual gate tanh(γ)');axes[0].set_title('Learned gate distribution')
    p=RUN/'frozen_sign_ablation.json';register(p);rows=json.loads(p.read_text())['scores']
    for j,f in enumerate([0,3]):
        q={r['ablation']:r['item']['spearman'] for r in rows if r['fold']==f}
        v=[q[k] for k in ['full','zero','positive_only','negative_only']]
        axes[1].scatter(v,np.arange(4)+j*.12,s=45,color=[COLORS[1],COLORS[2]][j],label=f'Setting {f}')
    axes[1].set_yticks(range(4),['Full','Zero','Positive only','Negative only'])
    axes[1].invert_yaxis();axes[1].set_xlabel('Item Spearman ρ');axes[1].set_title('Frozen-checkpoint intervention')
    axes[1].legend(frameon=False);axes[1].grid(axis='x',alpha=.15)
    save(fig,'fig08_gate_intervention_20261007','288 dependent learned coefficients per setting. Frozen gate sign intervention, no retraining or prediction-rule reselection.')

if __name__=='__main__':
    register(RUN/'complete_metrics.csv')
    reader_boxes();geometry_controls();scalp_maps();distributions();gate_figure()
    (AUDIT/'figure_provenance.json').write_text(json.dumps(PROVENANCE,indent=2))
    print('Generated six data figures, cohort statistics and provenance:',AUDIT)
