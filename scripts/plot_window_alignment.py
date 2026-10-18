#!/usr/bin/env python
"""Plot completed paired folds, never partial scores."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]/'results/window_alignment_20261008'
t=pd.read_csv(R/'complete_metrics.csv')
names=['E800_G800','E1200_G800','E1200_G1200'];colors=['#40988f','#355887','#cd884c']
plt.rcParams.update({'font.size':11,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(11.5,3.6))
for ax,(metric,label) in zip(axes,[('item_spearman','Item Spearman ρ'),('trial_spearman','Single-reading Spearman ρ'),('item_auc_hi_vs_lo','High–low item ROC AUC')]):
 pivot=t.pivot(index='fold',columns='window',values=metric)[names]
 for _,row in pivot.iterrows():ax.plot(range(3),row.values,color='#b6bdc7',lw=.8,zorder=1)
 for i,name in enumerate(names):
  ax.scatter([i]*7,pivot[name],color=colors[i],s=27,zorder=2)
  ax.scatter(i,pivot[name].mean(),marker='D',s=65,color=colors[i],edgecolor='black',lw=.8,zorder=3)
 ax.set_xticks(range(3),['800 / 800','1200 / 800','1200 / 1200']);ax.set_xlabel('EEG / Eye post-window (ms)');ax.set_ylabel(label)
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
fig.suptitle('Matched window-length diagnostic: seven validation settings, seed2026',fontsize=12)
fig.tight_layout();fig.savefig(R/'window_comparison.pdf',bbox_inches='tight');fig.savefig(R/'window_comparison.png',dpi=180,bbox_inches='tight')
