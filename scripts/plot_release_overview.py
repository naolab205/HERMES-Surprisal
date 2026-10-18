"""Draw the README comparison from seven observed archived fold scores."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/assets';OUT.mkdir(parents=True,exist_ok=True)
def main():
 arms=[('MM_mid','HERMES','#137F87'),('ABL_noeye','EEG only','#437DB8'),('ABL_nochar','No cross-slot block','#D79845')]
 fig,axes=plt.subplots(1,2,figsize=(10.5,3.2),layout='constrained')
 plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
 for ax,metric,title in zip(axes,['item_spearman','item_auc_hi_vs_lo'],['Item Spearman correlation','High / low item AUC']):
  for i,(arm,label,color) in enumerate(arms):
   f=pd.read_csv(ROOT/'results/tables'/f'charbag_{arm}_per_fold.csv').sort_values('fold')
   if f.fold.tolist()!=list(range(7)):raise ValueError(f'Incomplete fold table: {arm}')
   values=f[metric].to_numpy();offset=np.linspace(-.1,.1,len(values))
   ax.scatter(values,i+offset,s=31,color=color,alpha=.7,zorder=3)
   ax.scatter([values.mean()],[i],s=68,marker='D',color=color,edgecolor='white',linewidth=1,zorder=4)
  ax.set_yticks(range(3),[x[1] for x in arms]);ax.invert_yaxis();ax.set_ylim(2.5,-.5)
  ax.set_title(title,loc='left',fontsize=12,fontweight='bold',pad=13)
  ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
  for edge in ['top','right','left']:ax.spines[edge].set_visible(False)
  ax.spines['bottom'].set_color('#B9C9D1');ax.tick_params(axis='both',length=0,labelsize=10)
 fig.supxlabel('Seven retrospective item-block folds · dots = folds · diamonds = means',fontsize=10,color='#526878')
 for ext in ['png','svg']:fig.savefig(OUT/f'archived_results.{ext}',dpi=180,facecolor='white',bbox_inches='tight')
 plt.close(fig)
if __name__=='__main__':main()
