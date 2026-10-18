"""Measured row-onset scalp contrast and between-reader dispersion animations.

Each frame is a complete 200-ms window mean. There is no temporal interpolation.
Only MNE's spatial interpolation is used between the 128 observed electrodes.
"""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import mne
import numpy as np
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'analysis/verified_revision_20261007/row_onset_topography.npz'
ELECTRODES=SOURCE.parent/'electrodes.tsv'
OUT=ROOT/'docs/appendix/topography'
WINDOWS=np.array([[0,200],[200,400],[400,600],[600,800]])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False,'text.color':'#18324E',
                     'axes.labelcolor':'#18324E','axes.titleweight':'bold'})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def topology(ax,v,info,limit,cmap):
 image,_=mne.viz.plot_topomap(v,info,axes=ax,show=False,cmap=cmap,vlim=limit,
                              contours=0,sensors=True,sphere=(0,0,0,.12),extrapolate='head')
 return image
def draw_frames(values,info,limit,cmap,stem,title,label,series,series_label):
 frames=[]
 for j,(lo,hi) in enumerate(WINDOWS):
  fig=plt.figure(figsize=(8.5,6),facecolor='#F7F9FC')
  fig.suptitle(title,fontsize=18,y=.965,weight='bold')
  fig.text(.5,.887,f'{lo}–{hi} ms after observed row start',ha='center',fontsize=15,color='#167E83')
  ax=fig.add_axes([.12,.245,.54,.58]);ax.set_facecolor('#F7F9FC')
  im=topology(ax,values[j],info,limit,cmap)
  cax=fig.add_axes([.72,.32,.026,.41]);cb=fig.colorbar(im,cax=cax);cb.set_label(label,fontsize=11,labelpad=11)
  tax=fig.add_axes([.14,.11,.69,.075]);tax.set_ylim(0,max(series)*1.18)
  tax.plot(WINDOWS.mean(1),series,color='#B2C3CE',marker='o',markersize=5,lw=1.5)
  tax.scatter([WINDOWS[j].mean()],[series[j]],color='#167E83',s=55,zorder=4)
  tax.set_xticks(WINDOWS.mean(1),[f'{a}–{b}' for a,b in WINDOWS],fontsize=9)
  tax.set_yticks([]);tax.spines[['left','top','right']].set_visible(False);tax.spines['bottom'].set_color('#CBD6DF')
  tax.set_title(f'{series_label}: {series[j]:.3f} µV',loc='left',fontsize=10,pad=7)
  tax.tick_params(axis='x',length=0)
  fig.text(.5,.035,'128 sensors · 9 readers · training data only · fixed color scale',ha='center',fontsize=10,color='#607389')
  fig.canvas.draw();arr=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy();frames.append(Image.fromarray(arr))
  fig.savefig(OUT/f'{stem}_frame_{j+1:02d}.png',dpi=150,facecolor=fig.get_facecolor())
  plt.close(fig)
 frames[0].save(OUT/f'{stem}.gif',save_all=True,append_images=frames[1:],duration=1500,loop=0,optimize=True)
 frames[0].save(OUT/f'{stem}_poster.png')
 # Small palette quantization affects display only, never the saved source arrays.
 return len(frames)
def contact(values,info,limit,cmap,stem,label):
 fig,axes=plt.subplots(1,4,figsize=(10.5,2.8),layout='constrained')
 for ax,v,(lo,hi) in zip(axes,values,WINDOWS):
  im=topology(ax,v,info,limit,cmap);ax.set_title(f'{lo}–{hi} ms',fontsize=11)
 cb=fig.colorbar(im,ax=axes,shrink=.7,pad=.01);cb.set_label(label,fontsize=9)
 for ext in ['pdf','png']:fig.savefig(OUT/f'{stem}_keyframes.{ext}',dpi=220,bbox_inches='tight')
 plt.close(fig)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 a=np.load(SOURCE,allow_pickle=False);group=a['group_uv'];reader=a['reader_uv'];subjects=a['subjects']
 assert group.shape==(4,128) and reader.shape==(9,4,128)
 assert np.array_equal(a['windows_ms'],WINDOWS) and np.isfinite(reader).all()
 assert np.allclose(reader.mean(0),group,atol=1e-7)
 audit=json.loads((SOURCE.parent/'topography_audit.json').read_text())
 rms=np.sqrt((group**2).mean(1));assert np.allclose(rms,audit['group_rms_uv'],atol=1e-7)
 sd=reader.std(axis=0,ddof=1);sd_rms=np.sqrt((sd**2).mean(1))
 group_lim=float(np.ceil(abs(group).max()*100)/100)
 sd_lim=float(np.ceil(sd.max()*100)/100)
 e=pd.read_csv(ELECTRODES,sep='\t',encoding='utf-8-sig')
 assert e.name.tolist()==[f'E{i}' for i in range(1,129)]
 info=mne.create_info(e.name.tolist(),128,ch_types='eeg')
 info.set_montage(mne.channels.make_dig_montage(ch_pos={x['name']:np.array([x['x'],x['y'],x['z']]) for _,x in e.iterrows()},coord_frame='head'))
 n1=draw_frames(group,info,(-group_lim,group_lim),'RdBu_r','group_contrast','Group mean surprisal contrast','High − low voltage (µV)',rms,'Scalp RMS')
 n2=draw_frames(sd,info,(0,sd_lim),'YlOrRd','reader_dispersion','Reader dispersion of the same contrast','Between-reader SD (µV)',sd_rms,'Scalp RMS of reader SD')
 contact(group,info,(-group_lim,group_lim),'RdBu_r','group_contrast','High − low voltage (µV)')
 contact(sd,info,(0,sd_lim),'YlOrRd','reader_dispersion','Between-reader SD (µV)')
 np.savez_compressed(OUT/'animation_source_arrays.npz',group_uv=group,reader_uv=reader,reader_sd_uv=sd,
                     subjects=subjects,windows_ms=WINDOWS,electrode_names=e.name.to_numpy(dtype=str),xyz_m=e[['x','y','z']].to_numpy())
 pd.DataFrame({'start_ms':WINDOWS[:,0],'end_ms':WINDOWS[:,1],'group_rms_uv':rms,'reader_sd_rms_uv':sd_rms,'reader_sd_sensor_mean_uv':sd.mean(1)}).to_csv(OUT/'window_summary.csv',index=False)
 records={'source_npz':str(SOURCE.relative_to(ROOT)),'source_sha256':sha(SOURCE),'electrodes_sha256':sha(ELECTRODES),
          'subjects':subjects.tolist(),'windows_ms':WINDOWS.tolist(),'n_readers':9,'n_sensors':128,
          'group_rms_uv':rms.tolist(),'reader_sd_rms_uv':sd_rms.tolist(),'reader_sd_sensor_mean_uv':sd.mean(1).tolist(),
          'group_color_limits_uv':[-group_lim,group_lim],'reader_sd_color_limits_uv':[0,sd_lim],
          'group_frames':n1,'reader_frames':n2,'frame_duration_ms':1500,'temporal_interpolation':False,
          'reader_dispersion_definition':'Sample SD across nine signed reader-level contrasts (ddof=1); not SE or a significance test.',
          'data_scope':'First fixed-grid window anchored to observed row start; setting-0 training blocks 2–6 only; exactly ten complete slots; within-book training terciles; equal weight to books then readers; 13-sample baseline; recorded voltage before CSD.',
          'spatial_interpolation':'MNE topomap, sphere=(0,0,0,0.12), extrapolate=head, contours=0; same sensor montage as paper Figure 5.',
          'mne_version':mne.__version__}
 (OUT/'provenance.json').write_text(json.dumps(records,indent=2)+'\n')
 for stem in ['group_contrast','reader_dispersion']:
  im=Image.open(OUT/f'{stem}.gif');assert im.n_frames==4 and im.info['loop']==0
 print(json.dumps(records,indent=2))
if __name__=='__main__':main()
