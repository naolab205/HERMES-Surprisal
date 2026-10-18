"""Render finer measured 0–2000ms maps and a ten-reader scalp mosaic.

39 overlapping 100-ms means, shifted by 50ms, are shown at 100ms/frame.
No temporal morphing or fabricated participant is used.
"""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import mne
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/appendix/extended'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titleweight':'bold','text.color':'#1B324D','pdf.fonttype':42})

def make_info():
 e=pd.read_csv(ROOT/'analysis/verified_revision_20261007/electrodes.tsv',sep='\t',encoding='utf-8-sig')
 info=mne.create_info(e.name.tolist(),128,'eeg')
 info.set_montage(mne.channels.make_dig_montage(ch_pos={row['name']:np.array([row.x,row.y,row.z]) for _,row in e.iterrows()},coord_frame='head'))
 return info

def topo(ax,values,info,limit):
 return mne.viz.plot_topomap(values,info,axes=ax,show=False,cmap='RdBu_r',vlim=(-limit,limit),
                            contours=0,sensors=True,sphere=(0,0,0,.12),extrapolate='head')[0]
def capture(fig):
 fig.canvas.draw();return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
def gif(frames,name):
 frames[0].save(OUT/(name+'.gif'),save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
 frames[0].save(OUT/(name+'_poster.png'))
 im=Image.open(OUT/(name+'.gif'));assert im.n_frames==39 and im.info['loop']==0
 durations=[]
 for i in range(im.n_frames):im.seek(i);durations.append(im.info['duration'])
 assert durations==[100]*39
 print(name,im.n_frames,'frames',sum(durations),'ms', (OUT/(name+'.gif')).stat().st_size,'bytes',flush=True)

def group_movie(group,windows,info,limit):
 rms=np.sqrt((group**2).mean(1));frames=[];centers=windows.mean(1)
 fig=plt.figure(figsize=(8.5,6),facecolor='#F7F9FC')
 fig.suptitle('Extended row-onset scalp contrast',fontsize=18,y=.97,weight='bold')
 stamp=fig.text(.5,.895,'',ha='center',fontsize=15,color='#177F87')
 ax=fig.add_axes([.11,.25,.55,.59]);im=topo(ax,group[0],info,limit)
 # Replot each measured map with the same geometry and scale. No interpolation in time.
 cax=fig.add_axes([.72,.33,.026,.41]);fig.colorbar(ScalarMappable(norm=Normalize(-limit,limit),cmap='RdBu_r'),cax=cax,label='High − low voltage (µV)')
 tax=fig.add_axes([.14,.12,.69,.065]);tax.plot(centers,rms,color='#B9C9D2',lw=1.4)
 dot,=tax.plot([centers[0]],[rms[0]],'o',color='#167E83',markersize=7)
 tax.set_xlim(0,2000);tax.set_ylim(0,rms.max()*1.15);tax.set_xticks([0,500,1000,1500,2000]);tax.set_yticks([])
 tax.spines[['left','right','top']].set_visible(False);tax.spines['bottom'].set_color('#CBD6DF');tax.set_xlabel('Time after observed row start (ms)',fontsize=10)
 line=tax.set_title('',loc='left',fontsize=10,pad=6)
 fig.text(.5,.025,'10 healthy readers · 100-ms window / 50-ms step · measured summaries',ha='center',fontsize=9,color='#61758A')
 for i,(lo,hi) in enumerate(windows):
  ax.clear();topo(ax,group[i],info,limit);stamp.set_text(f'{lo}–{hi} ms after row start')
  dot.set_data([centers[i]],[rms[i]]);line.set_text(f'Scalp RMS: {rms[i]:.3f} µV')
  frames.append(capture(fig))
  if i in [0,19,38]:frames[-1].save(OUT/f'group_extended_frame_{i:02d}.png')
 plt.close(fig);gif(frames,'group_extended')

def reader_movie(reader,subjects,windows,info,limit):
 frames=[];rms=np.sqrt((reader**2).mean(-1))
 fig,axes=plt.subplots(2,5,figsize=(13.8,6.5),facecolor='#F7F9FC')
 fig.subplots_adjust(left=.025,right=.895,bottom=.12,top=.81,hspace=.4,wspace=.2)
 fig.suptitle('Ten-reader scalp contrasts',fontsize=21,y=.98,weight='bold')
 stamp=fig.text(.5,.89,'',ha='center',fontsize=16,color='#177F87')
 cax=fig.add_axes([.925,.255,.015,.43]);cb=fig.colorbar(ScalarMappable(norm=Normalize(-limit,limit),cmap='RdBu_r'),cax=cax)
 cb.set_label('High − low voltage (µV)',fontsize=11)
 fig.text(.5,.047,'Same signed scale for every reader and time window · training data only',ha='center',fontsize=10,color='#61758A')
 fig.text(.5,.015,'sub-15 has GarnettDream only; the other nine readers contribute two books.',ha='center',fontsize=9,color='#61758A')
 for j,(lo,hi) in enumerate(windows):
  stamp.set_text(f'{lo}–{hi} ms · 100-ms mean, 50-ms advance')
  for i,(ax,subject) in enumerate(zip(axes.flat,subjects)):
   ax.clear();topo(ax,reader[i,j],info,limit)
   ax.set_title(f'{subject}\nRMS {rms[i,j]:.2f} µV',fontsize=10,pad=7)
  frames.append(capture(fig))
  if j in [0,19,38]:frames[-1].save(OUT/f'readers_extended_frame_{j:02d}.png')
 plt.close(fig);gif(frames,'readers_extended')

def keyframes(group,reader,windows,info,glim,rlim):
 indices=[0,10,20,30,38]
 fig,axes=plt.subplots(1,5,figsize=(11,2.6),layout='constrained')
 for ax,j in zip(axes,indices):
  im=topo(ax,group[j],info,glim);lo,hi=windows[j];ax.set_title(f'{lo}–{hi} ms',fontsize=10)
 fig.colorbar(im,ax=axes,shrink=.7,pad=.01,label='High − low (µV)')
 fig.savefig(OUT/'extended_group_keyframes.pdf',bbox_inches='tight');plt.close(fig)

def main():
 a=np.load(OUT/'extended_source_arrays.npz',allow_pickle=False)
 group=a['group_uv'];reader=a['reader_uv'];windows=a['windows_ms'];subjects=a['subjects']
 assert group.shape==(39,128) and reader.shape==(10,39,128)
 assert np.allclose(group,reader.mean(0)) and np.isfinite(reader).all()
 info=make_info();glim=float(np.ceil(abs(group).max()*100)/100);rlim=float(np.ceil(abs(reader).max()*100)/100)
 print('Shared display limits:',glim,rlim,'µV',flush=True)
 group_movie(group,windows,info,glim);reader_movie(reader,subjects,windows,info,rlim)
 keyframes(group,reader,windows,info,glim,rlim)
 p={'source_sha256':hashlib.sha256((OUT/'extended_source_arrays.npz').read_bytes()).hexdigest(),
    'subjects':subjects.tolist(),'n_readers':10,'n_frames':39,'window_width_ms':100,'window_step_ms':50,'display_frame_ms':100,
    'animation_duration_ms':3900,'time_extent_ms':[0,2000],'temporal_interpolation':False,
    'group_color_limits_uv':[-glim,glim],'reader_color_limits_uv':[-rlim,rlim],
    'shared_across_readers_and_time':True,'mne_version':mne.__version__,
    'cohort_difference':'Available-book average for sub-15; do not equate to the balanced nine-reader main analysis.'}
 (OUT/'rendering_provenance.json').write_text(json.dumps(p,indent=2)+'\n')
 print('EXTENDED ANIMATIONS COMPLETE',flush=True)
if __name__=='__main__':main()
