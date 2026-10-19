"""Build a static README header and one animated architecture diagram.

Uses Pillow drawing primitives and type, not a generative image model.
The diagram shows the actual ordering of the primary architecture at a high level.
Recorded-data animations use a separate script.
"""
from pathlib import Path
import argparse
import math
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/assets'
W=1120;S=2;FRAMES=60;DURATION=100
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

def rgb(h):return tuple(bytes.fromhex(h.lstrip('#')))
def mix(a,b,t):return tuple(round(x+(y-x)*t) for x,y in zip(rgb(a) if isinstance(a,str) else a,rgb(b) if isinstance(b,str) else b))

def render(frame,hero):
 H=440 if hero else 270
 bg='#0D1728' if hero else '#F6F8FC'
 fg='#F0F5FF' if hero else '#1D2B42'
 muted='#96A8C0' if hero else '#64758B'
 panel='#14233A' if hero else '#FFFFFF'
 border='#2B3D55' if hero else '#DFE6EF'
 teal='#52D7C3' if hero else '#118B88'
 amber='#F3C17E' if hero else '#C77D20'
 purple='#B4A5FA' if hero else '#8064CD'
 im=Image.new('RGB',(W*S,H*S),rgb(bg));d=ImageDraw.Draw(im)
 def rect(box,fill,outline=None,r=12,width=1):d.rounded_rectangle(tuple(round(v*S) for v in box),radius=r*S,fill=fill,outline=outline,width=width*S)
 def line(points,fill,width=2):d.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=width*S,joint='curve')
 def dot(x,y,r,c):d.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=c)
 def text(x,y,t,size=16,c=fg,bold=False,center=False):
  font=ImageFont.truetype(BOLD if bold else FONT,size*S)
  if center:x-=d.textlength(t,font=font)/(2*S)
  d.text((round(x*S),round(y*S)),t,font=font,fill=c,stroke_width=0)
 t=frame/FRAMES
 phase=t*4
 if hero:
  # Low-contrast structure and a restrained editorial title.
  for x in range(640,W,36):line([(x,20),(x,214)],'#14233A',1)
  for y in range(22,215,36):line([(638,y),(1082,y)],'#14233A',1)
  text(42,27,'EEG  /  EYE TRACKING  /  SURPRISAL',12,c=teal,bold=True)
  text(38,49,'HERMES',67,bold=True)
  text(42,132,'From reading signals to row-level surprisal',19,c='#B3C2D7')
  text(42,169,'Separate encoders · Residual fusion · Cross-slot integration',13,c=muted)
  # Representation tiles are schematic, not EEG amplitudes or attention weights.
  for layer in range(3):
   ox=824+layer*62;oy=49+layer*14
   for row in range(4):
    for col in range(5):
     v=.11+.58*(.5+.5*math.sin(2*math.pi*(t-row*.1-col*.065-layer*.16)))
     color=mix('#1A2B43',teal if layer==0 else (purple if layer==1 else amber),v)
     rect((ox+col*13,oy+row*13,ox+col*13+9,oy+row*13+9),color,r=2)
  text(966,142,'WINDOWS  /  SLOTS  /  ROW',10,c=muted,center=True)
  line([(42,207),(1078,207)],'#26354A',1)
  offset=230
 else:
  text(32,17,'THE PRIMARY HIERARCHY',12,c=muted,bold=True)
  offset=58
 # Input branches include their independent within-window encoders.
 cards=[(34,offset,226,63,'EEG encoder','128 × 167 → slot embedding',teal),
        (34,offset+82,226,63,'Eye encoder','4 × 45 → slot embedding',amber),
        (333,offset+42,181,64,'Residual fusion','EEG + linear(Eye)',purple),
        (578,offset+42,222,64,'Cross-slot encoder','sequence integration',purple),
        (863,offset+42,222,64,'Row readout','masked mean + regression',teal)]
 # All paths remain visible; moving particles communicate conceptual progression.
 routes=[([(260,offset+32),(296,offset+32),(296,offset+74),(333,offset+74)],teal,0),
         ([(260,offset+114),(296,offset+114),(296,offset+74),(333,offset+74)],amber,0),
         ([(514,offset+74),(578,offset+74)],purple,1),
         ([(800,offset+74),(863,offset+74)],purple,2)]
 for points,color,k in routes:
  line(points,border,2)
  q=phase-k
  if 0<=q<1:
   lengths=[math.dist(a,b) for a,b in zip(points,points[1:])];dist=q*sum(lengths)
   for i,length in enumerate(lengths):
    if dist<=length:
     a,b=points[i],points[i+1];u=dist/length
     x=a[0]+u*(b[0]-a[0]);y=a[1]+u*(b[1]-a[1])
     for radius,alpha in [(10,.12),(6,.22),(3,1)]:dot(x,y,radius,mix(bg,color,alpha))
     break
    dist-=length
  a,b=points[-2],points[-1]
  line([(b[0]-6,b[1]-4),b,(b[0]-6,b[1]+4)],border,1)
 for index,(x,y,w,h,title,sub,color) in enumerate(cards):
  stage=0 if index<2 else index-1
  active=max(0,1-abs(phase-stage-.45)/.65)
  outline=mix(border,color,active*.8)
  rect((x,y,x+w,y+h),panel,outline,11,1)
  rect((x+1,y+13,x+4,y+h-13),color,r=1)
  text(x+15,y+12,title,17,c=fg,bold=True)
  text(x+15,y+37,sub,11,c=muted)
  if index==4:
   # A moving cursor, not an invented numerical score.
   line([(x+15,y+h+14),(x+w-15,y+h+14)],border,2)
   f=.5+.37*math.sin(2*math.pi*t)
   dot(x+15+f*(w-30),y+h+14,3,teal)
 # Indicator for the current stage; no fabricated performance counters.
 labels=['LOCAL ENCODING','RESIDUAL FUSION','CROSS-SLOT INTEGRATION','ROW PREDICTION']
 basey=H-32
 for i,label in enumerate(labels):
  x=42+i*258
  dot(x,basey+6,2.5,teal if int(phase)%4==i else border)
  text(x+12,basey,label,9,c=fg if int(phase)%4==i else muted,bold=True)
 if not hero:
  text(32,H-16,'Schematic architecture; the diagram does not show measured activations.',9,c=muted)
 return im.resize((W,H),Image.Resampling.LANCZOS)

def build_architecture_animation():
 frames=[render(i,False) for i in range(FRAMES)]
 samples=Image.new('RGB',(W,frames[0].height*4))
 for j,k in enumerate([0,15,30,45]):samples.paste(frames[k],(0,j*frames[0].height))
 palette=samples.quantize(colors=128,method=Image.Quantize.MEDIANCUT)
 quant=[f.quantize(palette=palette,dither=Image.Dither.NONE) for f in frames]
 path=OUT/'architecture_animated.gif'
 quant[0].save(path,save_all=True,append_images=quant[1:],duration=DURATION,loop=0,optimize=True,disposal=1)
 with Image.open(path) as image:
  assert image.n_frames==FRAMES and image.info['loop']==0
  for i in range(FRAMES):
   image.seek(i)
   assert image.info['duration']==DURATION
 print(f'Architecture animation: {FRAMES} frames, {FRAMES*DURATION} ms.')

def main():
 global OUT
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-dir',type=Path,default=OUT)
 OUT=parser.parse_args().output_dir
 OUT.mkdir(parents=True,exist_ok=True)
 render(0,True).crop((0,0,W,204)).save(OUT/'overview.png')
 render(0,False).save(OUT/'architecture.png')
 build_architecture_animation()
 print('Static header and architecture fallback written.')

if __name__=='__main__':main()
