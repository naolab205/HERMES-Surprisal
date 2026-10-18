#!/usr/bin/env python3
"""Same preregistered bootstrap for schedule vs baseline/capacity controls."""
import argparse,json
from pathlib import Path
import pandas as pd
from bootstrap_physical_screen import paired
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/single_model_breakthrough_20261006/physical_screen'
ap=argparse.ArgumentParser();ap.add_argument('--valid-normalization',action='store_true');args=ap.parse_args()
if args.valid_normalization:candidate,controls='phys_deconv_valid',['phys_valid','phys_unshifted_valid']
else:candidate,controls='phys_deconv',['phys_control','phys_unshifted']
manifest=pd.read_csv(ROOT/'results/response_mechanism_20261002/sentence_manifest.csv')
a=pd.read_csv(BASE/f'{candidate}/tables/charbag_{candidate}_predictions.csv');rows=[]
for control in controls:
    b=pd.read_csv(BASE/f'{control}/tables/charbag_{control}_predictions.csv')
    for fold in (0,3):rows.append({'candidate':candidate,'control':control,**paired(a,b,manifest,fold)})
(BASE/f'{candidate}_bootstrap.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
