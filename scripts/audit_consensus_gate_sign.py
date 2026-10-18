#!/usr/bin/env python3
"""Frozen-checkpoint mechanism audit; never selects a new prediction rule."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import run_support_consensus_cv as experiment
from foundation.gpu_store import GpuEpochStore,GpuBatcher
from foundation.data import ShardedEpochs,SURPRISAL_CSV
from cogalign.io.retrospective_splits import attach_retrospective_split
from summarize_retrospective_baseline import metrics,item_frame,conditional_rho
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/support_consensus_20261006'
CACHE=ROOT/'data_cache/char_epochs_fixed_grid_350ms_20261006'

def main():
    experiment.armmode='support'
    bags=experiment.equal_cohort(cache_dir=CACHE)
    store=GpuEpochStore(ShardedEpochs(bags.epochs),bags.slot_index,device=torch.device('cuda:0'))
    manifest=pd.read_csv(ROOT/'results/response_mechanism_20261002/sentence_manifest.csv')
    results=[];frames=[]
    for fold in [0,3]:
        checkpoint=torch.load(OUT/f'consensus_source/checkpoints/consensus_source_f{fold}.pt',map_location='cpu',weights_only=False)
        state=checkpoint['state_dict'];args=SimpleNamespace(**checkpoint['config'])
        spatial=experiment.global_runner.IdentityIndependentSpatial(state['spatial_op.matrices'].numpy(),np.ones(19,dtype=np.float32))
        model=experiment.ConsensusDecoder(spatial,d_model=args.d_model,stride=args.seq_stride,time_depth=args.seq_depth,
            char_depth=args.char_depth,block=args.seq_block,dropout=args.dropout,valid_slot_norm=args.valid_slot_norm).to('cuda:0')
        model.load_state_dict(state);model.eval()
        original=model.deconv.gates.detach().clone()
        f=attach_retrospective_split(bags.obs,manifest,fold,label_csv=str(SURPRISAL_CSV))
        f=f[f.split.eq('val')].reset_index(drop=True)
        loader=GpuBatcher(store,f.obs_index.to_numpy(),f.y.to_numpy(np.float32),f.group_id.to_numpy(),
                          batch_size=64,shuffle=False,drop_last=False)
        for name,gate in [('full',original),('zero',torch.zeros_like(original)),
                          ('positive_only',original.clamp_min(0)),('negative_only',original.clamp_max(0))]:
            with torch.no_grad():model.deconv.gates.copy_(gate)
            pred=experiment.runner.predict(model,loader,torch.device('cuda:0'),args=args)
            g=f.assign(pred=pred,fold=fold,ablation=name)
            if name=='full':
                archived=pd.read_csv(OUT/'consensus_source/tables/charbag_consensus_source_predictions.csv')
                pair=g.merge(archived[archived.fold.eq(fold)][['subject','book','run','row_idx','pred']],
                             on=['subject','book','run','row_idx'],suffixes=('_new','_old'),validate='one_to_one')
                if np.max(abs(pair.pred_new-pair.pred_old))>2e-6:raise ValueError('Frozen replay does not match original predictions')
            results.append({'fold':fold,'ablation':name,'trial':metrics(g),'item':metrics(item_frame(g)),
                            'conditional_rho':conditional_rho(item_frame(g))})
            frames.append(g[['subject','book','run','row_idx','item_id','y','pred','fold','ablation']])
        print(f'[gate sign audit] fold{fold} complete',flush=True)
    (OUT/'frozen_sign_ablation.json').write_text(json.dumps({'scope':'selected source checkpoints; no retraining or promotion selection',
        'scores':results},indent=2)+'\n')
    pd.concat(frames).to_csv(OUT/'frozen_sign_predictions.csv',index=False)
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
