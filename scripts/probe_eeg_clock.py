#!/usr/bin/env python3
"""Training-text-only clock probe on equal physical supports, without labels.

Fourier phase on two disjoint late windows; all frequency/channel selection
uses fit texts only. Duration is used solely for offline association reporting.
"""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foundation.data import load_char_bags,ShardedEpochs
from cogalign.io.splits import add_item_ids
from build_char_epoch_cache import canonical_montage_positions
CACHE=ROOT/'data_cache/char_epochs_fixed_grid_350ms_20261006'
OUT=ROOT/'results/eeg_clock_20261006'
FREQ=np.linspace(2.,3.6,65)
WINDOWS=((.8,2.1),(2.6,3.9))

def get_frames(bags,manifest):
    base=add_item_ids(bags.obs);base['obs_index']=np.arange(len(base))
    base=base.merge(manifest.loc[manifest.included,['item_id','block','sentence']],
                    on='item_id',validate='many_to_one')
    base=base[base.n_slots.eq(10)]
    frames={}
    for fold in [0,3]:
        f=base[~base.block.isin([fold,(fold+1)%7])].copy()
        rng=np.random.default_rng(2026+fold);chosen=[]
        for book,cap in [('GarnettDream',300),('LittlePrince',150)]:
            g=f[f.book.eq(book)].drop_duplicates('item_id');n=0
            for sentence in rng.permutation(g.sentence.unique()):
                ids=g.loc[g.sentence.eq(sentence),'item_id'].tolist()
                chosen.extend(ids);n+=len(ids)
                if n>=cap:break
        frames[fold]=f[f.item_id.isin(chosen)].copy()
    return frames

def extract(bags,indices):
    meta=pd.concat([pd.read_csv(CACHE/('meta_'+g.replace('|','_')+'.csv'))
                    for g in bags.groups],ignore_index=True)
    offsets=np.rint(np.arange(10)*.35*128).astype(int)
    length=int(offsets[-1]+167);shards=ShardedEpochs(bags.epochs)
    csd=np.load(CACHE/'csd_matrix.npy').astype(np.float32)
    times=(np.arange(length)-13)/128
    result=[]
    for lo in range(0,len(indices),32):
        ids=indices[lo:lo+32];ix=np.stack([bags.slot_index[int(i)] for i in ids])
        assert np.array_equal(meta.char_slot.to_numpy()[ix],np.broadcast_to(np.arange(10),ix.shape))
        x=shards.take(ix.ravel()).reshape(len(ids),10,128,167)
        # Unique physical trajectory: overlapping samples must agree exactly.
        wave=np.zeros((len(ids),128,length),dtype=np.float32);used=np.zeros(length,bool)
        for slot,start in enumerate(offsets):
            overlap=used[start:start+167]
            if overlap.any():
                a=wave[:,:,start:start+167][...,overlap];b=x[:,slot][...,overlap]
                if not np.array_equal(a,b):raise ValueError('Inconsistent duplicate physical samples')
            wave[:,:,start:start+167]=x[:,slot];used[start:start+167]=True
        assert used.all()
        features=[]
        for a,b in WINDOWS:
            keep=(times>=a)&(times<b);t=times[keep]
            w=np.hanning(keep.sum()).astype(np.float32)
            basis=(np.exp(-2j*np.pi*t[:,None]*FREQ[None])*w[:,None]).astype(np.complex64)
            z=detrend(wave[...,keep],axis=-1,type='linear')@basis
            z=np.einsum('cd,bdf->bcf',csd,z,optimize=True)
            features.append(z/(np.abs(z)+1e-12))
        result.append(np.stack(features,axis=1).astype(np.complex64))
        if lo%1024==0:print(f'[clock features] {lo}/{len(indices)}',flush=True)
    return np.concatenate(result)

def held_test(zfit,zheld,sentences,allowed,seed):
    mean=zfit.mean(0) # window,channel,frequency
    agreement=np.real(mean[0]*mean[1].conj())
    strength=np.minimum(abs(mean[0]),abs(mean[1]))*np.maximum(agreement,0)
    selected=np.argsort(strength[allowed],axis=0)[-8:]
    spectral=np.take_along_axis(strength[allowed],selected,axis=0).mean(0)
    k=int(spectral.argmax());channels=allowed[selected[:,k]]
    template=mean[:,channels,k];template=template/(abs(template)+1e-12)
    projections=zheld[:,:,channels,k]*template.conj()[None]
    score=float(np.min(np.real(projections.mean(axis=(0,2)))))
    _,groups=np.unique(sentences,return_inverse=True);rng=np.random.default_rng(seed)
    null=[]
    for _ in range(199):
        phase=np.exp(1j*rng.uniform(-np.pi,np.pi,groups.max()+1))[groups]
        null.append(float(np.min(np.real((projections*phase[:,None,None]).mean(axis=(0,2))))))
    p=(1+sum(v>=score for v in null))/200
    phase_agreement=float(np.real(np.mean(template[0]*template[1].conj())))
    # A tentative per-reading frequency, derived from EEG alone. This is not a
    # verified stimulus-onset estimator. SOA is not consulted here.
    m=mean[:,channels,:];m=m/(abs(m)+1e-12)
    alignment=np.real(zheld[:,:,channels,:]*m.conj()[None]).mean(axis=(1,2))
    predicted=FREQ[alignment.argmax(-1)]
    return {'frequency_hz':float(FREQ[k]),'held_min_projection':score,
            'phase_randomization_p':p,'early_late_phase_agreement':phase_agreement,
            'channels':channels.tolist(),'null_95pct':np.quantile(null,[.025,.975]).tolist()},predicted

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'scope':'training texts only; no surprisal labels loaded',
              'folds':[0,3],'inner':'leave-one-training-block-out',
              'input':'full ten-slot rows only; equal 570sample physical support; no duration, count or reader features',
              'windows_seconds':WINDOWS,'frequency_grid_hz':FREQ.tolist(),
              'fit':'training EEG selects8channels and global frequency by sustained phase coherence',
              'null':'199 sentence-shared phase rotations; preserves Fourier magnitudes and channel dependence',
              'gate':'in BOTH fold settings >=4/5 held blocks projection>.05, p<=.05, phase agreement>.5; selected-frequency SD<=.15Hz; posterior arm also required',
              'limitation':'phase locking is not surprisal specificity; estimated SOA is an offline nuisance diagnostic only'}
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    bags=load_char_bags(cache_dir=CACHE)
    manifest_path=ROOT/'results/response_mechanism_20261002/sentence_manifest.csv'
    manifest=pd.read_csv(manifest_path);frames=get_frames(bags,manifest)
    ids=np.unique(np.concatenate([f.obs_index.to_numpy(int) for f in frames.values()]))
    z=extract(bags,ids);lookup=dict(zip(ids,np.arange(len(ids))))
    xyz,names=canonical_montage_positions();assert len(names)==128
    arms={'all':np.arange(128),'posterior':np.flatnonzero(xyz[:,1]<=0)}
    results=[];predictions=[]
    for fold,f in frames.items():
        f['feature_index']=f.obs_index.map(lookup)
        for block in sorted(f.block.unique()):
            fit=f[f.block.ne(block)];held=f[f.block.eq(block)]
            for arm,channels in arms.items():
                r,pred=held_test(z[fit.feature_index],z[held.feature_index],
                                (held.book+'|'+held.sentence).to_numpy(),channels,2026+int(block))
                r.update(fold=fold,held_block=int(block),arm=arm,n_readings=len(held),n_items=held.item_id.nunique())
                # Reference SOA has not entered feature extraction, selection,
                # or frequency prediction. It is inspected only after fitting.
                h=held[['subject','book','run','row_idx','item_id','dur']].copy()
                h['estimated_reference_frequency']=10/h.dur
                h['predicted_eeg_frequency']=pred;h['fold']=fold;h['held_block']=int(block);h['arm']=arm
                r['diagnostic_frequency_rho']=float(spearmanr(h.estimated_reference_frequency,pred).statistic)
                results.append(r);predictions.append(h)
            print(f'[clock] fold{fold} heldblock{block} complete',flush=True)
    gates=[]
    for fold in [0,3]:
        for arm in arms:
            rows=[r for r in results if r['fold']==fold and r['arm']==arm]
            good=sum(r['held_min_projection']>.05 and r['phase_randomization_p']<=.05 and r['early_late_phase_agreement']>.5 for r in rows)
            sd=float(np.std([r['frequency_hz'] for r in rows]))
            gates.append({'fold':fold,'arm':arm,'passing_blocks':good,'frequency_sd_hz':sd,'passed':good>=4 and sd<=.15})
    pd.concat(predictions).to_csv(OUT/'frequency_predictions.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps({'results':results,'gates':gates,
        'clock_gate_passed':all(r['passed'] for r in gates)},indent=2)+'\n')
    np.savez_compressed(OUT/'phase_features.npz',features=z,obs_indices=ids,frequencies=FREQ)
    (OUT/'provenance.json').write_text(json.dumps({'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cache_audit_sha256':hashlib.sha256((CACHE/'CACHE_AUDIT.json').read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps(gates,indent=2),flush=True)
if __name__=='__main__':main()
