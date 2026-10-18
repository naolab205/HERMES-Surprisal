#!/usr/bin/env python3
"""Audit ledger of all completed arms; do not compare across input protocols."""
import json
from pathlib import Path
import pandas as pd
from summarize_retrospective_baseline import item_frame,metrics,conditional_rho
from foundation.data import load_char_bags
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/single_model_breakthrough_20261006'
counts=load_char_bags(cache_dir=ROOT/'data_cache/char_epochs_fixed_grid_350ms_20261006').obs[
    ['subject','book','run','row_idx','n_slots']]
records=[]
paths=list(OUT.glob('*/tables/charbag_*_predictions.csv'))+list(OUT.glob('physical_screen/*/tables/charbag_*_predictions.csv'))
for path in sorted(paths):
    arm=path.parent.parent.name
    protocol=('spliced_variable_clock' if arm in ['norm_control','norm_valid','shared_reference']
              else 'continuous_fixed_grid' if arm.startswith('fixed_') else 'continuous_variable_clock')
    p=pd.read_csv(path)
    assert set(p.fold)=={0,3} and set(p.eval_split)=={'val'}
    assert not p.duplicated(['fold','subject','book','run','row_idx']).any()
    p=p.merge(counts,on=['subject','book','run','row_idx'],validate='many_to_one')
    for fold,g in p.groupby('fold'):
        trial=metrics(g);item=metrics(item_frame(g))
        r={'arm':arm,'fold':int(fold),'seed':2026,'input_protocol':protocol,
           'synthetic_noise':arm.startswith('noise_') or arm=='fixed_noise',
           'prediction_file':str(path.relative_to(ROOT)),
           'conditional_rho':conditional_rho(item_frame(g)),
           **{'trial_'+k:v for k,v in trial.items()},**{'item_'+k:v for k,v in item.items()}}
        records.append(r)
pd.DataFrame(records).to_csv(OUT/'complete_experiment_ledger.csv',index=False)
(OUT/'complete_experiment_ledger.json').write_text(json.dumps({
    'scope':'retrospective development only; 2 folds; 1 seed; no ensemble',
    'comparison_rule':'architecture comparisons only within the same input and normalization protocol',
    'metrics_version':'tie-preserving item mean; numerical constant strata excluded',
    'scores':records},indent=2)+'\n')
print(f'{len(paths)} completed arms; {len(records)} fold records')
