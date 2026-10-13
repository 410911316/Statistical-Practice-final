"""Source-version-screened baseline/increment comparison and asymmetric stacking."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from analysis_v13.data import load,sha,eligible_training,KEYS
from analysis_v13.models import fit_model
from analysis_v13.fusion import solve,MODELS
from analysis_v11.evaluation import metrics,select_candidate,WINDOW
from run_analysis_v8 import write_json

ROOT=Path(__file__).resolve().parents[1]


def run(reuse=False):
    output=ROOT/'result/analysis_v13';private=ROOT/'.local/analysis_v13'
    (output/'tables').mkdir(parents=True,exist_ok=True);(private/'models').mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'configs/analysis_v13.json').read_text('utf-8'))
    base=json.loads((ROOT/'configs/analysis_v8_core.json').read_text('utf-8'))
    grid=json.loads((ROOT/cfg['grid_source']).read_text('utf-8'))['grids']
    data,availability=load(ROOT)
    data.to_csv(private/'analysis_input.csv',index=False,encoding='utf-8-sig')
    availability.to_csv(output/'tables/availability.csv',index=False,encoding='utf-8-sig')
    source_paths=[Path(__file__),ROOT/'scripts/collect_d1_baseline.py',*sorted((ROOT/'src/analysis_v13').glob('*.py')),
        ROOT/'src/analysis_v11/evaluation.py',ROOT/'src/analysis_v7/models.py',ROOT/'src/analysis_v5/models.py',
        ROOT/'src/analysis_v1/models.py',ROOT/'src/analysis_v6/models.py',ROOT/'src/analysis_v8/features.py',
        ROOT/'configs/analysis_v13.json',ROOT/'configs/analysis_v11.json',ROOT/'configs/analysis_v8_core.json',
        ROOT/'docs/d1_od_version_sources.json',ROOT/'docs/d1_show_end_sources.json',ROOT/'docs/d1_program_sources.json',
        private/'baseline_day_plan.json',private/'analysis_input.csv',ROOT/'.local/analysis_v12/predictions.csv']
    signature={p.relative_to(ROOT).as_posix():sha(p) for p in source_paths}
    choices=[];fits=[];weight_records=[];predictions=[];oof=[]
    histories={m:[] for m in MODELS};history_keys=[]
    with threadpool_limits(limits=1):
        for year in sorted(data.year.unique()):
            train=eligible_training(data,year)
            if train.event_date.nunique()<cfg['min_train_events']:continue
            valid=data.loc[data.year.eq(year)&data.station.eq('北門')].reset_index(drop=True)
            past=pd.concat(history_keys,ignore_index=True) if history_keys else pd.DataFrame()
            now=valid[KEYS+['year','y','available_baseline','label_version_at',*WINDOW]].copy()
            now['base_train_year']=int(train.year.max())
            for model in MODELS:
                path=private/f'core_candidates_{model}_{year}.npz';stamp=path.with_suffix('.json')
                if reuse and path.exists() and stamp.exists():
                    saved=json.loads(stamp.read_text('utf-8'))
                    if saved['signature']!=signature or sha(path)!=saved['sha256']:raise ValueError('Candidate cache changed')
                    matrix=np.load(path)['predictions']
                else:
                    matrix=np.empty((len(valid),len(grid[model])))
                    for j,params in enumerate(grid[model]):
                        fitted=fit_model(train,model,'core',base['candidate_groups'],params,base)
                        matrix[:,j]=fitted.predict(valid)
                    np.savez_compressed(path,predictions=matrix)
                    write_json(stamp,dict(signature=signature,sha256=sha(path)))
                if len(past) and past.event_date.nunique()>=cfg['min_tuning_events']:
                    known=pd.to_datetime(past.label_version_at,utc=True)<=pd.Timestamp(f'{year}-01-01',tz='Asia/Taipei')
                    best,loss,losses=select_candidate(past.loc[known].reset_index(drop=True),np.vstack(histories[model])[known],year,2,len(grid[model]))
                    params=grid[model][best];selection=dict(candidate_id=best,past_cost=loss,all_costs=losses.tolist(),
                        past_events=int(past.loc[known].event_date.nunique()),parameter_source='strictly_prior_core_OOF')
                else:
                    params=cfg['defaults'][model];selection=dict(candidate_id=None,past_cost=None,all_costs=[],past_events=0,parameter_source='prespecified_default_no_tuning_labels')
                choices.append(dict(year=int(year),model=model,params=params,**selection))
                for mode in ('core','direct','increment'):
                    groups=base['candidate_groups']+(['available_baseline'] if mode!='core' else [])
                    fitted=fit_model(train,model,mode,groups,params,base)
                    raw,pred=fitted.predict_components(valid)
                    now[f'{mode}_{model}']=pred
                    if mode=='increment':now[model]=pred;now[f'raw_delta_{model}']=raw
                    modelpath=private/'models'/f'{year}_{mode}_{model}.joblib';joblib.dump(fitted,modelpath)
                    fits.append(dict(year=int(year),model=model,mode=mode,params=params,
                        train_events=int(train.event_date.nunique()),train_start=str(train.event_date.min().date()),
                        train_end=str(train.event_date.max().date()),fit_at=f'{year}-01-01T00:00:00+08:00',
                        max_label_version=str(pd.to_datetime(train.label_version_at,utc=True).max()),
                        certificate=fitted.certificate,model_sha256=sha(modelpath)))
                histories[model].append(matrix)
                print(f'{year} {model}: frozen core parameters applied to direct/increment',flush=True)
            now['baseline_only']=valid.available_baseline.to_numpy()
            now['equal']=now[list(MODELS)].mean(axis=1)
            meta=pd.concat(oof,ignore_index=True) if oof else pd.DataFrame()
            for method,ratio,window in [('fusion_mae',1,False),('fusion_window1',1,True),('fusion_window2',2,True)]:
                if len(meta) and meta.event_date.nunique()>=cfg['min_fusion_events']:
                    w,audit=solve(meta,int(year),ratio,window)
                else:w=np.array([0.,0.,1.]);audit={'reason':'insufficient_prior_OOF_use_XGBoost','past_events':0}
                now[method]=now[list(MODELS)].to_numpy()@w
                weight_records.append(dict(year=int(year),method=method,weights=dict(zip(MODELS,w.tolist())),**audit))
            oof.append(now.copy());history_keys.append(valid[KEYS+['year','y','label_version_at',*WINDOW]].copy())
            if year in cfg['outer_years']:predictions.append(now.copy())
    outer=pd.concat(predictions,ignore_index=True)
    old=pd.read_csv(ROOT/'.local/analysis_v12/predictions.csv',parse_dates=['event_date'])
    old=old.loc[old.variant.eq('sensitivity_baseline')&old.model.eq('xgboost')&old.control.eq('fixed_core'),KEYS+['y','prediction']]
    outer=outer.merge(old.rename(columns={'y':'archive_y','prediction':'archived_474'}),on=KEYS,validate='one_to_one')
    if len(outer)!=54 or not outer.y.eq(outer.archive_y).all():raise ValueError('Archived comparison rows differ')
    methods=['archived_474','baseline_only']+[f'{mode}_{model}' for mode in ('core','direct','increment') for model in MODELS]+['equal','fusion_mae','fusion_window1','fusion_window2']
    outer.to_csv(private/'predictions.csv',index=False,encoding='utf-8-sig')
    pd.concat(oof,ignore_index=True).to_csv(private/'temporal_oof.csv',index=False,encoding='utf-8-sig')
    summary=[];annual=[];events=[]
    for method in methods:
        common={'method':method,'status':'archived_not_D1_comparable' if method=='archived_474' else 'official_version_screened'}
        summary.append({**common,**metrics(outer.reset_index(drop=True),outer[method])})
        for year,f in outer.groupby('year'):annual.append({**common,'year':int(year),**metrics(f.reset_index(drop=True),f[method])})
        for date,f in outer.groupby('event_date'):events.append({**common,'event_date':str(date.date()),**metrics(f.reset_index(drop=True),f[method])})
    for name,rows in [('summary',summary),('annual',annual),('events',events)]:pd.DataFrame(rows).to_csv(output/'tables'/f'{name}.csv',index=False,encoding='utf-8-sig')
    write_json(output/'selections.json',choices);write_json(output/'fits.json',fits);write_json(output/'weights.json',weight_records)
    write_json(output/'manifest.json',dict(config=cfg,source_hashes=signature,events=int(data.event_date.nunique()),
        excluded_dates=availability.loc[~availability.eligible,'event_date'].tolist(),outer_events=9,
        prediction_sha256=sha(private/'predictions.csv'),oof_sha256=sha(private/'temporal_oof.csv'),
        feature_version_gate_passed=True,training_label_version_gate_passed=True,historical_public_acl_verified=False,
        independent_unseen_test=False,deployed=False))
    print(pd.DataFrame(summary)[['method','mae','rmse','cost_2']].round(2).to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reuse',action='store_true');run(parser.parse_args().reuse)
