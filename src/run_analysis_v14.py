"""Controlled XGBoost experiments on the unchanged V12 original-input snapshot."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import platform
import joblib
import numpy as np
import pandas as pd
import xgboost
from threadpoolctl import threadpool_limits
from analysis_v13.data import sha
from analysis_v14.experiment import candidates,fit_final,select_past,shapes,panel_check,KEYS,WINDOW
from analysis_v11.evaluation import metrics
from run_analysis_v8 import write_json

ROOT=Path(__file__).resolve().parents[1]


def run(reuse=False):
    cfg=json.loads((ROOT/'configs/analysis_v14.json').read_text('utf-8'))
    out=ROOT/'result/analysis_v14';private=ROOT/'.local/analysis_v14'
    (out/'tables').mkdir(parents=True,exist_ok=True);(private/'models').mkdir(parents=True,exist_ok=True)
    oldmanifest=json.loads((ROOT/'result/analysis_v12/manifest.json').read_text('utf-8'))
    for path,value in oldmanifest['source_hashes'].items():
        if sha(ROOT/path)!=value:raise ValueError(f'Archived source changed: {path}')
    archived_path=ROOT/'.local/analysis_v12/predictions.csv'
    if sha(archived_path)!=oldmanifest['predictions_sha256']:raise ValueError('Archived predictions changed')
    source=pd.read_csv(ROOT/cfg['input'],parse_dates=['event_date']);panel_check(source)
    base=json.loads((ROOT/'configs/analysis_v8_core.json').read_text('utf-8'))
    canonical=json.loads((ROOT/'configs/analysis_v8_both.json').read_text('utf-8'))['candidate_groups']
    groups=[g for g in canonical if g in base['candidate_groups']]+['retrospective_baseline_56d']
    # Restrict inputs; all other V12 columns (weather, Google, attendance) are discarded.
    source=source[list(dict.fromkeys(KEYS+['year','y']+groups))].copy()
    source.to_csv(private/'analysis_input.csv',index=False,encoding='utf-8-sig')
    signature={**oldmanifest['source_hashes'],**{p.relative_to(ROOT).as_posix():sha(p) for p in
        [Path(__file__),*sorted((ROOT/'src/analysis_v14').glob('*.py')),ROOT/'configs/analysis_v14.json',
         ROOT/'src/analysis_v11/evaluation.py',ROOT/'result/analysis_v12/selections.json',
         ROOT/'result/analysis_v12/manifest.json',archived_path,private/'analysis_input.csv']}}
    versions=dict(python=platform.python_version(),xgboost=xgboost.__version__,numpy=np.__version__,pandas=pd.__version__)
    identity=dict(signature=signature,versions=versions)
    grid=shapes(cfg);fixed_grid=[dict(**p,n_estimators=n) for p in grid for n in cfg['fixed_trees']]
    oldchoices=json.loads((ROOT/'result/analysis_v12/selections.json').read_text('utf-8'))
    oldgrid=json.loads((ROOT/'configs/analysis_v11.json').read_text('utf-8'))['grids']['xgboost']
    archive=pd.read_csv(archived_path,parse_dates=['event_date'])
    archive=archive.loc[archive.variant.eq('sensitivity_baseline')&archive.model.eq('xgboost')]
    oldfixed={r['year']:oldgrid[r['fixed_candidate_id']] for r in oldchoices if r['variant']=='sensitivity_baseline' and r['model']=='xgboost'}
    histories={(mode,m):[] for mode in ['fixed','es'] for m in cfg['training_multipliers']}
    timeline=[];predictions=[];folds=[];selections=[];models=[];check474=[]
    def cached(year,m):
        path=private/f'candidates_{year}_w{m}.npz';meta=path.with_suffix('.json')
        if reuse and path.exists() and meta.exists():
            audit=json.loads(meta.read_text('utf-8'))
            if audit['identity']!=identity or audit['sha256']!=sha(path):raise ValueError('Changed V14 cache')
            with np.load(path) as z:matrix={k:z[k] for k in ['fixed','es']}
        else:
            matrix,audit=candidates(source,year,m,cfg,groups)
            np.savez_compressed(path,**matrix);audit.update(identity=identity,sha256=sha(path));write_json(meta,audit)
        return matrix,audit
    def save_fit(year,method,params,m,expected):
        train=source.loc[source.year.lt(year)].copy()
        valid=source.loc[source.year.eq(year)&source.station.eq('北門')].sort_values(KEYS)
        fitted=fit_final(train,year,params,m,cfg,groups);p=fitted.predict(valid)
        if expected is not None and not np.allclose(p,expected,rtol=0,atol=1e-5):
            raise ValueError(f'Refit differs from cached prefix or archived control: {year} {method} max={max(abs(p-expected))}')
        path=private/'models'/f'{year}_{method}.joblib';joblib.dump(fitted,path)
        models.append(dict(year=year,method=method,params=params,multiplier=m,sha256=sha(path),
                           train_events=int(train.event_date.nunique()),train_end=str(train.event_date.max().date())))
        # Preserve every float32 prediction exactly when serializing to CSV;
        # pandas otherwise prints a shortened decimal for float32 columns.
        return p.astype(np.float64)
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=3) as pool:
        for year in sorted(int(y) for y in source.year.unique()):
            if source.loc[source.year.lt(year),'event_date'].nunique()<cfg['min_base_events']:continue
            now=source.loc[source.year.eq(year)&source.station.eq('北門'),KEYS+['year','y',*WINDOW]].sort_values(KEYS).reset_index(drop=True)
            jobs={m:pool.submit(cached,year,m) for m in cfg['training_multipliers']}
            blocks={}
            for m,job in jobs.items():
                matrix,audit=job.result();blocks[m]=matrix
                folds.append({k:v for k,v in audit.items() if k not in ['identity','fixed_grid']})
            if year in cfg['outer_years']:
                past=pd.concat(timeline,ignore_index=True)
                for mode in ['fixed','es']:
                    for m in [*cfg['training_multipliers'],'selected']:
                        choice_weights=cfg['training_multipliers'] if m=='selected' else [m]
                        matrix=np.column_stack([np.vstack(histories[mode,w]) for w in choice_weights])
                        j,losses=select_past(past,matrix,year)
                        width=len(fixed_grid) if mode=='fixed' else len(grid)
                        picked_w=choice_weights[j//width];picked_j=j%width
                        audit=next(f for f in folds if f['year']==year and f['multiplier']==picked_w)
                        params=(fixed_grid[picked_j] if mode=='fixed' else
                                dict(**grid[picked_j],n_estimators=audit['stopping'][picked_j]['trees']))
                        method=f'local_{mode}_w{m}'
                        now[method]=save_fit(year,method,params,picked_w,blocks[picked_w][mode][:,picked_j])
                        selections.append(dict(year=year,method=method,multiplier=picked_w,candidate_id=picked_j,
                            params=params,past_events=int(past.event_date.nunique()),past_start=str(past.event_date.min().date()),
                            past_end=str(past.event_date.max().date()),past_mae=float(losses[j]),losses=losses.tolist(),
                            stopping=None if mode=='fixed' else audit['stopping'][picked_j]))
                for m in cfg['training_multipliers']:
                    method='archived_474' if m==1 else f'frozen_w{m}'
                    reference=archive.loc[archive.year.eq(year)&archive.control.eq('fixed_core')].sort_values(KEYS).prediction.to_numpy()
                    p=save_fit(year,method,oldfixed[year],m,reference if m==1 else None)
                    now[method]=p
                    if m==1:check474.extend(abs(p-now.y.to_numpy()))
                now['archived_retuned2']=archive.loc[archive.year.eq(year)&archive.control.eq('retuned')].sort_values(KEYS).prediction.to_numpy()
                # Re-score the original 180 cached candidates by past MAE only.
                old_past=[];old_now=None
                for yr in sorted(set(past.year)|{year}):
                    path=ROOT/f'.local/analysis_v12/sensitivity_baseline/xgboost/{yr}.npz'
                    stamp=json.loads(path.with_suffix('.json').read_text('utf-8'))
                    if sha(path)!=stamp['sha256']:raise ValueError('Old candidate cache changed')
                    a=np.load(path)['prediction']
                    if yr==year:old_now=a
                    else:old_past.append(a)
                j,loss=select_past(past,np.vstack(old_past),year)
                now['archived_retuned_mae']=old_now[:,j]
                selections.append(dict(year=year,method='archived_retuned_mae',candidate_id=j,params=oldgrid[j],
                    past_events=int(past.event_date.nunique()),past_mae=float(loss[j]),losses=loss.tolist(),stopping=None))
                predictions.append(now)
            timeline.append(now[KEYS+['year','y',*WINDOW]].copy())
            for m in cfg['training_multipliers']:
                for mode in ['fixed','es']:histories[mode,m].append(blocks[m][mode])
            print(f'{year}: completed 3 weights x (96 fixed-tree + 24 nested-stop candidates)',flush=True)
    if abs(np.mean(check474)-473.90951651114005)>1e-6:raise ValueError('Original 474 not reproduced')
    prediction=pd.concat(predictions,ignore_index=True)
    if len(prediction)!=54:raise ValueError('Outer comparison is not the same 54 rows')
    prediction.to_csv(private/'predictions.csv',index=False,encoding='utf-8-sig')
    pd.concat(timeline,ignore_index=True).to_csv(private/'timeline.csv',index=False,encoding='utf-8-sig')
    methods=[c for c in prediction if c.startswith(('local_','frozen_','archived_'))]
    summary=[];annual=[];events=[]
    for method in methods:
        summary.append(dict(method=method,**metrics(prediction.reset_index(drop=True),prediction[method])))
        for year,g in prediction.groupby('year'):annual.append(dict(method=method,year=int(year),**metrics(g.reset_index(drop=True),g[method])))
        for date,g in prediction.groupby('event_date'):events.append(dict(method=method,event_date=str(date.date()),**metrics(g.reset_index(drop=True),g[method])))
    for name,rows in [('summary',summary),('annual',annual),('events',events)]:
        pd.DataFrame(rows).to_csv(out/'tables'/f'{name}.csv',index=False,encoding='utf-8-sig')
    write_json(out/'selections.json',selections);write_json(out/'folds.json',folds);write_json(out/'models.json',models)
    write_json(out/'manifest.json',dict(config=cfg,source_hashes=signature,versions=versions,
        input_events=21,outer_events=9,outer_rows=54,groups=groups,prediction_sha256=sha(private/'predictions.csv'),
        timeline_sha256=sha(private/'timeline.csv'),original_474_reproduced=True,d1_deployable=False,independent_unseen_test=False,deployed=False,
        limitations=['Original OD/calendar vintages not verified; original-data sensitivity only',
            'Nine outer events repeatedly inspected; local grid informed by earlier results',
            'Only XGBoost changed; training weights differ from fusion weights and asymmetric underestimation loss']))
    print(pd.DataFrame(summary)[['method','mae','rmse','cost_2','peak_abs']].round(2).to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reuse',action='store_true');run(parser.parse_args().reuse)
