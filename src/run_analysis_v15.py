"""Compare seven prespecified prior-flow inputs under identical archived parameters."""
from pathlib import Path
import json
import duckdb
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from threadpoolctl import threadpool_limits
from analysis_v15.experiment import KEYS, VARIANTS, build_features, fit
from analysis_v14.experiment import panel_check
from analysis_v11.evaluation import metrics, WINDOW
from analysis_v13.data import sha
from run_analysis_v8 import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'result/analysis_v15'
PRIVATE = ROOT/'.local/analysis_v15'


def run():
    cfg = json.loads((ROOT/'configs/analysis_v15.json').read_text('utf8'))
    if (cfg['snapshot_hour']!=18 or cfg['windows']!=[1,3,5] or cfg['source_bins']!=list(range(13,18))
            or cfg['target_bins']!=list(range(18,24)) or cfg['conservative_target_bins']!=list(range(19,24))
            or cfg['outer_years']!=[2024,2025,2026] or cfg['models']!=['ridge','xgboost']):
        raise ValueError('This runner implements only the prespecified V15 protocol')
    old = json.loads((ROOT/'result/analysis_v14/manifest.json').read_text('utf8'))
    for path, digest in old['source_hashes'].items():
        if sha(ROOT/path) != digest:
            raise ValueError(f'Archived source changed: {path}')
    groups = old['groups']
    frame = pd.read_csv(ROOT/cfg['input'], parse_dates=['event_date']); panel_check(frame)
    with duckdb.connect() as con:
        hourly = con.execute('SELECT * FROM read_parquet(?)', [str(ROOT/cfg['hourly'])]).df()
    refs = pd.read_csv(ROOT/cfg['baseline_days'])
    features, audit = build_features(frame, hourly, refs, cfg['snapshot_hour'])
    frame = frame.merge(features, on=['event_date','station'], validate='many_to_one')
    # Validate that target labels mean the existing origin count on the same source bin.
    truth = hourly.rename(columns={'service_date':'event_date','source_hour':'hour'})
    truth['event_date'] = pd.to_datetime(truth.event_date)
    matched = frame.merge(truth[KEYS+['origin_count','origin_complete']], on=KEYS, validate='one_to_one')
    if len(matched)!=len(frame) or not matched.origin_complete.eq(True).all() or not np.allclose(matched.y,matched.origin_count):
        raise ValueError('Targets differ from source origin counts')
    (OUT/'tables').mkdir(parents=True, exist_ok=True); (PRIVATE/'models').mkdir(parents=True, exist_ok=True)
    frame.to_csv(PRIVATE/'analysis_input.csv', index=False, encoding='utf-8-sig')
    features.to_csv(PRIVATE/'snapshot_features.csv', index=False, encoding='utf-8-sig')
    write_json(PRIVATE/'feature_audit.json',audit)
    choices = json.loads((ROOT/'result/analysis_v12/selections.json').read_text('utf8'))
    grids = json.loads((ROOT/'configs/analysis_v11.json').read_text('utf8'))['grids']
    archive = pd.read_csv(ROOT/'.local/analysis_v12/predictions.csv',parse_dates=['event_date'])
    archive = archive.loc[archive.variant.eq('sensitivity_baseline') & archive.control.eq('fixed_core')]
    fits=[]; blocks=[]; check=[]
    with threadpool_limits(limits=1):
        for year in cfg['outer_years']:
            train = frame.loc[frame.year.lt(year)]
            valid = frame.loc[frame.year.eq(year)&frame.station.eq('北門')].sort_values(KEYS).reset_index(drop=True)
            block = valid[KEYS+['year','y',*WINDOW]].copy()
            for model in cfg['models']:
                choice = next(r for r in choices if r['variant']=='sensitivity_baseline' and r['model']==model and r['year']==year)
                params = grids[model][choice['fixed_candidate_id']]
                for variant in VARIANTS:
                    method = f'{model}_{variant}'
                    fitted = fit(train,year,model,variant,groups,params,cfg['seed'])
                    block[method] = fitted.predict(valid)
                    if variant=='none':
                        previous = archive.loc[archive.model.eq(model)&archive.year.eq(year)].sort_values(KEYS)
                        if not previous[KEYS].reset_index(drop=True).equals(valid[KEYS]):
                            raise ValueError('Archived keys differ')
                        error = float(np.max(abs(block[method].to_numpy()-previous.prediction.to_numpy())))
                        if error > 1e-5:
                            raise ValueError(f'Archived {model} not reproduced: {error}')
                        check.append(dict(year=year,model=model,max_abs_difference=error))
                    path=PRIVATE/'models'/f'{year}_{method}.joblib'; joblib.dump(fitted,path)
                    fits.append(dict(year=year,model=model,variant=variant,method=method,params=params,
                                     train_end=fitted.train_end,train_rows=fitted.train_rows,train_events=train.event_date.nunique(),
                                     features=fitted.encoder.names,path=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
            blocks.append(block)
            print(f'Completed year {year}: {train.event_date.nunique()} earlier events',flush=True)
    pred=pd.concat(blocks,ignore_index=True); pred.to_csv(PRIVATE/'predictions.csv',index=False,encoding='utf-8-sig')
    summary=[]; annual=[]; event=[]; conservative=[]
    for model in cfg['models']:
        for variant in VARIANTS:
            method=f'{model}_{variant}'; meta=dict(method=method,model=model,variant=variant)
            summary.append(dict(**meta,**metrics(pred,pred[method])))
            for year,g in pred.groupby('year'):
                annual.append(dict(**meta,year=year,**metrics(g,g[method])))
            for date,g in pred.groupby('event_date'):
                event.append(dict(**meta,event_date=str(date.date()),**metrics(g,g[method])))
            g=pred.loc[pred.hour.isin(cfg['conservative_target_bins'])]
            e=g[method].to_numpy()-g.y.to_numpy()
            conservative.append(dict(**meta,mae=float(np.mean(abs(e))),rmse=float(np.sqrt(np.mean(e*e))),
                                     bias=float(np.mean(e)),events=g.event_date.nunique(),rows=len(g)))
    for name,rows in [('summary',summary),('annual',annual),('event',event),('conservative',conservative)]:
        pd.DataFrame(rows).to_csv(OUT/'tables'/f'{name}.csv',index=False,encoding='utf-8-sig')
    write_json(OUT/'fits.json',fits)
    own=[ROOT/'configs/analysis_v15.json',Path(__file__),*sorted((ROOT/'src/analysis_v15').glob('*.py')),
         ROOT/cfg['hourly'],ROOT/cfg['baseline_days'],ROOT/'result/analysis_v14/manifest.json',
         ROOT/'result/analysis_v12/selections.json',PRIVATE/'analysis_input.csv',PRIVATE/'snapshot_features.csv',
         PRIVATE/'feature_audit.json',ROOT/'configs/analysis_v11.json',ROOT/'.local/analysis_v12/predictions.csv']
    signature={**old['source_hashes'],**{p.relative_to(ROOT).as_posix():sha(p) for p in own}}
    write_json(OUT/'manifest.json',dict(config=cfg,groups=groups,source_hashes=signature,
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                      sklearn=sklearn.__version__,xgboost=xgboost.__version__,duckdb=duckdb.__version__,joblib=joblib.__version__),
        prediction_sha256=sha(PRIVATE/'predictions.csv'),original_controls=check,
        input_events=frame.event_date.nunique(),input_rows=len(frame),test_events=pred.event_date.nunique(),test_rows=len(pred),
        deployed=False,real_time_availability_verified=False,source_hour_clock_semantics_verified=False,
        d1_model=False,independent_unseen_test=False,
        limitations=['Monthly data do not prove same-day availability',
                     'Source-bin clock basis and interval endpoints remain unverified',
                     'Original V12 history publication times remain partly unverified',
                     'Nine development events have been inspected repeatedly; all windows reported without choosing on test']))
    print(pd.DataFrame(summary)[['method','mae','rmse','peak_abs']].to_string(index=False))


if __name__=='__main__':
    run()
