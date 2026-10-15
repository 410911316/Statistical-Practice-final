"""Matched three/four-station ablation with all seven V15 inputs preserved."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from analysis_v16.experiment import check_panel, fit_cohort, common_metrics
from analysis_v15.experiment import KEYS
from analysis_v11.evaluation import metrics, WINDOW
from analysis_v13.data import sha
from run_analysis_v8 import write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/analysis_v16';PRIVATE=ROOT/'.local/analysis_v16'


def run():
    cfg=json.loads((ROOT/'configs/analysis_v16.json').read_text('utf8'))
    old=json.loads((ROOT/'result/analysis_v15/manifest.json').read_text('utf8'))
    if cfg['cohorts']!={'four':['北門','大橋頭站','雙連','民權西路'],'three':['北門','大橋頭站','雙連']}:
        raise ValueError('Use the user-confirmed three stations')
    if cfg['models']!=old['config']['models'] or cfg['variants']!=['none','origin_w1','origin_w3','origin_w5','both_w1','both_w3','both_w5']:
        raise ValueError('Prespecified model/input set changed')
    if cfg['outer_years']!=[2024,2025,2026] or cfg['evaluation_stations']!=cfg['cohorts']['three'] or cfg['seed']!=old['config']['seed']:
        raise ValueError('Comparison protocol changed')
    for p,digest in old['source_hashes'].items():
        if sha(ROOT/p)!=digest:raise ValueError(f'Changed archived input: {p}')
    if sha(ROOT/'.local/analysis_v15/predictions.csv')!=old['prediction_sha256']:
        raise ValueError('Changed reference predictions')
    frame=pd.read_csv(ROOT/cfg['input'],parse_dates=['event_date'])
    check_panel(frame,cfg['cohorts']['four'])
    oldp=pd.read_csv(ROOT/'.local/analysis_v15/predictions.csv',parse_dates=['event_date'])
    oldfits=json.loads((ROOT/'result/analysis_v15/fits.json').read_text('utf8'))
    groups=old['groups'];fits=[];blocks=[];reproductions=[]
    (OUT/'tables').mkdir(parents=True,exist_ok=True);(PRIVATE/'models').mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        for year in cfg['outer_years']:
            valid=frame.loc[frame.year.eq(year)&frame.station.isin(cfg['evaluation_stations'])].sort_values(KEYS).reset_index(drop=True)
            check_panel(valid,cfg['evaluation_stations'])
            block=valid[KEYS+['year','y',*WINDOW]].copy()
            for cohort,stations in cfg['cohorts'].items():
                for model in cfg['models']:
                    params=next(r['params'] for r in oldfits if r['year']==year and r['model']==model and r['variant']=='none')
                    for variant in cfg['variants']:
                        method=f'{cohort}_{model}_{variant}'
                        fitted,train=fit_cohort(frame,year,stations,model,variant,groups,params,cfg['seed'])
                        block[method]=fitted.predict(valid)
                        if cohort=='four':
                            reference=oldp.loc[oldp.year.eq(year)].sort_values(KEYS)
                            got=block.loc[block.station.eq('北門')].sort_values(KEYS)
                            assert got[KEYS].reset_index(drop=True).equals(reference[KEYS].reset_index(drop=True))
                            error=float(np.max(abs(got[method].to_numpy()-reference[f'{model}_{variant}'].to_numpy())))
                            if error>1e-6:raise ValueError(f'Four-station reference differs: {method} {year} {error}')
                            reproductions.append(dict(year=year,model=model,variant=variant,max_error=error))
                        path=PRIVATE/'models'/f'{year}_{method}.joblib';joblib.dump(fitted,path)
                        fits.append(dict(year=year,cohort=cohort,model=model,variant=variant,method=method,
                            stations=stations,params=params,train_rows=len(train),train_events=train.event_date.nunique(),
                            train_end=fitted.train_end,encoded_columns=len(fitted.encoder.names),
                            feature_names=fitted.encoder.names,
                            sklearn_alpha=float(fitted.estimator.alpha) if model=='ridge' else None,
                            path=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
            blocks.append(block);print(f'Completed {year}',flush=True)
    pred=pd.concat(blocks,ignore_index=True)
    pred.to_csv(PRIVATE/'predictions.csv',index=False,encoding='utf-8-sig')
    north=pred.loc[pred.station.eq('北門')].copy()
    outputs={name:[] for name in ['summary','annual','events','stations','common_three','conservative']}
    for cohort in cfg['cohorts']:
        for model in cfg['models']:
            for variant in cfg['variants']:
                method=f'{cohort}_{model}_{variant}';meta=dict(method=method,cohort=cohort,model=model,variant=variant)
                outputs['summary'].append(dict(**meta,**metrics(north,north[method])))
                for year,g in north.groupby('year'):
                    outputs['annual'].append(dict(**meta,year=year,**metrics(g,g[method])))
                for date,g in north.groupby('event_date'):
                    outputs['events'].append(dict(**meta,event_date=str(date.date()),**metrics(g,g[method])))
                for station,g in pred.groupby('station'):
                    outputs['stations'].append(dict(**meta,station=station,**common_metrics(g,g[method])))
                outputs['common_three'].append(dict(**meta,**common_metrics(pred,pred[method])))
                g=north.loc[north.hour.ge(19)]
                outputs['conservative'].append(dict(**meta,**common_metrics(g,g[method])))
    for name,records in outputs.items():
        pd.DataFrame(records).to_csv(OUT/'tables'/f'{name}.csv',index=False,encoding='utf-8-sig')
    paired=pd.DataFrame(outputs['summary']).pivot(index=['model','variant'],columns='cohort',values=['mae','rmse','peak_abs','cost_2'])
    changes=[]
    for (model,variant),row in paired.iterrows():
        r=dict(model=model,variant=variant)
        for metric in ['mae','rmse','peak_abs','cost_2']:
            r[f'{metric}_four']=float(row[metric,'four']);r[f'{metric}_three']=float(row[metric,'three'])
            r[f'{metric}_improvement_pct']=100*(1-row[metric,'three']/row[metric,'four'])
        changes.append(r)
    pd.DataFrame(changes).to_csv(OUT/'tables/paired_comparison.csv',index=False,encoding='utf-8-sig')
    write_json(OUT/'fits.json',fits)
    own=[ROOT/'configs/analysis_v16.json',Path(__file__),*sorted((ROOT/'src/analysis_v16').glob('*.py')),
         ROOT/'result/analysis_v15/manifest.json',ROOT/'result/analysis_v15/fits.json',ROOT/'.local/analysis_v15/predictions.csv']
    signature={**old['source_hashes'],**{p.relative_to(ROOT).as_posix():sha(p) for p in own}}
    write_json(OUT/'manifest.json',dict(config=cfg,groups=groups,source_hashes=signature,versions=old['versions'],
        prediction_sha256=sha(PRIVATE/'predictions.csv'),four_station_reproductions=reproductions,
        full_rows=len(frame),three_station_rows=int(frame.station.isin(cfg['cohorts']['three']).sum()),
        test_events=9,north_test_rows=len(north),common_three_test_rows=len(pred),
        deployed=False,independent_unseen_test=False,parameter_search=False,
        limitations=old['limitations']+['Cohort removal also reduces data and learned station effects; it is not a causal test of noise']))
    print(pd.DataFrame(changes)[['model','variant','mae_four','mae_three','mae_improvement_pct','peak_abs_four','peak_abs_three']].to_string(index=False))


if __name__=='__main__':run()
