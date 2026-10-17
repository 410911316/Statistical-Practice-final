"""Reproduce frozen final models and figures; no new tuning or SQL writes."""
from pathlib import Path
import argparse,hashlib,json,platform
import joblib,numpy as np,pandas as pd,xgboost,sklearn
from threadpoolctl import threadpool_limits
from forecasting.features import fit_one
from forecasting.weighted_tree import fit_final,panel_check,KEYS
from forecasting.evaluation import metrics
from forecasting.reporting import render
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def run():
    cfg=json.loads((ROOT/'configs/forecast.json').read_text('utf8'))
    out=ROOT/'result/forecast';private=ROOT/'.local/forecast'
    (private/'models').mkdir(parents=True,exist_ok=True);(out/'tables').mkdir(parents=True,exist_ok=True)
    inp=ROOT/cfg['input']
    if not inp.exists():raise FileNotFoundError('請私下取得 .local/forecast/analysis_input.csv；原始資料不在 GitHub。')
    if digest(inp)!=cfg['input_sha256']:raise ValueError('Frozen input changed; this command reproduces the agreed model, not a new-data fit')
    f=pd.read_csv(inp,parse_dates=['event_date']);panel_check(f)
    if f.event_date.nunique()!=21:raise ValueError('Expected 21 events')
    reference=private/'reference_predictions.csv'
    provenance=json.loads((out/'provenance.json').read_text('utf8'))
    if not reference.exists() or digest(reference)!=provenance['reference_sha256']:
        raise ValueError('Need matching privately shared reference_predictions.csv')
    reference=pd.read_csv(reference,parse_dates=['event_date']).sort_values(KEYS)
    blocks=[];fits=[];models={}
    with threadpool_limits(limits=1):
        for year in cfg['years']:
            train=f.loc[f.year.lt(year)].copy()
            test=f.loc[f.year.eq(year)&f.station.eq('北門')].sort_values(KEYS)
            if train.empty or not train.event_date.max()<test.event_date.min():raise ValueError('Training chronology violated')
            block=test.copy()
            for name in ['ridge','gam','xgboost','xgboost_weighted']:
                modelname='xgboost' if name=='xgboost_weighted' else name
                params=cfg['settings'][str(year)][modelname]
                fitted=(fit_final(train,year,params,cfg['training_multiplier'],cfg,cfg['groups'])
                        if name=='xgboost_weighted' else fit_one(train,name,tuple(cfg['groups']),params,cfg))
                block[name]=fitted.predict(test).astype(float)
                joblib.dump(fitted,private/'models'/f'{year}_{name}.joblib')
                records=dict(year=year,model=name,params=params,train_rows=len(train),
                    train_events=int(train.event_date.nunique()),train_end=str(train.event_date.max().date()),
                    encoded_columns=fitted.encoder.names,
                    dropped_dependencies=fitted.encoder.dropped_dependencies,
                    model_sha256=digest(private/'models'/f'{year}_{name}.joblib'))
                if name=='gam':records.update(terms=fitted.terms_audit,certificate=fitted.certificate)
                if name=='ridge':records.update(alpha=float(fitted.estimator.alpha),intercept=float(fitted.estimator.intercept_))
                fits.append(records);models[year,name]=fitted
            blocks.append(block)
    pred=pd.concat(blocks).sort_values(KEYS).reset_index(drop=True)
    ref=reference.reset_index(drop=True)
    if not pred[KEYS].equals(ref[KEYS]):raise ValueError('Reference keys differ')
    np.testing.assert_array_equal(pred.y,ref.y)
    differences={m:float(np.max(abs(pred[m]-ref[m]))) for m in ['ridge','gam','xgboost','xgboost_weighted']}
    if max(differences.values())>1e-5:raise ValueError(f'Consolidation altered predictions: {differences}')
    pred.to_csv(private/'all_predictions.csv',index=False,encoding='utf-8-sig')
    published=pred[KEYS+['year','y','xgboost_weighted']].rename(columns={'xgboost_weighted':'frozen_w4'})
    published.to_csv(private/'predictions.csv',index=False,encoding='utf-8-sig')
    summary=[];annual=[]
    for name in differences:
        summary.append(dict(model=name,**metrics(pred,pred[name])))
        for year,g in pred.groupby('year'):annual.append(dict(year=int(year),model=name,**metrics(g,g[name])))
    pd.DataFrame(summary).to_csv(out/'tables/metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(annual).to_csv(out/'tables/annual_metrics.csv',index=False,encoding='utf-8-sig')
    # Every event has six hours, so row and event-equal MAE coincide here.
    train=f.loc[f.year.lt(2026)];ridge=models[2026,'ridge']
    coefficients=pd.DataFrame({'feature':['intercept',*ridge.encoder.names],
        'coefficient':[ridge.estimator.intercept_,*ridge.estimator.coef_]})
    coefficients['training_mean']=coefficients.feature.map(ridge.encoder.mean)
    coefficients['training_scale']=coefficients.feature.map(ridge.encoder.scale)
    coefficients.to_csv(out/'tables/ridge_coefficients_2026.csv',index=False,encoding='utf-8-sig')
    encoder=ridge.encoder
    write_json(out/'ridge_equation_2026.json',dict(intercept=float(ridge.estimator.intercept_),
        coefficients=dict(zip(encoder.names,ridge.estimator.coef_.tolist())),
        mean=encoder.mean.to_dict(),scale=encoder.scale.to_dict(),
        imputation=encoder.imputation_values,station_reference=encoder.categories['station'][0],
        hour_reference=int(encoder.categories['hour'][0]),lambda_=cfg['settings']['2026']['ridge']['lambda'],
        sklearn_alpha=float(ridge.estimator.alpha),prediction_clipped_at_zero=True,
        train_until=str(train.event_date.max().date()),not_refit_on_2026=True))
    write_json(out/'fits.json',fits)
    write_json(out/'validation.json',dict(passed=True,reference_max_error=differences,
        models=12,independent_events=21,evaluation_events=9,evaluation_rows=54,
        d1_deployable=False,independent_unseen_test=False,new_tuning=False,
        input_sha256=digest(inp),prediction_sha256=digest(private/'predictions.csv'),
        versions=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,xgboost=xgboost.__version__)))
    render(pred,f,models,cfg,out)
    print(pd.DataFrame(summary)[['model','mae','rmse','peak_abs']].round(2).to_string(index=False))
    print('Exact archived controls reproduced:',differences)
if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
