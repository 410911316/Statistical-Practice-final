"""Independently reconstruct prior-bin features, predictions and all reported scores."""
from pathlib import Path
import json
import duckdb
import joblib
import numpy as np
import pandas as pd
from analysis_v13.data import sha
from verify_analysis_v13 import scores
from run_analysis_v8 import write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/analysis_v15';PRIVATE=ROOT/'.local/analysis_v15'


def run():
    m=json.loads((OUT/'manifest.json').read_text('utf8'));cfg=m['config']
    for name,digest in m['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Changed input/code: {name}')
    assert sha(PRIVATE/'predictions.csv')==m['prediction_sha256']
    f=pd.read_csv(PRIVATE/'analysis_input.csv',parse_dates=['event_date'])
    p=pd.read_csv(PRIVATE/'predictions.csv',parse_dates=['event_date'])
    keys=['event_date','station','hour']
    assert not f.duplicated(keys).any() and not p.duplicated(keys).any()
    assert len(f)==504 and f.event_date.nunique()==21 and len(p)==54 and p.event_date.nunique()==9
    with duckdb.connect() as con:
        h=con.execute('SELECT * FROM read_parquet(?)',[str(ROOT/cfg['hourly'])]).df()
    h['service_date']=pd.to_datetime(h.service_date)
    refs=pd.read_csv(ROOT/cfg['baseline_days'],parse_dates=['event_date','candidate_date'])
    refs=refs.loc[refs.selected.eq(True)]
    max_feature_error=0.;features_checked=0
    # Different implementation: pivot complete hourly series, compute daily totals.
    for (date,station),g in f.groupby(['event_date','station']):
        selected=refs.loc[refs.event_date.eq(date)&refs.station.eq(station),'candidate_date'].tolist()
        assert selected and all(d<date for d in selected)
        for w in [1,3,5]:
            for direction,value,complete in [('origin','origin_count','origin_complete'),
                    ('destination','destination_count_same_source_bin','destination_complete')]:
                rows=h.loc[h.station.eq(station)&h.service_date.isin([date,*selected])&
                           h.source_hour.between(18-w,17)]
                assert rows[complete].eq(True).all() and len(rows)==(len(selected)+1)*w
                mat=rows.pivot(index='service_date',columns='source_hour',values=value).sort_index(axis=1)
                mat=mat.astype(float)
                assert np.isfinite(mat.to_numpy()).all()
                observed=mat.loc[date];base=mat.loc[selected].sum(axis=1).mean()
                expected={'last':observed.iloc[-1],'excess':observed.sum()-base}
                if w>1:expected.update(sum=observed.sum(),slope=(observed.iloc[-1]-observed.iloc[0])/(w-1))
                for stat,v in expected.items():
                    col=f'lag_{direction}_{stat}_w{w}'
                    error=float(np.max(abs(g[col].to_numpy()-v)))
                    max_feature_error=max(max_feature_error,error);features_checked+=1
                    assert error<1e-8 and g[col].nunique()==1
    max_prediction_error=0.
    fits=json.loads((OUT/'fits.json').read_text('utf8'))
    choices=json.loads((ROOT/'result/analysis_v12/selections.json').read_text('utf8'))
    grids=json.loads((ROOT/'configs/analysis_v11.json').read_text('utf8'))['grids']
    for fit in fits:
        choice=next(r for r in choices if r['variant']=='sensitivity_baseline' and r['year']==fit['year'] and r['model']==fit['model'])
        assert fit['params']==grids[fit['model']][choice['fixed_candidate_id']]
        assert sha(ROOT/fit['path'])==fit['sha256']
        model=joblib.load(ROOT/fit['path'])
        train=f.loc[f.year.lt(fit['year'])]
        test=f.loc[f.year.eq(fit['year'])&f.station.eq('北門')].sort_values(keys)
        reported=p.loc[p.year.eq(fit['year'])].sort_values(keys)
        assert test[keys].reset_index(drop=True).equals(reported[keys].reset_index(drop=True))
        assert model.train_end==str(train.event_date.max().date())==fit['train_end']
        assert model.train_rows==len(train)==fit['train_rows']
        assert pd.Timestamp(model.train_end).year<fit['year']
        means=train[model.encoder.lag_columns].to_numpy(float).mean(axis=0)
        np.testing.assert_allclose(model.encoder.means,means)
        error=float(np.max(abs(model.predict(test)-reported[fit['method']].to_numpy())))
        max_prediction_error=max(max_prediction_error,error);assert error<1e-7
    max_metric_error=0.;metric_rows=0
    for name in ['summary','annual','event','conservative']:
        table=pd.read_csv(OUT/'tables'/f'{name}.csv')
        for row in table.to_dict('records'):
            g=p
            if name=='annual':g=g.loc[g.year.eq(row['year'])]
            if name=='event':g=g.loc[g.event_date.eq(pd.Timestamp(row['event_date']))]
            if name=='conservative':
                g=g.loc[g.hour.ge(19)]
                error=g[row['method']].to_numpy()-g.y.to_numpy()
                values=dict(mae=np.mean(abs(error)),rmse=np.sqrt(np.mean(error**2)),bias=np.mean(error))
            else:values=scores(g,g[row['method']].to_numpy())
            for metric,value in values.items():
                error=abs(row[metric]-value);max_metric_error=max(max_metric_error,error)
                assert error<1e-7,(name,metric)
            metric_rows+=1
    archive=pd.read_csv(ROOT/'.local/analysis_v12/predictions.csv',parse_dates=['event_date'])
    for model in ['ridge','xgboost']:
        a=archive.loc[archive.variant.eq('sensitivity_baseline')&archive.control.eq('fixed_core')&archive.model.eq(model)].sort_values(keys)
        np.testing.assert_allclose(a.prediction,p.sort_values(keys)[f'{model}_none'],rtol=0,atol=1e-5)
    summary=pd.read_csv(OUT/'tables/summary.csv').set_index('method')
    assert abs(summary.loc['xgboost_none','mae']-473.90951651114005)<1e-6
    result=dict(passed=True,source_hashes=len(m['source_hashes']),event_station_snapshots=84,
                feature_values_checked=features_checked,models_reloaded=len(fits),metric_rows=metric_rows,
                max_feature_error=max_feature_error,max_prediction_error=max_prediction_error,max_metric_error=max_metric_error,
                original_474_reproduced=True,source_bin_input_max=17,target_bin_min=18,
                input_clock_semantics_verified=False,same_day_availability_verified=False)
    write_json(OUT/'independent_validation.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':run()
