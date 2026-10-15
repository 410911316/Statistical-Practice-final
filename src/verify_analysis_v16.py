"""Independent station membership, chronology, fixed settings and numerical checks."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from analysis_v13.data import sha
from verify_analysis_v13 import scores
from run_analysis_v8 import write_json

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'result/analysis_v16';PRIVATE=ROOT/'.local/analysis_v16'


def run():
    m=json.loads((OUT/'manifest.json').read_text('utf8'));cfg=m['config']
    for file,digest in m['source_hashes'].items():assert sha(ROOT/file)==digest,file
    assert sha(PRIVATE/'predictions.csv')==m['prediction_sha256']
    f=pd.read_csv(ROOT/cfg['input'],parse_dates=['event_date'])
    p=pd.read_csv(PRIVATE/'predictions.csv',parse_dates=['event_date'])
    keys=['event_date','station','hour']
    assert len(p)==162 and not p.duplicated(keys).any()
    assert p.event_date.nunique()==9 and set(p.station)==set(cfg['evaluation_stations'])
    for _,g in p.groupby('event_date'):
        assert len(g)==18 and set(g.station)==set(cfg['evaluation_stations'])
        for _,h in g.groupby('station'):assert set(h.hour)==set(range(18,24))
    oldfits=json.loads((ROOT/'result/analysis_v15/fits.json').read_text('utf8'))
    fits=json.loads((OUT/'fits.json').read_text('utf8'));assert len(fits)==84
    max_prediction=0.
    for r in fits:
        assert sha(ROOT/r['path'])==r['sha256']
        expected=next(z for z in oldfits if z['year']==r['year'] and z['model']==r['model'] and z['variant']=='none')
        assert expected['params']==r['params']
        train=f.loc[f.year.lt(r['year']) & f.station.isin(cfg['cohorts'][r['cohort']])]
        test=f.loc[f.year.eq(r['year']) & f.station.isin(cfg['evaluation_stations'])].sort_values(keys)
        pred=p.loc[p.year.eq(r['year'])].sort_values(keys)
        fitted=joblib.load(ROOT/r['path'])
        assert fitted.train_end==str(train.event_date.max().date())==r['train_end']
        assert fitted.train_rows==len(train)==r['train_rows']
        assert pd.Timestamp(fitted.train_end).year<r['year']
        assert set(fitted.encoder.base.categories['station'])==set(r['stations'])
        assert len(train)==train.event_date.nunique()*6*len(r['stations'])
        if r['cohort']=='three':
            assert train.station.ne('民權西路').all()
            assert not any('station=民權西路' in n for n in fitted.encoder.names)
        if r['model']=='ridge':
            assert abs(fitted.estimator.alpha-len(train)*r['params']['lambda'])<1e-10
        np.testing.assert_allclose(fitted.encoder.means,train[fitted.encoder.lag_columns].to_numpy(float).mean(axis=0))
        assert test[keys].reset_index(drop=True).equals(pred[keys].reset_index(drop=True))
        error=float(np.max(abs(fitted.predict(test)-pred[r['method']].to_numpy())))
        max_prediction=max(max_prediction,error);assert error<1e-7
    north=p.loc[p.station.eq('北門')]
    ref=pd.read_csv(ROOT/'.local/analysis_v15/predictions.csv',parse_dates=['event_date']).sort_values(keys)
    max_control=0.
    for model in cfg['models']:
        for variant in cfg['variants']:
            name=f'{model}_{variant}'
            d=float(np.max(abs(north.sort_values(keys)[f'four_{name}'].to_numpy()-ref[name].to_numpy())))
            max_control=max(max_control,d);assert d<1e-6
    max_metric=0.;rows=0
    for name in ['summary','annual','events','stations','common_three','conservative']:
        for r in pd.read_csv(OUT/'tables'/f'{name}.csv').to_dict('records'):
            g=north if name in ['summary','annual','events','conservative'] else p
            if name=='annual':g=g.loc[g.year.eq(r['year'])]
            if name=='events':g=g.loc[g.event_date.eq(pd.Timestamp(r['event_date']))]
            if name=='stations':g=g.loc[g.station.eq(r['station'])]
            if name=='conservative':g=g.loc[g.hour.ge(19)]
            if name in ['summary','annual','events']:expected=scores(g,g[r['method']].to_numpy())
            else:
                errors=g[r['method']].to_numpy()-g.y.to_numpy()
                expected=dict(mae=np.mean(abs(errors)),rmse=np.sqrt(np.mean(errors**2)),bias=np.mean(errors))
                assert r['rows']==len(g)
            for key,value in expected.items():
                diff=abs(r[key]-value);max_metric=max(max_metric,diff);assert diff<1e-7
            rows+=1
    s=pd.read_csv(OUT/'tables/summary.csv').set_index('method')
    for r in pd.read_csv(OUT/'tables/paired_comparison.csv').to_dict('records'):
        for metric in ['mae','rmse','peak_abs','cost_2']:
            a=s.loc[f"four_{r['model']}_{r['variant']}",metric]
            b=s.loc[f"three_{r['model']}_{r['variant']}",metric]
            assert abs(r[f'{metric}_improvement_pct']-100*(1-b/a))<1e-8
    result=dict(passed=True,source_hashes=len(m['source_hashes']),models_checked=len(fits),metric_rows=rows,
        reference_comparisons=42,max_prediction_error=max_prediction,max_control_error=max_control,max_metric_error=max_metric,
        same_test_events=True,excluded_station_absent_from_three_training=True,training_years_strictly_before_test=True,
        same_fixed_hyperparameters=True,no_new_test_based_selection=True,deployed=False)
    write_json(OUT/'independent_validation.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':run()
