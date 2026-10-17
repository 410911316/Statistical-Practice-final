"""Recompute outputs independently and audit temporal/station exclusions."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from analysis_v13.data import sha
from verify_analysis_v13 import scores
from run_analysis_v8 import write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/analysis_v19'
PRIVATE=ROOT/'.local/analysis_v19'
KEYS=['event_date','station','hour']


def run():
    manifest=json.loads((OUT/'manifest.json').read_text('utf8'))
    cfg=manifest['config']
    for path,digest in manifest['source_hashes'].items():
        assert sha(ROOT/path)==digest,path
    assert sha(PRIVATE/'predictions.csv')==manifest['prediction_sha256']
    data=pd.read_csv(PRIVATE/'analysis_input.csv',parse_dates=['event_date'])
    pred=pd.read_csv(PRIVATE/'predictions.csv',parse_dates=['event_date'])
    assert len(pred)==108 and pred.event_date.nunique()==9 and not pred.duplicated(KEYS).any()
    assert set(pred.station)=={'北門','大橋頭站'}
    fits=json.loads((OUT/'fits.json').read_text('utf8'))
    assert len(fits)==27
    max_reload=0.
    for record in fits:
        assert sha(ROOT/record['path'])==record['sha256']
        train=data.loc[data.year.lt(record['year'])&data.station.isin(record['stations'])]
        test=data.loc[data.year.eq(record['year'])&data.station.isin(cfg['evaluation_stations'])].sort_values(KEYS)
        expected=pred.loc[pred.year.eq(record['year'])].sort_values(KEYS)
        assert test[KEYS].reset_index(drop=True).equals(expected[KEYS].reset_index(drop=True))
        np.testing.assert_array_equal(test.y,expected.y)
        assert record['train_end']==str(train.event_date.max().date())
        assert pd.Timestamp(record['train_end']).year < record['year']
        assert len(train)==record['train_rows']==6*len(record['stations'])*record['train_events']
        fitted=joblib.load(ROOT/record['path'])
        encoder=fitted.encoder if record['model']=='xgboost_w4' else fitted.encoder.base
        assert set(encoder.categories['station'])==set(record['stations'])
        if record['cohort']=='two':
            assert not any('station=雙連' in c or 'station=民權西路' in c for c in encoder.names)
            assert not any(c.startswith('lag_') for c in encoder.names)
        if record['model']=='ridge':
            assert abs(fitted.estimator.alpha-len(train)*record['params']['lambda'])<1e-10
        for audit in record['weight_audit']:
            assert abs(audit['weight_sum']-audit['rows'])<1e-10
            assert audit['rows']==6*len(record['stations'])
        delta=float(np.max(abs(fitted.predict(test)-expected[record['method']].to_numpy())))
        assert delta<1e-7
        max_reload=max(max_reload,delta)
    for year in cfg['outer_years']:
        for model in cfg['methods']:
            records=[r for r in fits if r['year']==year and r['model']==model]
            assert all(r['params']==records[0]['params'] for r in records)
    max_metric=0.;metric_rows=0
    north=pred.loc[pred.station.eq('北門')]
    for table in ('summary','annual','events','stations','common_two'):
        for row in pd.read_csv(OUT/'tables'/f'{table}.csv').to_dict('records'):
            f=north if table in ('summary','annual','events') else pred
            if table=='annual':f=f.loc[f.year.eq(row['year'])]
            if table=='events':f=f.loc[f.event_date.eq(pd.Timestamp(row['event_date']))]
            if table=='stations':f=f.loc[f.station.eq(row['station'])]
            p=f[row['method']].to_numpy()
            if table in ('stations','common_two'):
                e=p-f.y.to_numpy()
                expected=dict(mae=np.mean(abs(e)),rmse=np.sqrt(np.mean(e*e)),bias=np.mean(e))
                assert len(f)==row['rows']
            else:expected=scores(f,p)
            for key,value in expected.items():
                error=float(abs(row[key]-value));max_metric=max(max_metric,error)
                assert error<1e-7,(table,key,error)
            metric_rows+=1
    # Check all public replay records: a nonempty queue after boarding implies
    # the available space was exhausted. Otherwise do not force a full train.
    replay_path=ROOT/'app/frontend/public/data/rail-replay.json'
    replay=json.loads(replay_path.read_text('utf8'))
    train_count=0;left_count=0;violations=[]
    for date,cases in replay['events'].items():
        for name,pair in cases.items():
            remaining=replay['maximum']*float(name.split('-')[1])
            for direction,trace in pair.items():
                for train in trace['trains']:
                    train_count+=1
                    assert abs(train['boarded']-min(train['before'],remaining))<2e-6
                    if train['left']>1e-6:
                        left_count+=1
                        if abs(train['boarded']-remaining)>1e-6:violations.append([date,name,direction,train['time']])
    assert not violations
    example=next(t for t in replay['events']['2026-08-15']['broad-0.5']['north']['trains'] if t['time']==182)
    write_json(OUT/'queue_display_audit.json',dict(source_sha256=sha(replay_path),
        checked_train_records=train_count,records_with_left_behind=left_count,
        nonfull_departures_with_left_behind=violations,example_20260815_2102_north=example,
        interpretation='Before service shows a fixed post-alighting background assumption; boarding occurs at departure, not at animation approach.'))
    validation=dict(passed=True,source_hashes=len(manifest['source_hashes']),models_checked=len(fits),
        metric_rows=metric_rows,max_reload_error=max_reload,max_metric_error=max_metric,
        archive_controls=len(manifest['reference_checks']),max_control_error=max(r['max_difference'] for r in manifest['reference_checks']),
        same_nine_events=True,training_years_strictly_before_test=True,
        removed_stations_absent=True,same_fixed_params=True,no_same_day_lags=True,deployed=False)
    write_json(OUT/'independent_validation.json',validation)
    print(json.dumps(validation,ensure_ascii=False,indent=2))


if __name__=='__main__':run()
