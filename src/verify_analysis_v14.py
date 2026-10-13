"""Recompute V14 selection, stopping rules, training masses and reported errors."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from analysis_v13.data import sha
from verify_analysis_v13 import scores

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/'.local/analysis_v14';OUT=ROOT/'result/analysis_v14'
KEYS=['event_date','station','hour'];WINDOW=['planned_end_0_60_beimen','planned_end_60_120_beimen']


def run():
    manifest=json.loads((OUT/'manifest.json').read_text('utf-8'))
    for path,value in manifest['source_hashes'].items():assert sha(ROOT/path)==value,path
    for path,key in [('predictions.csv','prediction_sha256'),('timeline.csv','timeline_sha256')]:assert sha(PRIVATE/path)==manifest[key]
    source=pd.read_csv(PRIVATE/'analysis_input.csv',parse_dates=['event_date'])
    prediction=pd.read_csv(PRIVATE/'predictions.csv',parse_dates=['event_date'])
    timeline=pd.read_csv(PRIVATE/'timeline.csv',parse_dates=['event_date'])
    cfg=manifest['config'];curves=0;fallbacks=0;cap=0;weight_error=0.;cache={}
    for path in sorted(PRIVATE.glob('candidates_*.npz')):
        meta=json.loads(path.with_suffix('.json').read_text('utf-8'));assert sha(path)==meta['sha256']
        assert meta['identity']['signature']==manifest['source_hashes']
        yr=meta['year'];m=meta['multiplier'];train=source.loc[source.year.lt(yr)].copy()
        with np.load(path) as z:cache[yr,m]={k:z[k] for k in ['fixed','es']}
        assert pd.Timestamp(meta['train_end']).year<yr
        for record in meta['stopping']:
            if not record['stopping_active']:
                assert record['trees']==cfg['fallback_trees'];fallbacks+=1;continue
            stop_year=int(train.year.max());inner=train.loc[train.year.lt(stop_year)]
            stop=train.loc[train.year.eq(stop_year)&train.station.eq('北門')]
            assert record['inner_train_end']==str(inner.event_date.max().date())
            assert record['stop_start']==str(stop.event_date.min().date()) and record['stop_end']==str(stop.event_date.max().date())
            assert inner.year.max()<stop_year<yr
            history=np.asarray(record['curve']);best=int(np.argmin(history));assert record['trees']==best+1
            if len(history)<cfg['max_trees']:assert len(history)-1-best==cfg['patience']
            assert abs(history[best]-record['stop_score'])<.001
            curves+=1;cap+=int(record['cap_reached'])
        for audit in meta['weight_audit']:
            event=train.loc[train.event_date.eq(pd.Timestamp(audit['event_date']))];north=event.loc[event.station.eq('北門')]
            vals=north[WINDOW].to_numpy();q=np.zeros(6) if np.isnan(vals).all() else vals.sum(axis=1)
            total=24+(m-1)*q.sum();northmass=24*(6+(m-1)*q.sum())/total
            diff=abs(northmass-audit['beimen_weight']);weight_error=max(weight_error,diff);assert diff<1e-8
            assert abs(audit['event_weight']-24)<1e-8
    max_choice_error=0.;n_choices=0
    selections=json.loads((OUT/'selections.json').read_text('utf-8'))
    for record in selections:
        yr=record['year'];past=timeline.loc[timeline.year.lt(yr)].copy();assert past.event_date.nunique()==record['past_events']
        if record['method']=='archived_retuned_mae':
            matrix=np.vstack([np.load(ROOT/f'.local/analysis_v12/sensitivity_baseline/xgboost/{y}.npz')['prediction'] for y in sorted(past.year.unique())])
            chosen=record['candidate_id']
        else:
            mode=record['method'].split('_')[1];selected=record['method'].endswith('wselected')
            ws=cfg['training_multipliers'] if selected else [record['multiplier']]
            matrix=np.column_stack([np.vstack([cache[y,w][mode] for y in sorted(past.year.unique())]) for w in ws])
            width=96 if mode=='fixed' else 24
            chosen=ws.index(record['multiplier'])*width+record['candidate_id']
            current=prediction.loc[prediction.year.eq(yr)]
            assert np.allclose(current[record['method']],cache[yr,record['multiplier']][mode][:,record['candidate_id']],rtol=0,atol=1e-5)
        loss=np.mean(abs(matrix-past.y.to_numpy()[:,None]),axis=0)
        assert chosen==int(np.argmin(loss));diff=float(np.max(abs(loss-np.asarray(record['losses']))))
        max_choice_error=max(max_choice_error,diff);assert diff<1e-8;n_choices+=1
    max_reload=0.;models=json.loads((OUT/'models.json').read_text('utf-8'))
    for record in models:
        year=record['year'];path=PRIVATE/'models'/f"{year}_{record['method']}.joblib"
        assert sha(path)==record['sha256'];assert pd.Timestamp(record['train_end']).year<year
        fit=joblib.load(path);frame=source.loc[source.year.eq(year)&source.station.eq('北門')].sort_values(KEYS)
        values=np.maximum(fit.estimator.predict(fit.encoder.transform(frame)),0)
        expected=prediction.loc[prediction.year.eq(year)].sort_values(KEYS)[record['method']].to_numpy()
        diff=float(np.max(abs(values-expected)));max_reload=max(max_reload,diff);assert diff<1e-5
        assert fit.estimator.get_booster().num_boosted_rounds()==record['params']['n_estimators']
    old=pd.read_csv(ROOT/'.local/analysis_v12/predictions.csv',parse_dates=['event_date'])
    old=old.loc[old.variant.eq('sensitivity_baseline')&old.model.eq('xgboost')&old.control.eq('fixed_core')].sort_values(KEYS)
    assert np.allclose(prediction.sort_values(KEYS).archived_474,old.prediction,rtol=0,atol=1e-6)
    assert abs(scores(prediction,prediction.archived_474)['mae']-473.90951651114005)<1e-6
    n_metrics=0;max_metrics=0.
    for name,key in [('summary',None),('annual','year'),('events','event_date')]:
        for record in pd.read_csv(OUT/f'tables/{name}.csv').to_dict('records'):
            f=prediction if key is None else prediction.loc[prediction[key].eq(pd.Timestamp(record[key]) if key=='event_date' else record[key])]
            calculated=scores(f.reset_index(drop=True),f[record['method']])
            for metric,value in calculated.items():
                diff=abs(value-record[metric]);max_metrics=max(max_metrics,diff);assert diff<1e-6
            n_metrics+=1
    summary=dict(passed=True,checked_source_hashes=len(manifest['source_hashes']),verified_candidate_cache_files=len(cache),
        stopping_curves_checked=curves,insufficient_history_fallbacks=fallbacks,paths_reaching_cap=cap,
        max_training_mass_error=weight_error,choices_recomputed=n_choices,max_selection_difference=max_choice_error,
        models_reloaded=len(models),max_prediction_difference=max_reload,metric_rows=n_metrics,max_metric_difference=max_metrics,
        original_473_909516_reproduced=True,temporal_stopping_and_selection_verified=True,
        historical_D1_availability_verified=False,independent_unseen_test=False)
    (OUT/'independent_validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':run()
