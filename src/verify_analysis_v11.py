"""Independently reconstruct every tuning choice and score from archived outputs."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
KEYS=['event_date','station','hour']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def independent_scores(f,p):
    """Explicit per-event averaging, independent of production evaluation module."""
    p=np.asarray(p,float)
    if p.ndim==1:p=p[:,None]
    under=[];over=[];mse=[];mae=[];bias=[];peaku=[];peaka=[];peakh=[];fallback=0
    f=f.reset_index(drop=True)
    for _,e in f.groupby('event_date'):
        e=e.sort_values('hour');z=p[e.index];y=e.y.to_numpy()[:,None]
        assert len(e)==6 and list(e.hour)==list(range(18,24))
        q=e[['planned_end_0_60_beimen','planned_end_60_120_beimen']].to_numpy(float)
        if np.isnan(q).all() or (np.isfinite(q).all() and q.sum()==0):
            w=np.full(6,1/6);fallback+=1
        else:
            assert np.isfinite(q).all() and (q>=0).all() and (q.sum(axis=1)<=1+1e-8).all()
            w=q.sum(axis=1)/q.sum()
        error=z-y
        under.append(w@np.clip(-error,0,None));over.append(w@np.clip(error,0,None))
        mae.append(np.mean(abs(error),axis=0));mse.append(np.mean(error**2,axis=0));bias.append(error.mean(axis=0))
        peak=int(np.argmax(y[:,0]));delta=error[peak]
        peaku.append(np.clip(-delta,0,None));peaka.append(abs(delta));peakh.append(abs(peak-np.argmax(z,axis=0)))
    u,o=np.mean(under,axis=0),np.mean(over,axis=0)
    return dict(mae=np.mean(mae,axis=0),rmse=np.sqrt(np.mean(mse,axis=0)),bias=np.mean(bias,axis=0),
                window_under=u,window_over=o,**{f'cost_{r}':r*u+o for r in (1,2,3)},
                peak_under=np.mean(peaku,axis=0),peak_abs=np.mean(peaka,axis=0),peak_hour=np.mean(peakh,axis=0),
                events=f.event_date.nunique(),fallback_events=fallback)


def run():
    out=ROOT/'result/analysis_v11'
    cfg=json.loads((ROOT/'configs/analysis_v11.json').read_text('utf-8'))
    manifest=json.loads((out/'run_manifest.json').read_text('utf-8'))
    folds=json.loads((out/'folds.json').read_text('utf-8'))
    choices=json.loads((out/'selections.json').read_text('utf-8'))
    files=manifest['source_files']+manifest['source_and_output_hashes']
    for x in files:assert sha(ROOT/x['path'])==x['sha256'],x['path']
    models=joblib.load(ROOT/'.local/analysis_v11/selected_models.joblib')
    sources={};preds={};timeline={};matrices={}
    for v in cfg['variants']:
        sources[v]=pd.read_csv(ROOT/f'.local/analysis_v8/{v}/analysis_input.csv',parse_dates=['event_date']).sort_values(KEYS).reset_index(drop=True)
        assert len(sources[v])==504 and sources[v].event_date.nunique()==21
        preds[v]=pd.read_csv(ROOT/f'.local/analysis_v11/{v}_predictions.csv',parse_dates=['event_date'])
        old=pd.read_csv(ROOT/f'.local/analysis_v8/{v}/outer_predictions.csv',parse_dates=['event_date'])
        old=old.loc[old.station.eq('北門')].sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(preds[v][KEYS+['y']],old[KEYS+['y']],check_dtype=False)
        for m in ['ridge','gam','xgboost']:
            np.testing.assert_allclose(preds[v]['old_'+m],old[m],rtol=0,atol=1e-9)
            parts=[];values=[]
            records=sorted([f for f in folds if f['variant']==v and f['model']==m],key=lambda r:r['source']['validation_year'])
            for f in records:
                year=f['source']['validation_year'];s=sources[v];train=s.loc[s.year<year]
                valid=s.loc[s.year.eq(year)&s.station.eq('北門')].reset_index(drop=True)
                assert str(train.event_date.max().date())==f['source']['train_end']
                assert train.event_date.max()<valid.event_date.min() and len(train)==f['train_rows']
                assert train.event_date.nunique()==f['train_events'] and sorted(train.station.unique())==f['train_stations']
                for rel,digest in f['source']['signature'].items():assert sha(ROOT/rel)==digest
                path=ROOT/f'.local/analysis_v11/{v}/{m}/{year}.npz'
                assert sha(path)==f['sha256']
                z=np.load(path)['predictions'];assert z.shape==(len(valid),len(cfg['grids'][m]))
                parts.append(valid);values.append(z)
            timeline[v,m]=pd.concat(parts,ignore_index=True);matrices[v,m]=np.vstack(values)
    max_choice_diff=max_prediction_diff=max_metric_diff=0.
    for r in choices:
        v,m,year=r['variant'],r['model'],r['outer_year'];t=timeline[v,m];z=matrices[v,m]
        mask=t.year.lt(year).to_numpy();cost=independent_scores(t.loc[mask].reset_index(drop=True),z[mask,:r['candidate_count']])[f"cost_{r['ratio']}"]
        assert int(np.argmin(cost))==r['candidate_id']
        max_choice_diff=max(max_choice_diff,float(np.max(abs(cost-r['all_candidate_costs']))))
        assert r['params']==cfg['grids'][m][r['candidate_id']]
        assert pd.Timestamp(r['inner_end']).year<year
        assert r['inner_events']==t.loc[mask].event_date.nunique()
        assert r['inner_fallback_events']==independent_scores(t.loc[mask].reset_index(drop=True),z[mask,0])['fallback_events']
        input_frame=sources[v].loc[sources[v].year.eq(year)&sources[v].station.eq('北門')]
        replay=models[(v,year,m,r['candidate_id'])].predict(input_frame)
        name=f"{r['grid']}_r{r['ratio']}_{m}"
        expected=preds[v].loc[preds[v].year.eq(year),name].to_numpy()
        max_prediction_diff=max(max_prediction_diff,float(np.max(abs(replay-expected))))
        np.testing.assert_allclose(z[t.year.eq(year),r['candidate_id']],expected,atol=1e-9,rtol=0)
    rows=0
    for table in ['summary','annual','events']:
        records=pd.read_csv(out/'tables'/f'{table}.csv')
        for _,r in records.iterrows():
            f=preds[r['variant']]
            if table=='annual':f=f.loc[f.year.eq(r['year'])]
            if table=='events':f=f.loc[f.event_date.eq(pd.Timestamp(r['event_date']))]
            scores=independent_scores(f.reset_index(drop=True),f[r['method']])
            for name,value in scores.items():max_metric_diff=max(max_metric_diff,abs(float(np.asarray(value).ravel()[0])-float(r[name])))
            rows+=1
    assert max_choice_diff<1e-8 and max_prediction_diff<1e-7 and max_metric_diff<1e-8
    result=dict(status='passed',candidate_fits=sum(f['candidate_count'] for f in folds),chronological_folds=len(folds),
                selection_decisions=len(choices),distinct_models_replayed=len(models),metric_rows=rows,
                max_candidate_cost_difference=max_choice_diff,max_prediction_difference=max_prediction_diff,max_metric_difference=max_metric_diff,
                max_gam_normal_equation_residual=max(f['max_gam_normal_equation_residual'] for f in folds),
                note='Independent arithmetic and saved-model replay; not new external validation',
                checked_files=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in [Path(__file__),out/'run_manifest.json',out/'selections.json',out/'folds.json',*sorted((out/'tables').glob('*.csv'))]])
    (out/'independent_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checked_files'},ensure_ascii=False,indent=2))


if __name__=='__main__':run()
