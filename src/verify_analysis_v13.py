"""Independent numerical and temporal checks of completed V13 outputs."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,eye,hstack,vstack
from analysis_v13.data import sha

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/'.local/analysis_v13';OUT=ROOT/'result/analysis_v13'
MODELS=['ridge','gam','xgboost']
WINDOW=['planned_end_0_60_beimen','planned_end_60_120_beimen']


def weights(f,window):
    a=np.zeros(len(f));f=f.reset_index(drop=True)
    for _,g in f.groupby('event_date'):
        q=g[WINDOW].sum(axis=1,min_count=2).to_numpy(float) if window else np.ones(len(g))
        if np.isnan(q).all() or np.nansum(q)==0:q=np.ones(len(g))
        assert np.isfinite(q).all() and (q>=0).all()
        a[g.index]=q/q.sum()/f.event_date.nunique()
    return a


def scores(f,p):
    a=weights(f,True);b=weights(f,False);e=np.asarray(p)-f.y.to_numpy(float)
    peaks=[]
    for _,g in f.assign(p=np.asarray(p)).groupby('event_date'):
        g=g.sort_values('hour');t=g.loc[g.y.idxmax()];z=g.loc[g.p.idxmax()]
        peaks.append([max(0,t.y-t.p),abs(t.y-t.p),abs(t.hour-z.hour)])
    under=a@np.maximum(-e,0);over=a@np.maximum(e,0)
    return dict(mae=b@abs(e),rmse=np.sqrt(b@(e*e)),bias=b@e,window_under=under,window_over=over,
                cost_1=under+over,cost_2=2*under+over,cost_3=3*under+over,
                **dict(zip(['peak_under','peak_abs','peak_hour'],np.mean(peaks,axis=0))))


def run():
    manifest=json.loads((OUT/'manifest.json').read_text('utf-8'))
    for name,value in manifest['source_hashes'].items():assert sha(ROOT/name)==value,name
    for file,key in [('predictions.csv','prediction_sha256'),('temporal_oof.csv','oof_sha256')]:
        assert sha(PRIVATE/file)==manifest[key]
    data=pd.read_csv(PRIVATE/'analysis_input.csv',parse_dates=['event_date'])
    assert data.year.eq(data.event_date.dt.year).all()
    assert data.day_of_year.eq(data.event_date.dt.dayofyear).all()
    assert data.is_weekend.eq(data.event_date.dt.dayofweek.ge(5).astype(int)).all()
    pred=pd.read_csv(PRIVATE/'predictions.csv',parse_dates=['event_date'])
    oof=pd.read_csv(PRIVATE/'temporal_oof.csv',parse_dates=['event_date'])
    panel=pd.read_csv(PRIVATE/'official_hourly.csv');panel['date']=pd.to_datetime(panel.date)
    ledger=json.loads((ROOT/'docs/d1_od_version_sources.json').read_text('utf-8'))
    versions={r['month']:pd.Timestamp(r['version_at']) for r in ledger['source_records']}
    assert all(r['transactional_range_md5_verified'] for r in ledger['source_records'])
    max_baseline_error=0.
    for entry in json.loads((PRIVATE/'baseline_day_plan.json').read_text('utf-8')):
        if not entry['eligible']:continue
        date=pd.Timestamp(entry['event_date']);cutoff=date.tz_localize('Asia/Taipei')-pd.Timedelta(hours=6)
        dates=pd.to_datetime(entry['days']);assert all(versions[d.strftime('%Y%m')]<=cutoff for d in dates)
        expected=panel.loc[panel.date.isin(dates)].groupby(['station','hour']).y.mean()
        observed=data.loc[data.event_date.eq(date)].set_index(['station','hour']).available_baseline
        diff=float(abs(observed-expected).max());max_baseline_error=max(max_baseline_error,diff)
        assert diff<1e-8
    assert (pd.to_datetime(data.baseline_version_at,utc=True)<=pd.to_datetime(data.cutoff_at,utc=True)).all()
    fits=json.loads((OUT/'fits.json').read_text('utf-8'));max_pred_error=0.
    for fit in fits:
        year=fit['year'];path=PRIVATE/'models'/f"{year}_{fit['mode']}_{fit['model']}.joblib"
        assert sha(path)==fit['model_sha256']
        assert pd.Timestamp(fit['max_label_version'])<=pd.Timestamp(fit['fit_at'])
        assert pd.Timestamp(fit['train_end']).year<year
        f=data.loc[data.year.eq(year)&data.station.eq('北門')].sort_values(['event_date','station','hour'])
        model=joblib.load(path);p=model.estimator.predict(model.encoder.transform(f))
        if fit['mode']=='increment':p=p+f.available_baseline.to_numpy()
        p=np.maximum(p,0)
        target=oof.loc[oof.year.eq(year)].sort_values(['event_date','station','hour'])[f"{fit['mode']}_{fit['model']}"]
        diff=float(np.max(abs(p-target.to_numpy())));max_pred_error=max(max_pred_error,diff);assert diff<1e-6
    selections=json.loads((OUT/'selections.json').read_text('utf-8'));n_selections=0
    for choice in selections:
        if choice['candidate_id'] is None:continue
        year=choice['year'];parts=[];keys=[]
        for yr in sorted(oof.year.unique()):
            if yr>=year:continue
            g=oof.loc[oof.year.eq(yr)].reset_index(drop=True)
            matrix=np.load(PRIVATE/f"core_candidates_{choice['model']}_{yr}.npz")['predictions']
            known=pd.to_datetime(g.label_version_at,utc=True)<=pd.Timestamp(f'{year}-01-01',tz='Asia/Taipei')
            keys.append(g.loc[known]);parts.append(matrix[known])
        f=pd.concat(keys,ignore_index=True);p=np.vstack(parts);r=f.y.to_numpy()[:,None]-p
        losses=weights(f,True)@(2*np.maximum(r,0)+np.maximum(-r,0))
        assert int(np.argmin(losses))==choice['candidate_id']
        assert np.allclose(losses,choice['all_costs'],rtol=0,atol=1e-6);n_selections+=1
    n_lp=0;max_lp_error=0.
    for record in json.loads((OUT/'weights.json').read_text('utf-8')):
        w=np.array([record['weights'][m] for m in MODELS]);year=record['year']
        assert (w>=0).all() and abs(w.sum()-1)<1e-10
        current=oof.loc[oof.year.eq(year)];assert np.allclose(current[MODELS].to_numpy()@w,current[record['method']])
        if 'reason' in record:assert np.array_equal(w,[0,0,1]);continue
        f=oof.loc[oof.year.lt(year)].reset_index(drop=True);assert f.event_date.nunique()==record['past_events']
        assert (pd.to_datetime(f.label_version_at,utc=True)<=pd.Timestamp(f'{year}-01-01',tz='Asia/Taipei')).all()
        p=f[MODELS].to_numpy();y=f.y.to_numpy();a=weights(f,record['window']);r=record['ratio'];n=len(f)
        # Epigraph of max(r*(y-Pw), Pw-y), independently of production u-v LP.
        A=vstack([hstack([-r*csr_matrix(p),-eye(n)]),hstack([csr_matrix(p),-eye(n)])]).tocsr()
        sol=linprog(np.r_[np.zeros(3),a],A_ub=A,b_ub=np.r_[-r*y,y],
                    A_eq=csr_matrix(np.r_[np.ones(3),np.zeros(n)][None,:]),b_eq=[1],bounds=(0,None),method='highs')
        assert sol.success
        error=abs(sol.fun-record['objective']);max_lp_error=max(max_lp_error,error);assert error<1e-6;n_lp+=1
    count=0;max_metric_error=0.
    for filename,key in [('summary',None),('annual','year'),('events','event_date')]:
        table=pd.read_csv(OUT/f'tables/{filename}.csv')
        for row in table.to_dict('records'):
            f=pred if key is None else pred.loc[pred[key].eq(pd.Timestamp(row[key]) if key=='event_date' else row[key])]
            calculated=scores(f.reset_index(drop=True),f[row['method']])
            for metric,value in calculated.items():
                error=abs(value-row[metric]);max_metric_error=max(max_metric_error,error);assert error<1e-6,(filename,metric)
            count+=1
    assert abs(scores(pred,pred.archived_474)['mae']-473.909517)<1e-5
    result=dict(passed=True,source_hashes=len(manifest['source_hashes']),verified_official_months=len(versions),
                reconstructed_baseline_events=data.event_date.nunique(),max_baseline_difference=max_baseline_error,
                reloaded_fits=len(fits),max_prediction_difference=max_pred_error,parameter_choices=n_selections,
                independently_solved_fusions=n_lp,max_LP_objective_difference=max_lp_error,
                metric_rows=count,max_metric_difference=max_metric_error,legacy_MAE_473_909517_reproduced=True,
                historical_public_ACL_proven=False,independent_unseen_test=False)
    (OUT/'independent_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':run()
