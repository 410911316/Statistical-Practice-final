"""Separate parameter, chronology, epigraph LP, prediction and metric audit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from analysis_v15.experiment import fit
from analysis_v13.data import sha
from verify_analysis_v13 import scores, weights
from run_analysis_v8 import write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"result/analysis_v18";PRIVATE=ROOT/".local/analysis_v18"
KEYS=["event_date","station","hour"]
WINDOW=["planned_end_0_60_beimen","planned_end_60_120_beimen"]


def run():
    manifest=json.loads((OUT/"manifest.json").read_text("utf8"));cfg=manifest["config"]
    for path,digest in {**manifest["source_hashes"],**manifest["output_hashes"]}.items():
        assert sha(ROOT/path)==digest,path
    source=pd.read_csv(ROOT/cfg["input"],parse_dates=["event_date"])
    archive=pd.read_csv(ROOT/".local/analysis_v8/core/analysis_input.csv",parse_dates=["event_date"])
    grids=json.loads((ROOT/"configs/analysis_v11.json").read_text("utf8"))["grids"]
    oof=pd.read_csv(PRIVATE/"temporal_oof.csv",parse_dates=["event_date"])
    pred=pd.read_csv(PRIVATE/"predictions.csv",parse_dates=["event_date"])
    fits=json.loads((OUT/"fits.json").read_text("utf8"))
    corrections=json.loads((OUT/"corrections.json").read_text("utf8"))
    assert len(fits)==12 and len(corrections)==12
    assert not oof.duplicated(["model",*KEYS]).any()
    assert not pred.duplicated(["model",*KEYS]).any()
    assert len(pred)==108 and pred.event_date.nunique()==9
    max_pred=max_metric=max_lp=0.
    with threadpool_limits(limits=1):
        for record in fits:
            year,model=record["year"],record["model"]
            past=[];matrix=[]
            for earlier in range(2019,year):
                past.append(archive.loc[archive.year.eq(earlier)&archive.station.eq("北門")].sort_values(KEYS))
                matrix.append(np.load(ROOT/f".local/analysis_v11/core/{model}/{earlier}.npz")["predictions"])
            past=pd.concat(past,ignore_index=True);matrix=np.concatenate(matrix)
            assert past.year.max()==record["parameter_validation_end_year"]<year
            e=matrix-past.y.to_numpy()[:,None];w=weights(past,True)
            cost=w@(2*np.maximum(-e,0)+np.maximum(e,0))
            choice=int(np.argmin(cost))
            assert choice==record["candidate_index"]
            assert grids[model][choice]==record["params"]
            assert abs(cost[choice]-record["validation_cost"])<1e-7
            train=source.loc[source.year.lt(year)&source.station.isin(cfg["stations"])]
            assert len(train)==record["train_rows"]==18*record["train_events"]
            assert str(train.event_date.max().date())==record["train_end"]
            assert train.station.ne("民權西路").all() and train.year.max()<year
            target=source.loc[source.year.eq(year)&source.station.eq("北門")].sort_values(KEYS)
            saved=oof.loc[oof.year.eq(year)&oof.model.eq(model)].sort_values(KEYS)
            assert target[KEYS].reset_index(drop=True).equals(saved[KEYS].reset_index(drop=True))
            np.testing.assert_allclose(target[["y",*WINDOW]],saved[["y",*WINDOW]],equal_nan=True)
            fitted=fit(train,year,model,"none",manifest["groups"],record["params"],cfg["model_seed"])
            assert fitted.encoder.names==record["feature_names"]
            delta=float(np.max(abs(fitted.predict(target)-saved.base_prediction.to_numpy())))
            max_pred=max(max_pred,delta);assert delta<1e-7

    for c in corrections:
        year,model,ratio=c["cutoff_year"],c["model"],c["under_ratio"]
        history=oof.loc[oof.year.lt(year)&oof.model.eq(model)].reset_index(drop=True)
        assert (pd.to_datetime(history.base_train_end).dt.year<history.year).all()
        assert (history.parameter_validation_end_year<history.year).all()
        q=history[WINDOW].to_numpy(float)
        known=np.isfinite(q).all(axis=1)
        q=np.nan_to_num(q)
        w=np.zeros(len(history));dates=[]
        for date,g in history.groupby("event_date"):
            if known[g.index].all() and q[g.index].sum()>0:
                w[g.index]=q[g.index].sum(axis=1)/q[g.index].sum()
                dates.append(str(date.date()))
        assert dates==c["eligible_dates"] and len(dates)=={2024:6,2025:9,2026:12}[year]
        w/=len(dates);base=history.base_prediction.to_numpy()
        scale=max(1.,float(w@base));assert abs(scale-c["scale"])<1e-7
        idx=w>0
        X=base[idx,None]*q[idx]/scale;r=(history.y.to_numpy()[idx]-base[idx])/scale
        a=w[idx];n=len(r)
        # Independent epigraph formulation: loss >= rho*(r-Xtheta), loss >= Xtheta-r.
        dim=2+n+2;A=np.zeros((2*n+4,dim));b=np.r_[-ratio*r,r,np.zeros(4)]
        A[:n,:2]=-ratio*X;A[:n,2:2+n]=-np.eye(n)
        A[n:2*n,:2]=X;A[n:2*n,2:2+n]=-np.eye(n)
        A[2*n:2*n+2,:2]=np.eye(2);A[2*n:2*n+2,-2:]=-np.eye(2)
        A[-2:,:2]=-np.eye(2);A[-2:,-2:]=-np.eye(2)
        solution=linprog(np.r_[np.zeros(2),a,np.full(2,c["penalty"])],
                         A_ub=A,b_ub=b,bounds=[(-c["bound"],c["bound"])]*2+[(0,None)]*(n+2),method="highs")
        assert solution.success
        theta=np.array(c["theta"])
        assert (abs(theta)<=c["bound"]+1e-9).all()
        residual=r-X@theta
        value=float(a@(ratio*np.maximum(residual,0)+np.maximum(-residual,0))+c["penalty"]*abs(theta).sum())
        delta=max(abs(value-solution.fun),abs(value-c["objective"]))
        max_lp=max(max_lp,delta);assert delta<1e-8
        target=pred.loc[pred.year.eq(year)&pred.model.eq(model)].copy()
        qtest=target[WINDOW].fillna(0).to_numpy()
        expected=target.base.to_numpy()*(1+qtest@theta)
        actual=target[f"correction_{int(ratio)}"].to_numpy()
        delta=float(np.max(abs(expected-actual)));max_pred=max(max_pred,delta);assert delta<1e-7
        assert np.isfinite(actual).all() and (actual>=0).all()
        np.testing.assert_allclose(actual[qtest.sum(axis=1)==0],target.base.to_numpy()[qtest.sum(axis=1)==0])
        assert (abs(actual-target.base.to_numpy())<=cfg["coefficient_bound"]*target.base.to_numpy()+1e-7).all()

    # Recompute every published aggregate from saved rows, including each event.
    count=0
    for name in ["summary","annual","events"]:
        table=pd.read_csv(OUT/f"tables/{name}.csv")
        for row in table.to_dict("records"):
            g=pred.loc[pred.model.eq(row["model"])]
            if name=="annual":g=g.loc[g.year.eq(row["year"])]
            if name=="events":g=g.loc[g.event_date.eq(pd.Timestamp(row["event_date"]))]
            computed=scores(g,g[row["method"]].to_numpy())
            for key,value in computed.items():
                delta=abs(value-row[key]);max_metric=max(max_metric,delta);assert delta<1e-7
            if name=="events":
                peak=g.sort_values("hour").loc[g.y.idxmax()]
                assert int(peak.hour)==row["actual_peak_hour"] and peak.y==row["actual_peak"]
                assert abs(peak[row["method"]]-row["predicted_at_actual_peak"])<1e-7
            count+=1
    result=dict(passed=True,source_hashes=len(manifest["source_hashes"]),output_hashes=len(manifest["output_hashes"]),
        historical_parameter_choices_checked=len(fits),base_models_refitted=len(fits),epigraph_lp_checks=len(corrections),
        metric_rows=count,max_prediction_difference=max_pred,max_metric_difference=max_metric,max_lp_objective_difference=max_lp,
        strictly_past_labels=True,unknown_windows_unchanged=True,correction_cap=.25,
        d1_publication_fully_verified=False,independent_unseen_test=False,deployed=False)
    write_json(OUT/"independent_validation.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":run()
