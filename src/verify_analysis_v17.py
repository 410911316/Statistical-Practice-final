"""Independent chronology, subset composition, refit and metric checks for V17."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from analysis_v15.experiment import fit
from analysis_v13.data import sha
from verify_analysis_v13 import scores
from run_analysis_v8 import write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"result/analysis_v17"; PRIVATE=ROOT/".local/analysis_v17"
KEYS=["event_date","station","hour"]
WINDOW=["planned_end_0_60_beimen","planned_end_60_120_beimen"]


def run():
    manifest=json.loads((OUT/"manifest.json").read_text("utf8"));cfg=manifest["config"]
    for path,digest in manifest["source_hashes"].items():
        assert sha(ROOT/path)==digest,path
    for path,key in [(PRIVATE/"predictions.csv","prediction_sha256"),(OUT/"fits.json","fits_sha256"),
                     (OUT/"tables/scores.csv","scores_sha256"),(OUT/"tables/curves.csv","curves_sha256")]:
        assert sha(path)==manifest[key]
    f=pd.read_csv(ROOT/cfg["input"],parse_dates=["event_date"])
    p=pd.read_csv(PRIVATE/"predictions.csv",parse_dates=["event_date"])
    s=pd.read_csv(OUT/"tables/scores.csv")
    curves=pd.read_csv(OUT/"tables/curves.csv")
    fits=json.loads((OUT/"fits.json").read_text("utf8"))
    old=json.loads((ROOT/"result/analysis_v16/fits.json").read_text("utf8"))
    ref=pd.read_csv(ROOT/".local/analysis_v16/predictions.csv",parse_dates=["event_date"])
    assert len(fits)==222 and len(s)==444 and len(curves)==144
    assert not p.duplicated(["run_id","scope",*KEYS]).any()
    max_pred=max_score=max_scale=max_control=0.
    with threadpool_limits(limits=1):
        for r in fits:
            year=r["year"];n=r["n_events"]
            eligible=f.loc[f.year.lt(year)&f.station.isin(cfg["stations"])]
            all_dates=sorted(pd.to_datetime(eligible.event_date.unique()))
            dates=pd.to_datetime(r["train_dates"])
            if r["protocol"]=="full":
                expected=all_dates
            elif r["protocol"]=="oldest_first":
                expected=all_dates[:n]
            elif r["protocol"]=="recent_first":
                expected=all_dates[-n:]
            else:
                assert r["protocol"]=="event_subsample" and 0<=r["replicate"]<10
                rng=np.random.default_rng(np.random.SeedSequence([cfg["subset_seed"],year,r["replicate"]]))
                expected=sorted(all_dates[i] for i in rng.permutation(len(all_dates))[:n])
            assert list(dates)==expected and len(dates)==n and (dates.year<year).all()
            train=eligible.loc[eligible.event_date.isin(dates)]
            assert len(train)==n*18==r["train_rows"] and train.station.ne("民權西路").all()
            assert str(train.event_date.max().date())==r["train_end"]
            for date,g in train.groupby("event_date"):
                assert len(g)==18 and set(g.station)==set(cfg["stations"])
                for _,station in g.groupby("station"):assert set(station.hour)==set(range(18,24))
            parameter=next(v["params"] for v in old if v["year"]==year and v["cohort"]=="three"
                           and v["model"]==r["model"] and v["variant"]=="none")
            assert parameter==r["params"]
            fitted=fit(train,year,r["model"],"none",manifest["groups"],parameter,cfg["model_seed"])
            assert fitted.encoder.names==r["feature_names"]
            assert set(fitted.encoder.base.categories["station"])==set(cfg["stations"])
            if r["model"]=="ridge":
                assert abs(fitted.estimator.alpha-len(train)*parameter["lambda"])<1e-9
            for column,actual in r["train_mean"].items():
                values=train[column]
                if column in r["imputation"]:
                    available=values.dropna()
                    value=0. if available.empty else float(available.median())
                    assert abs(value-r["imputation"][column])<1e-9
                    values=values.fillna(value)
                mean=0. if column=="is_weekend" else values.mean()
                scale=1. if column=="is_weekend" else values.std(ddof=0)
                if scale==0:scale=1.
                delta=max(abs(mean-actual),abs(scale-r["train_scale"][column]))
                max_scale=max(max_scale,delta);assert delta<1e-8
            for scope in ["train","validation"]:
                expected_frame=train if scope=="train" else f.loc[f.year.eq(year)&f.station.isin(cfg["stations"])]
                expected_frame=expected_frame.loc[expected_frame.station.eq("北門")].sort_values(KEYS)
                observed=p.loc[p.run_id.eq(r["run_id"])&p.scope.eq(scope)].sort_values(KEYS)
                assert expected_frame[KEYS].reset_index(drop=True).equals(observed[KEYS].reset_index(drop=True))
                np.testing.assert_allclose(expected_frame[["y",*WINDOW]].to_numpy(),observed[["y",*WINDOW]].to_numpy(),equal_nan=True)
                if scope=="validation":assert len(observed)==18 and observed.event_date.nunique()==3
                predicted=fitted.predict(expected_frame)
                err=float(np.max(abs(predicted-observed.prediction.to_numpy())))
                max_pred=max(max_pred,err);assert err<1e-7
                truth=scores(observed,observed.prediction.to_numpy())
                row=s.loc[s.run_id.eq(r["run_id"])&s.scope.eq(scope)]
                assert len(row)==1
                for metric,value in truth.items():
                    diff=abs(value-row.iloc[0][metric])
                    max_score=max(max_score,diff);assert diff<1e-7
                if r["protocol"]=="full" and scope=="validation":
                    control=ref.loc[ref.year.eq(year)&ref.station.eq("北門")].sort_values(KEYS)
                    err=float(np.max(abs(predicted-control[f"three_{r['model']}_none"].to_numpy())))
                    max_control=max(max_control,err);assert err<1e-6
    # Recompute each published band's descriptive statistics without production aggregate().
    for r in curves.to_dict("records"):
        g=s.loc[s.year.eq(r["year"])&s.model.eq(r["model"])&s.scope.eq(r["scope"])&s.n_events.eq(r["n_events"])]
        g=g.loc[g.protocol.isin([r["protocol"],"full"])]
        assert len(g)==r["n_fits"]
        assert len(g)==(1 if r["protocol"]!="event_subsample" or (g.protocol=="full").all() else 10)
        for metric in ["mae","rmse","peak_abs","peak_under","cost_2"]:
            v=g[metric].to_numpy()
            for label,value in [("mean",v.mean()),("p10",np.percentile(v,10)),("p90",np.percentile(v,90))]:
                assert abs(r[f"{metric}_{label}"]-value)<1e-7
    # Ensure nested subsets by replicate / order and that dates agree across the two models.
    records=pd.DataFrame(fits)
    for _,g in records.groupby(["year","protocol","replicate","model"]):
        sets=[set(v) for v in g.sort_values("n_events").train_dates]
        assert all(a<b for a,b in zip(sets,sets[1:]))
    for _,g in records.groupby("subset_id"):
        assert len(g)==2 and g.iloc[0].train_dates==g.iloc[1].train_dates
    result=dict(passed=True,source_hashes=len(manifest["source_hashes"]),models_refitted=len(fits),
                metric_rows=len(s),curve_rows=len(curves),full_controls=6,
                max_prediction_difference=max_pred,max_metric_difference=max_score,
                max_preprocessing_difference=max_scale,max_control_difference=max_control,
                fixed_validation_events_per_year=3,whole_events=True,nested_subsets=True,
                future_labels_excluded=True,same_parameters_per_year=True,
                bands_are_confidence_intervals=False,deployed=False)
    write_json(OUT/"independent_validation.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":run()
