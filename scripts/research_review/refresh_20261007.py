"""Refresh only the statistical methods fixed on 2026-10-05, using frozen inputs.

Primary: original four-week baseline and legacy 18..23 weather labels.
Sensitivity: already-approved calendar-matched eight-week baseline; separately,
verified 19..24 weather report-hour ends. These are not new methods or causal tests.
All database calls use the existing read-only transaction helper.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import basic_statistics as basic
import station_time_statistics as station
import weather_regression_statistics as weather
import extended_summary as extended

VERSION = "2026-10-07-refresh"
POLICY = "same_weekday_calendar_56d_nearest4_v1"
SOURCE_FILES = [
    "data/processed/station_hourly.parquet",
    "data/processed/regression_event_station.parquet",
    "data/processed/calendar_baseline/candidate_audit.csv",
    "data/processed/calendar_baseline/event_station_baseline.csv",
    "data/processed/calendar_baseline/summary.json",
    "data/processed/event_station_hourly_context.csv",
    "data/original_data/calendar/calendar_2017_2026.csv",
    "results/preprocessing/summary.json",
    "results/weather/source_verification.csv",
    "results/weather/source_verification_summary.json",
    "results/research_review/2026-10-05/statistics/summary.json",
    "results/research_review/2026-10-05/statistics/primary_beimen_events.csv",
    "results/research_review/2026-10-05/station_time/summary.json",
    "results/research_review/2026-10-05/weather_regression/summary.json",
    "results/research_review/2026-10-05/extended_statistics/summary.json",
]


def write_json(path, data):
    basic.write_json(path, data)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="")


def quiet_call(fn, *args):
    with redirect_stdout(io.StringIO()):
        return fn(*args)


def snapshot_inputs(root, private, reuse):
    index = private / "input_manifest.json"
    frozen = private / "source_data"
    if reuse:
        manifest = read_json(index)
        for item in manifest["files"]:
            path = private / item["snapshot_relative_path"]
            if basic.sha(path) != item["sha256"]:
                raise ValueError(f"Frozen input changed: {path}")
        return frozen
    if index.exists() or (private / "statistics_snapshot.json").exists():
        raise ValueError("Existing refresh inputs must use --reuse-snapshot; use another --snapshot-root for a new capture")
    private.mkdir(parents=True, exist_ok=True)
    files=[]
    for relative in SOURCE_FILES:
        src=root/relative
        if not src.exists():
            if relative == "results/weather/source_verification_summary.json":
                continue
            raise FileNotFoundError(src)
        dst=frozen/relative
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        files.append({"source_path":relative,"snapshot_relative_path":dst.relative_to(private).as_posix(),"sha256":basic.sha(dst),"bytes":dst.stat().st_size})
    basic.extract(root,private/"statistics_snapshot.json",False)
    weather.extract(root,False,private/"weather_snapshot.json",True)
    (private/"statistics_queries.sql").write_text(basic.sql_queries(),encoding="utf-8")
    (private/"weather_queries.sql").write_text(weather.query(True),encoding="utf-8")
    sys.path.insert(0,str(root))
    from app.backend.database import _read_json
    sql="""SELECT JSON_OBJECT('event_date',event_date,'station',station,'policy_name',policy_name,
      'baseline_n_days',baseline_n_days,'baseline_dates',baseline_dates,
      'baseline_origin_mean',baseline_origin_mean,'event_origin_total',event_origin_total,
      'eligible',eligible,'source_run_sha256',source_run_sha256)
      FROM taipei_metro_analysis.analysis_calendar_baseline ORDER BY event_date,station;
      SELECT JSON_OBJECT('verification_total',COUNT(*),'values_matched',SUM(values_match),
      'cloud_methods_verified',SUM(cloud_method_verified)) FROM taipei_metro_analysis.weather_source_verification;"""
    meta=_read_json(sql)
    write_json(private/"supplement_snapshot.json",{"records":meta,"read_only":True,"query_sha256":hashlib.sha256(sql.encode()).hexdigest(),"extracted_at_utc":datetime.now(timezone.utc).isoformat()})
    (private/"supplement_queries.sql").write_text(sql,encoding="utf-8")
    for name in ["statistics_snapshot.json","weather_snapshot.json","supplement_snapshot.json","statistics_queries.sql","weather_queries.sql","supplement_queries.sql"]:
        p=private/name
        files.append({"source_path":"fresh_read_only_mysql" if name.endswith(".json") else "fixed_read_only_query", "snapshot_relative_path":name,"sha256":basic.sha(p),"bytes":p.stat().st_size})
    write_json(index,{"captured_at_utc":datetime.now(timezone.utc).isoformat(),"files":files,"read_only":True,"warning":"Original source files and prior analysis outputs are not modified"})
    return frozen


def eight_week_snapshot(private,frozen,out):
    """Reconstruct hourly baselines from the previously fixed selected-date audit."""
    out.mkdir(parents=True,exist_ok=True)
    source=read_json(private/"statistics_snapshot.json")
    base=pd.read_csv(frozen/"data/processed/calendar_baseline/event_station_baseline.csv",encoding="utf-8-sig")
    candidates=pd.read_csv(frozen/"data/processed/calendar_baseline/candidate_audit.csv",encoding="utf-8-sig")
    supplement=read_json(private/"supplement_snapshot.json")["records"]
    db=pd.DataFrame([r for r in supplement if "event_date" in r])
    if len(base)!=104 or base.duplicated(["event_date","station"]).any() or set(base.policy_name)!={POLICY}:
        raise ValueError("Calendar baseline shape/policy changed")
    joined=base.merge(db,on=["event_date","station"],suffixes=("_file","_db"),validate="one_to_one")
    for field in ["baseline_n_days","baseline_origin_mean","event_origin_total","eligible"]:
        if not np.allclose(joined[field+"_file"],joined[field+"_db"],equal_nan=True):
            raise ValueError(f"Calendar DB/CSV mismatch: {field}")
    for field in ["policy_name","source_run_sha256"]:
        if not joined[field+"_file"].eq(joined[field+"_db"]).all():
            raise ValueError(f"Calendar DB/CSV provenance mismatch: {field}")
    for r in joined.itertuples():
        a=json.loads(r.baseline_dates_file)
        b=r.baseline_dates_db
        if isinstance(b,str):b=json.loads(b)
        if a!=b:raise ValueError("Calendar DB/CSV dates mismatch")
    selected=candidates.loc[candidates.selected].copy()
    selected=selected.sort_values(["event_date","station","lag_days"])
    event_days={r["event_date"] for r in source["events"]}
    calendar=pd.read_csv(frozen/"data/original_data/calendar/calendar_2017_2026.csv",keep_default_na=False).set_index("calendar_date").to_dict("index")
    # Reapply the existing date/calendar/completeness ordering rule, independently
    # of the chosen outcomes; do not import the ETL entry point that writes data.
    def is_candidate(r):
        a,b=calendar.get(r.event_date),calendar.get(r.candidate_date)
        return (r.candidate_date not in event_days and a is not None and b is not None and
          (a["is_holiday"],a["is_makeup_workday"],a["day_note"])==(b["is_holiday"],b["is_makeup_workday"],b["day_note"]) and
          r.complete_hours==6 and r.observed_hours==6)
    for (day,s),g in candidates.groupby(["event_date","station"]):
        ordered=g.sort_values("lag_days")
        expected=ordered.loc[ordered.apply(is_candidate,axis=1)].head(4).candidate_date.tolist()
        b=base.loc[base.event_date.eq(day)&base.station.eq(s)].iloc[0]
        if expected!=json.loads(b.baseline_dates) or expected!=ordered.loc[ordered.selected,"candidate_date"].tolist() or len(expected)!=b.baseline_n_days:
            raise ValueError(f"Calendar selection rule mismatch: {day}/{s}")
    con=duckdb.connect()
    con.register("selected",selected)
    history=con.execute("""SELECT c.event_date,c.station,c.candidate_date,c.lag_days,s.source_hour,
        s.origin_count,s.origin_complete FROM selected c JOIN read_parquet(?) s
        ON CAST(c.candidate_date AS DATE)=s.service_date AND c.station=s.station
        AND s.source_hour>=18 AND s.source_hour<24 ORDER BY c.event_date,c.station,c.lag_days,s.source_hour""",
        [str(frozen/"data/processed/station_hourly.parquet")]).df()
    con.close()
    if len(history)!=len(selected)*6 or not history.origin_complete.all() or history.origin_count.isna().any():
        raise ValueError("Selected eight-week comparison days lack six complete hours")
    aggregate=history.groupby(["event_date","station","source_hour"]).origin_count.agg(mean="mean",n="count")
    # Validate every candidate total directly before using per-hour averages.
    totals=history.groupby(["event_date","station","candidate_date"]).origin_count.sum().rename("recomputed")
    selected_totals=selected.merge(totals,on=["event_date","station","candidate_date"],validate="one_to_one")
    if not np.allclose(selected_totals.candidate_origin_total,selected_totals.recomputed):
        raise ValueError("Eight-week candidate totals disagree with source hours")
    derived=deepcopy(source)
    base_index=base.set_index(["event_date","station"])
    for r in derived["windows"]:
        b=base_index.loc[(r["event_date"],r["station"])]
        if r["origin_count_window"]!=b.event_origin_total:raise ValueError("Event observation changed between baseline inputs")
        r["baseline_origin_window"]=float(b.baseline_origin_mean) if pd.notna(b.baseline_origin_mean) else None
        r["baseline_min_days"]=int(b.baseline_n_days)
    for r in derived["hourly"]:
        key=(r["event_date"],r["station"],pd.Timestamp(r["hour_start"]).hour)
        b=aggregate.loc[key] if key in aggregate.index else None
        r["baseline_origin_mean_28d"]=float(b["mean"]) if b is not None else None
        r["baseline_n_days"]=int(b["n"]) if b is not None else 0
    derived["baseline_policy"]=POLICY
    derived["baseline_field_compatibility_note"]="The legacy baseline_origin_mean_28d field name carries the separately derived 56d sensitivity; not a change to original DB or primary analysis."
    derived["derived_at_utc"]=read_json(private/"input_manifest.json")["captured_at_utc"]
    derived["parent_snapshot_sha256"]=basic.sha(private/"statistics_snapshot.json")
    write_json(private/"eight_week_snapshot.json",derived)
    csv(out/"selected_baseline_source_hours.csv",history)
    csv(out/"baseline_candidates.csv",candidates)
    csv(out/"calendar_baseline_windows.csv",base)
    return derived


def eight_week_basic(root,private,out,data):
    events,joined,hourly,checks=basic.prepare(data)
    primary=joined.loc[joined.station.eq("北門")&joined.eligible].sort_values("event_date").copy()
    if len(primary)!=21:raise ValueError("Expected all 21 held events for approved 8-week sensitivity")
    years=primary.groupby("year").log1p_difference.mean()
    loo=pd.DataFrame([{"excluded_year":int(y),"remaining_events":int(primary.year.ne(y).sum()),
      "mean_log1p_difference":float(primary.loc[primary.year.ne(y),"log1p_difference"].mean()),
      "geometric_ratio_plus_one":float(np.exp(primary.loc[primary.year.ne(y),"log1p_difference"].mean()))} for y in years.index])
    rng=np.random.default_rng(basic.SEED)
    ci=np.quantile(rng.choice(years.to_numpy(),size=(20000,len(years)),replace=True).mean(axis=1),[.025,.975])
    sw=stats.shapiro(primary.log1p_difference)
    description=[]
    for s,g in joined.loc[joined.event_date.isin(primary.event_date)].groupby("station"):
        description.append({"station":s,"eligible_events":len(g),"mean_origin_eligible":float(g.origin_count_window.mean()),"mean_baseline_eligible":float(g.baseline_origin_window.mean()),"mean_excess_eligible":float(g.excess_vs_baseline.mean()),"geometric_ratio_plus_one":float(np.exp(g.log1p_difference.mean()))})
    summary={"design":{"baseline_policy":POLICY,"analysis_role":"sensitivity only; eight-week calendar rule changes baseline and eligibility","hours":[18,24],"seed":basic.SEED,"fixed_test":"two-sided event paired log1p contrast; F=t^2","inference_status":"exploratory; event independence and common distribution not assured"},
      "coverage":{"events":len(events),"status_counts":events.event_status.value_counts().to_dict(),"beimen_primary_n":len(primary),"beimen_primary_dates":list(primary.event_date),"beimen_primary_years":list(map(int,years.index)),"eligible_min_baseline_days_counts":{str(int(k)):int(v) for k,v in primary.baseline_min_days.value_counts().items()}},
      "primary_F_test":basic.contrast_summary(primary.log1p_difference),"year_equal_weight_sensitivity":basic.contrast_summary(years),
      "year_bootstrap":{"unit":"year mean, each year equally weighted","replications":20000,"mean_log1p_difference_percentile_ci95":ci.tolist(),"geometric_ratio_plus_one_ci95":np.exp(ci).tolist(),"warning":"Few years; exploratory"},
      "diagnostics":{"shapiro_W":float(sw.statistic),"shapiro_p":float(sw.pvalue),"leave_one_year_geometric_ratio_range":[float(loo.geometric_ratio_plus_one.min()),float(loo.geometric_ratio_plus_one.max())]},"station_descriptive":description}
    for name,frame in [("primary_beimen_events",primary),("event_station_summary",joined),("hourly_window_reconciliation",checks),("station_descriptive",pd.DataFrame(description)),("leave_one_year_out",loo),("year_mean_contrasts",years.rename("mean_log1p_difference").reset_index()),("cancelled_postponed",joined.loc[~joined.event_status.eq("held")])]:csv(out/f"{name}.csv",frame)
    write_json(out/"summary.json",summary)
    basic.plots(out,joined,hourly,primary)
    write_json(out/"source_manifest.json",{"snapshot_sha256":basic.sha(private/"eight_week_snapshot.json"),"input_manifest_sha256":basic.sha(private/"input_manifest.json"),"baseline_policy":POLICY,"script_sha256":basic.sha(Path(__file__)),"outputs":{p.name:basic.sha(p) for p in out.glob("*") if p.is_file() and p.name!="source_manifest.json"}})


def run_extensions(root,private,base,snapshot,n):
    primary=base/"statistics"
    quiet_call(station.run,SimpleNamespace(root=root,output=base/"station_time",snapshot=snapshot,primary_csv=primary/"primary_beimen_events.csv",permutations=station.PERMUTATIONS))
    quiet_call(weather.run,SimpleNamespace(root=root,output=base/"weather_regression",snapshot=private/"weather_snapshot.json",reuse_snapshot=True,verified_weather=True,primary_directory=primary,expected_n=n,label_mode="legacy_18_23"))
    data=extended.build(root,base)
    directory=base/"extended_statistics";directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/"summary.json",data);csv(directory/"tests.csv",pd.DataFrame(data["tests"]))
    quiet_call(weather.run,SimpleNamespace(root=root,output=base/"weather_aligned",snapshot=private/"weather_snapshot.json",reuse_snapshot=True,verified_weather=True,primary_directory=primary,expected_n=n,label_mode="verified_hour_end_19_24"))
    return data


def annotate_outputs(root,out):
    for policy in ["four_week_primary","eight_week_sensitivity"]:
        for p in (out/policy).rglob("summary.json"):
            d=read_json(p)
            d["refresh_context"]={"analysis_date":"2026-10-07","baseline_policy":"original_previous_7_14_21_28_days" if policy=="four_week_primary" else POLICY,
              "role":"primary_same_rules_as_2026_10_05" if policy=="four_week_primary" else "previously_approved_baseline_sensitivity",
              "weather_timing_sensitivity":p.parent.name=="weather_aligned",
              "multiplicity_scope":"Seven-question Holm applies to each legacy-label scenario separately; aligned weather has only two-test Holm. Neither covers all sensitivities or all historical research."}
            write_json(p,d)
    # Extended summaries include source summary hashes; update after annotation.
    for p in out.rglob("extended_statistics/summary.json"):
        d=read_json(p)
        for source in d["sources"]:
            source["sha256"]=basic.sha(root/source["path"])
        write_json(p,d)
    for policy in ["four_week_primary","eight_week_sensitivity"]:
        for module in ["statistics","station_time","weather_regression","weather_aligned"]:
            p=out/policy/module/"source_manifest.json"
            d=read_json(p)
            if module.startswith("weather"):
                d["previous_manifest_sha256"]=basic.sha(out/policy/"statistics/source_manifest.json")
            d["outputs"]={q.name:basic.sha(q) for q in p.parent.glob("*") if q.is_file() and q.name!="source_manifest.json"}
            write_json(p,d)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output",type=Path)
    parser.add_argument("--snapshot-root",type=Path)
    parser.add_argument("--report",type=Path)
    parser.add_argument("--reuse-snapshot",action="store_true")
    args=parser.parse_args();root=args.root.resolve()
    out=(args.output or root/f"results/research_review/{VERSION}").resolve()
    private=(args.snapshot_root or root/f".local/research_review/{VERSION}").resolve()
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"savefig.bbox":"tight","savefig.pad_inches":0.15})
    frozen=snapshot_inputs(root,private,args.reuse_snapshot)
    primary=out/"four_week_primary"
    quiet_call(basic.run,SimpleNamespace(root=root,output=primary/"statistics",snapshot=private/"statistics_snapshot.json",reuse_snapshot=True,source_root=frozen))
    if read_json(primary/"statistics/summary.json")["coverage"]["beimen_primary_n"]!=19:raise ValueError("Expected 19 primary events")
    run_extensions(root,private,primary,private/"statistics_snapshot.json",19)
    sensitivity=out/"eight_week_sensitivity"
    data=eight_week_snapshot(private,frozen,sensitivity/"statistics")
    eight_week_basic(root,private,sensitivity/"statistics",data)
    run_extensions(root,private,sensitivity,private/"eight_week_snapshot.json",21)
    annotate_outputs(root,out)
    report_path=(args.report or root/"docs/research_review/analysis_2026-10-07.md").resolve()
    write_comparison_and_report(root,private,frozen,out,report_path)
    manifest={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"input_manifest_path":str((private/"input_manifest.json").relative_to(root)),"input_manifest_sha256":basic.sha(private/"input_manifest.json"),"scripts":{name:basic.sha(Path(__file__).parent/name) for name in ["refresh_20261007.py","basic_statistics.py","station_time_statistics.py","weather_regression_statistics.py","extended_summary.py"]},"outputs":{p.relative_to(out).as_posix():basic.sha(p) for p in out.rglob("*") if p.is_file() and (p.relative_to(out).parts[0] in ["four_week_primary","eight_week_sensitivity"] or p.name in ["comparison_20261005_vs_20261007.csv","weather_timing_sensitivity.csv","comparison_summary.json","comparison_overview.png"])},"methods":"Only 2026-10-05 existing analyses and previously approved baseline/timing sensitivities", "report_sha256":basic.sha(report_path)}
    write_json(out/"source_manifest.json",manifest)
    print(json.dumps({"output":str(out),"input_snapshot":str(private),"primary_n":19,"eight_week_n":21},ensure_ascii=False,indent=2))


# Report writer is kept below for one-command regeneration of numbers and prose.

def write_comparison_and_report(root,private,frozen,out,report_path=None):
    import os
    report_path=report_path or root/"docs/research_review/analysis_2026-10-07.md"
    scenarios={"2026-10-05 四週13場":frozen/"results/research_review/2026-10-05", "2026-10-07 四週19場":out/"four_week_primary", "2026-10-07 八週21場（敏感度）":out/"eight_week_sensitivity"}
    results=[]; summaries={}
    for label,base in scenarios.items():
        b=read_json(base/"statistics/summary.json");s=read_json(base/"station_time/summary.json");w=read_json(base/"weather_regression/summary.json")
        summaries[label]=(b,s,w)
        north=next(r for r in s["station_descriptive"] if r["station"]=="北門")
        f=b["primary_F_test"];reg=w["baseline_regression"]
        results.append({"scenario":label,"n_events":f["n"],"n_years":len(b["coverage"]["beimen_primary_years"]),"geometric_ratio_plus_one":f["geometric_ratio_plus_one"],"ratio_ci_low":f["geometric_ratio_plus_one_ci95"][0],"ratio_ci_high":f["geometric_ratio_plus_one_ci95"][1],"F":f["F"],"F_p_conditional":f["p"],"sign_p":f["sign_test_two_sided_p"],"friedman_Q":s["station_omnibus"]["friedman_Q"],"kendall_W":s["station_omnibus"]["kendall_W"],"observed_late_share":north["observed_late_share_mean"],"baseline_late_share":north["baseline_late_share_mean"],"late_difference_pp":north["late_share_difference_mean"]*100,"OLS_slope":reg["slope"],"OLS_R2":reg["r_squared_in_sample"],"LOYO_MAE":reg["loyo_mae_log1p"],"LOYO_naive_MAE":reg["loyo_intercept_only_mae_log1p"],"LOYO_RMSE":reg["loyo_rmse_log1p"],"LOYO_naive_RMSE":reg["loyo_intercept_only_rmse_log1p"]})
    comparison=pd.DataFrame(results);csv(out/"comparison_20261005_vs_20261007.csv",comparison)
    timing=[]
    for policy in ["four_week_primary","eight_week_sensitivity"]:
        for mode,folder in [("18..23 原整點標籤","weather_regression"),("19..24 已核對小時終點","weather_aligned")]:
            w=read_json(out/policy/folder/"summary.json")
            for r in w["weather_correlations"]:
                timing.append({"baseline_policy":policy,"weather_timing":mode,"variable":r["variable"],"n":r["n"],"rho":r["rho"],"permutation_p":r["permutation_p"],"holm_two_p":r["holm_two_tests_p"],"year_equal_weight_rho":r["annual_mean_rho_descriptive"],"leave_year_rho_min":r["leave_one_year_rho_range"][0],"leave_year_rho_max":r["leave_one_year_rho_range"][1]})
    csv(out/"weather_timing_sensitivity.csv",pd.DataFrame(timing))
    old=pd.read_csv(frozen/"results/research_review/2026-10-05/statistics/primary_beimen_events.csv",encoding="utf-8-sig")
    new=pd.read_csv(out/"four_week_primary/statistics/primary_beimen_events.csv",encoding="utf-8-sig")
    pair=old.merge(new,on=["event_date","station"],suffixes=("_old","_new"),validate="one_to_one")
    changes={field:int((~np.isclose(pair[field+"_old"],pair[field+"_new"])).sum()) for field in ["origin_count_window","baseline_origin_window","baseline_min_days"]}
    new_dates=sorted(set(new.event_date)-set(old.event_date))
    validation={"unchanged_prior_sample_n":len(pair),"old_sample_value_differences":changes,"added_primary_dates":new_dates,"old_sample_using_latest_input":basic.contrast_summary(pair.log1p_difference_new),"weather_report_hour_end_verified_rows":int(pd.DataFrame(read_json(private/"weather_snapshot.json")["weather"]).values_match.eq(1).sum()),"supplement_counts":[r for r in read_json(private/"supplement_snapshot.json")["records"] if "verification_total" in r]}
    write_json(out/"comparison_summary.json",{"scenarios":results,"weather_timing":timing,"validation":validation,"limitations":"Exploratory; events within years and shared baselines dependent; calendar controls are limited; weather is retrospective; LOYO is not forecasting; no family-wise coverage across all sensitivity runs"})
    # Same visual definitions for all three samples, no p-value-driven selection.
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout="constrained")
    labels=["10/5 四週\n13 場", "10/7 四週\n19 場（主）", "10/7 八週\n21 場（敏感度）"]
    x=np.arange(3);v=comparison.geometric_ratio_plus_one.to_numpy()
    axes[0].errorbar(x,v,yerr=np.vstack([v-comparison.ratio_ci_low.to_numpy(),comparison.ratio_ci_high.to_numpy()-v]),fmt="o",color="#333333",capsize=5)
    axes[0].axhline(1,color="#aaaaaa",linestyle="--");axes[0].set_ylabel("(觀測+1)/(基準+1) 幾何平均")
    axes[0].set_title("北門相對自身基準\n誤差線為常規條件式95% CI")
    axes[1].plot(x,comparison.late_difference_pp,"o-",color="#333333");axes[1].set_ylabel("活動占比 - 基準占比（百分點）");axes[1].set_title("北門21–24時／18–24時\n各場占比差再平均")
    axes[2].plot(x,comparison.LOYO_MAE,"o-",label="單一基準回歸",color="#333333")
    axes[2].plot(x,comparison.LOYO_naive_MAE,"s--",label="同折訓練平均",color="#888888")
    axes[2].set_ylabel("整年留出 MAE（log1p尺度）");axes[2].set_title("回顧性比較\n訓練含未來年份，非forecasting");axes[2].legend(fontsize=8)
    for ax in axes:ax.set_xticks(x,labels);ax.grid(axis="y",alpha=.15)
    fig.suptitle("沿用既有方法：擴充樣本與基準規則敏感度分開解讀")
    fig.savefig(out/"comparison_overview.png",dpi=160);plt.close(fig)
    b,s,w=summaries["2026-10-07 四週19場"];b8,s8,w8=summaries["2026-10-07 八週21場（敏感度）"]
    f=b["primary_F_test"];fy=b["year_equal_weight_sensitivity"];reg=w["baseline_regression"]
    north=next(r for r in s["station_descriptive"] if r["station"]=="北門")
    peak=next(r for r in s["peak_and_time_descriptive"] if r["station"]=="北門")
    e=read_json(out/"four_week_primary/extended_statistics/summary.json")
    e8=read_json(out/"eight_week_sensitivity/extended_statistics/summary.json")
    def link(path):return os.path.relpath(path,report_path.parent).replace("\\","/")
    def p(value):return f"{value:.6g}"
    def interval(values,digits=4):return "–".join(f"{v:.{digits}f}" for v in values)
    compare_table="\n".join(f"| {r['scenario']} | {r['n_events']}/{r['n_years']} | {r['geometric_ratio_plus_one']:.4f} | {r['F']:.4f} | {r['friedman_Q']:.1f}/{r['kendall_W']:.3f} | {r['late_difference_pp']:.2f} | {r['OLS_R2']:.4f} | {r['LOYO_MAE']:.4f}/{r['LOYO_naive_MAE']:.4f} |" for r in results)
    station_table="\n".join(f"| {r['station']} | {r['mean_origin_eligible']:,.1f} | {r['mean_baseline_eligible']:,.1f} | {r['mean_excess_eligible']:,.1f} | {r['geometric_ratio_plus_one']:.4f} |" for r in b["station_descriptive"])
    tests_table="\n".join(f"| {r['question']} | {r['n']} | {p(r['raw_p'])} | {p(r['holm_seven_p'])} |" for r in e["tests"])
    weather_table="\n".join(f"| {'四週19場' if r['baseline_policy']=='four_week_primary' else '八週21場'} | {r['weather_timing']} | {'氣溫' if r['variable'].startswith('temperature') else '風速'} | {r['n']} | {r['rho']:.4f} | {p(r['permutation_p'])} | {p(r['holm_two_p'])} |" for r in timing)
    tests8="\n".join(f"| {r['question']} | {r['n']} | {p(r['raw_p'])} | {p(r['holm_seven_p'])} |" for r in e8["tests"])
    sensitivity_table="\n".join(f"| {'氣溫' if r['variable'].startswith('temperature') else '風速'} | {r['annual_equal_weight_n']} | {r['annual_mean_rho_descriptive']:.4f} | {interval(r['leave_one_year_rho_range'])} |" for r in w["weather_correlations"])
    rain={r["rain_label_class"]:r["events"] for r in w["rain_groups"] if r["scope"]=="primary_19_dates"}
    report=f'''# 2026-10-07：補齊資料後沿用既有方法的完整重跑

**主要比較由四週基準的13場、9年擴充為19場、10年；北門19場都高於自身歷史基準，配對比值幾何平均由3.6297降為{f['geometric_ratio_plus_one']:.4f}。** 四站相對增幅排序仍一致，晚間後段占比高於基準；原兩項天氣關聯仍無清楚證據，單一基準回歸的整年留出誤差仍高於同折訓練平均。本次只是補資料後重跑原方法，並非獨立新樣本確認或因果估計。

| 跟10/5相比變了什麼 | 10/5 | 本次四週主結果 |
|---|---:|---:|
| 可比較活動／年份 | 13場／9年 | 19場／10年 |
| 北門相對歷史基準幾何比值 | 3.6297 | {f['geometric_ratio_plus_one']:.4f} |
| 每場北門是否都高於基準 | 13/13 | 19/19 |
| 北門後段占比平均高出基準 | 31.17百分點 | {north['late_share_difference_mean']*100:.2f}百分點 |
| 四站增幅排序 | 每場一致 | 每場仍一致 |
| 兩项天氣關聯、回歸留年比較 | 無清楚關聯；回歸較差 | 結論未改變 |

完整檢定、區間與敏感度列於後文。八週21場屬基準規則敏感度；不取代四週19場主結果。

## 1. 來源、樣本及本次改動

- 來源：新建唯讀MySQL快照（活動表、四站逐時／視窗、臺北主測站及已核對天氣視圖）；同時凍結本機Parquet、日曆、已核定八週候選日、10/5比較輸出。來源擷取時間、查詢與SHA-256見[总manifest]({link(out/'source_manifest.json')})；私有快照位於`{private.relative_to(root).as_posix()}`，`--reuse-snapshot`完全使用凍結來源。
- 補齊201907、202306、202507後共17個月份。26日期仍為21已舉行、4取消、1延期；四站×26日期×6時段共624列，104個視窗均與逐時總量／基準一致。
- **主規則沿用10/5：**18≤來源時段<24；前7/14/21/28天，同站同時段，排除活動表全部日期，每小時至少2個完整比較日，不補零、不改時段、不加日曆配對。19場中基準2/3/4日分別6/5/8場。2023-08-09、2023-08-16仍因四週候選全落在活動表而不能納入。
- 新增主要日期：{ '、'.join(new_dates) }。舊13場的觀測總量、基準及基準日數變動筆數各為{changes['origin_count_window']}、{changes['baseline_origin_window']}、{changes['baseline_min_days']}；主結果差異因納入額外6場，並非更改舊13場公式。
- **八週是規則改變的敏感度：**沿用已同意的`{POLICY}`，前8個同星期日期、排除全部活動日、相同政府日曆假日／補班類別及註記、取最近至多4個完整日、最低2日，可比較21場；不是把它替換成主要規則。104筆視窗及候選日均以凍結的逐時Parquet重建核對。
- 天氣官方值與來源標籤已有1008筆核對；本次主站504筆全部核對一致。23:59已解釋為翌日00:00的小時終點。主结果仍保留18..23整點；另列19..24小時終點，分開呈現時間定義敏感度。來源時間核對**不代表OD刷卡分箱與氣象曝露已精確對齊**。

同一年可能共享比較日；19場不是19個已證實獨立的實驗，76站別列、456逐時列更不能冒充獨立事件。財務資料的10個預算年、5個宣導年沒有展開為大量獨立樣本，本輪未新增多因子、因果或排隊模型。年度預算、宣導所需經費及其比值另見[年度財務描述表]({link(root/"results/research_review/2026-10-07-refresh/data_inventory/年度預算與宣導經費.csv")})；「所需／原列法定」不是實際支出占比，僅作描述。

## 2. 與10/5的數值對照

| 版本／規則 | 場／年 | 北門幾何比值 | 配對F=t² | Friedman Q／W | 後段占比差（百分點） | OLS R² | 留年MAE：回歸／訓練平均 |
|---|---:|---:|---:|---:|---:|---:|---:|
{compare_table}

比值定義為`exp(mean(log(1+Y)−log(1+B)))`，不是兩個平均相除。晚間後段為來源分箱21、22、23時，占18..23六小時總量；MAE為log1p尺度。[完整CSV]({link(out/'comparison_20261005_vs_20261007.csv')})／[JSON及舊樣本核對]({link(out/'comparison_summary.json')})。

![各版本比較]({link(out/'comparison_overview.png')})

## 3. 四週19場主結果：配對增幅、年份敏感度

對每場北門計算`d=log(1+Y)−log(1+B)`，既有單樣本t之平方F檢查`E(d)=0`。不是將觀測與基準當成38個獨立樣本。

| 指標 | 19場主結果 |
|---|---:|
| 平均log1p差 | {f['mean_log1p_difference']:.6f} |
| F(1,18)，條件式雙尾p | {f['F']:.6f}；{p(f['p'])} |
| 幾何比值、常規95% CI | {f['geometric_ratio_plus_one']:.4f}；{interval(f['geometric_ratio_plus_one_ci95'])} |
| 正／零差場數，符號檢定p | {f['positive']}／{f['zero']}；{p(f['sign_test_two_sided_p'])} |
| 10年等權：幾何比值、95% CI | {fy['geometric_ratio_plus_one']:.4f}；{interval(fy['geometric_ratio_plus_one_ci95'])} |
| 10年等權：F(1,9)、條件式p | {fy['F']:.4f}；{p(fy['p'])} |
| 逐次移除一整年：幾何比值範圍 | {interval(b['diagnostics']['leave_one_year_geometric_ratio_range'])} |
| 年均值bootstrap：比值百分位95%区間 | {interval(b['year_bootstrap']['geometric_ratio_plus_one_ci95'])} |
| Shapiro–Wilk W、p | {b['diagnostics']['shapiro_W']:.4f}；{p(b['diagnostics']['shapiro_p'])} |

F與CI需要事件差值獨立、可比且近似常態等工作假設；目前未證實。符號檢定改測方向機率而非平均幅度，仍不能解決同年／共用基準相依。年度等權改變估計目標，留年與固定種子20261005、20,000次年度bootstrap均屬敏感度；10年不足以保證區間涵蓋率。基準僅2–4日，其估計誤差未另建模型。[JSON]({link(out/'four_week_primary/statistics/summary.json')})／[19場CSV]({link(out/'four_week_primary/statistics/primary_beimen_events.csv')})／[留年CSV]({link(out/'four_week_primary/statistics/leave_one_year_out.csv')})。

![北門觀測與基準]({link(out/'four_week_primary/statistics/01_beimen_event_totals.png')})

## 4. 四站、晚間占比與固定七項Holm

同19場四站的18–24時總量描述如下；差值為觀測減自身歷史基準，未扣除其他同日因素。

| 站名 | 觀測平均人次 | 基準平均人次 | 平均差值 | 幾何比值 |
|---|---:|---:|---:|---:|
{station_table}

Friedman Q={s['station_omnibus']['friedman_Q']:.1f}、Kendall W={s['station_omnibus']['kendall_W']:.3f}，19場相對增幅皆為北門＞大橋頭站＞雙連＞民權西路。199,999次場內標籤置換p={p(s['station_omnibus']['permutation']['p_plus_one'])}為本次解析度下限，不是精確尾機率或p=0；W為排名一致度，不是人流解釋比例。固定站別的可交換性未經隨機設計保證。

北門對各其他站皆19正、0負、0零。北門後段占比平均{north['observed_late_share_mean']:.2%}，基準{north['baseline_late_share_mean']:.2%}，平均配對差{north['late_share_difference_mean']*100:.2f}個百分點；19場均正。全21場描述中，北門{peak['peak_includes_21_events']}/21場峰在21時，最大一小時占六小時總量中位數{peak['peak_share_median']:.2%}。來源時段不能當作實際展演結束或候車時間。

| 既有固定問題 | 場／非零配對數 | 原始p | 七項Holm p |
|---|---:|---:|---:|
{tests_table}

七項為四站整體1項、北門對三站3項、北門占比1項、原時段天氣2項；**未包含基本F、OLS係數、年度敏感度、八週規則與另一時間對齊**，不可聲稱涵蓋所有研究嘗試。Holm無法修復單一p因相依、混雜或不可交換而失效。所有結果仍探索性。[七項CSV]({link(out/'four_week_primary/extended_statistics/tests.csv')})／[四站JSON含年度結果]({link(out/'four_week_primary/station_time/summary.json')})／[配對CSV]({link(out/'four_week_primary/station_time/station_pairwise_sign.csv')})／[尖峰CSV]({link(out/'four_week_primary/station_time/station_peak_descriptive.csv')})。

![四站相對增幅]({link(out/'four_week_primary/station_time/01_paired_station_contrasts.png')})
![北門後段占比]({link(out/'four_week_primary/station_time/03_beimen_late_share_pairs.png')})

## 5. 天氣：維持原時段，另列已核對小時終點

來源是臺北主測站466920，並非現地量測；21個已舉行日期各有24筆主站觀測，兩個固定時窗各選6點，氣溫／風速均需6點有效才摘要。氣溫是各觀測時點值；風速為六個「整點前10分鐘均值」之平均，**不是連續六小時平均風速**。19..24把原18:00換為日末23:59（正規化翌日00:00）；21場兩個變數的摘要皆改變。

兩項Spearman以`abs(rho)`作雙尾配對置換，99,999次、原種子20261005／20261006、加一p；最低10個有效配對且各有至少5個不同值。仍以北門log1p相對增幅為反應量，沒有改變變數或按p選時段。

| 基準 | 氣象時窗 | 變數 | n | rho | 置換p | 本時窗兩項Holm p |
|---|---|---|---:|---:|---:|---:|
{weather_table}

19場原時段的年度及留年描述如下；年度均值相關不是事件層相關，同年與季節差異仍可能主導。

| 變數 | 年數 | 年平均相關rho | 逐次留年rho範圍 |
|---|---:|---:|---:|
{sensitivity_table}

各情境尚無清楚單調關聯證據，不能宣称天氣無影響；置換仍依賴事件間可交換性。這是事後天氣，不能當作活動前已知預測。雨量T保留微量有雨，&保留後續累積而不補零；主19場原時段有雨證據{rain.get('wet_evidence',0)}場、六點數值零{rain.get('all_six_labels_numeric_zero',0)}場。取消／延期5日沒有主站配對，不能用此涵蓋解釋取消原因。雲量只描述核對與覆蓋，不加新檢定。

[時段敏感度CSV]({link(out/'weather_timing_sensitivity.csv')})／[原時段JSON]({link(out/'four_week_primary/weather_regression/summary.json')})／[19–24終點JSON]({link(out/'four_week_primary/weather_aligned/summary.json')})／[逐場天氣CSV]({link(out/'four_week_primary/weather_aligned/weather_event_summary.csv')})。

![原時段天氣關聯]({link(out/'four_week_primary/weather_regression/01_weather_associations.png')})
![已核對終點敏感度]({link(out/'four_week_primary/weather_aligned/01_weather_associations.png')})

## 6. 原單一回歸與整年留出

只擬合`log(1+Y)=alpha+beta log(1+B)+error`，n=19、殘差自由度17。截距{reg['intercept']:.6f}；斜率{reg['slope']:.6f}、條件95% CI {interval(reg['slope_ci95'])}，樣本內R²={reg['r_squared_in_sample']:.4f}。CI將歷史基準視為固定且未加其估計誤差，假定獨立、近似常態與等變異殘差。Shapiro殘差p={p(reg['shapiro_residual_p_diagnostic'])}，不能由未拒絕推得模型已驗證；最大Cook距離{reg['max_cooks_distance']:.4f}為{reg['max_cooks_event']}，沒有據此刪樣本。

| 19筆整年留出結果（log1p尺度） | MAE | RMSE |
|---|---:|---:|
| 原單一基準回歸 | {reg['loyo_mae_log1p']:.6f} | {reg['loyo_rmse_log1p']:.6f} |
| 同折訓練活動平均 | {reg['loyo_intercept_only_mae_log1p']:.6f} | {reg['loyo_intercept_only_rmse_log1p']:.6f} |

回歸仍未優於簡單參考；逐折斜率範圍{interval(reg['loyo_slope_range'])}。整年留出防止同年事件同時進入訓練／測試，但較早測試年仍使用較晚資料，**是回顧性敏感度，不是未來預測回測**。未把exp回轉當作條件平均人次。[模型JSON]({link(out/'four_week_primary/weather_regression/summary.json')})／[逐折CSV]({link(out/'four_week_primary/weather_regression/regression_loyo.csv')})／[診斷CSV]({link(out/'four_week_primary/weather_regression/regression_event_diagnostics.csv')})。

![回歸與整年留出]({link(out/'four_week_primary/weather_regression/02_baseline_regression.png')})
![殘差診斷]({link(out/'four_week_primary/weather_regression/03_regression_diagnostics.png')})

## 7. 八週21場僅作同方法敏感度

八週相同日曆規則把另外2場纳入且可能改變其他場基準。21場北門皆正，比值{b8['primary_F_test']['geometric_ratio_plus_one']:.4f}，常規95% CI {interval(b8['primary_F_test']['geometric_ratio_plus_one_ci95'])}；F(1,20)={b8['primary_F_test']['F']:.4f}、条件p={p(b8['primary_F_test']['p'])}。年度等權比值{b8['year_equal_weight_sensitivity']['geometric_ratio_plus_one']:.4f}，移除一年範圍{interval(b8['diagnostics']['leave_one_year_geometric_ratio_range'])}。Friedman Q={s8['station_omnibus']['friedman_Q']:.1f}、W={s8['station_omnibus']['kendall_W']:.3f}。完整年度結果、配對符號CI、時段、天氣與OLS皆按原方法重算。

| 同七問題在八週情境中的敏感度 | n | 原始p | 情境內七項Holm p |
|---|---:|---:|---:|
{tests8}

本表的七項校正只限八週情境自身，不把它視為新增七項獨立驗證或所有敏感度的全域校正。政府办公日分類不等於所有人工作日，八週仍有季節趨勢、未列活動、同期節慶和共享基準；不能只因样本增加就選為唯一結論。[八週JSON]({link(out/'eight_week_sensitivity/statistics/summary.json')})／[21場CSV]({link(out/'eight_week_sensitivity/statistics/primary_beimen_events.csv')})／[候選日稽核]({link(out/'eight_week_sensitivity/statistics/baseline_candidates.csv')})／[逐時重建來源]({link(out/'eight_week_sensitivity/statistics/selected_baseline_source_hours.csv')})／[七項CSV]({link(out/'eight_week_sensitivity/extended_statistics/tests.csv')})。

## 8. 重跑與交付

在專案根目錄使用現有環境，不新增套件、不寫MySQL；未覆寫10/5或今日稍早10/7成果。

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 scripts/research_review/refresh_20261007.py --reuse-snapshot
# 若要另建一份當下來源，指定新的私有來源與輸出路徑，省略 --reuse-snapshot：
.\\.venv\\Scripts\\python.exe -X utf8 scripts/research_review/refresh_20261007.py --snapshot-root .local/research_review/my-new-run --output results/research_review/my-new-run --report .local/research_review/my-new-run/report.md
```

[重跑程式]({link(root/'scripts/research_review/refresh_20261007.py')})沿用既有四支統計模組，保留舊預設；新版路徑／n可指定。每個模組有CSV、JSON、圖與來源manifest；全套來源檔凍結與hash防止重現時靜默讀到改版輸入。新結果是探索性關聯：不能等同現地觀眾、站間個人移動、排隊長度、煙火因果倍數或已驗證的營運預測。

方法依據與10/5相同：[SciPy單樣本t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html)、[Friedman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html)、[精確二項檢定](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)、[置換](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)、[OLS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html)、[Holm](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)。本輪未引入新的網路方法。
'''
    # Keep the requested Traditional Chinese terminology consistent.
    for a,z in [("总manifest","總manifest"),("主结果","主結果"),("区間","區間"),("宣称","宣稱"),("条件","條件"),("纳入","納入"),("办公日","辦公日"),("样本","樣本"),("两项","兩項"),("兩项","兩項")]:report=report.replace(a,z)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(report,encoding="utf-8")


if __name__ == "__main__":
    main()
