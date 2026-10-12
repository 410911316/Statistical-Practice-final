"""Reproducible exploratory event-level statistics; MySQL reads are read-only.

Fixed before outcome inspection: held events, Beimen, 18 <= source_hour < 24,
six complete hours and >=2 historical comparison days in every hour. Primary
contrast is log1p(observed window)-log1p(baseline window), H0: E[contrast]=0.
The single conventional F test is the one-sample paired-contrast test F=t**2.
It is exploratory: overlapping baselines/year effects can violate independence.
Other stations are descriptive only. No model selection, no null-to-zero fill.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

STATIONS = ["北門", "大橋頭站", "雙連", "民權西路"]
MIN_BASELINE_DAYS = 2
SEED = 20261005
SOURCE_URLS = {
    "ttest_1samp": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html",
    "f_test": "https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.f_test.html",
    "binomtest": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html",
    "shapiro": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html",
}
DB = "taipei_metro_analysis"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sql_queries() -> str:
    station_sql = ",".join("CONVERT(0x" + s.encode("utf-8").hex() + " USING utf8mb4)" for s in STATIONS)
    queries = []
    specs = [
        ("events", "dadaocheng_fireworks_events", ["event_date", "event_name", "event_status", "announced_show_start_time", "fireworks_duration_seconds", "has_drone_show", "not_held_reason", "transferred_to_date", "official_source_url"], "", "event_date"),
        ("windows", "v_regression_event_station", ["event_date", "station", "window_start_hour", "window_end_hour_exclusive", "expected_hours", "observed_origin_hours", "observed_destination_hours", "origin_count_window", "destination_count_window", "peak_origin_hour_count", "baseline_origin_window", "baseline_min_days", "origin_window_complete", "source_run_sha256", "updated_at", "observed_event_status", "trends_missing", "trends_cross_event_calibrated"], f"WHERE station IN ({station_sql}) AND window_start_hour=18 AND window_end_hour_exclusive=24", "event_date,station"),
        ("hourly", "analysis_event_station_hourly", ["event_date", "hour_start", "station", "origin_count", "destination_count_same_source_bin", "origin_complete", "destination_complete", "baseline_origin_mean_28d", "baseline_n_days", "source_run_sha256"], f"WHERE station IN ({station_sql}) AND hour_start>=TIMESTAMPADD(HOUR,18,CAST(event_date AS DATETIME)) AND hour_start<TIMESTAMPADD(HOUR,24,CAST(event_date AS DATETIME))", "event_date,hour_start,station"),
    ]
    for kind, table, columns, where, order in specs:
        pairs = ",".join(f"'{c}',{c}" for c in columns)
        queries.append(f"SELECT JSON_OBJECT('kind','{kind}','record',JSON_OBJECT({pairs})) FROM {DB}.{table} {where} ORDER BY {order};")
    queries.append(f"""SELECT JSON_OBJECT('kind','network_totals','record',JSON_OBJECT(
      'event_date',event_date, 'origin_total',SUM(origin_count),
      'destination_total',SUM(destination_count_same_source_bin),
      'rows',COUNT(*),'origin_nonnull',COUNT(origin_count),'destination_nonnull',COUNT(destination_count_same_source_bin),
      'origin_incomplete',SUM(origin_complete=0),'destination_incomplete',SUM(destination_complete=0)))
      FROM {DB}.analysis_event_station_hourly
      WHERE hour_start>=TIMESTAMPADD(HOUR,18,CAST(event_date AS DATETIME))
      AND hour_start<TIMESTAMPADD(HOUR,24,CAST(event_date AS DATETIME)) GROUP BY event_date ORDER BY event_date;""")
    return "\n".join(queries)


def extract(root: Path, snapshot: Path, reuse: bool) -> dict:
    if reuse:
        return json.loads(snapshot.read_text(encoding="utf-8"))
    sys.path.insert(0, str(root))
    from app.backend.database import _read_json
    query = sql_queries()
    rows = _read_json(query)
    result = {k: [] for k in ["events", "windows", "hourly", "network_totals"]}
    for row in rows:
        result[row["kind"]].append(row["record"])
    result["extracted_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["sql_sha256"] = hashlib.sha256(query.encode("utf-8")).hexdigest()
    write_json(snapshot, result)
    return result


def contrast_summary(values) -> dict:
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 3 or not np.isfinite(x).all() or np.std(x, ddof=1) == 0:
        raise ValueError("Need >=3 finite nonconstant independent units for exploratory test")
    t = stats.ttest_1samp(x, 0, alternative="two-sided", nan_policy="raise")
    f = float(t.statistic**2)
    ci = t.confidence_interval(confidence_level=0.95)
    assert np.isclose(stats.f.sf(f, 1, n-1), t.pvalue, rtol=1e-9)
    nonzero = x[x != 0]
    sign = stats.binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5, alternative="two-sided")
    return dict(n=n, mean_log1p_difference=float(x.mean()), sd_log1p_difference=float(x.std(ddof=1)),
                t=float(t.statistic), F=f, df_num=1, df_denom=n-1, p=float(t.pvalue),
                mean_log1p_difference_ci95=[float(ci.low), float(ci.high)],
                geometric_ratio_plus_one=float(np.exp(x.mean())),
                geometric_ratio_plus_one_ci95=[float(np.exp(ci.low)), float(np.exp(ci.high))],
                paired_standardized_mean=float(x.mean()/x.std(ddof=1)),
                positive=int((x>0).sum()), zero=int((x==0).sum()), sign_test_two_sided_p=float(sign.pvalue))


def prepare(data: dict):
    events, windows, hourly = (pd.DataFrame(data[k]) for k in ["events", "windows", "hourly"])
    for df in [events, windows, hourly]:
        df["event_date"] = pd.to_datetime(df.event_date).dt.strftime("%Y-%m-%d")
    assert not events.duplicated(["event_date"]).any()
    assert not windows.duplicated(["event_date","station"]).any()
    assert not hourly.duplicated(["event_date","hour_start","station"]).any()
    assert set(windows.station) == set(STATIONS)
    hourly["hour"] = pd.to_datetime(hourly.hour_start).dt.hour
    hourly["origin_count"] = pd.to_numeric(hourly.origin_count)
    hourly["baseline_origin_mean_28d"] = pd.to_numeric(hourly.baseline_origin_mean_28d)
    numeric = ["origin_count_window", "destination_count_window", "peak_origin_hour_count", "baseline_origin_window", "baseline_min_days", "observed_origin_hours", "expected_hours"]
    for column in numeric:
        windows[column] = pd.to_numeric(windows[column])
    joined = windows.merge(events[["event_date","event_status","fireworks_duration_seconds"]], on="event_date", validate="many_to_one")
    assert (joined.observed_event_status == joined.event_status).all()
    checks = []
    for (date, station), group in hourly.groupby(["event_date","station"], sort=True):
        window = joined.loc[(joined.event_date==date)&(joined.station==station)].iloc[0]
        complete = set(group.hour)==set(range(18,24)) and group.origin_count.notna().all() and group.origin_complete.eq(1).all()
        total = float(group.origin_count.sum()) if complete else np.nan
        b_complete = len(group)==6 and group.baseline_origin_mean_28d.notna().all()
        baseline = float(group.baseline_origin_mean_28d.sum()) if b_complete else np.nan
        def equal_or_missing(a, b):
            return (pd.isna(a) and pd.isna(b)) or np.isclose(a,b,rtol=1e-10,atol=1e-6)
        ok = bool(equal_or_missing(total,window.origin_count_window) and equal_or_missing(baseline,window.baseline_origin_window) and complete==bool(window.origin_window_complete))
        if not ok:
            raise ValueError(f"Hourly/window mismatch {date} {station}")
        if complete and not np.isclose(group.origin_count.max(),window.peak_origin_hour_count):
            raise ValueError(f"Peak mismatch {date} {station}")
        checks.append(dict(event_date=date,station=station,hourly_rows=len(group),complete=complete,
                           hourly_origin_sum=total,window_origin_sum=window.origin_count_window,
                           min_baseline_days=int(group.baseline_n_days.min()),hourly_window_match=ok))
    checks = pd.DataFrame(checks)
    assert len(checks)==len(joined)==len(events)*4
    assert np.array_equal(checks.sort_values(["event_date","station"]).min_baseline_days.to_numpy(),joined.sort_values(["event_date","station"]).baseline_min_days.to_numpy())
    joined["year"] = pd.to_datetime(joined.event_date).dt.year
    joined["held_complete"] = joined.event_status.eq("held") & joined.origin_window_complete.eq(1) & joined.origin_count_window.notna()
    joined["eligible"] = joined.held_complete & joined.baseline_min_days.ge(MIN_BASELINE_DAYS) & joined.baseline_origin_window.notna()
    joined["exclusion_reason"] = np.select([~joined.event_status.eq("held"),~joined.origin_window_complete.eq(1)|joined.origin_count_window.isna(),joined.baseline_origin_window.isna(),joined.baseline_min_days.lt(MIN_BASELINE_DAYS)], ["cancelled_or_postponed","incomplete_window","missing_baseline","fewer_than_2_baseline_days"], default="")
    joined["excess_vs_baseline"] = joined.origin_count_window-joined.baseline_origin_window
    joined["log1p_difference"] = np.log1p(joined.origin_count_window)-np.log1p(joined.baseline_origin_window)
    joined["ratio_plus_one"] = (joined.origin_count_window+1)/(joined.baseline_origin_window+1)
    return events, joined, hourly, checks


def validate_baseline_candidates(candidates: pd.DataFrame) -> None:
    """Window-mean reconstruction is valid only for all-six-or-no-hour candidates."""
    partial = candidates.loc[~candidates.complete_origin_hours.isin([0, 6])]
    if not partial.empty:
        row = partial.iloc[0]
        raise ValueError(
            "Partial baseline candidate: "
            f"event_date={row.event_date}, station={row.station}, "
            f"candidate_date={row.candidate_date}, "
            f"complete_origin_hours={int(row.complete_origin_hours)}; "
            "expected 0 or 6. Stop: unequal-hour baselines require per-hour reconstruction."
        )


def self_test_partial_baseline() -> dict:
    # Realistic missing-hour scenario: aggregate six source-hour rows, then remove
    # the 20:00 row. Missing is not a zero and cannot pass the window-level audit.
    hours = pd.DataFrame({"source_hour":range(18,24), "origin_complete":[True]*6})
    candidate = {"event_date":"2017-08-26", "station":"北門", "candidate_date":"2017-08-19"}
    complete = pd.DataFrame([{**candidate,"complete_origin_hours":int(hours.origin_complete.sum())}])
    validate_baseline_candidates(complete)
    incomplete_hours = hours.loc[hours.source_hour.ne(20)]
    partial = pd.DataFrame([{**candidate,"complete_origin_hours":int(incomplete_hours.origin_complete.sum())}])
    try:
        validate_baseline_candidates(partial)
    except ValueError as exc:
        message = str(exc)
        if "complete_origin_hours=5" not in message or "2017-08-19" not in message:
            raise AssertionError("Partial-baseline error omitted the failed candidate") from exc
        return {"complete_six_hour_candidate":"accepted", "removed_source_hour":20,
                "partial_five_hour_candidate":"rejected", "error_message":message}
    raise AssertionError("Five-hour baseline candidate was silently accepted")


def compare_local(root, data, joined, hourly, source_root=None):
    """Read existing Parquet via DuckDB; compare values and live event metadata."""
    import duckdb
    conn = duckdb.connect()
    source_root = source_root or root
    paths = [source_root/"data/processed/regression_event_station.parquet",source_root/"data/processed/station_hourly.parquet",source_root/"results/preprocessing/summary.json"]
    info = [{"path":str(p.relative_to(root)),"bytes":p.stat().st_size,"modified_at_utc":datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),"sha256":sha(p)} for p in paths if p.exists()]
    local = conn.execute("SELECT * FROM read_parquet(?)",[str(paths[0])]).df()
    local["event_date"] = pd.to_datetime(local.event_date).dt.strftime("%Y-%m-%d")
    local = local.loc[local.station.isin(STATIONS)&local.window_start_hour.eq(18)&local.window_end_hour_exclusive.eq(24)]
    merged = joined.merge(local,on=["event_date","station"],suffixes=("_db","_file"),validate="one_to_one")
    fields = ["origin_count_window","baseline_origin_window","baseline_min_days","observed_event_status","trends_missing"]
    diffs = {}
    for field in fields:
        a,b=merged[field+"_db"],merged[field+"_file"]
        if field in {"observed_event_status","trends_missing"}:
            same=(a==b)|(a.isna()&b.isna())
        else:
            same=np.isclose(a.astype(float),b.astype(float),equal_nan=True)
        diffs[field]=int((~same).sum())
    dates = list(pd.DataFrame(data["events"]).event_date)
    conn.register("event_dates",pd.DataFrame({"event_date":pd.to_datetime(dates)}))
    totals=conn.execute("""SELECT CAST(service_date AS VARCHAR) AS event_date,
      SUM(origin_count) AS local_origin_total,SUM(destination_count_same_source_bin) AS local_destination_total
      FROM read_parquet(?) s JOIN event_dates e ON CAST(s.service_date AS DATE)=CAST(e.event_date AS DATE)
      WHERE source_hour>=18 AND source_hour<24 GROUP BY service_date ORDER BY service_date""",[str(paths[1])]).df()
    totals.event_date=totals.event_date.str[:10]
    network=pd.DataFrame(data["network_totals"])
    network = network.rename(columns={"origin_total":"available_origin_sum",
                                      "destination_total":"available_destination_sum"})
    totals = totals.rename(columns={"local_origin_total":"local_available_origin_sum",
                                    "local_destination_total":"local_available_destination_sum"})
    network.event_date=pd.to_datetime(network.event_date).dt.strftime("%Y-%m-%d")
    network=network.merge(totals,on="event_date",validate="one_to_one")
    network["origin_null_rows"] = network.rows-network.origin_nonnull
    network["destination_null_rows"] = network.rows-network.destination_nonnull
    network["all_origin_label_hours_complete"] = network.origin_incomplete.eq(0)&network.origin_null_rows.eq(0)
    network["all_destination_label_hours_complete"] = network.destination_incomplete.eq(0)&network.destination_null_rows.eq(0)
    network["all_label_hours_complete"] = network.all_origin_label_hours_complete&network.all_destination_label_hours_complete
    network["origin_equals_destination"]=network.available_origin_sum.eq(network.available_destination_sum)
    network["db_equals_processed"]=network.available_origin_sum.eq(network.local_available_origin_sum)&network.available_destination_sum.eq(network.local_available_destination_sum)
    if not network.origin_equals_destination.all() or not network.db_equals_processed.all() or len(network)!=len(dates):
        raise ValueError("Network available-row event window reconciliation failed")
    pre=json.loads(paths[2].read_text(encoding="utf-8"))
    candidates = pd.DataFrame([
        {"event_date":date,"station":station,"lag_days":lag,
         "candidate_date":pd.Timestamp(date)-pd.Timedelta(days=lag)}
        for date in dates for station in STATIONS for lag in [7,14,21,28]])
    conn.register("candidates",candidates)
    candidates = conn.execute("""SELECT c.event_date,c.station,c.lag_days,
        CAST(c.candidate_date AS DATE) candidate_date,COUNT(s.source_hour) observed_hours,
        COUNT(s.origin_count) observed_origin_hours,
        SUM(CASE WHEN s.origin_complete THEN 1 ELSE 0 END) complete_origin_hours,
        SUM(s.origin_count) candidate_origin_total
        FROM candidates c LEFT JOIN read_parquet(?) s
          ON CAST(c.candidate_date AS DATE)=s.service_date AND c.station=s.station
          AND s.source_hour>=18 AND s.source_hour<24
        GROUP BY c.event_date,c.station,c.lag_days,c.candidate_date
        ORDER BY c.event_date,c.station,c.lag_days""",[str(paths[1])]).df()
    candidates["candidate_date"]=pd.to_datetime(candidates.candidate_date).dt.strftime("%Y-%m-%d")
    candidates["excluded_event_date"]=candidates.candidate_date.isin(dates)
    candidates["available_month"]=pd.to_datetime(candidates.candidate_date).dt.strftime("%Y%m").isin(pre["months"])
    candidates["usable"]=~candidates.excluded_event_date & candidates.observed_origin_hours.eq(6) & candidates.complete_origin_hours.eq(6)
    candidates["reason"]=np.select([candidates.excluded_event_date,~candidates.available_month,~candidates.usable], ["event_date_excluded","month_not_downloaded","incomplete_or_missing_station_hours"], default="usable")
    candidates.loc[~candidates.complete_origin_hours.eq(6),"candidate_origin_total"]=np.nan
    validate_baseline_candidates(candidates)
    for (date,station),group in candidates.groupby(["event_date","station"]):
        row=joined.loc[joined.event_date.eq(date)&joined.station.eq(station)].iloc[0]
        # In this snapshot each candidate is either complete for all six hours or absent.
        # Stop rather than silently reconstruct an unequal-hour baseline incorrectly.
        assert int(group.usable.sum())==int(row.baseline_min_days)
        expected=group.loc[group.usable,"candidate_origin_total"].mean()
        assert (pd.isna(expected) and pd.isna(row.baseline_origin_window)) or np.isclose(expected,row.baseline_origin_window)
    conn.close()
    return {"files":info,"selected_rows_compared":len(merged),"snapshot_value_differences":diffs,
            "preprocessing_generated_at_utc":pre["generated_at_utc"],"preprocessing_all_month_totals":{k:pre[k] for k in ["od_passengers","origin_passengers","destination_passengers","source_rows","clean_rows"]},
            "prior_all_month_reconciliation_not_recomputed":True},network,candidates


def plots(out, joined, hourly, primary):
    font_path=Path("C:/Windows/Fonts/msjh.ttc")
    if font_path.exists():
        plt.rcParams["font.family"]=FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"axes.unicode_minus":False,"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":130,"savefig.dpi":160})
    b=joined.loc[joined.station.eq("北門")&joined.held_complete].sort_values("event_date")
    fig,ax=plt.subplots(figsize=(10,7),layout="constrained")
    y=np.arange(len(b))
    ax.barh(y,b.origin_count_window,color="#30343b",height=.63,label="活動日觀測")
    ax.scatter(b.baseline_origin_window,y,color="#cf6b32",s=32,label="歷史基準",zorder=3)
    ax.set_yticks(y,b.event_date+np.where(b.eligible,""," *"))
    ax.invert_yaxis(); ax.set_xlabel("18–24 時起站 OD 人次")
    ax.set_title("北門：已舉行活動的晚間總量與歷史基準\n* 基準不足 2 日，僅描述、未納入主要檢定",loc="left")
    ax.legend(loc="lower right"); ax.grid(axis="x",alpha=.15)
    fig.savefig(out/"01_beimen_event_totals.png"); plt.close(fig)
    eligible_dates=set(joined.loc[joined.station.eq("北門")&joined.eligible,"event_date"])
    profile=hourly.loc[hourly.event_date.isin(eligible_dates)&hourly.origin_complete.eq(1)].copy()
    profile_stats=profile.groupby(["station","hour"]).origin_count.agg(n="count",median="median",q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75)).reset_index()
    profile_stats.to_csv(out/"hourly_profile.csv",index=False,encoding="utf-8-sig")
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True,layout="constrained")
    for station,ax in zip(STATIONS,axes.flat):
        g=profile_stats.loc[profile_stats.station.eq(station)]
        ax.plot(g.hour,g["median"],color="#30343b",marker="o")
        ax.fill_between(g.hour,g.q25,g.q75,color="#9babb7",alpha=.45)
        ax.set_title(station); ax.set_xticks(range(18,24)); ax.grid(alpha=.16)
        ax.set_xlabel("來源時段"); ax.set_ylabel("起站 OD 人次")
    fig.suptitle(f"四站同場配對描述（{len(eligible_dates)} 場）｜中位數與四分位範圍，非信賴區間")
    fig.savefig(out/"02_station_hour_profiles.png"); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout="constrained")
    stats.probplot(primary.log1p_difference.to_numpy(),dist="norm",plot=axes[0])
    axes[0].set_title("主要對比的常態 Q–Q 圖"); axes[0].set_xlabel("理論常態分位數"); axes[0].set_ylabel("事件 log1p 差")
    axes[1].scatter(pd.to_datetime(primary.event_date),primary.log1p_difference,color="#30343b")
    axes[1].axhline(0,color="#999",linestyle="--")
    axes[1].set_title("各事件 log1p 差與年份"); axes[1].set_ylabel("log(觀測+1) - log(基準+1)")
    axes[1].tick_params(axis="x",rotation=30)
    fig.savefig(out/"03_contrast_diagnostics.png"); plt.close(fig)


def run(args):
    root=args.root.resolve(); out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    snapshot=args.snapshot.resolve() if args.snapshot else root/".local/research_review/2026-10-05/statistics_snapshot.json"
    data=extract(root,snapshot,args.reuse_snapshot)
    events,joined,hourly,checks=prepare(data)
    local,network,candidates=compare_local(root,data,joined,hourly,getattr(args,"source_root",None))
    primary=joined.loc[joined.station.eq("北門")&joined.eligible].sort_values("event_date").copy()
    primary_result=contrast_summary(primary.log1p_difference)
    year_values=primary.groupby("year").log1p_difference.mean()
    yearly_sensitivity=contrast_summary(year_values)
    loo=[]
    for year in sorted(primary.year.unique()):
        sub=primary.loc[primary.year.ne(year)]
        loo.append(dict(excluded_year=int(year),remaining_events=len(sub),mean_log1p_difference=float(sub.log1p_difference.mean()),geometric_ratio_plus_one=float(np.exp(sub.log1p_difference.mean()))))
    rng=np.random.default_rng(SEED)
    # Resample whole years with equal year weights; exploratory percentile CI.
    boot=rng.choice(year_values.to_numpy(),size=(20000,len(year_values)),replace=True).mean(axis=1)
    year_boot_ci=np.quantile(boot,[.025,.975]).tolist()
    desc=[]
    for station in STATIONS:
        all_held=joined.loc[joined.station.eq(station)&joined.held_complete]
        paired=joined.loc[joined.station.eq(station)&joined.event_date.isin(primary.event_date)]
        assert len(paired)==len(primary) and paired.eligible.all()
        desc.append(dict(station=station,held_complete_events=len(all_held),eligible_events=len(paired),
           median_origin_all_held=float(all_held.origin_count_window.median()),mean_origin_all_held=float(all_held.origin_count_window.mean()),
           median_origin_eligible=float(paired.origin_count_window.median()),mean_origin_eligible=float(paired.origin_count_window.mean()),
           mean_baseline_eligible=float(paired.baseline_origin_window.mean()),mean_excess_eligible=float(paired.excess_vs_baseline.mean()),
           median_peak_hour_count=float(paired.peak_origin_hour_count.median()),geometric_ratio_plus_one=float(np.exp(paired.log1p_difference.mean()))))
    counts=events.assign(year=pd.to_datetime(events.event_date).dt.year).groupby(["year","event_status"]).size().unstack(fill_value=0).reset_index()
    for c in ["held","cancelled","postponed"]:
        if c not in counts: counts[c]=0
    counts["eligible_beimen_events"]=counts.year.map(primary.groupby("year").size()).fillna(0).astype(int)
    for name,df in [("event_station_summary",joined),("primary_beimen_events",primary),("excluded_beimen_events",joined.loc[joined.station.eq("北門")&~joined.eligible]),("cancelled_postponed",joined.loc[~joined.event_status.eq("held")]),("hourly_window_reconciliation",checks),("network_reconciliation",network),("baseline_candidates",candidates),("year_counts",counts),("station_descriptive",pd.DataFrame(desc)),("leave_one_year_out",pd.DataFrame(loo)),("year_mean_contrasts",year_values.rename("mean_log1p_difference").reset_index())]:
        df.to_csv(out/f"{name}.csv",index=False,encoding="utf-8-sig",na_rep="")
    shapiro=stats.shapiro(primary.log1p_difference)
    summary={"design":{"primary_station":"北門","stations":STATIONS,"hours":[18,24],"minimum_baseline_days_per_hour":MIN_BASELINE_DAYS,"fixed_test":"two-sided event paired log1p contrast; F=t^2", "independent_working_unit":"event_date", "inference_status":"exploratory; event independence and common distribution not assured", "seed":SEED},
       "coverage":{"events":len(events),"status_counts":events.event_status.value_counts().to_dict(),"selected_window_rows":len(joined),"selected_hourly_rows":len(hourly),"complete_origin_window_rows":int(joined.origin_window_complete.sum()),"beimen_primary_n":len(primary),"beimen_primary_dates":list(primary.event_date),"beimen_primary_years":list(map(int,year_values.index)),"eligible_min_baseline_days_counts":{str(int(k)):int(v) for k,v in primary.baseline_min_days.value_counts().items()},"hourly_missing_origin":int(hourly.origin_count.isna().sum()),"hourly_missing_baseline":int(hourly.baseline_origin_mean_28d.isna().sum())},
       "primary_F_test":primary_result,"year_equal_weight_sensitivity":yearly_sensitivity,
       "year_bootstrap":{"unit":"year mean, each year equally weighted","replications":20000,"mean_log1p_difference_percentile_ci95":year_boot_ci,"geometric_ratio_plus_one_ci95":np.exp(year_boot_ci).tolist(),"warning":"few years; exploratory, not a guarantee of valid coverage"},
       "diagnostics":{"shapiro_W":float(shapiro.statistic),"shapiro_p":float(shapiro.pvalue),"shapiro_interpretation":"non-rejection is not proof of normality; QQ and design also matter","leave_one_year_geometric_ratio_range":[min(x["geometric_ratio_plus_one"] for x in loo),max(x["geometric_ratio_plus_one"] for x in loo)]},"station_descriptive":desc,"freshness":local,
       "reconciliation":{"selected_hourly_window_matches":int(checks.hourly_window_match.sum()),"network_event_windows":len(network),
           "network_available_origin_sum":int(network.available_origin_sum.sum()),
           "network_available_destination_sum":int(network.available_destination_sum.sum()),
           "network_db_equals_processed":bool(network.db_equals_processed.all()),
           "network_station_hour_rows":int(network.rows.sum()),
           "network_origin_incomplete_rows":int(network.origin_incomplete.sum()),
           "network_destination_incomplete_rows":int(network.destination_incomplete.sum()),
           "network_origin_null_rows":int(network.origin_null_rows.sum()),
           "network_destination_null_rows":int(network.destination_null_rows.sum()),
           "network_dates_with_incomplete_label_hours":list(network.loc[~network.all_label_hours_complete,"event_date"]),
           "network_all_label_hours_complete":bool(network.all_label_hours_complete.all()),
           "meaning":"SUM ignores NULL; matching available-row sums do not establish whole-network coverage. Incomplete label-hours may reflect historical labels absent in a source period, not proven reporting omissions. Four selected stations are complete."}}
    write_json(out/"summary.json",summary)
    plots(out,joined,hourly,primary)
    manifest={"extracted_at_utc":data["extracted_at_utc"],"generated_at_utc":datetime.now(timezone.utc).isoformat(),"database":DB,"read_only":True,"sql_sha256":data["sql_sha256"],"snapshot_local_path":str(snapshot.relative_to(root)),"snapshot_sha256":sha(snapshot),"script_sha256":sha(Path(__file__)),"source_run_sha256":sorted(joined.source_run_sha256.unique()),"python":platform.python_version(),"packages":{k:importlib.metadata.version(k) for k in ["numpy","pandas","scipy","matplotlib","duckdb"]},"method_sources":SOURCE_URLS,"source_files":local["files"],"outputs":{p.name:sha(p) for p in sorted(out.glob("*")) if p.is_file() and p.name!="source_manifest.json"}}
    write_json(out/"source_manifest.json",manifest)
    print(json.dumps({"output":str(out),"coverage":summary["coverage"],"primary":primary_result,"year_sensitivity":yearly_sensitivity,"freshness":local["snapshot_value_differences"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output",type=Path)
    parser.add_argument("--source-root",type=Path,help="Frozen local source tree for reconciliation; defaults to --root")
    parser.add_argument("--snapshot",type=Path,help="Private snapshot path; use a new path for a new analysis run")
    parser.add_argument("--reuse-snapshot",action="store_true",help="Use the original private JSON snapshot; no fresh MySQL read")
    parser.add_argument("--self-test",action="store_true",help="Exercise the partial-baseline failure path without source data or MySQL writes")
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test_partial_baseline(),ensure_ascii=False,indent=2))
        sys.exit(0)
    if args.output is None: args.output=args.root/"results/research_review/2026-10-05/statistics"
    run(args)
