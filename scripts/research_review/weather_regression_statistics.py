"""Exploratory weather associations and ONE baseline regression.

Rules fixed for this extension before inspecting associations (not preregistration):
primary station, exact 18:00..23:00 source labels, all six valid temperature/wind
values required. Exclude 23:59; rain codes are never zero-filled; cloud coverage
only. Keep the previous 13 Beimen events and their original response definition.
Two Spearman tests: n>=10 and >=5 distinct values, abs-rho pairings permutation,
99,999 resamples with fixed seeds; Holm within this two-test family. One OLS:
log1p(Y) ~ intercept + log1p(B). Whole-year holdout is retrospective sensitivity.
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

SEED = 20261005
RESAMPLES = 99999
VARIABLES = ["temperature_six_label_mean_c", "wind_six_label_mean_m_s"]
LABELS = ["六個整點氣溫摘要（°C）", "六個整點風速摘要（m/s）"]
SOURCES = {
    "CODiS": "https://codis.cwa.gov.tw/pdf_data/Readme.pdf",
    "spearmanr": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html",
    "permutation_test": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html",
    "linregress": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def query(verified=False):
    fields = ["observed_at", "datetime_tst_raw", "time_alignment_status", "air_temperature_c",
        "temperature_quality_flag", "hourly_precipitation_mm", "precipitation_raw_value",
        "precipitation_status", "precipitation_quality_flag", "mean_wind_speed_m_s",
        "wind_speed_quality_flag", "total_cloud_amount_tenths", "cloud_measurement_type",
        "cloud_status", "cloud_method_review_required", "source", "source_file_sha256",
        "source_row_number", "imported_at"]
    if verified:
        fields += ["official_report_hour_end", "values_match", "cloud_method_verified", "effective_cloud_review_required", "response_sha256", "verified_at_utc"]
    pairs = ",".join(f"'{key}',w.{key}" for key in fields)
    table = "v_weather_verified" if verified else "weather_observation"
    return f"""SELECT JSON_OBJECT('kind','events','record',JSON_OBJECT(
      'event_date',e.event_date,'event_status',e.event_status,
      'station_id',m.station_id,'station_name',s.station_name,
      'distance_to_dadaocheng_km',m.distance_to_dadaocheng_km))
      FROM taipei_metro_analysis.dadaocheng_fireworks_events e
      LEFT JOIN taipei_metro_analysis.event_weather_station m
        ON e.event_date=m.event_date AND m.station_role='primary'
      LEFT JOIN taipei_metro_analysis.weather_station s ON m.station_id=s.station_id
      ORDER BY e.event_date;
    SELECT JSON_OBJECT('kind','weather','record',JSON_OBJECT(
      'event_date',m.event_date,'station_id',m.station_id,{pairs}))
      FROM taipei_metro_analysis.event_weather_station m
      JOIN taipei_metro_analysis.{table} w ON m.station_id=w.station_id
       AND w.observed_at>=CAST(m.event_date AS DATETIME)
       AND w.observed_at<TIMESTAMPADD(DAY,1,CAST(m.event_date AS DATETIME))
      WHERE m.station_role='primary' ORDER BY m.event_date,w.observed_at;"""


def extract(root, reuse, snapshot=None, verified=False):
    snapshot = snapshot or root/".local/research_review/2026-10-05/weather_extended/weather_snapshot.json"
    sql = query(verified)
    if reuse:
        data = json.loads(snapshot.read_text(encoding="utf-8"))
        if data["query_sha256"] != hashlib.sha256(sql.encode()).hexdigest():
            raise ValueError("Existing weather snapshot uses different SQL; do not relabel its provenance")
    else:
        sys.path.insert(0, str(root))
        from app.backend.database import _read_json
        rows = _read_json(sql)
        data = {"events":[], "weather":[], "extracted_at_utc":datetime.now(timezone.utc).isoformat(),
                "query_sha256":hashlib.sha256(sql.encode()).hexdigest(), "read_only":True}
        for row in rows:
            data[row["kind"]].append(row["record"])
        write_json(snapshot, data)
    return data, snapshot


def load_primary(root, directory=None, expected_n=13):
    directory = directory or root/"results/research_review/2026-10-05/statistics"
    path = directory/"primary_beimen_events.csv"
    manifest = json.loads((directory/"source_manifest.json").read_text(encoding="utf-8"))
    if sha(path) != manifest["outputs"][path.name]:
        raise ValueError("Previous primary event file differs from its audited manifest")
    data = pd.read_csv(path, encoding="utf-8-sig", dtype={"event_date":str})
    if len(data)!=expected_n or data.event_date.duplicated().any() or not data.station.eq("北門").all():
        raise ValueError(f"This extension requires {expected_n} unique Beimen events from the matching snapshot")
    if not data.event_status.eq("held").all() or not data.baseline_min_days.ge(2).all():
        raise ValueError("Prior eligibility unexpectedly changed")
    data["response_log1p_difference"] = np.log1p(data.origin_count_window)-np.log1p(data.baseline_origin_window)
    if not np.allclose(data.response_log1p_difference, data.log1p_difference):
        raise ValueError("Prior response does not reconcile")
    return data, path


def summarize_weather(data, label_mode="legacy_18_23"):
    events = pd.DataFrame(data["events"])
    weather = pd.DataFrame(data["weather"])
    events.event_date = pd.to_datetime(events.event_date).dt.strftime("%Y-%m-%d")
    weather.event_date = pd.to_datetime(weather.event_date).dt.strftime("%Y-%m-%d")
    weather["timestamp"] = pd.to_datetime(weather.observed_at)
    if events.event_date.duplicated().any() or weather.duplicated(["event_date","station_id","observed_at"]).any():
        raise ValueError("Duplicate primary station mapping or observation")
    if label_mode == "legacy_18_23":
        expected = {f"{h:02d}:00:00" for h in range(18,24)}
        weather["analysis_label"] = weather.timestamp.dt.strftime("%H:%M:%S")
        weather["selected_exact_label"] = weather.analysis_label.isin(expected)
        weather["excluded_end_of_day"] = weather.timestamp.dt.strftime("%H:%M").eq("23:59") | weather.time_alignment_status.eq("end_of_day_unverified")
    elif label_mode == "verified_hour_end_19_24":
        if "values_match" not in weather or not weather.values_match.eq(1).all():
            raise ValueError("Aligned sensitivity requires verified observations, not inferred timestamps")
        stamp = pd.to_datetime(weather.official_report_hour_end)
        delta = (stamp - pd.to_datetime(weather.event_date)).dt.total_seconds() / 3600
        expected = set(range(19,25))
        weather["analysis_label"] = delta
        weather["selected_exact_label"] = delta.isin(expected)
        weather["excluded_end_of_day"] = False
    else:
        raise ValueError(f"Unknown label mode: {label_mode}")
    weather["source_label_verified"] = weather["values_match"].eq(1) if "values_match" in weather else False
    selected = weather.loc[weather.selected_exact_label & ~weather.excluded_end_of_day].copy()
    for column in ["air_temperature_c","mean_wind_speed_m_s","hourly_precipitation_mm","total_cloud_amount_tenths"]:
        selected[column] = pd.to_numeric(selected[column])
    rows=[]
    for event in events.to_dict("records"):
        g = selected.loc[selected.event_date.eq(event["event_date"])]
        complete_labels = len(g)==6 and set(g.analysis_label)==expected
        temp_valid = g.air_temperature_c.notna() & g.temperature_quality_flag.fillna("").eq("")
        wind_valid = g.mean_wind_speed_m_s.notna() & g.wind_speed_quality_flag.fillna("").eq("")
        rain_valid = g.precipitation_quality_flag.fillna("").eq("")
        positive = rain_valid & (g.precipitation_status.eq("trace_T") | (g.precipitation_status.eq("observed") & g.hourly_precipitation_mm.gt(0)))
        zero = rain_valid & g.precipitation_status.eq("observed") & g.hourly_precipitation_mm.eq(0)
        known = positive | zero
        wet_status = "wet_evidence" if positive.any() else "all_six_labels_numeric_zero" if complete_labels and zero.all() else "unknown"
        rows.append({**event, "year":int(event["event_date"][:4]), "selected_labels":len(g),
          "six_exact_labels_complete":bool(complete_labels), "temperature_valid_labels":int(temp_valid.sum()),
          "wind_valid_labels":int(wind_valid.sum()),
          "temperature_six_label_mean_c":float(g.air_temperature_c.mean()) if complete_labels and temp_valid.all() else np.nan,
          "wind_six_label_mean_m_s":float(g.mean_wind_speed_m_s.mean()) if complete_labels and wind_valid.all() else np.nan,
          "rain_label_class":wet_status, "rain_positive_or_trace_labels":int(positive.sum()),
          "rain_trace_labels":int(g.precipitation_status.eq("trace_T").sum()),
          "rain_accumulated_later_labels":int(g.precipitation_status.eq("accumulated_later_ampersand").sum()),
          "rain_unknown_or_missing_labels":int(6-known.sum()),
          "cloud_numeric_labels":int(g.total_cloud_amount_tenths.notna().sum()),
          "cloud_method_review_labels":int(g.cloud_method_review_required.sum()) if len(g) else 0,
          "weather_source_time_semantics_verified":bool(len(g) and g.source_label_verified.all()), "weather_label_mode":label_mode})
    return pd.DataFrame(rows), weather, selected


def association(merged, column, seed):
    pair = merged[["event_date","year",column,"response_log1p_difference"]].dropna().copy()
    result = {"variable":column,"n":len(pair),"distinct_x":int(pair[column].nunique()),
              "distinct_y":int(pair.response_log1p_difference.nunique()),"seed":seed,
              "n_resamples":RESAMPLES,"dates":list(pair.event_date)}
    if len(pair)<10 or pair[column].nunique()<5 or pair.response_log1p_difference.nunique()<5:
        result.update({"status":"not_tested_insufficient_pairs_or_distinct_values","rho":None,"permutation_p":None,"holm_two_tests_p":None})
        return result, pd.DataFrame(), pd.DataFrame()
    x=pair[column].to_numpy(); y=pair.response_log1p_difference.to_numpy()
    rx=stats.rankdata(x); ry=stats.rankdata(y); ry=ry-ry.mean()
    denominator=np.sqrt(np.sum((rx-rx.mean())**2)*np.sum(ry**2))
    def statistic(permuted, axis=-1):
        return np.abs(np.sum((permuted-permuted.mean(axis=axis,keepdims=True))*ry,axis=axis)/denominator)
    test=stats.permutation_test((rx,),statistic,permutation_type="pairings",alternative="greater",
          n_resamples=RESAMPLES,batch=1000,vectorized=True,rng=np.random.default_rng(seed))
    rho=float(stats.spearmanr(x,y).statistic)
    tolerance=100*np.finfo(float).eps*abs(float(test.statistic))
    extreme=int(np.count_nonzero(test.null_distribution>=abs(rho)-tolerance))
    p_manual=(extreme+1)/(RESAMPLES+1)
    if not np.isclose(test.pvalue,p_manual):
        raise ValueError("Permutation plus-one p-value does not reconcile")
    years=pair.groupby("year")[[column,"response_log1p_difference"]].mean().reset_index()
    annual_rho=float(stats.spearmanr(years[column],years.response_log1p_difference).statistic)
    loo=[]
    for year in sorted(pair.year.unique()):
        d=pair.loc[pair.year.ne(year)]
        value=float(stats.spearmanr(d[column],d.response_log1p_difference).statistic) if len(d)>=3 and d[column].nunique()>1 and d.response_log1p_difference.nunique()>1 else np.nan
        loo.append({"variable":column,"excluded_year":int(year),"remaining_events":len(d),"rho":value})
    result.update({"status":"exploratory_tested","rho":rho,"permutation_p":float(test.pvalue),
        "extreme_count":extreme,"p_convention":"(count(abs(rho_permuted)>=abs(rho_observed))+1)/(99999+1), floating tolerance as SciPy",
        "annual_equal_weight_n":len(years),"annual_mean_rho_descriptive":annual_rho,
        "leave_one_year_rho_range":[float(np.nanmin([r["rho"] for r in loo])),float(np.nanmax([r["rho"] for r in loo]))]})
    years["variable"]=column
    years=years.rename(columns={column:"annual_mean_weather"})
    return result, pd.DataFrame(loo), years


def fit_baseline(merged):
    n=len(merged); x=np.log1p(merged.baseline_origin_window.to_numpy()); y=np.log1p(merged.origin_count_window.to_numpy())
    fit=stats.linregress(x,y)
    fitted=fit.intercept+fit.slope*x; residual=y-fitted; df=n-2
    sse=float(np.sum(residual**2)); mse=sse/df
    design=np.column_stack([np.ones(n),x]); cov=mse*np.linalg.inv(design.T@design)
    if not np.allclose(np.sqrt(np.diag(cov)),[fit.intercept_stderr,fit.stderr]):
        raise ValueError("OLS standard errors do not reconcile with matrix calculation")
    leverage=np.sum((design@np.linalg.inv(design.T@design))*design,axis=1)
    cooks=(residual**2/(2*mse))*leverage/(1-leverage)**2
    q=stats.t.ppf(.975,df)
    diagnostics=merged[["event_date","year","origin_count_window","baseline_origin_window"]].copy()
    diagnostics["x_log1p_baseline"]=x; diagnostics["y_log1p_observed"]=y
    diagnostics["fitted_log1p"]=fitted; diagnostics["residual_log1p"]=residual
    diagnostics["leverage"]=leverage; diagnostics["cooks_distance"]=cooks
    loyo=[]
    for year in sorted(merged.year.unique()):
        train=merged.year.ne(year).to_numpy(); test=~train
        sub=stats.linregress(x[train],y[train]); prediction=sub.intercept+sub.slope*x[test]
        naive=np.full(int(test.sum()),y[train].mean())
        for idx,pred,base in zip(np.flatnonzero(test),prediction,naive):
            loyo.append({"event_date":merged.iloc[idx].event_date,"held_out_year":int(year),"training_n":int(train.sum()),
               "training_years":";".join(map(str,sorted(merged.loc[train,"year"].unique()))),
               "training_contains_later_years":bool(merged.loc[train,"year"].gt(year).any()),
               "training_intercept":float(sub.intercept),"training_slope":float(sub.slope),
               "observed_log1p":float(y[idx]),"fitted_holdout_log1p":float(pred),"intercept_only_holdout_log1p":float(base),
               "error_log1p":float(y[idx]-pred),"intercept_only_error_log1p":float(y[idx]-base)})
    loyo=pd.DataFrame(loyo)
    sw=stats.shapiro(residual)
    result={"formula":"log1p(origin_count_window) ~ intercept + log1p(baseline_origin_window)",
        "n":n,"residual_df":df,"intercept":float(fit.intercept),"slope":float(fit.slope),
        "intercept_ci95":[float(fit.intercept-q*fit.intercept_stderr),float(fit.intercept+q*fit.intercept_stderr)],
        "slope_ci95":[float(fit.slope-q*fit.stderr),float(fit.slope+q*fit.stderr)],
        "slope_standard_error":float(fit.stderr),"slope_p_conditional":float(fit.pvalue),
        "r_squared_in_sample":float(fit.rvalue**2),"residual_sd_log1p":float(np.sqrt(mse)),
        "shapiro_residual_W":float(sw.statistic),"shapiro_residual_p_diagnostic":float(sw.pvalue),
        "max_cooks_distance":float(cooks.max()),"max_cooks_event":str(merged.iloc[int(cooks.argmax())].event_date),
        "loyo_n":len(loyo),"loyo_mae_log1p":float(loyo.error_log1p.abs().mean()),
        "loyo_rmse_log1p":float(np.sqrt(np.mean(loyo.error_log1p**2))),
        "loyo_intercept_only_mae_log1p":float(loyo.intercept_only_error_log1p.abs().mean()),
        "loyo_intercept_only_rmse_log1p":float(np.sqrt(np.mean(loyo.intercept_only_error_log1p**2))),
        "loyo_slope_range":[float(loyo.training_slope.min()),float(loyo.training_slope.max())],
        "warning":"Conditional OLS CI assumes independent normal homoskedastic errors and treats estimated baselines as fixed. LOYO includes future years and is not forecasting validation. No weather variable was added."}
    return result, diagnostics, loyo


def draw(out, merged, correlations, regression, residuals, loyo, caption=None):
    font=Path("C:/Windows/Fonts/msjh.ttc")
    if font.exists(): plt.rcParams["font.family"]=FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus":False,"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":160})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
    for idx,(column,ax) in enumerate(zip(VARIABLES,axes)):
        d=merged[[column,"response_log1p_difference","year","event_date"]].dropna()
        ax.scatter(d[column],d.response_log1p_difference,color="#343b42",s=40)
        for row in (d.itertuples(index=False,name=None) if len(merged)<=13 else []):
            offset=(4,-12) if column==VARIABLES[1] and row[3]=="2024-07-17" else (4,4)
            ax.annotate(str(int(row[2])),(row[0],row[1]),xytext=offset,textcoords="offset points",fontsize=8,color="#62676c")
        result=correlations[idx]
        ax.set_title(f"n={result['n']}，Spearman rho={result['rho']:.3f}\n兩項 Holm p={result['holm_two_tests_p']:.3f}" if result["rho"] is not None else "樣本不足，不計算檢定")
        ax.set_xlabel(LABELS[idx]); ax.set_ylabel("log(北門觀測+1) - log(歷史基準+1)"); ax.grid(alpha=.15); ax.margins(x=.1,y=.12)
    fig.suptitle(caption or ("天氣與相對人流：標籤摘要的探索性關聯（每點代表一場活動）" if len(merged)>13 else "天氣與相對人流：標籤摘要的探索性關聯（數字標籤為年份）"))
    fig.savefig(out/"01_weather_associations.png"); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
    x=residuals.x_log1p_baseline; y=residuals.y_log1p_observed
    axes[0].scatter(x,y,color="#343b42"); grid=np.linspace(x.min(),x.max(),100)
    axes[0].plot(grid,regression["intercept"]+regression["slope"]*grid,color="#bb713f")
    axes[0].set_xlabel("log(歷史基準+1)"); axes[0].set_ylabel("log(活動觀測+1)")
    axes[0].set_title(f"單一基準回歸｜樣本內 R²={regression['r_squared_in_sample']:.3f}")
    axes[1].scatter(loyo.observed_log1p,loyo.fitted_holdout_log1p,color="#343b42",label="基準回歸")
    lo=min(loyo.observed_log1p.min(),loyo.fitted_holdout_log1p.min()); hi=max(loyo.observed_log1p.max(),loyo.fitted_holdout_log1p.max())
    axes[1].plot([lo,hi],[lo,hi],color="#999",linestyle="--",label="觀測=預估")
    axes[1].set_xlabel("留出事件的 log(觀測+1)"); axes[1].set_ylabel("留整年模型的預估 log(人次+1)")
    axes[1].set_title("留一年敏感度｜訓練可能含未來，非事前預測"); axes[1].legend()
    for ax in axes: ax.grid(alpha=.15)
    fig.savefig(out/"02_baseline_regression.png"); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
    axes[0].scatter(residuals.fitted_log1p,residuals.residual_log1p,color="#343b42")
    axes[0].axhline(0,color="#999",linestyle="--")
    label_dates = set(residuals.loc[residuals.residual_log1p.abs().nlargest(3).index,"event_date"]) | {str(residuals.loc[residuals.cooks_distance.idxmax(),"event_date"])}
    for row in residuals.itertuples():
        if len(merged)>13 and row.event_date not in label_dates:
            continue
        offset=(3,-12) if row.event_date=="2024-07-17" or (len(merged)>13 and row.residual_log1p<0 and row.residual_log1p!=residuals.residual_log1p.min()) else (3,4)
        axes[0].annotate(str(row.event_date),(row.fitted_log1p,row.residual_log1p),xytext=offset,textcoords="offset points",fontsize=7)
    axes[0].set_xlabel("樣本內預估 log(人次+1)"); axes[0].set_ylabel("殘差（log1p 尺度）"); axes[0].set_title("殘差與擬合值\n日期標示：|殘差|最高3場與Cook最大場" if len(merged)>13 else "殘差與擬合值")
    stats.probplot(residuals.residual_log1p,plot=axes[1])
    axes[1].set_title("殘差常態 Q-Q 圖"); axes[1].set_xlabel("理論常態分位數"); axes[1].set_ylabel("殘差（log1p 尺度）")
    fig.savefig(out/"03_regression_diagnostics.png"); plt.close(fig)


def self_test():
    # A missing exact hour and a 23:59 value must never become a six-label mean.
    event={"event_date":"2026-08-15","event_status":"held","station_id":"S","station_name":"test","distance_to_dadaocheng_km":1}
    rows=[]
    for time in ["18:00:00","19:00:00","20:00:00","21:00:00","22:00:00","23:59:00"]:
        rows.append({"event_date":"2026-08-15","station_id":"S","observed_at":"2026-08-15 "+time,
        "time_alignment_status":"source_label","air_temperature_c":30,"temperature_quality_flag":None,
        "mean_wind_speed_m_s":1,"wind_speed_quality_flag":None,"hourly_precipitation_mm":None,
        "precipitation_status":"accumulated_later_ampersand","precipitation_quality_flag":None,
        "total_cloud_amount_tenths":None,"cloud_method_review_required":1})
    agg,_,_=summarize_weather({"events":[event],"weather":rows})
    assert agg.selected_labels.iloc[0]==5 and pd.isna(agg.temperature_six_label_mean_c.iloc[0]) and pd.isna(agg.wind_six_label_mean_m_s.iloc[0])
    assert agg.rain_label_class.iloc[0]=="unknown"
    rows[-1]["observed_at"]="2026-08-15 23:00:00"; rows[-1]["precipitation_status"]="trace_T"
    agg,_,_=summarize_weather({"events":[event],"weather":rows})
    assert agg.temperature_six_label_mean_c.iloc[0]==30 and agg.rain_label_class.iloc[0]=="wet_evidence"
    assert agg.rain_accumulated_later_labels.iloc[0]==5
    aligned_rows=[]
    for hour in range(19,25):
        row=rows[0].copy()
        row["observed_at"]="2026-08-15 "+(f"{hour:02d}:00:00" if hour<24 else "23:59:00")
        row["official_report_hour_end"]=(pd.Timestamp("2026-08-15")+pd.Timedelta(hours=hour)).isoformat()
        row["values_match"]=1
        aligned_rows.append(row)
    aligned,_,selected=summarize_weather({"events":[event],"weather":aligned_rows},"verified_hour_end_19_24")
    assert aligned.six_exact_labels_complete.iloc[0] and len(selected)==6
    assert selected.analysis_label.tolist()==list(range(19,25))
    incomplete,_,_=summarize_weather({"events":[event],"weather":aligned_rows[:-1]},"verified_hour_end_19_24")
    assert not incomplete.six_exact_labels_complete.iloc[0] and pd.isna(incomplete.temperature_six_label_mean_c.iloc[0])
    aligned_rows[-1]["values_match"]=0
    try:
        summarize_weather({"events":[event],"weather":aligned_rows},"verified_hour_end_19_24")
    except ValueError:
        pass
    else:
        raise AssertionError("Unverified hour-end entered aligned sensitivity")
    return {"five_exact_plus_2359":"mean unavailable","ampersand":"unknown, not zero","trace_T":"wet evidence, unknown amount",
            "verified_19_to_24":"six labels accepted; missing 24 not imputed; unverified rows rejected"}


def run(args):
    root=args.root.resolve(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    data,snapshot=extract(root,args.reuse_snapshot,getattr(args,"snapshot",None),getattr(args,"verified_weather",False))
    primary,primary_path=load_primary(root,getattr(args,"primary_directory",None),getattr(args,"expected_n",13))
    label_mode=getattr(args,"label_mode","legacy_18_23")
    weather_summary,weather,selected=summarize_weather(data,label_mode)
    merged=primary.merge(weather_summary,on=["event_date","year","event_status"],how="left",validate="one_to_one",indicator=True)
    if not merged._merge.eq("both").all(): raise ValueError("Previous event key is missing from weather catalog")
    merged=merged.drop(columns="_merge")
    results=[]; leave=[]; yearly=[]
    for i,column in enumerate(VARIABLES):
        result,loo,annual=association(merged,column,SEED+i)
        results.append(result)
        if not loo.empty: leave.append(loo)
        if not annual.empty: yearly.append(annual)
    # Fixed family size = 2; unavailable tests are conservatively assigned p=1
    # for multiplicity bookkeeping, while their own reported result remains null.
    raw=np.array([r["permutation_p"] if r["permutation_p"] is not None else 1 for r in results])
    order=np.argsort(raw); adjusted=np.minimum(1,np.maximum.accumulate(raw[order]*np.array([2,1])))
    for rank,index in enumerate(order):
        if results[index]["permutation_p"] is not None: results[index]["holm_two_tests_p"]=float(adjusted[rank])
    regression,residuals,loyo=fit_baseline(merged)
    rain_counts=[]
    for scope,df in [("all_26_dates",weather_summary),("held_21_dates",weather_summary.loc[weather_summary.event_status.eq("held")]),(f"primary_{len(merged)}_dates",merged)]:
        for category in ["wet_evidence","all_six_labels_numeric_zero","unknown"]:
            rain_counts.append({"scope":scope,"rain_label_class":category,"events":int(df.rain_label_class.eq(category).sum())})
    tables={"weather_event_summary":weather_summary,"primary_weather_merged":merged,
        "weather_correlations":pd.DataFrame(results),"weather_leave_one_year":pd.concat(leave,ignore_index=True) if leave else pd.DataFrame(),
        "weather_yearly_means":pd.concat(yearly,ignore_index=True) if yearly else pd.DataFrame(),
        "rain_group_counts":pd.DataFrame(rain_counts),"regression_event_diagnostics":residuals,"regression_loyo":loyo}
    for name,df in tables.items(): df.to_csv(out/f"{name}.csv",index=False,encoding="utf-8-sig",na_rep="")
    coverage={"events_in_catalog":len(weather_summary),"primary_weather_station_ids":sorted(weather.station_id.unique()),
      "primary_daily_weather_rows":len(weather),"primary_end_of_day_rows_excluded":int(weather.excluded_end_of_day.sum()),
      "selected_exact_labels":len(selected),"events_with_six_labels":int(weather_summary.six_exact_labels_complete.sum()),
      "dates_without_primary_station":list(weather_summary.loc[weather_summary.station_id.isna(),"event_date"]),
      "temperature_complete_event_means":int(weather_summary[VARIABLES[0]].notna().sum()),"wind_complete_event_means":int(weather_summary[VARIABLES[1]].notna().sum()),
      "cloud_numeric_selected_labels":int(selected.total_cloud_amount_tenths.notna().sum()),
      "cloud_method_review_selected_labels":int(selected.cloud_method_review_required.sum()),
      "cloud_method_review_after_verification_selected_labels":int(selected.effective_cloud_review_required.sum()) if "effective_cloud_review_required" in selected else int(selected.cloud_method_review_required.sum()),
      "cloud_method_counts":{str(k):int(v) for k,v in selected.cloud_measurement_type.value_counts(dropna=False).items()},
      "selected_precipitation_status_counts":{str(k):int(v) for k,v in selected.precipitation_status.value_counts(dropna=False).items()},
      "source_time_intervals_verified":False,
      "weather_report_hour_end_verified":bool(len(selected) and selected.source_label_verified.all()),
      "od_to_weather_exact_exposure_verified":False}
    summary={"design":{"primary_events":f"{len(primary)} Beimen dates from the matching primary CSV", "weather_labels":label_mode, "wind_semantics":"mean of six preceding-10-minute averages, not continuous six-hour mean", "correlation_response":"log1p(Y)-log1p(B)","fixed_weather_family":VARIABLES,"pairwise_min_n":10,"minimum_distinct_per_variable":5,"resampling_count":RESAMPLES,"seed":SEED,"multiplicity":"Holm within these two weather tests; not global study-wide correction", "inference":"exploratory: source timing, exchangeability, same-year dependence and baseline uncertainty remain"},
      "weather_coverage":coverage,"rain_groups":rain_counts,"weather_correlations":results,"baseline_regression":regression,"self_test":self_test()}
    write_json(out/"summary.json",summary)
    draw(out,merged,results,regression,residuals,loyo, "19–24 時小時終點敏感度：事後觀測天氣（每點代表一場活動）" if label_mode=="verified_hour_end_19_24" else None)
    manifest={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"weather_extracted_at_utc":data["extracted_at_utc"],
      "weather_query_sha256":data["query_sha256"],"snapshot_path":str(snapshot.relative_to(root)),"snapshot_sha256":sha(snapshot),
      "weather_source_file_sha256":sorted(weather.source_file_sha256.unique()),"primary_input_path":str(primary_path.relative_to(root)),
      "primary_input_sha256":sha(primary_path),"previous_manifest_sha256":sha(primary_path.parent/"source_manifest.json"),
      "script_sha256":sha(Path(__file__)),"python":platform.python_version(),"packages":{k:importlib.metadata.version(k) for k in ["numpy","pandas","scipy","matplotlib"]},
      "method_sources":SOURCES,"outputs":{p.name:sha(p) for p in sorted(out.glob("*")) if p.is_file() and p.name!="source_manifest.json"}}
    write_json(out/"source_manifest.json",manifest)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output",type=Path)
    parser.add_argument("--snapshot",type=Path)
    parser.add_argument("--primary-directory",type=Path)
    parser.add_argument("--expected-n",type=int,default=13)
    parser.add_argument("--verified-weather",action="store_true")
    parser.add_argument("--label-mode",choices=["legacy_18_23","verified_hour_end_19_24"],default="legacy_18_23")
    parser.add_argument("--reuse-snapshot",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),ensure_ascii=False,indent=2)); sys.exit(0)
    if args.output is None: args.output=args.root/"results/research_review/2026-10-05/weather_regression"
    run(args)
