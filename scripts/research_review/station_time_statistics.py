"""Event-block station comparisons and source-hour concentration, read-only.

Fixed extension: 4 stations on the original 13 eligible held events; Friedman
rank statistic with within-event Monte Carlo label permutations; Beimen versus
3 stations via two-sided exact sign tests with Holm adjustment. Descriptions
use all 21 complete held events. The only time-share inference compares Beimen
21 <= source_hour < 24 share with its baseline on the same 13 eligible events.
These are observational, exploratory contrasts, not randomized causal tests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import permutations, product
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter

STATIONS = ["北門", "大橋頭站", "雙連", "民權西路"]
SEED = 20261005
PERMUTATIONS = 199999
SOURCES = {
    "Friedman": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html",
    "paired_permutation": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html",
    "exact_binomial": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html",
    "Holm": "https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_or_missing(a, b):
    return (pd.isna(a) and pd.isna(b)) or bool(np.isclose(a, b, rtol=1e-10, atol=1e-6))


def load_snapshot(snapshot, primary_csv):
    data = json.loads(Path(snapshot).read_text(encoding="utf-8"))
    events, windows, hourly = (pd.DataFrame(data[k]) for k in ("events", "windows", "hourly"))
    for df in (events, windows, hourly):
        df["event_date"] = pd.to_datetime(df.event_date).dt.strftime("%Y-%m-%d")
    require(not events.duplicated("event_date").any(), "Duplicate event dates")
    require(not windows.duplicated(["event_date", "station"]).any(), "Duplicate windows")
    require(not hourly.duplicated(["event_date", "station", "hour_start"]).any(), "Duplicate hours")
    require(set(windows.station) == set(hourly.station) == set(STATIONS), "Unexpected station labels")
    require(set(windows.event_date) == set(hourly.event_date) == set(events.event_date), "Date coverage mismatch")
    for col in ("origin_count_window", "baseline_origin_window", "baseline_min_days", "peak_origin_hour_count"):
        windows[col] = pd.to_numeric(windows[col], errors="raise")
    for col in ("origin_count", "baseline_origin_mean_28d", "baseline_n_days"):
        hourly[col] = pd.to_numeric(hourly[col], errors="raise")
    hour_ts = pd.to_datetime(hourly.hour_start)
    hourly["hour"] = hour_ts.dt.hour
    require(hour_ts.dt.strftime("%Y-%m-%d").equals(hourly.event_date), "Hour outside event date")
    require(hour_ts.dt.minute.eq(0).all() and hour_ts.dt.second.eq(0).all(), "Non-hour-aligned source bins")
    require(windows.window_start_hour.eq(18).all() and windows.window_end_hour_exclusive.eq(24).all(), "Wrong window")
    require(len(windows) == len(events) * 4, "Not all event/station windows are present")
    windows = windows.merge(events[["event_date", "event_status"]], on="event_date", validate="many_to_one")
    require(windows.observed_event_status.eq(windows.event_status).all(), "Conflicting event status")
    validation = []
    for (date, station), g in hourly.groupby(["event_date", "station"], sort=True):
        w = windows.loc[windows.event_date.eq(date) & windows.station.eq(station)].iloc[0]
        require(len(g) == 6 and set(g.hour) == set(range(18, 24)), f"Missing/extra source hour: {date} {station}")
        require(g.origin_count.dropna().ge(0).all(), f"Negative origin count: {date} {station}")
        require(g.baseline_origin_mean_28d.dropna().ge(0).all(), f"Negative baseline: {date} {station}")
        complete = g.origin_complete.eq(1).all() and g.origin_count.notna().all()
        baseline_complete = g.baseline_origin_mean_28d.notna().all()
        total = float(g.origin_count.sum()) if complete else np.nan
        base = float(g.baseline_origin_mean_28d.sum()) if baseline_complete else np.nan
        require(equal_or_missing(total, w.origin_count_window), f"Total mismatch: {date} {station}")
        require(equal_or_missing(base, w.baseline_origin_window), f"Baseline mismatch: {date} {station}")
        require(bool(w.origin_window_complete) == bool(complete), f"Completeness mismatch: {date} {station}")
        require(int(g.baseline_n_days.min()) == int(w.baseline_min_days), f"Baseline day mismatch: {date} {station}")
        if complete:
            require(equal_or_missing(float(g.origin_count.max()), w.peak_origin_hour_count), f"Peak mismatch: {date} {station}")
        validation.append({"event_date": date, "station": station, "hours": 6, "origin_complete": bool(complete),
                           "baseline_complete": bool(baseline_complete), "matches_window": True})
    require(len(validation) == len(windows), "Missing event/station groups")
    windows["eligible"] = windows.event_status.eq("held") & windows.origin_window_complete.eq(1) & windows.origin_count_window.notna() & windows.baseline_origin_window.notna() & windows.baseline_min_days.ge(2)
    windows["held_complete"] = windows.event_status.eq("held") & windows.origin_window_complete.eq(1) & windows.origin_count_window.notna()
    windows["year"] = pd.to_datetime(windows.event_date).dt.year
    windows["log1p_contrast"] = np.log1p(windows.origin_count_window) - np.log1p(windows.baseline_origin_window)
    primary = pd.read_csv(primary_csv, encoding="utf-8-sig")
    require(not primary.duplicated("event_date").any(), "Duplicate dates in prior primary CSV")
    dates = set(primary.event_date)
    require(dates == set(windows.loc[windows.station.eq("北門") & windows.eligible, "event_date"]), "Original primary dates changed")
    same_dates = windows.loc[windows.event_date.isin(dates)].copy()
    require(same_dates.eligible.all() and len(same_dates) == len(dates)*4, "A comparison station lacks a valid baseline on primary dates")
    left = same_dates.loc[same_dates.station.eq("北門")].set_index("event_date").sort_index()
    right = primary.set_index("event_date").sort_index()
    for col in ("origin_count_window", "baseline_origin_window", "baseline_min_days"):
        require(np.allclose(left[col], right[col], rtol=1e-10, atol=1e-6), f"Prior primary CSV changed: {col}")
    return data, windows, hourly, same_dates, pd.DataFrame(validation)


def ranks_and_correction(matrix):
    matrix = np.asarray(matrix, dtype=float)
    require(matrix.ndim == 2 and matrix.shape[0] >= 2 and matrix.shape[1] >= 3, "Need blocks x >=3 stations")
    require(np.isfinite(matrix).all(), "Missing/nonfinite comparison value")
    n, k = matrix.shape
    ranks = stats.rankdata(matrix, axis=1, method="average")
    tie_sum = 0
    for row in matrix:
        counts = np.unique(row, return_counts=True)[1]
        tie_sum += int(np.sum(counts**3-counts))
    correction = 1 - tie_sum/(n*k*(k*k-1))
    require(correction > 0, "All within-block station values are tied; Friedman undefined")
    return ranks, correction


def friedman_from_ranks(ranks, correction):
    n, k = ranks.shape[-2:]
    # Leading dimensions can batch Monte Carlo replications.
    sums = np.sum(ranks, axis=-2)
    return (12*np.sum(sums*sums, axis=-1)/(n*k*(k+1)) - 3*n*(k+1))/correction


def friedman_permutation(matrix, repetitions, seed):
    ranks, correction = ranks_and_correction(matrix)
    n, k = ranks.shape
    observed = float(friedman_from_ranks(ranks, correction))
    scipy_result = stats.friedmanchisquare(*np.asarray(matrix).T, nan_policy="raise")
    require(np.isclose(observed, scipy_result.statistic, rtol=1e-11, atol=1e-11), "Independent Friedman formula mismatch")
    permutations_index = np.array(list(permutations(range(k))))
    require(repetitions > 0, "Positive permutation count required")
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, repetitions, 2000):
        size = min(2000, repetitions-start)
        indices = permutations_index[rng.integers(len(permutations_index), size=(size, n))]
        shuffled = np.take_along_axis(np.broadcast_to(ranks, (size, n, k)), indices, axis=2)
        null_q = friedman_from_ranks(shuffled, correction)
        exceed += int(np.count_nonzero(null_q >= observed - 1e-10))
    mc_ci = stats.binomtest(exceed, repetitions).proportion_ci(confidence_level=.95, method="exact")
    return {"n_blocks": n, "k_stations": k, "friedman_Q": observed, "df_asymptotic": k-1,
            "asymptotic_p_reference_only": float(scipy_result.pvalue), "tie_correction": float(correction),
            "kendall_W": observed/(n*(k-1)), "mean_ranks": ranks.mean(axis=0).tolist(),
            "permutation": {"method": "independent station-label permutation within each block; Q upper tail",
                "repetitions": repetitions, "seed": seed, "exceedances": exceed,
                "p_plus_one": (exceed+1)/(repetitions+1), "resolution": 1/(repetitions+1),
                "mc_tail_probability_ci95": [float(mc_ci.low), float(mc_ci.high)],
                "mc_ci_meaning": "Monte Carlo sampling error only, conditional on supplied data and exchangeability model; not an effect CI"}}


def exact_sign(values):
    x = np.asarray(values, dtype=float)
    require(np.isfinite(x).all(), "Missing sign-test difference")
    # Only arithmetic zero is omitted. No data-dependent near-zero threshold.
    positive, negative, zero = int((x>0).sum()), int((x<0).sum()), int((x==0).sum())
    require(positive + negative > 0, "All differences zero; sign test undefined")
    result = stats.binomtest(positive, positive+negative, p=.5, alternative="two-sided")
    ci = result.proportion_ci(confidence_level=.95, method="exact")
    return {"n_total": len(x), "positive": positive, "negative": negative, "ties_omitted": zero,
            "n_nonzero": positive+negative, "p_two_sided_exact": float(result.pvalue),
            "positive_proportion": positive/(positive+negative), "positive_probability_ci95": [float(ci.low), float(ci.high)],
            "mean_difference": float(np.mean(x)), "median_difference": float(np.median(x)),
            "difference_q25": float(np.quantile(x,.25)), "difference_q75": float(np.quantile(x,.75))}


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    require(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), "Invalid p values")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def time_features(windows, hourly):
    held = windows.loc[windows.held_complete].copy()
    rows, shares = [], []
    for w in held.itertuples():
        g = hourly.loc[hourly.event_date.eq(w.event_date) & hourly.station.eq(w.station)].sort_values("hour")
        total = float(w.origin_count_window)
        require(total > 0, f"Zero total has undefined shares: {w.event_date} {w.station}")
        max_count = float(g.origin_count.max())
        peak_hours = g.loc[g.origin_count.eq(max_count), "hour"].astype(int).tolist()
        late = float(g.loc[g.hour.ge(21), "origin_count"].sum())
        base_ok = g.baseline_origin_mean_28d.notna().all() and pd.notna(w.baseline_origin_window) and w.baseline_origin_window > 0
        base_late = float(g.loc[g.hour.ge(21), "baseline_origin_mean_28d"].sum()) if base_ok else np.nan
        base_share = base_late/float(w.baseline_origin_window) if base_ok else np.nan
        rows.append({"event_date": w.event_date, "year": int(w.year), "station": w.station,
             "eligible": bool(w.eligible), "total_18_24": total, "peak_hour_count": max_count,
             "peak_hour_share": max_count/total, "peak_hours_json": json.dumps(peak_hours), "peak_tie_count": len(peak_hours),
             "late_21_24_count": late, "late_21_24_share": late/total,
             "baseline_total_18_24": w.baseline_origin_window, "baseline_late_21_24_count": base_late,
             "baseline_late_21_24_share": base_share, "late_share_difference": late/total-base_share})
        for h in g.itertuples():
            shares.append({"event_date": w.event_date, "station": w.station, "hour": int(h.hour), "eligible": bool(w.eligible),
                 "origin_count": float(h.origin_count), "observed_share": float(h.origin_count)/total,
                 "baseline_count": h.baseline_origin_mean_28d,
                 "baseline_share": float(h.baseline_origin_mean_28d)/w.baseline_origin_window if base_ok else np.nan})
    time_df, share_df = pd.DataFrame(rows), pd.DataFrame(shares)
    require(np.allclose(share_df.groupby(["event_date", "station"]).observed_share.sum(), 1), "Observed shares do not sum to 1")
    require(np.allclose(share_df.loc[share_df.eligible].groupby(["event_date", "station"]).baseline_share.sum(), 1), "Eligible baseline shares do not sum to 1")
    descriptions = []
    for station in STATIONS:
        g = time_df.loc[time_df.station.eq(station)]
        row = {"station": station, "events": len(g), "events_with_tied_peak": int(g.peak_tie_count.gt(1).sum()),
              "peak_share_median": float(g.peak_hour_share.median()), "peak_share_q25": float(g.peak_hour_share.quantile(.25)),
              "peak_share_q75": float(g.peak_hour_share.quantile(.75)), "late_share_median": float(g.late_21_24_share.median())}
        for hour in range(18,24):
            row[f"peak_includes_{hour}_events"] = int(g.peak_hours_json.apply(lambda s: hour in json.loads(s)).sum())
        descriptions.append(row)
    return time_df, share_df, pd.DataFrame(descriptions)


def plots(out, contrast_matrix, shares, time_df):
    font = Path("C:/Windows/Fonts/msjh.ttc")
    if font.exists():
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130, "savefig.dpi": 160})
    fig, ax = plt.subplots(figsize=(9,5), layout="constrained")
    for row in contrast_matrix.to_numpy():
        ax.plot(range(4), row, color="#aaaaaa", alpha=.55, linewidth=1, marker="o", markersize=3)
    ax.plot(range(4), contrast_matrix.mean(), color="#222222", marker="D", linewidth=2.3, label="各站平均 log1p 差")
    ax.axhline(0, color="#999999", linestyle="--", linewidth=1)
    ax.set_xticks(range(4), STATIONS)
    ax.set_ylabel("log(活動觀測總量+1) - log(本站基準總量+1)")
    ax.set_title(f"四站增幅的同場比較（{len(contrast_matrix)} 場）\n每條灰線代表同一場活動；高度比較相對本站基準的增幅", loc="left")
    ax.grid(axis="y", alpha=.15); ax.legend()
    fig.savefig(out/"01_paired_station_contrasts.png"); plt.close(fig)

    s = shares.loc[shares.eligible]
    fig, axes = plt.subplots(2,2, figsize=(10,7), sharex=True, sharey=True, layout="constrained")
    for station, ax in zip(STATIONS, axes.flat):
        g = s.loc[s.station.eq(station)]
        for column, color, label, style in [("observed_share", "#252525", "活動觀測", "-"), ("baseline_share", "#638c9a", "歷史基準", "--")]:
            grouped = g.groupby("hour")[column]
            med, q25, q75 = grouped.median(), grouped.quantile(.25), grouped.quantile(.75)
            ax.plot(med.index, med, color=color, marker="o", linestyle=style, label=label)
            ax.fill_between(med.index, q25, q75, color=color, alpha=.12)
        ax.set_xticks(range(18,24)); ax.set_title(station)
        ax.set_xlabel("來源時段（21 表示 21:00–22:00 分箱）")
        ax.set_ylabel("占該場本站 18–24 時總量比例")
        ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.15)
    axes[0,0].legend()
    fig.suptitle("同一組可比活動的逐時占比\n線為各場占比的中位數；陰影為四分位範圍，非信賴區間")
    fig.savefig(out/"02_hourly_share_profiles.png"); plt.close(fig)

    b = time_df.loc[time_df.station.eq("北門") & time_df.eligible].sort_values("event_date")
    fig, ax = plt.subplots(figsize=(9,6.5), layout="constrained")
    y = np.arange(len(b))
    for i, row in enumerate(b.itertuples()):
        ax.plot([row.baseline_late_21_24_share, row.late_21_24_share], [i,i], color="#aaaaaa")
    ax.scatter(b.baseline_late_21_24_share, y, color="#638c9a", label="歷史基準", s=35)
    ax.scatter(b.late_21_24_share, y, color="#252525", label="活動觀測", s=35)
    ax.set_yticks(y,b.event_date); ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.set_xlabel("21–23 時三個分箱合計／18–23 時六個分箱合計")
    ax.set_title("北門：晚間人流是否更集中在 21:00–24:00？\n同場配對的來源時段占比；不是實際散場時間", loc="left")
    ax.grid(axis="x",alpha=.15); ax.legend(loc="lower right")
    fig.savefig(out/"03_beimen_late_share_pairs.png"); plt.close(fig)


def self_test():
    matrix = np.array([[1,1,3],[2,1,4]], dtype=float)
    ranks, correction = ranks_and_correction(matrix)
    all_perms = list(permutations(range(3)))
    for inds in product(all_perms, repeat=2):
        shuffled = np.take_along_axis(matrix, np.asarray(inds), axis=1)
        sr, sc = ranks_and_correction(shuffled)
        require(np.isclose(friedman_from_ranks(sr, sc), stats.friedmanchisquare(*shuffled.T).statistic), "Tied exact-enumeration discrepancy")
    require(np.allclose(holm([.04,.01,.03]), [.06,.03,.06]), "Holm order restoration failed")
    sign = exact_sign([1,1,-1,0])
    require(sign["ties_omitted"] == 1 and sign["p_two_sided_exact"] == 1, "Sign ties failed")
    require(friedman_permutation(matrix, 99, 7) == friedman_permutation(matrix, 99, 7), "Seed reproducibility failed")
    return {"tied_friedman_permutations_verified":36, "holm":"passed", "sign_ties":"passed", "fixed_seed":"passed"}


def run(args):
    root = args.root.resolve(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    snapshot = args.snapshot.resolve() if args.snapshot else root/".local/research_review/2026-10-05/statistics_snapshot.json"
    prior_csv = getattr(args,"primary_csv",None) or root/"results/research_review/2026-10-05/statistics/primary_beimen_events.csv"
    data, windows, hourly, paired, validation = load_snapshot(snapshot, prior_csv)
    matrix = paired.pivot(index="event_date", columns="station", values="log1p_contrast").reindex(columns=STATIONS).sort_index()
    friedman = friedman_permutation(matrix.to_numpy(), args.permutations, SEED)
    pairs = []
    pair_differences = pd.DataFrame({"event_date": matrix.index})
    for station in STATIONS[1:]:
        difference = matrix["北門"]-matrix[station]
        result = exact_sign(difference)
        pairs.append({"reference": "北門", "comparison": station, **result,
                      "geometric_ratio_of_plus_one_ratios": float(np.exp(difference.mean()))})
        pair_differences[station] = difference.to_numpy()
    for result, adj in zip(pairs, holm([r["p_two_sided_exact"] for r in pairs])):
        result["p_holm_three_comparisons"] = float(adj)
    year_matrix = matrix.assign(year=pd.to_datetime(matrix.index).year).groupby("year").mean().reindex(columns=STATIONS)
    year_friedman = friedman_permutation(year_matrix.to_numpy(), args.permutations, SEED+1)
    year_pairs = [{"comparison": s, **exact_sign(year_matrix["北門"]-year_matrix[s])} for s in STATIONS[1:]]
    for r, adj in zip(year_pairs, holm([r["p_two_sided_exact"] for r in year_pairs])):
        r["p_holm_three_comparisons"] = float(adj)
    loo = []
    years = pd.to_datetime(matrix.index).year
    for excluded_year in sorted(set(years)):
        m = matrix.loc[years != excluded_year]
        for station in STATIONS[1:]:
            d = m["北門"]-m[station]
            loo.append({"excluded_year": int(excluded_year), "remaining_events": len(m), "comparison": station,
                        "mean_log1p_difference_of_contrasts": float(d.mean()), "positive_events": int((d>0).sum()),
                        "negative_events": int((d<0).sum())})
    time_df, shares, time_desc = time_features(windows, hourly)
    late = time_df.loc[time_df.eligible].copy()
    north = late.loc[late.station.eq("北門")]
    late_sign = exact_sign(north.late_share_difference)
    late_years = north.groupby("year").late_share_difference.mean()
    late_year_sign = exact_sign(late_years)
    late_loo = [{"excluded_year": int(y), "remaining_events": int(north.year.ne(y).sum()),
                 "mean_share_difference": float(north.loc[north.year.ne(y), "late_share_difference"].mean())} for y in sorted(north.year.unique())]
    station_desc = []
    for i, station in enumerate(STATIONS):
        g = paired.loc[paired.station.eq(station)]
        ls = late.loc[late.station.eq(station)]
        station_desc.append({"station": station, "n": len(g), "mean_log1p_contrast": float(g.log1p_contrast.mean()),
              "geometric_ratio_plus_one": float(np.exp(g.log1p_contrast.mean())), "mean_rank": friedman["mean_ranks"][i],
              "observed_late_share_mean": float(ls.late_21_24_share.mean()), "baseline_late_share_mean": float(ls.baseline_late_21_24_share.mean()),
              "late_share_difference_mean": float(ls.late_share_difference.mean()), "late_share_difference_median": float(ls.late_share_difference.median())})
    summary = {"design": {"stations":STATIONS, "source_hours":[18,24], "late_source_hours":[21,24],
            "minimum_baseline_days":2, "unit":"event_date; repeated stations within each event",
            "station_tests":"one Friedman omnibus plus three Beimen paired sign tests, Holm family of three",
            "time_test":"one pre-specified two-sided Beimen late-share sign test; other stations descriptive",
            "cross_question_multiplicity":"module retains family-of-three pairwise Holm; overall extension report additionally adjusts seven planned questions. No whole-study FWER guarantee.",
            "design_timing":"fixed for this extension after prior descriptive results had been seen; not pre-registration or independent confirmation",
            "inference":"conditional exploratory; neither station exchangeability nor independence is established",
            "fixed_seed":SEED},
        "coverage":{"snapshot_extracted_at_utc":data["extracted_at_utc"], "scheduled_dates":int(windows.event_date.nunique()),
            "held_dates":int(windows.loc[windows.held_complete].event_date.nunique()), "paired_events":len(matrix),
            "paired_years":list(map(int,year_matrix.index)), "paired_dates":list(matrix.index),
            "hourly_rows_validated":len(hourly), "window_rows_validated":len(validation), "origin_missing_rows":int(hourly.origin_count.isna().sum())},
        "station_omnibus":friedman, "station_pairwise":pairs, "station_descriptive":station_desc,
        "year_equal_weight_sensitivity":{"omnibus":year_friedman,"pairwise":year_pairs},
        "leave_year_out_station":loo, "peak_and_time_descriptive":time_desc.to_dict(orient="records"),
        "beimen_late_share":{"event_sign_test":late_sign, "year_mean_sign_test":late_year_sign, "leave_year_out":late_loo},
        "checks":self_test()}
    frames = {"event_station_contrasts":paired[["event_date","year","station","origin_count_window","baseline_origin_window","baseline_min_days","log1p_contrast"]],
              "station_descriptive":pd.DataFrame(station_desc), "pairwise_contrast_differences":pair_differences,
              "station_pairwise_sign":pd.DataFrame(pairs), "year_station_contrasts":year_matrix.reset_index(),
              "leave_year_out_station":pd.DataFrame(loo), "event_station_time_features":time_df,
              "hourly_shares":shares, "station_peak_descriptive":time_desc, "late_share_comparisons":late,
              "beimen_year_late_share_difference":late_years.rename("mean_share_difference").reset_index(),
              "beimen_leave_year_out_late_share":pd.DataFrame(late_loo), "input_reconciliation":validation}
    for name, df in frames.items():
        # Tuple/list fields remain JSON text, rather than locale-dependent display strings.
        for c in df.columns:
            if df[c].map(lambda x:isinstance(x,(list,tuple))).any():
                df[c] = df[c].map(lambda x:json.dumps(x) if isinstance(x,(list,tuple)) else x)
        df.to_csv(out/f"{name}.csv", index=False, encoding="utf-8-sig", na_rep="")
    write_json(out/"summary.json",summary)
    plots(out,matrix,shares,time_df)
    manifest = {"generated_at_utc":datetime.now(timezone.utc).isoformat(), "read_only_input":True,
        "snapshot_path":str(snapshot), "snapshot_sha256":sha(snapshot), "snapshot_extracted_at_utc":data["extracted_at_utc"],
        "prior_primary_csv":str(prior_csv.relative_to(root)), "prior_primary_csv_sha256":sha(prior_csv),
        "script_sha256":sha(__file__), "python":platform.python_version(),
        "packages":{p:importlib.metadata.version(p) for p in ("numpy","pandas","scipy","matplotlib")},
        "method_sources":SOURCES, "sources_checked_on":"2026-10-05", "outputs":{p.name:sha(p) for p in sorted(out.glob("*")) if p.is_file() and p.name!="source_manifest.json"}}
    write_json(out/"source_manifest.json",manifest)
    print(json.dumps({"output":str(out), "coverage":summary["coverage"], "omnibus":friedman,
                      "pairs":pairs,"time_descriptions":summary["peak_and_time_descriptive"],"late_share":summary["beimen_late_share"],
                      "station_descriptive":station_desc,"year_omnibus":year_friedman},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--snapshot",type=Path)
    parser.add_argument("--primary-csv",type=Path,help="Paired Beimen sample from the matching statistics snapshot")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--permutations",type=int,default=PERMUTATIONS)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),ensure_ascii=False,indent=2))
    else:
        if args.output is None:
            args.output=args.root/"results/research_review/2026-10-05/station_time"
        run(args)
