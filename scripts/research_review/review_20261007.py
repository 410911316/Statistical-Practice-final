"""Independent 2026-10-07 review: read-only SQL and independent source reconstruction.

Does not import author analysis functions. New methods are deliberately excluded.
Private source snapshots are saved under .local; only audit summaries are public.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import permutations
import hashlib
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from load_data import mysql

STATIONS = ["北門", "大橋頭站", "雙連", "民權西路"]
PRIVATE = ROOT / ".local/research_review/2026-10-07-independent-review"
OUTPUT = ROOT / "results/research_review/2026-10-07-refresh/review.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_table(table, fields, where="", order=""):
    columns = ",".join(f"'{field}',{field}" for field in fields)
    return f"SELECT JSON_OBJECT({columns}) FROM taipei_metro_analysis.{table} {where} {order};"


def source_snapshot(reuse=False):
    path = PRIVATE / "source_snapshot.json"
    if reuse:
        return json.loads(path.read_text(encoding="utf-8"))
    station_literals = ",".join("CONVERT(0x" + s.encode().hex() + " USING utf8mb4)" for s in STATIONS)
    specs = {
        "events": read_table("dadaocheng_fireworks_events", ["event_date", "event_status"], order="ORDER BY event_date"),
        "windows": read_table("v_regression_event_station", ["event_date", "station", "origin_count_window", "baseline_origin_window", "baseline_min_days", "origin_window_complete", "peak_origin_hour_count"], f"WHERE station IN ({station_literals}) AND window_start_hour=18 AND window_end_hour_exclusive=24", "ORDER BY event_date,station"),
        "hourly": read_table("analysis_event_station_hourly", ["event_date", "hour_start", "station", "origin_count", "origin_complete", "baseline_origin_mean_28d", "baseline_n_days"], f"WHERE station IN ({station_literals}) AND HOUR(hour_start)>=18 AND HOUR(hour_start)<24 AND DATE(hour_start)=event_date", "ORDER BY event_date,station,hour_start"),
        "calendar_baseline": read_table("analysis_calendar_baseline", ["event_date", "station", "policy_name", "baseline_n_days", "baseline_dates", "baseline_origin_mean", "event_origin_total", "eligible"], order="ORDER BY event_date,station"),
        "weather_mapping": read_table("event_weather_station", ["event_date", "station_id", "station_role"], order="ORDER BY event_date,station_id"),
        "weather": read_table("v_weather_verified", ["station_id", "observed_at", "official_report_hour_end", "values_match", "air_temperature_c", "temperature_quality_flag", "mean_wind_speed_m_s", "wind_speed_quality_flag", "hourly_precipitation_mm", "precipitation_raw_value", "precipitation_status", "precipitation_quality_flag", "total_cloud_amount_tenths", "cloud_measurement_type", "cloud_method_verified", "effective_cloud_review_required", "source_file_sha256", "response_sha256"], order="ORDER BY station_id,observed_at"),
        "finance": read_table("v_dadaocheng_annual_finance", ["event_year", "planned_budget_ntd", "required_publicity_cost_ntd", "budget_basis", "budget_scope_note"], order="ORDER BY event_year"),
    }
    result = {"extracted_at_utc": datetime.now(timezone.utc).isoformat(), "read_only": True}
    for name, query in specs.items():
        response = mysql("START TRANSACTION READ ONLY;\n" + query + "\nROLLBACK;")
        result[name] = [json.loads(s) for s in response.splitlines() if s]
    result["queries"] = specs
    save(path, result)
    return result


def sign(values):
    a = np.asarray(values)
    plus, minus, ties = int((a > 0).sum()), int((a < 0).sum()), int((a == 0).sum())
    from math import comb
    n = plus + minus
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(plus, minus) + 1)) / 2**n)
    return {"positive": plus, "negative": minus, "ties": ties, "p": p}


def contrast(values):
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    se = sd / np.sqrt(n)
    F = mean * mean / (se * se)
    ci = mean + np.array([-1, 1]) * stats.t.ppf(.975, n - 1) * se
    return {"n": n, "mean": mean, "sd": sd, "F": float(F), "df": [1, n - 1], "p": float(stats.f.sf(F, 1, n - 1)), "geometric_ratio_plus_one": float(np.exp(mean)), "geometric_ratio_ci95": np.exp(ci).tolist(), "sign": sign(a)}


def check_cached_weather(rows):
    from decimal import Decimal, InvalidOperation
    cache = ROOT / "data/original_data/weather_verification"
    cached = {}
    checked = 0
    def equal(a, b):
        if a in (None, "") or b in (None, ""):
            return a in (None, "") and b in (None, "")
        try:
            return Decimal(str(a)) == Decimal(str(b))
        except InvalidOperation:
            return str(a) == str(b)
    fields = [("air_temperature_c", "AirTemperature", "Instantaneous"),
              ("temperature_quality_flag", "AirTemperature", "Instantaneousf"),
              ("mean_wind_speed_m_s", "WindSpeed", "Mean"),
              ("wind_speed_quality_flag", "WindSpeed", "Meanf"),
              ("precipitation_raw_value", "Precipitation", "Accumulation"),
              ("precipitation_quality_flag", "Precipitation", "Accumulationf")]
    for row in rows:
        stamp = pd.Timestamp(row["observed_at"])
        key = (row["station_id"], stamp.strftime("%Y-%m-%d"))
        if key not in cached:
            path = cache / f"{key[0]}_{key[1]}.json"
            raw = json.loads(path.read_bytes())
            assert raw["code"] == 200 and raw["metadata"]["timezone"] == "+08:00"
            assert len(raw["data"]) == 1 and raw["data"][0]["StationID"] == key[0]
            records = raw["data"][0]["dts"]
            assert len(records) == len({r["DataTime"] for r in records}) == 24
            cached[key] = ({r["DataTime"]: r for r in records}, sha(path))
        records, digest = cached[key]
        assert digest == row["response_sha256"]
        remote = records[stamp.isoformat()]
        for local_field, group, field in fields:
            assert equal(row[local_field], remote.get(group, {}).get(field)), (key, stamp, local_field)
        cloud_field = {"manual_instantaneous": "Instantaneous", "satellite_retrieved": "SatRetrieved"}.get(row["cloud_measurement_type"])
        if cloud_field:
            assert equal(row["total_cloud_amount_tenths"], remote["TotalCloudAmount"][cloud_field])
        else:
            assert row["total_cloud_amount_tenths"] is None
            assert remote.get("TotalCloudAmount", {}).get("Instantaneous") is None
            assert remote.get("TotalCloudAmount", {}).get("SatRetrieved") is None
        expected_end = stamp.ceil("h") if stamp.strftime("%H:%M") == "23:59" else stamp
        assert expected_end == pd.Timestamp(row["official_report_hour_end"])
        checked += 1
    return {"rows_checked": checked, "cached_station_day_responses": len(cached), "response_hashes_match_sql": True, "remote_redownload_performed": False}


def preflight(data):
    events = pd.DataFrame(data["events"])
    events.event_date = pd.to_datetime(events.event_date).dt.strftime("%Y-%m-%d")
    event_days = set(events.event_date)
    windows = pd.DataFrame(data["windows"])
    windows.event_date = pd.to_datetime(windows.event_date).dt.strftime("%Y-%m-%d")
    windows = windows.merge(events, on="event_date", validate="many_to_one")
    windows["year"] = pd.to_datetime(windows.event_date).dt.year
    hourly = pd.DataFrame(data["hourly"])
    hourly.event_date = pd.to_datetime(hourly.event_date).dt.strftime("%Y-%m-%d")
    hourly["hour"] = pd.to_datetime(hourly.hour_start).dt.hour
    assert not hourly.duplicated(["event_date", "station", "hour"]).any()
    assert len(hourly) == 624 and len(windows) == 104
    for (_, _), g in hourly.groupby(["event_date", "station"]):
        assert len(g) == 6 and set(g.hour) == set(range(18, 24))
    assert hourly.origin_complete.eq(1).all() and hourly.origin_count.notna().all()
    db_hourly = hourly.set_index(["event_date", "station", "hour"]).sort_index()
    con = duckdb.connect()
    local = con.execute("SELECT service_date,source_hour,station,origin_count,origin_complete FROM read_parquet(?) WHERE station IN (?,?,?,?) AND source_hour>=18 AND source_hour<24", [str(ROOT / "data/processed/station_hourly.parquet"), *STATIONS]).df()
    local["event_date"] = pd.to_datetime(local.service_date).dt.strftime("%Y-%m-%d")
    local = local.rename(columns={"source_hour": "hour"}).set_index(["event_date", "station", "hour"]).sort_index()
    assert not local.index.duplicated().any()
    selected_local = local.loc[db_hourly.index]
    assert np.array_equal(db_hourly.origin_count, selected_local.origin_count)
    assert np.array_equal(db_hourly.origin_complete, selected_local.origin_complete.astype(int))
    calendar = pd.read_csv(ROOT / "data/original_data/calendar/calendar_2017_2026.csv", keep_default_na=False).set_index("calendar_date").to_dict("index")
    baseline8 = {(str(r["event_date"])[:10], r["station"]): r for r in data["calendar_baseline"]}
    four_reasons = {"event_excluded": 0, "missing_hours": 0, "usable": 0}
    eight_reasons = {}
    reconstructed = []
    reconstructed_hourly = []
    def candidates(day, station, horizon):
        result = []
        for k in range(1, horizon + 1):
            d = (pd.Timestamp(day) - pd.Timedelta(days=7*k)).strftime("%Y-%m-%d")
            idx = [(d, station, h) for h in range(18, 24)]
            g = local.reindex(idx)
            valid = g.origin_count.notna().all() and g.origin_complete.fillna(False).all()
            result.append((d, g, valid))
        return result
    for row in windows.itertuples():
        h = hourly.loc[hourly.event_date.eq(row.event_date) & hourly.station.eq(row.station)].sort_values("hour")
        assert h.origin_count.sum() == row.origin_count_window and h.origin_count.max() == row.peak_origin_hour_count
        selected4 = []
        for d, g, valid in candidates(row.event_date, row.station, 4):
            reason = "event_excluded" if d in event_days else "usable" if valid else "missing_hours"
            four_reasons[reason] += 1
            if reason == "usable":
                selected4.append(g.origin_count.to_numpy())
        assert len(selected4) == row.baseline_min_days
        if selected4:
            hourly_average = np.mean(selected4, axis=0)
            assert np.allclose(hourly_average, h.baseline_origin_mean_28d)
            assert np.isclose(sum(hourly_average), row.baseline_origin_window)
        else:
            assert h.baseline_origin_mean_28d.isna().all() and pd.isna(row.baseline_origin_window)
        selected8 = []
        for d, g, valid in candidates(row.event_date, row.station, 8):
            a, b = calendar[row.event_date], calendar[d]
            reason = ("event_excluded" if d in event_days else
                      "class_mismatch" if (a["is_holiday"], a["is_makeup_workday"]) != (b["is_holiday"], b["is_makeup_workday"]) else
                      "note_mismatch" if (a["day_note"] or "") != (b["day_note"] or "") else
                      "missing_hours" if not valid else "selected" if len(selected8) < 4 else "beyond_nearest_four")
            eight_reasons[reason] = eight_reasons.get(reason, 0) + 1
            if reason == "selected":
                selected8.append((d, g.origin_count.to_numpy()))
        expected8 = baseline8[(row.event_date, row.station)]
        assert expected8["baseline_dates"] == [d for d, _ in selected8]
        assert expected8["baseline_n_days"] == len(selected8)
        b8 = float(np.mean([x.sum() for _, x in selected8])) if selected8 else None
        assert b8 is None and expected8["baseline_origin_mean"] is None or np.isclose(b8, expected8["baseline_origin_mean"])
        eligible4 = row.event_status == "held" and len(selected4) >= 2 and row.origin_window_complete == 1
        eligible8 = row.event_status == "held" and len(selected8) >= 2 and row.origin_window_complete == 1
        assert bool(expected8["eligible"]) == eligible8
        base8_hours = np.mean([x for _, x in selected8], axis=0) if selected8 else [np.nan]*6
        for hour_row, base8_hour in zip(h.itertuples(), base8_hours):
            reconstructed_hourly.append({"event_date": row.event_date, "station": row.station, "hour": int(hour_row.hour), "observed": hour_row.origin_count, "baseline4": hour_row.baseline_origin_mean_28d, "baseline8": base8_hour})
        reconstructed.append({"event_date": row.event_date, "station": row.station, "year": int(row.year), "held": row.event_status == "held", "eligible4": eligible4, "eligible8": eligible8, "observed": row.origin_count_window, "baseline4": row.baseline_origin_window, "baseline8": b8, "late_observed_share": h.loc[h.hour.ge(21), "origin_count"].sum()/row.origin_count_window, "late_baseline_share4": h.loc[h.hour.ge(21), "baseline_origin_mean_28d"].sum()/row.baseline_origin_window if pd.notna(row.baseline_origin_window) else None, "late_baseline_share8": float(np.mean([x[3:].sum() for _, x in selected8])) / b8 if b8 else None})
    r = pd.DataFrame(reconstructed)
    r.to_csv(PRIVATE / "independent_event_station.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(reconstructed_hourly).to_csv(PRIVATE / "independent_hourly_baselines.csv", index=False, encoding="utf-8-sig")
    context = pd.read_csv(ROOT / "data/processed/event_station_hourly_context.csv")
    assert len(context) == 624 and not context.duplicated(["event_date", "station", "hour_start"]).any()
    assert context.weather_station_id.notna().sum() == 504
    weather = pd.DataFrame(data["weather"])
    assert len(weather) == 1008 and weather.values_match.eq(1).all()
    assert weather.cloud_method_verified.eq(1).all()
    mapping = pd.DataFrame(data["weather_mapping"])
    assert len(mapping) == 42 and len(mapping.loc[mapping.station_role.eq("primary")]) == 21
    weather["event_date"] = pd.to_datetime(weather.observed_at).dt.strftime("%Y-%m-%d")
    primary_weather = weather.merge(mapping.loc[mapping.station_role.eq("primary")], on=["event_date", "station_id"], validate="many_to_one")
    primary_weather["report_hour"] = (pd.to_datetime(primary_weather.official_report_hour_end) - pd.to_datetime(primary_weather.event_date)).dt.total_seconds() / 3600
    selected_weather = primary_weather.loc[primary_weather.report_hour.between(19, 24)].copy()
    assert len(selected_weather) == 126 and selected_weather.groupby("event_date").size().eq(6).all()
    assert selected_weather.temperature_quality_flag.fillna("").eq("").all()
    assert selected_weather.wind_speed_quality_flag.fillna("").eq("").all()
    assert selected_weather[["air_temperature_c", "mean_wind_speed_m_s"]].notna().all().all()
    means = selected_weather.groupby("event_date")[["air_temperature_c", "mean_wind_speed_m_s"]].mean()
    means.to_csv(PRIVATE / "independent_weather_means.csv", encoding="utf-8-sig")
    old = primary_weather.loc[pd.to_datetime(primary_weather.observed_at).dt.strftime("%H:%M:%S").isin([f"{h:02}:00:00" for h in range(18, 24)])].groupby("event_date")[["air_temperature_c", "mean_wind_speed_m_s"]].mean()
    financial = pd.DataFrame(data["finance"])
    summary = {"events": len(events), "status_counts": events.event_status.value_counts().to_dict(), "four_station_hours": len(hourly), "four_station_windows": len(windows), "fourweek_eligible_held_dates": int(r.loc[r.station.eq("北門"), "eligible4"].sum()), "eightweek_eligible_held_dates": int(r.loc[r.station.eq("北門"), "eligible8"].sum()), "fourweek_candidate_counts": four_reasons, "eightweek_candidate_counts": eight_reasons, "calendar_days": len(calendar), "context_weather_rows": int(context.weather_station_id.notna().sum()), "weather_rows_values_matched": len(weather), "weather_station_day_mappings": len(mapping), "weather_primary_rows": len(primary_weather), "selected_weather_rows": len(selected_weather), "raw_2359_rows": int(pd.to_datetime(weather.observed_at).dt.strftime("%H:%M").eq("23:59").sum()), "weather_hours": [19, 20, 21, 22, 23, 24], "weather_changed_event_counts": {c: int((~np.isclose(means[c], old[c])).sum()) for c in means.columns}, "cached_weather_source_reconciliation": check_cached_weather(data["weather"]), "finance_annual_rows": len(financial), "budget_years": int(financial.planned_budget_ntd.notna().sum()), "publicity_cost_years": int(financial.required_publicity_cost_ntd.notna().sum()), "annual_finance": data["finance"], "financial_unit": "event_year; must not duplicate annual amounts as independent event or hourly observations", "sql_parquet_core_hourly_exact_match": True, "fourweek_hourly_baselines_reconstructed": 104, "eightweek_baselines_reconstructed": 104}
    return summary, r, {"legacy_18_23": old, "verified_hour_end_19_24": means}


def independent_statistics(r, means, horizon):
    d = r.loc[r[f"eligible{horizon}"]].copy()
    d["response"] = np.log1p(d.observed) - np.log1p(d[f"baseline{horizon}"])
    b = d.loc[d.station.eq("北門")].sort_values("event_date").copy()
    years = b.groupby("year").response.mean()
    ci = np.quantile(np.random.default_rng(20261005).choice(years.to_numpy(), size=(20000, len(years)), replace=True).mean(axis=1), [.025, .975])
    matrix = d.pivot(index="event_date", columns="station", values="response").reindex(columns=STATIONS).sort_index()
    ranks = stats.rankdata(matrix.to_numpy(), axis=1)
    n, k = ranks.shape
    tie_sum = sum(sum(c**3 - c for c in np.unique(row, return_counts=True)[1]) for row in matrix.to_numpy())
    correction = 1 - tie_sum / (n * (k**3-k))
    q = (12*np.square(ranks.sum(axis=0)).sum() / (n*k*(k+1)) - 3*n*(k+1))/correction
    seed = 20261007
    rng = np.random.default_rng(seed)
    perms = np.array(list(permutations(range(k))))
    exceeded = 0
    for start in range(0, 199999, 2000):
        size = min(2000, 199999-start)
        indices = perms[rng.integers(24, size=(size, n))]
        shuffled = np.take_along_axis(np.broadcast_to(ranks, (size, n, k)), indices, axis=2)
        simulated = (12*np.square(shuffled.sum(axis=1)).sum(axis=1)/(n*k*(k+1))-3*n*(k+1))/correction
        exceeded += int((simulated >= q-1e-10).sum())
    pairs = {s: sign(matrix["北門"] - matrix[s]) for s in STATIONS[1:]}
    late = b.late_observed_share - b[f"late_baseline_share{horizon}"]
    weather = {}
    for mode, means_frame in means.items():
        joined = b.merge(means_frame, on="event_date", validate="one_to_one")
        weather[mode] = {}
        for j, col in enumerate(means_frame.columns):
            x = stats.rankdata(joined[col]); y = stats.rankdata(joined.response)
            x = x-x.mean(); y = y-y.mean()
            den = np.sqrt(np.square(x).sum()*np.square(y).sum())
            rho = float(x@y/den)
            rng = np.random.default_rng(seed+j)
            extreme = 0
            for start in range(0, 99999, 1000):
                size = min(1000, 99999-start)
                permuted = np.array([rng.permutation(x) for _ in range(size)])
                simulated = permuted@y/den
                extreme += int((np.abs(simulated) >= abs(rho) - 100*np.finfo(float).eps*abs(rho)).sum())
            weather[mode][col] = {"rho": rho, "permutation_p_independent_seed": (extreme+1)/100000, "seed": seed+j, "n": len(joined)}
    x = np.log1p(b[f"baseline{horizon}"].to_numpy()); y = np.log1p(b.observed.to_numpy())
    design = np.column_stack([np.ones(len(b)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y-design@beta
    r2 = 1 - np.square(residuals).sum()/np.square(y-y.mean()).sum()
    rows = []
    for year in sorted(b.year.unique()):
        test = b.year.eq(year).to_numpy(); train = ~test
        coeff = np.linalg.lstsq(design[train], y[train], rcond=None)[0]
        for date, observed, predicted in zip(b.loc[test, "event_date"], y[test], design[test]@coeff):
            rows.append({"event_date": date, "year": int(year), "observed": float(observed), "fitted": float(predicted), "intercept_only": float(y[train].mean())})
    folds = pd.DataFrame(rows)
    folds.to_csv(PRIVATE / f"independent_loyo_{horizon}.csv", index=False)
    ols = {"n": len(b), "intercept": float(beta[0]), "slope": float(beta[1]), "r_squared": float(r2)}
    for col in ["fitted", "intercept_only"]:
        err = folds.observed-folds[col]
        ols[f"{col}_mae"] = float(np.abs(err).mean()); ols[f"{col}_rmse"] = float(np.sqrt(np.square(err).mean()))
    return {"primary": contrast(b.response), "primary_dates": list(b.event_date), "years": list(map(int, years.index)), "year_equal_weight": contrast(years), "year_bootstrap_ci95": ci.tolist(), "leave_one_year_ratio_range": [float(min(np.exp(b.loc[b.year.ne(y), "response"].mean()) for y in years.index)), float(max(np.exp(b.loc[b.year.ne(y), "response"].mean()) for y in years.index))], "friedman": {"Q": float(q), "W": float(q/(n*(k-1))), "mean_ranks": ranks.mean(axis=0).tolist(), "seed": seed, "exceedances": exceeded, "permutation_p": (exceeded+1)/200000}, "station_pairs": pairs, "late_share": {"observed_mean": float(b.late_observed_share.mean()), "baseline_mean": float(b[f"late_baseline_share{horizon}"].mean()), "difference_mean": float(late.mean()), "sign": sign(late)}, "weather": weather, "ols": ols}


def check_author_outputs(directory, independent, r, means):
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    def close(a, b):
        assert np.allclose(a, b, rtol=1e-9, atol=1e-10, equal_nan=True), (a, b)
    def holm(p):
        order = sorted(range(len(p)), key=lambda i: p[i])
        result = [0.0]*len(p); largest = 0.0
        for rank, i in enumerate(order):
            largest = max(largest, p[i]*(len(p)-rank)); result[i] = min(1, largest)
        return result
    checks = {}
    source_hours = pd.read_csv(PRIVATE / "independent_hourly_baselines.csv").set_index(["event_date", "station", "hour"])
    for policy, key, horizon in [("four_week_primary", "fourweek", 4), ("eight_week_sensitivity", "eightweek_existing_policy_sensitivity", 8)]:
        base = directory / policy; known = independent[key]
        basic = read(base / "statistics/summary.json")
        stations = read(base / "station_time/summary.json")
        assert basic["coverage"]["beimen_primary_dates"] == known["primary_dates"]
        for section, authored in [("primary", basic["primary_F_test"]), ("year_equal_weight", basic["year_equal_weight_sensitivity"])]:
            a = known[section]
            for ka, kb in [("n", "n"), ("F", "F"), ("p", "p"), ("mean", "mean_log1p_difference"), ("sd", "sd_log1p_difference"), ("geometric_ratio_plus_one", "geometric_ratio_plus_one"), ("geometric_ratio_ci95", "geometric_ratio_plus_one_ci95")]:
                close(a[ka], authored[kb])
            close(a["sign"]["p"], authored["sign_test_two_sided_p"])
        close(known["year_bootstrap_ci95"], basic["year_bootstrap"]["mean_log1p_difference_percentile_ci95"])
        close(known["leave_one_year_ratio_range"], basic["diagnostics"]["leave_one_year_geometric_ratio_range"])
        close(known["friedman"]["Q"], stations["station_omnibus"]["friedman_Q"])
        close(known["friedman"]["W"], stations["station_omnibus"]["kendall_W"])
        close(known["friedman"]["permutation_p"], stations["station_omnibus"]["permutation"]["p_plus_one"])
        station_rows = r.loc[r[f"eligible{horizon}"]].copy()
        station_rows["response"] = np.log1p(station_rows.observed)-np.log1p(station_rows[f"baseline{horizon}"])
        year_matrix = station_rows.groupby(["year", "station"]).response.mean().unstack().reindex(columns=STATIONS)
        year_csv = pd.read_csv(base / "station_time/year_station_contrasts.csv").set_index("year").reindex(columns=STATIONS)
        close(year_matrix, year_csv)
        for pair in stations["year_equal_weight_sensitivity"]["pairwise"]:
            expected = sign(year_matrix["北門"]-year_matrix[pair["comparison"]])
            close(pair["p_two_sided_exact"], expected["p"])
        for pair in stations["station_pairwise"]:
            expected = known["station_pairs"][pair["comparison"]]
            assert pair["positive"] == expected["positive"] and pair["negative"] == expected["negative"]
            close(pair["p_two_sided_exact"], expected["p"])
        north = next(s for s in stations["station_descriptive"] if s["station"] == "北門")
        for k, v in [("observed_late_share_mean", "observed_mean"), ("baseline_late_share_mean", "baseline_mean"), ("late_share_difference_mean", "difference_mean")]:
            close(north[k], known["late_share"][v])
        t = pd.read_csv(base / "station_time/event_station_time_features.csv")
        assert len(t) == 84
        for row in t.itertuples():
            h = source_hours.loc[(row.event_date, row.station)]
            max_count = h.observed.max()
            close(row.total_18_24, h.observed.sum()); close(row.peak_hour_share, max_count/h.observed.sum())
            assert json.loads(row.peak_hours_json) == list(map(int, h.index[h.observed.eq(max_count)]))
            close(row.late_21_24_share, h.loc[h.index >= 21, "observed"].sum()/h.observed.sum())
            if h[f"baseline{horizon}"].notna().all():
                close(row.baseline_late_21_24_share, h.loc[h.index >= 21, f"baseline{horizon}"].sum()/h[f"baseline{horizon}"].sum())
        shares = pd.read_csv(base / "station_time/hourly_shares.csv")
        assert len(shares) == 504
        for row in shares.itertuples():
            h = source_hours.loc[(row.event_date, row.station)]
            close(row.observed_share, h.loc[row.hour, "observed"]/h.observed.sum())
            if h[f"baseline{horizon}"].notna().all():
                close(row.baseline_share, h.loc[row.hour, f"baseline{horizon}"]/h[f"baseline{horizon}"].sum())
        weather_p = []
        mc_comparisons = []
        for mode, folder in [("legacy_18_23", "weather_regression"), ("verified_hour_end_19_24", "weather_aligned")]:
            w = read(base / folder / "summary.json")
            merged = pd.read_csv(base / folder / "primary_weather_merged.csv").set_index("event_date").sort_index()
            assert list(merged.index) == known["primary_dates"]
            for a, col in zip(w["weather_correlations"], ["air_temperature_c", "mean_wind_speed_m_s"]):
                close(a["rho"], known["weather"][mode][col]["rho"])
                close(merged[a["variable"]], means[mode].loc[merged.index, col])
                annual = merged.groupby("year")[[a["variable"], "response_log1p_difference"]].mean()
                close(a["annual_mean_rho_descriptive"], stats.spearmanr(annual[a["variable"]], annual.response_log1p_difference).statistic)
                leave_rhos = []
                for yr in merged.year.unique():
                    remaining = merged.loc[merged.year.ne(yr)]
                    leave_rhos.append(float(stats.spearmanr(remaining[a["variable"]], remaining.response_log1p_difference).statistic))
                close(a["leave_one_year_rho_range"], [min(leave_rhos), max(leave_rhos)])
                p, q = a["permutation_p"], known["weather"][mode][col]["permutation_p_independent_seed"]
                se = np.sqrt((p*(1-p)+q*(1-q))/100000)
                assert abs(p-q) < 5*se + 1e-5, (p, q, se)
                mc_comparisons.append({"mode": mode, "variable": col, "author_p": p, "independent_seed_p": q, "difference_in_mc_standard_errors": abs(p-q)/se})
                if mode == "legacy_18_23": weather_p.append(p)
            close([v["holm_two_tests_p"] for v in w["weather_correlations"]], holm([v["permutation_p"] for v in w["weather_correlations"]]))
            fit = w["baseline_regression"]
            for ka, kb in [("intercept", "intercept"), ("slope", "slope"), ("r_squared", "r_squared_in_sample"), ("fitted_mae", "loyo_mae_log1p"), ("fitted_rmse", "loyo_rmse_log1p"), ("intercept_only_mae", "loyo_intercept_only_mae_log1p"), ("intercept_only_rmse", "loyo_intercept_only_rmse_log1p")]:
                close(known["ols"][ka], fit[kb])
            folds = pd.read_csv(base / folder / "regression_loyo.csv").set_index("event_date").sort_index()
            expected = pd.read_csv(PRIVATE / f"independent_loyo_{horizon}.csv").set_index("event_date").sort_index()
            for ka, kb in [("observed_log1p", "observed"), ("fitted_holdout_log1p", "fitted"), ("intercept_only_holdout_log1p", "intercept_only")]:
                close(folds[ka], expected[kb])
            eligible = r.loc[r.station.eq("北門") & r[f"eligible{horizon}"]].set_index("event_date").sort_index()
            x, y = np.log1p(eligible[f"baseline{horizon}"].to_numpy()), np.log1p(eligible.observed.to_numpy())
            X = np.column_stack([np.ones(len(x)), x]); beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residual = y-X@beta; mse = np.square(residual).sum()/(len(x)-2)
            inv = np.linalg.inv(X.T@X); leverage = np.einsum("ij,jk,ik->i", X, inv, X)
            cook = (residual**2/(2*mse))*leverage/(1-leverage)**2
            diag = pd.read_csv(base / folder / "regression_event_diagnostics.csv").set_index("event_date").sort_index()
            for col, array in [("residual_log1p", residual), ("fitted_log1p", X@beta), ("leverage", leverage), ("cooks_distance", cook)]: close(diag[col], array)
            q = stats.t.ppf(.975, len(x)-2); errors = np.sqrt(np.diag(mse*inv))
            close(fit["slope_ci95"], beta[1]+np.array([-1, 1])*q*errors[1])
            close(fit["intercept_ci95"], beta[0]+np.array([-1, 1])*q*errors[0])
        family = read(base / "extended_statistics/summary.json")["tests"]
        raw = [known["friedman"]["permutation_p"], *[known["station_pairs"][s]["p"] for s in STATIONS[1:]], known["late_share"]["sign"]["p"], *weather_p]
        assert len(family) == 7
        close([x["raw_p"] for x in family], raw)
        close([x["holm_seven_p"] for x in family], holm(raw))
        checks[policy] = {"all_numeric_checks_pass": True, "time_feature_rows": len(t), "hourly_share_rows": len(shares), "holm_seven": holm(raw), "weather_monte_carlo_comparisons": mc_comparisons}
    hashes = 0
    for path in directory.rglob("source_manifest.json"):
        manifest = read(path)
        for name, digest in manifest.get("outputs", {}).items():
            assert sha(path.parent/name) == digest, (path, name)
            hashes += 1
    checks["manifest_output_entries_verified"] = hashes
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-snapshot", action="store_true")
    parser.add_argument("--check-author", type=Path, help="Check completed author outputs against independent results")
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    data = source_snapshot(args.reuse_snapshot)
    coverage, r, means = preflight(data)
    result = {"review_status": "pending_author_outputs_and_visual_review", "review_generated_at_utc": datetime.now(timezone.utc).isoformat(), "source_snapshot_extracted_at_utc": data["extracted_at_utc"], "source_snapshot_sha256": sha(PRIVATE / "source_snapshot.json"), "review_script_sha256": sha(__file__), "independent_of_author_functions": True, "preflight": coverage, "fourweek": independent_statistics(r, means, 4), "eightweek_existing_policy_sensitivity": independent_statistics(r, means, 8), "limitations": ["Source coverage and calculations do not establish causality or event independence.", "SQL weather was reconciled against cached CODiS responses; no remote redownload was performed.", "SQL tables were read in separate read-only transactions, not one atomic snapshot.", "Whole-year holdout may train on future years and is not forward forecasting.", "Holm does not fix invalid single-test assumptions or control all historical exploration.", "Annual financial amounts are not event-level independent predictors."]}
    if args.check_author:
        result["author_comparison"] = check_author_outputs(args.check_author.resolve(), result, r, means)
        result["review_status"] = "numeric_checks_pass_pending_visual_and_reproduction_review"
    save(OUTPUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
