"""Calendar-matched eight-week sensitivity baseline; preserves the original 28d tables."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json

import duckdb
import pandas as pd

from import_event_finance import sql_value
from load_data import event_snapshot, mysql
from project_paths import ROOT, PROCESSED

STATIONS = ["北門", "大橋頭站", "雙連", "民權西路"]
POLICY = "same_weekday_calendar_56d_nearest4_v1"
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS taipei_metro_analysis.analysis_calendar_baseline (
 event_date DATE NOT NULL,
 station VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
 policy_name VARCHAR(64) NOT NULL,
 window_start_hour SMALLINT UNSIGNED NOT NULL,
 window_end_hour_exclusive SMALLINT UNSIGNED NOT NULL,
 baseline_n_days TINYINT UNSIGNED NOT NULL,
 baseline_dates JSON NOT NULL,
 baseline_origin_mean DOUBLE NULL,
 event_origin_total BIGINT UNSIGNED NULL,
 eligible BOOLEAN NOT NULL,
 source_run_sha256 CHAR(64) NOT NULL,
 PRIMARY KEY(event_date,station,policy_name,window_start_hour,window_end_hour_exclusive),
 FOREIGN KEY(event_date) REFERENCES taipei_metro_analysis.dadaocheng_fireworks_events(event_date),
 FOREIGN KEY(station) REFERENCES taipei_metro_analysis.analysis_station_label(station),
 CHECK(baseline_n_days BETWEEN 0 AND 4),
 CHECK(window_start_hour=18 AND window_end_hour_exclusive=24)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def choose_candidates(candidates: pd.DataFrame, event_days: set, calendar: dict) -> pd.DataFrame:
    """Classify all candidates before selecting by date, never by passenger outcome."""
    rows = candidates.copy().sort_values(["event_date", "station", "lag_days"])
    def reason(r):
        if r.candidate_date in event_days:
            return "event_date_excluded"
        a, b = calendar.get(r.event_date), calendar.get(r.candidate_date)
        if a is None or b is None:
            return "calendar_missing"
        if (a["is_holiday"], a["is_makeup_workday"]) != (b["is_holiday"], b["is_makeup_workday"]):
            return "calendar_class_mismatch"
        if (b["day_note"] or "") != (a["day_note"] or ""):
            return "holiday_note_mismatch"
        if r.complete_hours != 6 or r.observed_hours != 6:
            return "missing_or_incomplete_hours"
        return "eligible_candidate"
    rows["reason"] = rows.apply(reason, axis=1)
    rows["selected"] = False
    for _, g in rows.groupby(["event_date", "station"]):
        idx = g.loc[g.reason.eq("eligible_candidate")].head(4).index
        rows.loc[idx, "selected"] = True
    rows.loc[rows.reason.eq("eligible_candidate") & ~rows.selected, "reason"] = "beyond_nearest_four"
    rows.loc[rows.selected, "reason"] = "selected"
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mysql", action="store_true")
    parser.add_argument("--login-path", default="codex-local")
    args = parser.parse_args()
    events = event_snapshot(args.login_path)
    days = {r["event_date"] for r in events}
    cal = pd.read_csv(ROOT/"data/original_data/calendar/calendar_2017_2026.csv", keep_default_na=False)
    calendar = cal.set_index("calendar_date").to_dict("index")
    grid = pd.DataFrame([dict(event_date=e["event_date"], station=s, lag_days=7*k,
                             candidate_date=(datetime.fromisoformat(e["event_date"])-timedelta(days=7*k)).strftime("%Y-%m-%d"))
                         for e in events for s in STATIONS for k in range(1,9)])
    con = duckdb.connect()
    con.register("grid", grid)
    candidates = con.execute("""SELECT g.*,count(s.origin_count) AS observed_hours,
      count(*) FILTER(WHERE s.origin_complete) AS complete_hours,
      CASE WHEN count(s.origin_count)=6 AND count(*) FILTER(WHERE s.origin_complete)=6
        THEN sum(s.origin_count) END AS candidate_origin_total
      FROM grid g LEFT JOIN read_parquet(?) s ON s.service_date=CAST(g.candidate_date AS DATE)
       AND s.station=g.station AND s.source_hour>=18 AND s.source_hour<24
      GROUP BY ALL""", [str(PROCESSED/"station_hourly.parquet")]).df()
    candidates = choose_candidates(candidates, days, calendar)
    windows = pd.read_csv(PROCESSED/"regression_event_station.csv").set_index(["event_date", "station"])
    digest = hashlib.sha256((ROOT/"results/preprocessing/summary.json").read_bytes()+
                            (ROOT/"data/original_data/calendar/calendar_2017_2026.csv").read_bytes()+
                            __file__.encode()+open(__file__,"rb").read()).hexdigest()
    result = []
    status = {e["event_date"]:e["event_status"] for e in events}
    for (day, station), group in candidates.groupby(["event_date", "station"]):
        selected = group.loc[group.selected]
        window = windows.loc[(day,station)]
        n = len(selected)
        result.append(dict(event_date=day,station=station,policy_name=POLICY,
                           window_start_hour=18,window_end_hour_exclusive=24,
                           baseline_n_days=n,baseline_dates=json.dumps(selected.candidate_date.tolist()),
                           baseline_origin_mean=float(selected.candidate_origin_total.mean()) if n else None,
                           event_origin_total=int(window.origin_count_window) if pd.notna(window.origin_count_window) else None,
                           eligible=int(status[day]=="held" and n>=2 and bool(window.origin_window_complete)),
                           source_run_sha256=digest))
    output = PROCESSED/"calendar_baseline"
    output.mkdir(exist_ok=True)
    candidates.to_csv(output/"candidate_audit.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(result).to_csv(output/"event_station_baseline.csv",index=False,encoding="utf-8-sig")
    if args.mysql:
        mysql(SCHEMA_SQL,args.login_path)
        cols=list(result[0]); updates=",".join(f"{c}=incoming.{c}" for c in cols if c not in {"event_date","station","policy_name","window_start_hour","window_end_hour_exclusive"})
        values=",".join("("+",".join(sql_value(r[c]) for c in cols)+")" for r in result)
        mysql(f"START TRANSACTION; INSERT INTO taipei_metro_analysis.analysis_calendar_baseline ({','.join(cols)}) VALUES {values} AS incoming ON DUPLICATE KEY UPDATE {updates}; COMMIT;", args.login_path)
        count=int(mysql(f"SELECT COUNT(*) FROM taipei_metro_analysis.analysis_calendar_baseline WHERE source_run_sha256='{digest}';",args.login_path))
        if count != len(result):raise RuntimeError("Baseline database row count mismatch")
    summary=dict(policy_name=POLICY,generated_at_utc=datetime.now(timezone.utc).isoformat(),
                 rules="Previous 8 same weekdays; exclude all listed event dates including cancelled/postponed; same holiday/makeup class and day note; six complete 18-24 hours; nearest <=4; minimum 2.",
                 warning="Sensitivity analysis, not causal effect. Other festival activities and unlisted events can still affect comparison days; 8-week trend/season changes remain.",
                 rows=len(result),eligible_beimen_events=sum(r["eligible"] for r in result if r["station"]=="北門"),
                 excluded_beimen_dates=[r["event_date"] for r in result if r["station"]=="北門" and not r["eligible"]],
                 source_run_sha256=digest)
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":main()
