"""Read-only current-data inventory and annual finance description; no inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.backend.database import _read_json

DB = "taipei_metro_analysis"
FINANCE_COLUMNS = {
    "event_year": "年度",
    "event_label": "活動名稱",
    "planned_budget_ntd": "原列法定活動預算_元",
    "required_publicity_cost_ntd": "年度宣導案所需經費_元",
    "publicity_to_planned_budget_ratio": "宣導所需經費除以原列法定預算",
    "budget_basis": "預算口徑",
    "budget_scope_note": "預算範圍說明",
    "budget_source_url": "預算來源網址",
    "publicity_source_label": "宣導經費來源",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/research_review/2026-10-07-refresh/data_inventory")
    parser.add_argument("--snapshot", type=Path, default=ROOT / ".local/research_review/2026-10-07-refresh/inventory_snapshot.json")
    parser.add_argument("--reuse-snapshot", action="store_true")
    args = parser.parse_args()
    pairs = ",".join(f"'{name}',{name}" for name in FINANCE_COLUMNS)
    sql = f"""
SELECT JSON_OBJECT('kind','finance','record',JSON_OBJECT({pairs}))
FROM {DB}.v_dadaocheng_annual_finance ORDER BY event_year;
SELECT JSON_OBJECT('kind','status','record',JSON_OBJECT('status',event_status,'dates',COUNT(*)))
FROM {DB}.dadaocheng_fireworks_events GROUP BY event_status ORDER BY event_status;
SELECT JSON_OBJECT('kind','cloud','record',JSON_OBJECT('method',cloud_measurement_type,'rows',COUNT(*),
'observed_values',COUNT(total_cloud_amount_tenths)))
FROM {DB}.v_weather_verified GROUP BY cloud_measurement_type ORDER BY cloud_measurement_type;
SELECT JSON_OBJECT('kind','coverage','record',JSON_OBJECT(
'calendar_days',(SELECT COUNT(*) FROM {DB}.calendar_day),
'verified_weather_rows',(SELECT COUNT(*) FROM {DB}.v_weather_verified WHERE values_match=1),
'cloud_review_pending',(SELECT COUNT(*) FROM {DB}.v_weather_verified WHERE effective_cloud_review_required<>0),
'timing_dates',(SELECT COUNT(*) FROM {DB}.event_timing_context),
'nominal_show_ends',(SELECT COUNT(nominal_show_end) FROM {DB}.event_timing_context),
'actual_show_ends',(SELECT COUNT(actual_show_end) FROM {DB}.event_timing_context),
'calendar_baseline_windows',(SELECT COUNT(*) FROM {DB}.analysis_calendar_baseline),
'all_station_event_hours',(SELECT COUNT(*) FROM {DB}.analysis_event_station_hourly)));
"""
    if args.reuse_snapshot:
        snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    else:
        pre_path = ROOT / "results/preprocessing/summary.json"
        prep = json.loads(pre_path.read_text(encoding="utf-8"))
        month_paths = [ROOT / f"results/preprocessing/monthly/{month}.json" for month in prep["months"]]
        context_path = ROOT / "data/processed/event_station_hourly_context.csv"
        context = pd.read_csv(context_path)
        snapshot = {
            "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
            "query_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            "database_records": _read_json(sql),
            "preprocessing": prep,
            "monthly_quality": [json.loads(path.read_text(encoding="utf-8")) for path in month_paths],
            "context_rows": len(context),
            "file_sha256": {str(path.relative_to(ROOT)): sha(path) for path in [pre_path, context_path, *month_paths]},
        }
        write_json(args.snapshot, snapshot)
    finance = pd.DataFrame([r["record"] for r in snapshot["database_records"] if r["kind"] == "finance"])[list(FINANCE_COLUMNS)]
    if finance.event_year.duplicated().any():
        raise ValueError("Annual finance keys must be unique")
    valid = finance.required_publicity_cost_ntd.notna()
    for row in finance.loc[valid].itertuples(index=False):
        expected = row.required_publicity_cost_ntd / row.planned_budget_ntd
        if abs(expected - row.publicity_to_planned_budget_ratio) > 0.00000051:
            raise ValueError("Finance ratio differs from required publicity / planned budget")
    if finance.loc[~valid, "publicity_to_planned_budget_ratio"].notna().any():
        raise ValueError("Unknown publicity must have an unknown ratio")
    args.output.mkdir(parents=True, exist_ok=True)
    for column in ["event_year", "planned_budget_ntd", "required_publicity_cost_ntd"]:
        finance[column] = finance[column].astype("Int64")
    chinese_finance = finance.rename(columns=FINANCE_COLUMNS)
    chinese_finance.insert(5, "宣導所需經費占原列法定預算_百分比", (100 * finance.required_publicity_cost_ntd / finance.planned_budget_ntd).round(2))
    chinese_finance.to_csv(args.output / "年度預算與宣導經費.csv", index=False, encoding="utf-8-sig")
    monthly = pd.DataFrame(snapshot["monthly_quality"])
    monthly[["source_file", "source_rows", "clean_rows", "invalid_rows", "conflicting_keys", "exact_duplicate_rows_removed", "passengers"]].to_csv(args.output / "monthly_data_quality.csv", index=False, encoding="utf-8-sig")
    prep = snapshot["preprocessing"]
    if int(monthly.clean_rows.sum()) != prep["clean_rows"]:
        raise ValueError("Monthly and total clean-row counts differ")
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_extracted_at_utc": snapshot["extracted_at_utc"],
        "source_snapshot_sha256": sha(args.snapshot),
        "script_sha256": sha(Path(__file__)),
        "file_sha256": snapshot["file_sha256"],
        "months": prep["months"], "clean_od_rows": prep["clean_rows"],
        "od_passengers": prep["od_passengers"], "context_rows": snapshot["context_rows"],
        "database_inventory": [r for r in snapshot["database_records"] if r["kind"] != "finance"],
        "annual_finance": {"years": len(finance), "publicity_years": int(valid.sum()),
                           "missing_publicity_years": finance.loc[~valid, "event_year"].astype(int).tolist()},
        "limitations": [
            "OD rows/passenger counts are not independent events; finance is one row per year.",
            "Publicity is required campaign funding; denominator is original statutory event budget. Neither is final actual spending.",
            "Missing publicity remains blank, not zero. Different campaign/event scopes can limit between-year comparison.",
            "No financial hypothesis test, regression, return-on-investment or causal inference was run.",
            "Nominal show end is not verified actual show end; source-bin destinations are not verified exit clock time.",
        ],
    }
    write_json(args.output / "inventory_summary.json", summary)
    print(json.dumps({"output": str(args.output), "clean_od_rows": prep["clean_rows"], "finance_years": len(finance), "publicity_years": int(valid.sum())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
