"""Run with: uv run python src/main.py --mysql (omit --mysql for file outputs only)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json

import duckdb

from aggregate_flows import build_analysis
from data_preprocessing import clean_month, validated_clean_cache
from load_data import event_snapshot
from project_paths import RAW, INTERIM, PROCESSED, REPORTS, ROOT, prepare_directories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--login-path",default="codex-local")
    parser.add_argument("--months",nargs="+",help="Optional YYYYMM list; default all local monthly CSVs")
    parser.add_argument("--start-hour",type=int,default=18)
    parser.add_argument("--end-hour",type=int,default=24,help="Exclusive; 26 means next day 02:00")
    parser.add_argument("--force",action="store_true")
    parser.add_argument("--reuse-clean",action="store_true",
                        help="Include hash-verified clean Parquet months even when their source CSV is archived")
    parser.add_argument("--mysql",action="store_true",help="Also upsert derived analysis tables into MySQL")
    args=parser.parse_args()
    if not 0 <= args.start_hour < args.end_hour <= 48:
        parser.error("0 <= start-hour < end-hour <= 48 is required")
    prepare_directories()
    paths=sorted(p for p in RAW.glob("20????.csv") if not args.months or p.stem in args.months)
    cached=sorted(p for p in (INTERIM/"clean_od").glob("20????.parquet")
                  if args.reuse_clean and (not args.months or p.stem in args.months)
                  and p.stem not in {r.stem for r in paths})
    included={p.stem for p in paths+cached}
    if not included or (args.months and set(args.months)-included):
        parser.error("Some requested monthly files are missing")
    con=duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=4")
    con.execute("SET temp_directory=?",[str(ROOT/".local"/"duckdb_tmp")])
    records=event_snapshot(args.login_path)
    monthly=[]
    for path in paths:
        print(f"Cleaning/checking {path.name}",flush=True)
        monthly.append(clean_month(con,path,INTERIM/"clean_od",REPORTS/"monthly",args.force))
    for path in cached:
        print(f"Verifying clean cache {path.name}",flush=True)
        monthly.append(validated_clean_cache(path, REPORTS/"monthly"))
    print("Building regression tables and OD graph...",flush=True)
    report=build_analysis(con,[INTERIM/"clean_od"/f"{month}.parquet" for month in sorted(included)],
                          records,PROCESSED,args.start_hour,args.end_hour)
    report["generated_at_utc"]=datetime.now(timezone.utc).isoformat()
    report["source_rows"]=sum(r["source_rows"] for r in monthly)
    report["clean_rows"]=sum(r["clean_rows"] for r in monthly)
    report["exact_duplicate_rows_removed"]=sum(r["exact_duplicate_rows_removed"] for r in monthly)
    report["months"]=sorted(included)
    if args.mysql:
        from export_mysql import publish
        report["mysql"]=publish(con,args.login_path)
    (REPORTS/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
