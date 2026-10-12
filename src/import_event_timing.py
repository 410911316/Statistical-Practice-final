"""Separate announced/derived show endpoints from unverified actual endpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json

import pandas as pd

from import_event_finance import sql_value
from load_data import event_snapshot, mysql
from project_paths import ROOT, PROCESSED

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS taipei_metro_analysis.event_timing_context (
 event_date DATE PRIMARY KEY,
 nominal_show_end DATETIME NULL COMMENT 'Announced start plus documented duration; not actual end',
 nominal_end_basis VARCHAR(64) NOT NULL,
 announced_show_window_start TIME NULL,
 announced_show_window_end TIME NULL,
 scheduled_stage_start TIME NULL,
 scheduled_stage_end TIME NULL,
 stage_scope VARCHAR(255) NULL,
 actual_show_end DATETIME NULL,
 actual_stage_end DATETIME NULL,
 actual_time_source_url TEXT NULL,
 show_source_url TEXT NOT NULL,
 additional_source_url TEXT NULL,
 evidence_note TEXT NOT NULL,
 source_sha256 CHAR(64) NOT NULL,
 FOREIGN KEY(event_date) REFERENCES taipei_metro_analysis.dadaocheng_fireworks_events(event_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def timing_row(event, additional, digest):
    day=event["event_date"]
    held=event["event_status"]=="held"
    nominal=None
    basis="not_held" if not held else "unknown_exact_start"
    if held and event.get("announced_show_start_time"):
        duration=event.get("combined_show_duration_seconds") or event.get("fireworks_duration_seconds")
        if duration:
            nominal=(datetime.fromisoformat(day+"T"+event["announced_show_start_time"])+timedelta(seconds=duration)).isoformat(sep=" ")
            basis="announced_start_plus_documented_duration"
    evidence=additional.get(day,{})
    if held and evidence.get("scheduled_show_end_time"):
        nominal=day+" "+evidence["scheduled_show_end_time"]
        basis="official_announced_end"
    return dict(event_date=day,nominal_show_end=nominal,nominal_end_basis=basis,
                announced_show_window_start="20:15:00" if day=="2018-08-18" else None,
                announced_show_window_end="20:30:00" if day=="2018-08-18" else None,
                scheduled_stage_start=evidence.get("scheduled_stage_start_time"),
                scheduled_stage_end=evidence.get("scheduled_stage_end_time"),stage_scope=evidence.get("stage_scope"),
                actual_show_end=None,actual_stage_end=None,actual_time_source_url=None,
                show_source_url=event["official_source_url"],additional_source_url=evidence.get("source_url"),
                evidence_note="實際結束未核實；推算時刻不能代表實際散場。 "+(evidence.get("locator") or "")+" "+(event.get("notes") or ""),source_sha256=digest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mysql",action="store_true")
    parser.add_argument("--login-path",default="codex-local")
    args=parser.parse_args()
    events=event_snapshot(args.login_path)
    evidence_path=ROOT/"docs/event_timing_sources.json"
    evidence=json.loads(evidence_path.read_text(encoding="utf8"))
    additional={day:item for item in evidence["entries"] for day in item["dates"]}
    digest=hashlib.sha256(evidence_path.read_bytes()+json.dumps(events,sort_keys=True).encode()).hexdigest()
    rows=[timing_row(e,additional,digest) for e in events]
    pd.DataFrame(rows).to_csv(PROCESSED/"event_timing_context.csv",index=False,encoding="utf-8-sig")
    if args.mysql:
        mysql(SCHEMA_SQL,args.login_path)
        cols=list(rows[0]);values=",".join("("+",".join(sql_value(r[c]) for c in cols)+")" for r in rows)
        # Future separately verified actual times must survive rerunning this scheduled-time importer.
        preserve={"event_date","actual_show_end","actual_stage_end","actual_time_source_url"}
        updates=",".join(f"{c}=incoming.{c}" for c in cols if c not in preserve)
        mysql(f"START TRANSACTION; INSERT INTO taipei_metro_analysis.event_timing_context ({','.join(cols)}) VALUES {values} AS incoming ON DUPLICATE KEY UPDATE {updates}; COMMIT;",args.login_path)
    print(json.dumps(dict(rows=len(rows),nominal_show_end=sum(r["nominal_show_end"] is not None for r in rows),
                          scheduled_stage_end=sum(r["scheduled_stage_end"] is not None for r in rows),
                          newly_verified_actual_end=0),ensure_ascii=False))


if __name__=="__main__":main()
