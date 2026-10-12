"""Download official 2017–2026 DGPA calendars, validate every date, and upsert MySQL.

Run: python -X utf8 src/import_calendar.py [--check-only]
These are retrospective government-office calendars, not every person's work schedule.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path

import requests

from import_event_finance import sql_value
from load_data import mysql
from project_paths import ROOT

INDEX_URL = "https://data.gov.tw/dataset/14718"
# Links listed by the official index; 2025 uses the explicitly revised 1141020 file.
SOURCE_FILES = {
    2017: "202407/f9efbeb6-af1b-4adf-9034-3c8a95a5a192.csv",
    2018: "202407/d645842e-7b26-43e4-9f08-1172c61dde91.csv",
    2019: "202407/5f2ff9c5-a235-4819-8c62-f064fc5bffb8.csv",
    2020: "202407/4d215402-15c7-4ad9-ab90-a33f34a63f5b.csv",
    2021: "202407/daec35f0-ae43-4466-aa26-bafb39aa8e0a.csv",
    2022: "202407/cbe02c09-d3ce-4912-ace4-69b7129efa9f.csv",
    2023: "202407/dd89fe8c-25f5-4035-a31e-7c14b44a74d7.csv",
    2024: "202407/777152e9-fdd1-4a61-876c-2733e7692538.csv",
    2025: "202510/b84cb88a-803c-4621-a843-d637b2775615.csv",
    2026: "202506/a52331bd-a189-466b-b0f0-cae3062bbf74.csv",
}
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS taipei_metro_analysis.calendar_day (
 calendar_date DATE NOT NULL PRIMARY KEY,
 iso_weekday TINYINT UNSIGNED NOT NULL,
 is_weekend BOOLEAN NOT NULL,
 official_holiday_code TINYINT UNSIGNED NOT NULL COMMENT '官方原碼：0上班、2放假',
 is_holiday BOOLEAN NOT NULL COMMENT '政府行政機關放假日，包含例行週末',
 is_makeup_workday BOOLEAN NOT NULL COMMENT '由週六/日且官方原碼0推導',
 day_note VARCHAR(255) NULL,
 source_url VARCHAR(1000) NOT NULL,
 source_file_sha256 CHAR(64) NOT NULL,
 source_version VARCHAR(80) NOT NULL,
 CONSTRAINT chk_calendar_weekday CHECK(iso_weekday BETWEEN 1 AND 7),
 CONSTRAINT chk_calendar_code CHECK(official_holiday_code IN (0,2)),
 CONSTRAINT chk_calendar_holiday CHECK(is_holiday=(official_holiday_code=2)),
 CONSTRAINT chk_calendar_weekend CHECK(is_weekend=(iso_weekday>=6)),
 CONSTRAINT chk_calendar_makeup CHECK(is_makeup_workday=(iso_weekday>=6 AND official_holiday_code=0))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='人事行政總處辦公日曆；回顧用修訂版本';
CREATE OR REPLACE VIEW taipei_metro_analysis.v_dadaocheng_event_calendar AS
SELECT e.event_date,e.event_year,e.event_status,c.iso_weekday,c.is_weekend,
 c.is_holiday,c.is_makeup_workday,c.day_note,c.source_version
FROM taipei_metro_analysis.dadaocheng_fireworks_events e
JOIN taipei_metro_analysis.calendar_day c ON c.calendar_date=e.event_date;
"""


def parse_calendar(content: bytes, year: int, url: str) -> list[dict]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("cp950")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != ["西元日期", "星期", "是否放假", "備註"]:
        raise ValueError(f"Unexpected calendar columns in {year}: {reader.fieldnames}")
    digest = hashlib.sha256(content).hexdigest()
    rows = []
    for raw in reader:
        day = datetime.strptime(raw["西元日期"].strip(), "%Y%m%d").date()
        code = int(raw["是否放假"])
        if day.year != year or code not in {0, 2}:
            raise ValueError(f"Wrong year/holiday code on {day}")
        if raw["星期"].strip() != "一二三四五六日"[day.weekday()]:
            raise ValueError(f"Weekday mismatch on {day}")
        rows.append(dict(calendar_date=day.isoformat(), iso_weekday=day.isoweekday(),
                         is_weekend=int(day.weekday()>=5), official_holiday_code=code,
                         is_holiday=int(code==2), is_makeup_workday=int(day.weekday()>=5 and code==0),
                         day_note=raw["備註"].strip() or None, source_url=url,
                         source_file_sha256=digest,
                         source_version="1141020 revised" if year==2025 else SOURCE_FILES[year]))
    start, end = date(year,1,1), date(year+1,1,1)
    expected = {(start+timedelta(days=i)).isoformat() for i in range((end-start).days)}
    if len(rows) != len(expected) or {r["calendar_date"] for r in rows} != expected:
        raise ValueError(f"Calendar dates incomplete/duplicated for {year}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--login-path", default="codex-local")
    args = parser.parse_args()
    output = ROOT / "data/original_data/calendar"
    output.mkdir(parents=True, exist_ok=True)
    rows, sources = [], []
    for year, file in SOURCE_FILES.items():
        url = "https://www.dgpa.gov.tw/FileConversion?filename=dgpa%2Ffiles%2F" + file.replace("/","%2F")
        path = output / f"{year}.csv"
        if not path.exists():
            response = requests.get(url, timeout=(15,60))
            response.raise_for_status()
            parsed = parse_calendar(response.content, year, url)
            path.write_bytes(response.content)
        else:
            parsed = parse_calendar(path.read_bytes(), year, url)
        rows.extend(parsed)
        sources.append(dict(year=year, url=url, sha256=parsed[0]["source_file_sha256"],
                            rows=len(parsed), holidays=sum(r["is_holiday"] for r in parsed),
                            makeup_workdays=sum(r["is_makeup_workday"] for r in parsed)))
        print(f"Calendar {year}: {len(parsed)} verified days", flush=True)
    if not args.check_only:
        mysql(SCHEMA_SQL, args.login_path)
        cols = list(rows[0])
        statements = ["START TRANSACTION;"]
        for i in range(0,len(rows),250):
            values = ",".join("("+",".join(sql_value(r[c]) for c in cols)+")" for r in rows[i:i+250])
            updates = ",".join(f"{c}=incoming.{c}" for c in cols if c!="calendar_date")
            statements.append(f"INSERT INTO taipei_metro_analysis.calendar_day ({','.join(cols)}) VALUES {values} AS incoming ON DUPLICATE KEY UPDATE {updates};")
        statements.append("COMMIT;")
        mysql("\n".join(statements), args.login_path)
        count = int(mysql("SELECT COUNT(*) FROM taipei_metro_analysis.calendar_day WHERE calendar_date BETWEEN '2017-01-01' AND '2026-12-31';", args.login_path))
        if count != len(rows):
            raise RuntimeError("Calendar database count mismatch")
    with (output / "calendar_2017_2026.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest=dict(index_url=INDEX_URL,verified_at_utc=datetime.now(timezone.utc).isoformat(),
                  interpretation="retrospective government-office calendar, not historical as-of forecast information",
                  rows=len(rows),sources=sources)
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(rows=len(rows),imported=not args.check_only),ensure_ascii=False))


if __name__ == "__main__":
    main()
