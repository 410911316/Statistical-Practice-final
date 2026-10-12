"""Import Dadaocheng event years, annual publicity needs, and enacted event budgets.

Run from the project root::

    .\.venv\Scripts\python.exe -X utf8 src/import_event_finance.py
    .\.venv\Scripts\python.exe -X utf8 src/import_event_finance.py --check-only

The source CSV repeats annual publicity values on every fireworks date. This importer
checks those repeats agree, then stores one publicity row per year. Annual budget
figures are the original enacted activity appropriations, not actual expenditure.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path

from load_data import mysql
from project_paths import ROOT

DB = "taipei_metro_analysis"
DEFAULT_CSV = ROOT / "data/original_data/大稻埕量化資料.csv"
MEDIA = {
    "平面媒體": "has_print_media",
    "電視媒體": "has_television_media",
    "網路媒體": "has_online_media",
    "廣播媒體": "has_radio_media",
    "其他": "has_other_media",
}
HEADER = [
    "活動日期", "活動類型", "活動狀態", "已舉辦", "已取消", "已延期",
    "實際煙火分鐘", "煙火秒數", "無人機演出", "無人機狀態已知",
    "移至日期", "移轉類型", "是否原訂場次", "接收合併演出",
    "接收延期演出", "年份", "月份", "星期序號", "是否週末",
    "相關宣導案所需經費合計(元)", *MEDIA,
]

# The original enacted, activity-level appropriation shown in the Taipei DOIT budget.
# It can include planning, execution, and insurance. It is not actual spending.
BUDGET = {
    2017: (5_600_000, "https://www-ws.gov.taipei/001/Upload/public/Attachment/72913581075.pdf", 53,
           "國際河岸音樂季系列活動企劃執行及保險等相關費用"),
    2018: (5_600_000, "https://www-ws.gov.taipei/001/Upload/308/relfile/13816/7598523/c338bb07-0ed2-41cb-b54d-effeaa299896.pdf", 47,
           "國際河岸音樂季系列活動企劃執行及保險等相關業務"),
    2019: (5_600_000, "https://www-ws.gov.taipei/001/Upload/308/relfile/10123/7945923/2d9a84dd-7180-4faa-9b9b-233927550a14.pdf", 44,
           "臺北河岸音樂季系列活動企劃執行及保險等相關業務"),
    2020: (5_600_000, "https://www-ws.gov.taipei/001/Upload/308/relfile/13816/8148424/12559eeb-ef0f-4285-b389-a9561edd3e45.pdf", 46,
           "臺北河岸音樂季系列活動企劃執行及保險等相關業務"),
    2021: (5_600_000, "https://www-ws.gov.taipei/001/Upload/308/relfile/13816/8352392/6006f239-03a8-40d5-8513-7e09dbaf0e07.pdf", 52,
           "大稻埕情人節活動企劃執行及保險等相關業務"),
    2022: (6_500_000, "https://www-ws.gov.taipei/001/Upload/308/relfile/13816/8528059/a66fb0ea-a098-4990-9c79-8ee0592363bf.pdf", 48,
           "大稻埕情人節活動企劃執行及保險等相關業務費用"),
    2023: (9_000_000, "https://www-ws.gov.taipei/001/Upload/308/relfile/13816/8904219/4712ef5c-c475-43ed-9cea-081f37e54e1a.pdf", 48,
           "預算書列大稻埕情人節；活動後以大稻埕夏日節辦理"),
    2024: (12_000_000, "https://www-ws.gov.taipei/001/Upload/308/relfile/13816/9131731/daf64db2-f1f8-4eb6-9d7d-df3615966b9e.pdf", 45,
           "大稻埕活動企劃執行；年度科目移入河岸童樂會300萬元，不宜將跨年增額全視為煙火加碼"),
    2025: (12_000_000, "https://www-ws.gov.taipei/001/Upload/308/relfile/13816/9358459/a69520a8-97d9-4fa7-a925-7986eda834b0.pdf", 47,
           "大稻埕活動企劃執行等相關業務"),
    2026: (20_000_000, "https://www-ws.gov.taipei/001/Upload/308/relfile/13816/9542874/6a7a47b2-8bd8-48be-81b2-32d261ed71e9.pdf", 45,
           "大稻埕活動企劃執行等相關業務"),
}

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {DB}.dadaocheng_event_years (
  event_year SMALLINT UNSIGNED NOT NULL PRIMARY KEY,
  event_label VARCHAR(120) NOT NULL,
  CONSTRAINT chk_dadaocheng_year CHECK (event_year BETWEEN 2000 AND 2100)
) ENGINE=InnoDB COMMENT='每年一屆活動；多個煙火日期共用一個年度';

CREATE TABLE IF NOT EXISTS {DB}.dadaocheng_planned_event_budgets (
  event_year SMALLINT UNSIGNED NOT NULL PRIMARY KEY,
  planned_budget_ntd BIGINT UNSIGNED NOT NULL,
  budget_basis VARCHAR(80) NOT NULL,
  budget_scope_note VARCHAR(500) NOT NULL,
  source_url VARCHAR(1000) NOT NULL,
  source_pdf_page SMALLINT UNSIGNED NOT NULL,
  CONSTRAINT fk_dadaocheng_budget_year FOREIGN KEY (event_year)
    REFERENCES {DB}.dadaocheng_event_years(event_year),
  CONSTRAINT chk_dadaocheng_budget_positive CHECK (planned_budget_ntd > 0)
) ENGINE=InnoDB COMMENT='每年一筆原列法定活動預算；不是執行實支';

CREATE TABLE IF NOT EXISTS {DB}.dadaocheng_annual_publicity (
  event_year SMALLINT UNSIGNED NOT NULL PRIMARY KEY,
  required_publicity_cost_ntd BIGINT UNSIGNED NOT NULL,
  has_print_media BOOLEAN NULL,
  has_television_media BOOLEAN NULL,
  has_online_media BOOLEAN NULL,
  has_radio_media BOOLEAN NULL,
  has_other_media BOOLEAN NULL,
  source_file_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  source_label VARCHAR(255) NOT NULL,
  CONSTRAINT fk_dadaocheng_publicity_year FOREIGN KEY (event_year)
    REFERENCES {DB}.dadaocheng_event_years(event_year),
  CONSTRAINT chk_dadaocheng_publicity_positive CHECK (required_publicity_cost_ntd > 0),
  CONSTRAINT chk_dadaocheng_publicity_media CHECK (
    (has_print_media IN (0,1) OR has_print_media IS NULL) AND
    (has_television_media IN (0,1) OR has_television_media IS NULL) AND
    (has_online_media IN (0,1) OR has_online_media IS NULL) AND
    (has_radio_media IN (0,1) OR has_radio_media IS NULL) AND
    (has_other_media IN (0,1) OR has_other_media IS NULL))
) ENGINE=InnoDB COMMENT='CSV年度相關宣導案所需經費合計；非各場次費用、非實付金額';

CREATE OR REPLACE VIEW {DB}.v_dadaocheng_annual_finance AS
SELECT y.event_year, y.event_label,
       b.planned_budget_ntd, b.budget_basis, b.budget_scope_note,
       p.required_publicity_cost_ntd,
       CAST(ROUND(CAST(p.required_publicity_cost_ntd AS DECIMAL(20,6)) /
                  NULLIF(b.planned_budget_ntd, 0), 6) AS DECIMAL(18,6))
         AS publicity_to_planned_budget_ratio,
       p.has_print_media, p.has_television_media, p.has_online_media,
       p.has_radio_media, p.has_other_media,
       b.source_url AS budget_source_url, p.source_label AS publicity_source_label
FROM {DB}.dadaocheng_event_years y
LEFT JOIN {DB}.dadaocheng_planned_event_budgets b ON b.event_year=y.event_year
LEFT JOIN {DB}.dadaocheng_annual_publicity p ON p.event_year=y.event_year;

CREATE OR REPLACE VIEW {DB}.v_dadaocheng_event_quantitative AS
SELECT e.event_date, e.event_year, e.event_name, e.event_status,
       (e.event_status='held') AS was_held,
       (e.event_status='cancelled') AS was_cancelled,
       (e.event_status='postponed') AS was_postponed,
       e.fireworks_duration_seconds,
       ROUND(e.fireworks_duration_seconds / 60, 4) AS fireworks_duration_minutes,
       e.has_drone_show,
       (e.has_drone_show IS NOT NULL) AS drone_status_known,
       e.transferred_to_date, e.transfer_type,
       (e.event_status IN ('cancelled','postponed')) AS original_date_not_held,
       EXISTS(SELECT 1 FROM {DB}.dadaocheng_fireworks_events x
              WHERE x.transferred_to_date=e.event_date AND x.transfer_type='merged')
         AS receives_merged_show,
       EXISTS(SELECT 1 FROM {DB}.dadaocheng_fireworks_events x
              WHERE x.transferred_to_date=e.event_date AND x.transfer_type='rescheduled')
         AS receives_rescheduled_show,
       MONTH(e.event_date) AS event_month,
       WEEKDAY(e.event_date)+1 AS iso_weekday,
       (WEEKDAY(e.event_date)>=5) AS is_weekend
FROM {DB}.dadaocheng_fireworks_events e;
"""


def sql_value(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, Decimal)):
        return str(value)
    return "CONVERT(0x" + str(value).encode("utf-8").hex() + " USING utf8mb4)"


def parse_optional_int(text: str, label: str, allowed: set[int] | None = None) -> int | None:
    if text == "":
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid {label}: {text!r}") from exc
    if allowed is not None and value not in allowed:
        raise ValueError(f"Invalid {label}: {text!r}")
    return value


def parse_csv(path: Path) -> tuple[list[dict], dict[int, dict], str]:
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    reader = csv.DictReader(content.decode("utf-8-sig").splitlines())
    if reader.fieldnames != HEADER:
        raise ValueError("Quantitative CSV header does not match expected columns")
    events: list[dict] = []
    annual: dict[int, dict] = {}
    seen_dates: set[str] = set()
    for line, raw in enumerate(reader, 2):
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError(f"Malformed CSV row {line}")
        row = {key: value.strip() for key, value in raw.items()}
        event_date = date.fromisoformat(row["活動日期"])
        year = event_date.year
        if row["活動日期"] in seen_dates:
            raise ValueError(f"Duplicate date on CSV row {line}")
        seen_dates.add(row["活動日期"])
        if (parse_optional_int(row["年份"], "year") != year
                or parse_optional_int(row["月份"], "month") != event_date.month
                or parse_optional_int(row["星期序號"], "weekday") != event_date.isoweekday()
                or parse_optional_int(row["是否週末"], "weekend", {0, 1}) != int(event_date.isoweekday() >= 6)):
            raise ValueError(f"Date-derived fields disagree on CSV row {line}")
        status = row["活動狀態"]
        if status not in {"held", "cancelled", "postponed"}:
            raise ValueError(f"Invalid event status on CSV row {line}")
        for column, expected in [("已舉辦", status == "held"),
                                 ("已取消", status == "cancelled"),
                                 ("已延期", status == "postponed"),
                                 ("是否原訂場次", status != "held")]:
            if parse_optional_int(row[column], column, {0, 1}) != int(expected):
                raise ValueError(f"{column} disagrees with status on CSV row {line}")
        seconds = parse_optional_int(row["煙火秒數"], "fireworks seconds")
        try:
            minutes = Decimal(row["實際煙火分鐘"])
        except InvalidOperation as exc:
            raise ValueError(f"Invalid fireworks minutes on CSV row {line}") from exc
        if seconds is None and (status == "held" or minutes != 0):
            raise ValueError(f"Missing or inconsistent fireworks seconds on CSV row {line}")
        if seconds is not None and (status != "held" or abs(minutes * 60 - seconds) > Decimal("0.01")):
            raise ValueError(f"Fireworks minutes/seconds disagree on CSV row {line}")
        drone = parse_optional_int(row["無人機演出"], "drone", {0, 1})
        drone_known = parse_optional_int(row["無人機狀態已知"], "drone known", {0, 1})
        if drone_known != int(drone is not None):
            raise ValueError(f"Drone-known flag disagrees on CSV row {line}")
        target = row["移至日期"] or None
        if target:
            date.fromisoformat(target)
        transfer = row["移轉類型"] or None
        if (target is None) != (transfer is None) or (transfer is not None and transfer not in {"merged", "rescheduled"}):
            raise ValueError(f"Invalid transfer on CSV row {line}")
        event = {"event_date": str(event_date), "event_year": year, "event_label": row["活動類型"],
                 "event_status": status, "fireworks_duration_seconds": seconds,
                 "has_drone_show": drone, "transferred_to_date": target,
                 "transfer_type": transfer,
                 "receives_merged_show": parse_optional_int(row["接收合併演出"], "receives merged", {0, 1}),
                 "receives_rescheduled_show": parse_optional_int(row["接收延期演出"], "receives postponed", {0, 1})}
        events.append(event)
        cost = parse_optional_int(row["相關宣導案所需經費合計(元)"], "annual publicity cost")
        if cost is not None and cost <= 0:
            raise ValueError(f"Annual publicity cost must be positive on CSV row {line}")
        annual_row = {"event_year": year, "event_label": row["活動類型"],
                      "required_publicity_cost_ntd": cost,
                      **{sql_name: parse_optional_int(row[csv_name], csv_name, {0, 1})
                         for csv_name, sql_name in MEDIA.items()}}
        previous = annual.setdefault(year, annual_row)
        if annual_row != previous:
            raise ValueError(f"Annual fields vary between dates in {year}; cannot deduplicate")
        if (cost is None) != all(annual_row[column] is None for column in MEDIA.values()):
            raise ValueError(f"Annual publicity amount/media flags incomplete in {year}")
    if not events:
        raise ValueError("Empty quantitative CSV")
    return events, annual, digest


def validate_against_database(events: list[dict], login_path: str) -> None:
    raw = mysql(f"""SELECT DATE_FORMAT(event_date,'%Y-%m-%d'),event_status,
      IFNULL(CAST(fireworks_duration_seconds AS CHAR),'NULL'),
      IFNULL(CAST(has_drone_show AS CHAR),'NULL'),
      IFNULL(DATE_FORMAT(transferred_to_date,'%Y-%m-%d'),'NULL'),
      IFNULL(transfer_type,'NULL')
      FROM {DB}.dadaocheng_fireworks_events ORDER BY event_date;""", login_path)
    db_events = {}
    for line in raw.splitlines():
        event_date, status, seconds, drone, target, transfer = line.split("\t")
        db_events[event_date] = (status, None if seconds == "NULL" else int(seconds),
                                 None if drone == "NULL" else int(drone),
                                 None if target == "NULL" else target,
                                 None if transfer == "NULL" else transfer)
    by_date = {event["event_date"]: event for event in events}
    if set(by_date) != set(db_events):
        raise ValueError(f"CSV/DB event-date mismatch: CSV-only={sorted(set(by_date)-set(db_events))}, "
                         f"DB-only={sorted(set(db_events)-set(by_date))}")
    for event in events:
        actual = db_events[event["event_date"]]
        expected = tuple(event[column] for column in (
            "event_status", "fireworks_duration_seconds", "has_drone_show",
            "transferred_to_date", "transfer_type"))
        if actual != expected:
            raise ValueError(f"CSV/DB event conflict on {event['event_date']}: {actual} vs {expected}")
        incoming = [other for other in events if other["transferred_to_date"] == event["event_date"]]
        if event["receives_merged_show"] != int(any(x["transfer_type"] == "merged" for x in incoming)):
            raise ValueError(f"Merged-show flag disagrees on {event['event_date']}")
        if event["receives_rescheduled_show"] != int(any(x["transfer_type"] == "rescheduled" for x in incoming)):
            raise ValueError(f"Rescheduled-show flag disagrees on {event['event_date']}")


def schema_exists(login_path: str, table: str, column: str | None = None) -> bool:
    predicate = f" AND column_name={sql_value(column)}" if column else ""
    return mysql(f"SELECT COUNT(*) FROM information_schema.columns WHERE table_schema={sql_value(DB)} "
                 f"AND table_name={sql_value(table)}{predicate};", login_path) != "0"


def install_schema(login_path: str) -> None:
    # MySQL DDL auto-commits; data writes below are a separate transaction.
    statements = SCHEMA_SQL.split("CREATE OR REPLACE VIEW", 1)[0]
    mysql(statements, login_path)
    if not schema_exists(login_path, "dadaocheng_fireworks_events", "event_year"):
        mysql(f"""ALTER TABLE {DB}.dadaocheng_fireworks_events
          ADD COLUMN event_year SMALLINT UNSIGNED GENERATED ALWAYS AS (YEAR(event_date)) STORED;
          CREATE INDEX ix_dadaocheng_event_year ON {DB}.dadaocheng_fireworks_events(event_year);""", login_path)


def install_event_year_fk(login_path: str) -> None:
    # Insert all year rows before enforcing this FK against existing event dates.
    fk_count = mysql(f"""SELECT COUNT(*) FROM information_schema.referential_constraints
      WHERE constraint_schema={sql_value(DB)} AND table_name='dadaocheng_fireworks_events'
      AND constraint_name='fk_dadaocheng_event_year';""", login_path)
    if fk_count == "0":
        mysql(f"""ALTER TABLE {DB}.dadaocheng_fireworks_events
          ADD CONSTRAINT fk_dadaocheng_event_year FOREIGN KEY (event_year)
          REFERENCES {DB}.dadaocheng_event_years(event_year);""", login_path)


def insert_rows(login_path: str, annual: dict[int, dict], digest: str) -> None:
    years = sorted(annual)
    if set(years) != set(BUDGET):
        raise ValueError("Budget evidence years do not match CSV years")
    statements = ["SET SESSION sql_mode='STRICT_ALL_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO';",
                  "START TRANSACTION;"]
    for year in years:
        label = annual[year]["event_label"]
        statements.append(f"""INSERT INTO {DB}.dadaocheng_event_years(event_year,event_label)
          VALUES ({year},{sql_value(label)}) AS incoming
          ON DUPLICATE KEY UPDATE event_label=incoming.event_label;""")
        amount, url, page, note = BUDGET[year]
        statements.append(f"""INSERT INTO {DB}.dadaocheng_planned_event_budgets
          (event_year,planned_budget_ntd,budget_basis,budget_scope_note,source_url,source_pdf_page)
          VALUES ({year},{amount},{sql_value('原列法定活動預算')},{sql_value(note)},
                  {sql_value(url)},{page}) AS incoming
          ON DUPLICATE KEY UPDATE planned_budget_ntd=incoming.planned_budget_ntd,
            budget_basis=incoming.budget_basis,budget_scope_note=incoming.budget_scope_note,
            source_url=incoming.source_url,source_pdf_page=incoming.source_pdf_page;""")
        publicity = annual[year]
        if publicity["required_publicity_cost_ntd"] is not None:
            columns = ["event_year", "required_publicity_cost_ntd", *MEDIA.values(),
                       "source_file_sha256", "source_label"]
            values = [year, publicity["required_publicity_cost_ntd"],
                      *(publicity[column] for column in MEDIA.values()),
                      digest, "使用者提供的大稻埕量化資料.csv；年度相關宣導案所需經費合計"]
            updates = ",".join(f"{column}=incoming.{column}" for column in columns[1:])
            statements.append(f"""INSERT INTO {DB}.dadaocheng_annual_publicity
              ({','.join(columns)}) VALUES ({','.join(sql_value(x) for x in values)}) AS incoming
              ON DUPLICATE KEY UPDATE {updates};""")
    statements.append("COMMIT;")
    mysql("\n".join(statements), login_path)


def audit(login_path: str, expected_years: int, expected_publicity: int) -> dict:
    raw = mysql(f"""SELECT JSON_OBJECT(
      'event_dates',(SELECT COUNT(*) FROM {DB}.dadaocheng_fireworks_events),
      'event_years',(SELECT COUNT(*) FROM {DB}.dadaocheng_event_years),
      'planned_budgets',(SELECT COUNT(*) FROM {DB}.dadaocheng_planned_event_budgets),
      'annual_publicity',(SELECT COUNT(*) FROM {DB}.dadaocheng_annual_publicity),
      'ratios_available',(SELECT COUNT(publicity_to_planned_budget_ratio)
        FROM {DB}.v_dadaocheng_annual_finance),
      'event_view_rows',(SELECT COUNT(*) FROM {DB}.v_dadaocheng_event_quantitative));""", login_path)
    result = json.loads(raw)
    if (result["event_years"] != expected_years or result["planned_budgets"] != expected_years
            or result["annual_publicity"] != expected_publicity
            or result["ratios_available"] != expected_publicity
            or result["event_dates"] != result["event_view_rows"]):
        raise RuntimeError(f"Unexpected annual/event cardinality after import: {result}")
    joined = mysql(f"""SELECT COUNT(*) FROM {DB}.dadaocheng_fireworks_events e
      JOIN {DB}.dadaocheng_event_years y ON y.event_year=e.event_year
      LEFT JOIN {DB}.dadaocheng_planned_event_budgets b ON b.event_year=y.event_year
      LEFT JOIN {DB}.dadaocheng_annual_publicity p ON p.event_year=y.event_year;""", login_path)
    if int(joined) != result["event_dates"]:
        raise RuntimeError("Annual finance join multiplied event dates")
    result["event_rows_after_year_joins"] = int(joined)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--login-path", default="codex-local")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    events, annual, digest = parse_csv(args.csv)
    validate_against_database(events, args.login_path)
    result = {"csv": str(args.csv.resolve()), "csv_sha256": digest,
              "event_dates": len(events), "years": len(annual),
              "publicity_years": sum(row["required_publicity_cost_ntd"] is not None
                                     for row in annual.values())}
    if not args.check_only:
        install_schema(args.login_path)
        insert_rows(args.login_path, annual, digest)
        install_event_year_fk(args.login_path)
        view_sql = "CREATE OR REPLACE VIEW" + SCHEMA_SQL.split("CREATE OR REPLACE VIEW", 1)[1]
        mysql(view_sql, args.login_path)
        result["database_audit"] = audit(args.login_path, len(annual), result["publicity_years"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
