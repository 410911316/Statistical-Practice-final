"""Validate monthly OD files; keep zero counts and exact station labels."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

VERSION = 1
HEADER = ["日期", "時段", "進站", "出站", "人次"]


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validated_clean_cache(path: Path, report_dir: Path):
    """Reuse a cleaned month only when its original validation manifest still matches."""
    report_path = report_dir / f"{path.stem}.json"
    prior = json.loads(report_path.read_text(encoding="utf-8"))
    if (prior.get("pipeline_version") != VERSION
            or prior.get("source_file") != f"{path.stem}.csv"
            or prior.get("output_sha256") != digest(path)
            or prior.get("invalid_rows") != 0 or prior.get("conflicting_keys") != 0):
        raise ValueError(f"Clean cache validation failed: {path.name}; restore its source CSV")
    return {**prior, "cached": True}


def clean_month(con, path: Path, output_dir: Path, report_dir: Path, force=False):
    month = path.stem
    if not re.fullmatch(r"20\d{4}", month):
        raise ValueError(f"Expected YYYYMM.csv: {path.name}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if next(csv.reader(stream)) != HEADER:
            raise ValueError(f"Unexpected header: {path.name}")
    sha = digest(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{month}.parquet"
    report_path = report_dir / f"{month}.json"
    if not force and destination.exists() and report_path.exists():
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        if prior.get("source_sha256") == sha and prior.get("pipeline_version") == VERSION:
            if prior.get("output_sha256") == digest(destination):
                return {**prior, "cached": True}
    con.execute("DROP TABLE IF EXISTS candidate")
    con.execute("""CREATE TEMP TABLE candidate AS SELECT
      row_number() OVER ()::BIGINT AS source_row,
      trim("日期") AS date_text, trim("時段") AS hour_text,
      trim("進站") AS origin_station, trim("出站") AS destination_station,
      trim("人次") AS count_text,
      try_cast(trim("日期") AS DATE) AS service_date,
      try_cast(trim("時段") AS SMALLINT) AS source_hour,
      try_cast(trim("人次") AS BIGINT) AS passengers
      FROM read_csv(?, all_varchar=true, header=true, strict_mode=true)""", [str(path)])
    invalid_condition = """service_date IS NULL OR source_hour IS NULL OR passengers IS NULL
      OR NOT regexp_full_match(date_text, '[0-9]{4}-[0-9]{2}-[0-9]{2}')
      OR NOT regexp_full_match(hour_text, '[0-9]{1,2}')
      OR NOT regexp_full_match(count_text, '[0-9]+')
      OR source_hour NOT BETWEEN 0 AND 23 OR passengers < 0
      OR origin_station IS NULL OR origin_station='' OR destination_station IS NULL
      OR destination_station='' OR strftime(service_date, '%Y%m') <> """ + literal(month)
    invalid = con.execute(f"SELECT count(*) FROM candidate WHERE {invalid_condition}").fetchone()[0]
    if invalid:
        samples = con.execute(f"SELECT source_row FROM candidate WHERE {invalid_condition} LIMIT 10").fetchall()
        raise ValueError(f"{month}: {invalid} invalid rows; first source rows: {samples}")
    conflicts = con.execute("""SELECT count(*) FROM (
      SELECT service_date,source_hour,origin_station,destination_station
      FROM candidate GROUP BY ALL HAVING min(passengers) <> max(passengers))""").fetchone()[0]
    if conflicts:
        raise ValueError(f"{month}: {conflicts} OD keys have conflicting counts; no output published")
    con.execute("DROP TABLE IF EXISTS clean")
    con.execute("""CREATE TEMP TABLE clean AS SELECT service_date,source_hour,
      origin_station,destination_station,min(passengers)::BIGINT AS passengers,
      min(source_row)::BIGINT AS source_row FROM candidate GROUP BY ALL""")
    raw_count = con.execute("SELECT count(*) FROM candidate").fetchone()[0]
    clean_count, total, zeros = con.execute(
        "SELECT count(*),sum(passengers),count(*) FILTER(WHERE passengers=0) FROM clean").fetchone()
    part = destination.with_suffix(".part")
    con.execute(f"""COPY (SELECT *, {literal(month)} AS source_month,
      service_date + source_hour * INTERVAL '1 hour' AS hour_start
      FROM clean ORDER BY service_date,source_hour,origin_station,destination_station)
      TO {literal(part.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)""")
    part.replace(destination)
    report = dict(pipeline_version=VERSION, source_file=path.name, source_sha256=sha,
                  source_rows=raw_count, clean_rows=clean_count,
                  exact_duplicate_rows_removed=raw_count-clean_count,
                  invalid_rows=invalid, conflicting_keys=conflicts,
                  passengers=int(total), zero_rows=zeros,
                  output_sha256=digest(destination))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    con.execute("DROP TABLE candidate")
    con.execute("DROP TABLE clean")
    return report
