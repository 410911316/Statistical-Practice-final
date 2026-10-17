"""Collect archived operational GFS forecasts with pre-cutoff publication evidence.

Downloads only a public bucket listing and point CSV subsets, never global GRIBs.
NCAR supplies the operational-cycle values; NOAA's public AWS object timestamps
provide a conservative latest-known publication time for the same forecast file.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parents[1]
AWS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
NCAR = "https://tds.gdex.ucar.edu/thredds/ncss/grid/files/g/d084001/"
TAIPEI = timezone(timedelta(hours=8))
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
FEATURES = ("forecast_temperature_c", "forecast_wind_speed_m_s", "forecast_cloud_cover_pct")
LAT, LON = 25.056, 121.510


def fetch(url: str, cache: Path) -> tuple[bytes, dict]:
    """Reuse exact URL-addressed public responses and keep retrieval provenance."""
    key = hashlib.sha256(url.encode()).hexdigest()
    body_path, meta_path = cache / f"{key}.body", cache / f"{key}.json"
    if body_path.exists() and meta_path.exists():
        content = body_path.read_bytes()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["url"] != url or meta["sha256"] != hashlib.sha256(content).hexdigest():
            raise ValueError(f"Cache identity/hash mismatch: {url}")
        return content, meta
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(15, 60))
            response.raise_for_status()
            content = response.content
            meta = {"url": url, "checked_at": datetime.now(timezone.utc).isoformat(),
                    "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
                    "http_status": response.status_code}
            body_path.write_bytes(content)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return content, meta
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    raise AssertionError("unreachable")


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def point_values(run_day: str, lead: int, group: str, cache: Path) -> dict:
    variables, level = {
        "temperature": (["Temperature_height_above_ground"], 2),
        "wind": (["u-component_of_wind_height_above_ground", "v-component_of_wind_height_above_ground"], 10),
        "cloud": (["Total_cloud_cover_entire_atmosphere"], None),
    }[group]
    issued = datetime.strptime(run_day, "%Y%m%d").replace(tzinfo=timezone.utc)
    valid = issued + timedelta(hours=lead)
    params = {"var": ",".join(variables), "latitude": LAT, "longitude": LON,
              "time": valid.isoformat().replace("+00:00", "Z"), "accept": "csv"}
    if level is not None:
        params["vertCoord"] = level
    base = f"{NCAR}{run_day[:4]}/{run_day}/gfs.0p25.{run_day}00.f{lead:03d}.grib2"
    url = base + "?" + urlencode(params)
    raw, meta = fetch(url, cache)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    rows = list(reader)
    if len(rows) != 1 or timestamp(rows[0]["time"]) != valid:
        raise ValueError(f"Unexpected point valid time/count: {url}: {rows}")
    row = rows[0]
    def column(prefix):
        names = [name for name in row if name.startswith(prefix + "[unit=")]
        if len(names) != 1:
            raise ValueError(f"Unexpected variable/units: {url}: {list(row)}")
        return names[0]
    lat, lon = float(row[column("latitude")]), float(row[column("longitude")])
    if (lat, lon) != (25.0, 121.5):
        raise ValueError(f"Grid changed: {(lat, lon)}")
    vals = {}
    for var in variables:
        name = column(var)
        expected = "K" if group == "temperature" else "m/s" if group == "wind" else "%"
        if name != f'{var}[unit="{expected}"]':
            raise ValueError(f"Unexpected units: {name}")
        number = float(row[name])
        if not math.isfinite(number):
            raise ValueError(f"Missing forecast value: {url}")
        vals[var] = number
    return {"values": vals, "source_url": url, "sha256": meta["sha256"],
            "checked_at": meta["checked_at"], "forecast_valid_at": valid.isoformat(),
            "grid_latitude": lat, "grid_longitude": lon}


def collect_event(event_date: str, cache: Path) -> tuple[list[dict], dict]:
    event = date.fromisoformat(event_date)
    prev = event - timedelta(days=1)
    day = prev.strftime("%Y%m%d")
    cutoff = datetime.combine(prev, datetime.min.time(), tzinfo=TAIPEI).replace(hour=18)
    issued = datetime.combine(prev, datetime.min.time(), tzinfo=timezone.utc)
    prefix = f"gfs.{day}/00/atmos/gfs.t00z.pgrb2.0p25.f03"
    listing_url = AWS + "?" + urlencode({"list-type": 2, "prefix": prefix, "max-keys": 100})
    raw, listing_meta = fetch(listing_url, cache)
    xml = ET.fromstring(raw)
    objects = {c.find("s:Key", NS).text: c.find("s:LastModified", NS).text
               for c in xml.findall("s:Contents", NS)}
    summary = {"event_date": event_date, "cutoff": cutoff.isoformat(),
               "model_initialised_at": issued.isoformat(), "aws_listing_url": listing_url,
               "aws_listing_sha256": listing_meta["sha256"]}
    required = [f"gfs.{day}/00/atmos/gfs.t00z.pgrb2.0p25.f{lead:03d}" for lead in (33, 36, 39)]
    if any(k not in objects for k in required):
        summary.update(status="unknown", reason="No matching AWS publication record for every required lead.")
        rows = [{"event_date": event_date, "hour": hour, "feature": feature, "value": None,
                 "published_at": None, "source_url": listing_url,
                 "evidence_text": "Required AWS archival publication timestamps not found; forecast existence at other archives is not ruled out.",
                 "availability_status": "unknown", "forecast_issued_at": None,
                 "forecast_valid_at": datetime.combine(event, datetime.min.time(), tzinfo=TAIPEI).replace(hour=hour).isoformat(),
                 "checked_at": listing_meta["checked_at"], "notes": "Do not impute from event-day observations."}
                for hour in range(18, 24) for feature in FEATURES]
        return rows, summary
    for key in required:
        if not issued <= timestamp(objects[key]) <= cutoff:
            raise ValueError(f"Publication outside D-1 cutoff: {key}: {objects[key]}")
    points = {(lead, group): point_values(day, lead, group, cache)
              for lead in (33, 36, 39) for group in ("temperature", "wind", "cloud")}
    output = []
    for hour in range(18, 24):
        lead_target = hour + 16  # D 18:00+08 is run D-1 00Z + 34h.
        endpoints = [lead_target] if lead_target in (33, 36, 39) else ([33, 36] if lead_target < 36 else [36, 39])
        coefficient = 0.0 if len(endpoints) == 1 else (lead_target - endpoints[0]) / (endpoints[1] - endpoints[0])
        def interpolate(group, variable):
            a = points[endpoints[0], group]["values"][variable]
            b = points[endpoints[-1], group]["values"][variable]
            return a + coefficient * (b - a)
        vals = {
            FEATURES[0]: interpolate("temperature", "Temperature_height_above_ground") - 273.15,
            FEATURES[1]: math.hypot(interpolate("wind", "u-component_of_wind_height_above_ground"),
                                   interpolate("wind", "v-component_of_wind_height_above_ground")),
            FEATURES[2]: interpolate("cloud", "Total_cloud_cover_entire_atmosphere"),
        }
        pub_objects = [{"source_url": AWS + f"gfs.{day}/00/atmos/gfs.t00z.pgrb2.0p25.f{lead:03d}",
                        "last_modified": objects[f"gfs.{day}/00/atmos/gfs.t00z.pgrb2.0p25.f{lead:03d}"]}
                       for lead in endpoints]
        published = max(timestamp(o["last_modified"]) for o in pub_objects)
        valid = datetime.combine(event, datetime.min.time(), tzinfo=TAIPEI).replace(hour=hour)
        for feature, group in zip(FEATURES, ("temperature", "wind", "cloud")):
            evidence = [points[lead, group] for lead in endpoints]
            output.append({"event_date": event_date, "hour": hour, "feature": feature,
                           "value": round(vals[feature], 6), "published_at": published.isoformat(),
                           "source_url": evidence[0]["source_url"],
                           "evidence_text": "Operational NOAA GFS cycle; all NOAA AWS GRIB object LastModified timestamps precede D-1 18:00+08. Values subset from NCAR archive of the matching cycle and lead.",
                           "availability_status": "verified_pre_cutoff",
                           "forecast_issued_at": issued.isoformat(), "forecast_valid_at": valid.isoformat(),
                           "checked_at": max(e["checked_at"] for e in evidence),
                           "notes": "forecast_issued_at is model INITIALISATION, not release; published_at uses AWS object LastModified. Instantaneous target-hour forecast, not an hourly mean. Linear interpolation within the same pre-cutoff forecast cycle; wind u/v interpolated before vector norm. Cloud is instantaneous entire-atmosphere total cloud, not six-hour mean. All four MRT stations share nearest grid 25.0N,121.5E; no rainfall probability inferred.",
                           "publication_evidence": pub_objects,
                           "subset_evidence": evidence, "interpolation_weight": coefficient,
                           "lead_hours_used": endpoints, "grid_latitude": 25.0, "grid_longitude": 121.5})
    summary.update(status="verified_pre_cutoff", weather_rows=len(output),
                   earliest_published_at=min(objects[k] for k in required),
                   latest_published_at=max(objects[k] for k in required))
    return output, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=ROOT / ".local/forecast/analysis_input.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/d1_weather_sources.json")
    parser.add_argument("--cache", type=Path, default=ROOT / ".local/d1_weather")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    with args.events.open(encoding="utf-8-sig", newline="") as stream:
        dates = sorted({r["event_date"] for r in csv.DictReader(stream)})
    records, coverage = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for rows, summary in executor.map(lambda day: collect_event(day, args.cache), dates):
            records.extend(rows)
            coverage.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    payload = {"schema_version": 1, "source_kind": "operational_GFS_forecast_not_reanalysis",
               "cutoff_rule": "D-1 18:00 Asia/Taipei", "cycle_rule": "D-1 00:00 UTC",
               "location_requested": {"latitude": LAT, "longitude": LON},
               "nearest_grid": {"latitude": 25.0, "longitude": 121.5},
               "time_semantics": "Instantaneous forecast at target hour; f033/f036/f039 interpolated as needed within same pre-cutoff cycle. forecast_issued_at means model initialisation; published_at is latest NOAA AWS object timestamp of used endpoints.",
               "publication_caveat": "AWS LastModified demonstrates object availability by that time, not the original forecast dissemination timestamp. NCAR subset and NOAA AWS are matched by operational model, initialisation and lead; byte identity between archives is not asserted.",
               "coverage": coverage, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "events": len(dates), "verified_events": sum(x["status"] == "verified_pre_cutoff" for x in coverage), "records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
