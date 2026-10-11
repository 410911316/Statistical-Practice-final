"""Derive station-specific hourly overlap with planned show-end windows.

The end of the announced fireworks/sky program is a timing proxy, not observed
egress, a stage closing time, or measured walking delay. No target or observed
event-time field participates in this transformation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v1.data import _manifest_entry
from analysis_v1.d1_features import aware_time

STATION_CODES = {
    "beimen": "北門", "daqiaotou": "大橋頭站", "shuanglian": "雙連",
    "minquan_west": "民權西路",
}
END_FEATURES = tuple(f"planned_end_{window}_{station}"
                     for window in ("0_60", "60_120") for station in STATION_CODES)
PARENT_SOURCE = "docs/d1_program_sources.json"
COMBINED_URL = ("https://www.gov.taipei/News_Content.aspx?"
                "n=F0DDAF49B89E9413&s=FF7935502CB3CC88")
# This is a separately curated structured fact from the quoted announcement.
# It is NOT extracted by parsing evidence prose or treating unknown fireworks
# start as known. Extension to other announcements requires a reviewed record.
COMBINED_START_HOURS = {"2026-07-25": 20.0, "2026-08-15": 20.0}


def record_digest(record):
    raw = json.dumps(record, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _date(value):
    date = pd.Timestamp(value)
    if pd.isna(date) or date.tzinfo is not None or date != date.normalize():
        raise ValueError("Show-end event_date must be a local calendar date")
    return date


def _effective(record):
    """Use the latest source-version time, conservatively for date-only fields."""
    times = [aware_time(record.get("published_at"), "published_at")]
    for field in ("original_published_at", "source_updated_at", "updated_at"):
        value = record.get(field)
        if value:
            if isinstance(value, str) and len(value) == 10:
                value = f"{value}T23:59:59+08:00"
            times.append(aware_time(value, field))
    return max(times)


def _same_number(actual, expected, label):
    if expected is None:
        if actual is not None:
            raise ValueError(f"{label}: unknown source cannot be promoted")
        return
    if isinstance(actual, bool) or actual is None:
        raise ValueError(f"{label}: numeric value required")
    try:
        valid = np.isfinite(float(actual)) and np.isclose(
            float(actual), float(expected), atol=1e-10, rtol=0)
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError(f"{label}: derived value differs from validated parent")


def _load_validated(root, source_path):
    path = (root / source_path).resolve()
    path.relative_to(root)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if (payload.get("schema_version") != 1 or payload.get("timezone") != "Asia/Taipei"
            or payload.get("parent_source_path") != PARENT_SOURCE):
        raise ValueError("Invalid show-end source schema or parent source")
    parent_path = root / PARENT_SOURCE
    parent_entry = _manifest_entry(root, parent_path)
    if parent_entry["sha256"] != payload.get("parent_source_sha256"):
        raise ValueError("Show-end parent source changed; re-audit the derivative")
    parent = json.loads(parent_path.read_text(encoding="utf-8-sig"))
    parent_records = {}
    for item in parent["records"]:
        key = (item["event_date"], item["feature"])
        if key in parent_records:
            raise ValueError("Show-end derivative requires unique parent evidence records")
        parent_records[key] = item
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Show-end evidence requires records")
    validated = {}
    for record in records:
        date = _date(record["event_date"])
        key = date.strftime("%Y-%m-%d")
        if key in validated:
            raise ValueError("Duplicate show-end event")
        base_cutoff = date.tz_localize("Asia/Taipei") - pd.Timedelta(days=1) + pd.Timedelta(hours=18)
        if aware_time(record.get("cutoff_at"), "cutoff_at") != base_cutoff:
            raise ValueError("Derived source cutoff does not match D-1 18:00")
        refs = record.get("parent_references", [])
        wanted = (["planned_program_minutes"] if key in COMBINED_START_HOURS else
                  ["planned_fireworks_start_hour", "planned_fireworks_minutes"])
        if [ref.get("feature") for ref in refs] != wanted:
            raise ValueError("Show-end parent references do not match curated source scope")
        linked = []
        for ref in refs:
            original = parent_records.get((key, ref["feature"]))
            if original is None or record_digest(original) != ref.get("record_sha256"):
                raise ValueError("Derived parent reference differs from validated source")
            linked.append(original)
        available = max(_effective(item) for item in linked)
        if aware_time(record.get("published_at"), "derived published_at") != available:
            raise ValueError("Derived availability cannot precede parent source version")
        urls = list(dict.fromkeys(item["source_url"] for item in linked))
        if record.get("source_urls") != urls:
            raise ValueError("Derived source URL differs from parent evidence")
        eligible = all(item.get("availability_status") == "verified_pre_cutoff"
                       and item.get("value") is not None for item in linked) and available <= base_cutoff
        if key in COMBINED_START_HOURS:
            start = COMBINED_START_HOURS[key]
            duration = linked[0].get("value")
            fact = record.get("structured_start_evidence", {})
            if (fact.get("source_url") != COMBINED_URL or urls != [COMBINED_URL]
                    or fact.get("evidence_text") != linked[0].get("evidence_text")
                    or fact.get("parent_record_sha256") != record_digest(linked[0])):
                raise ValueError("Combined sky program needs its separately curated start evidence")
            _same_number(fact.get("value"), start, "combined program start")
            if aware_time(fact.get("published_at"), "structured start published_at") != available:
                raise ValueError("Combined start availability must match source")
            scope = "combined_sky_program"
        else:
            start, duration = (item.get("value") for item in linked)
            scope = "fireworks_only"
        if eligible:
            if not (0 <= float(start) < 24 and 0 < float(duration) <= 360):
                raise ValueError("Invalid planned show start or duration")
            end = float(start) + float(duration) / 60.0
            expected_status = "verified_pre_cutoff"
        else:
            start = duration = end = None
            scope, expected_status = "unknown", "unknown"
        if record.get("scope") != scope or record.get("availability_status") != expected_status:
            raise ValueError("Unknown or late source cannot be promoted to an eligible show end")
        for field, expected in (("planned_start_hour", start), ("planned_duration_minutes", duration),
                                ("planned_end_hour", end)):
            _same_number(record.get(field), expected, field)
        validated[date] = {**record, "effective_available_at": available,
                           "eligible": eligible, "parent_features": wanted}
    expected_dates = {_date(date) for date, feature in parent_records
                      if feature == "planned_fireworks_start_hour"}
    if set(validated) != expected_dates:
        raise ValueError("Show-end source must preserve all parent event dates, including unknowns")
    return validated, [_manifest_entry(root, path), parent_entry]


def attach_end_features(frame, metadata, root, source_path="docs/d1_show_end_sources.json", cutoff_hour=18):
    """Return copied frame, extended metadata and one provenance row per event.

    Values are interval-overlap lengths in hours, each multiplied by a station
    indicator. Unknown end time leaves all eight columns missing, including
    columns for other stations. Existing inputs and labels are never modified.
    """
    if type(cutoff_hour) is not int or not 0 <= cutoff_hour <= 23:
        raise ValueError("D1 cutoff hour must be an integer 0..23")
    if not {"event_date", "station", "hour"} <= set(frame):
        raise ValueError("Show-end features require event_date/station/hour")
    if set(END_FEATURES) & set(frame):
        raise ValueError("Show-end feature columns already exist")
    if frame.empty or frame[["event_date", "station", "hour"]].isna().any().any():
        raise ValueError("Show-end input has empty or missing keys")
    if not set(frame.station) <= set(STATION_CODES.values()):
        raise ValueError("Show-end features require one of the four audited stations")
    dates = frame.event_date.map(_date)
    hours = pd.to_numeric(frame.hour, errors="raise")
    if not (np.isfinite(hours) & hours.between(0, 23) & hours.eq(np.floor(hours))).all():
        raise ValueError("Show-end hour must identify an integer hourly bin 0..23")
    root = Path(root).resolve()
    evidence, manifests = _load_validated(root, source_path)
    result, info = frame.copy(), copy.deepcopy(metadata)
    for name in END_FEATURES:
        result[name] = np.nan
    audit = []
    for date in sorted(dates.unique()):
        date = pd.Timestamp(date)
        record = evidence.get(date)
        cutoff = date.tz_localize("Asia/Taipei") - pd.Timedelta(days=1) + pd.Timedelta(hours=cutoff_hour)
        eligible = bool(record and record["eligible"] and record["effective_available_at"] <= cutoff)
        reason = ("verified_pre_cutoff" if eligible else "no_verified_record" if not record else
                  "published_after_cutoff" if record["effective_available_at"] > cutoff else
                  "unknown_or_unverified_source_version")
        mask = dates.eq(date)
        if eligible:
            end = record["planned_end_hour"]
            for window, offset in (("0_60", 0), ("60_120", 1)):
                overlap = (np.minimum(hours.loc[mask] + 1, end + offset + 1)
                           - np.maximum(hours.loc[mask], end + offset)).clip(0, 1)
                for code, station in STATION_CODES.items():
                    result.loc[mask, f"planned_end_{window}_{code}"] = (
                        overlap * frame.loc[mask, "station"].eq(station))
        audit.append({"event_date": date.strftime("%Y-%m-%d"), "eligible": eligible,
                      "reason": reason, "scope": record["scope"] if eligible else "unknown",
                      "planned_start_hour": record["planned_start_hour"] if eligible else np.nan,
                      "planned_duration_minutes": record["planned_duration_minutes"] if eligible else np.nan,
                      "planned_end_hour": record["planned_end_hour"] if eligible else np.nan,
                      "cutoff_at": cutoff.isoformat(),
                      "effective_available_at": record["effective_available_at"].isoformat() if record else "",
                      "source_urls": json.dumps(record["source_urls"] if record else [], ensure_ascii=False),
                      "parent_features": json.dumps(record["parent_features"] if record else [])})
    audit = pd.DataFrame(audit)
    for name in END_FEATURES:
        known = int(result[name].notna().sum())
        info.setdefault("feature_eligibility", {})[name] = {
            "use": "verified_D1_planned_show_end_proxy" if known else "all_unknown_no_learnable_values",
            "reason": "Audited planned show end; station-specific 0–60/60–120 minute overlap. Not actual egress.",
            "known_rows": known, "unknown_rows": int(len(result) - known),
        }
        info.setdefault("missing_summary", {})[name] = int(len(result) - known)
    existing = {(item["path"], item["sha256"]) for item in info.setdefault("manifest", [])}
    info["manifest"].extend(item for item in manifests if (item["path"], item["sha256"]) not in existing)
    audit_info = info.setdefault("audit", {})
    for key in ("new_d1_features", "model_feature_whitelist"):
        audit_info.setdefault(key, []).extend(name for name in END_FEATURES if name not in audit_info.get(key, []))
    audit_info["planned_show_end_sources"] = manifests
    audit_info["planned_show_end_summary"] = {
        "events": int(len(audit)), "known_events": int(audit.eligible.sum()),
        "unknown_events": int((~audit.eligible).sum()),
        "known_by_scope": audit.loc[audit.eligible, "scope"].value_counts().to_dict(),
        "policy": "station_specific_overlap_0_60_and_60_120_minutes",
        "cutoff": f"D-1 {cutoff_hour:02d}:00 Asia/Taipei",
        "limitation": "Announcement-version fields are audited; no archived historical website version. Planned end is not observed egress or station arrival.",
    }
    return result, info, audit
