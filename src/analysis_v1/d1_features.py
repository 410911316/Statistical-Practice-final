"""Join only source versions demonstrably available before D-1 18:00 Taipei.

This verifies new covariates, not the publication history of existing OD data.
Unknown/late values stay missing; observed weather is never a substitute.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import _manifest_entry

PROGRAM_FEATURES = (
    "planned_drone", "planned_fireworks_minutes", "planned_fireworks_start_hour",
    "planned_program_minutes",
)
WEATHER_FEATURES = (
    "forecast_temperature_c", "forecast_precip_probability",
    "forecast_wind_speed_m_s", "forecast_cloud_cover_pct",
)
BOUNDS = {"planned_drone": (0, 1), "planned_fireworks_minutes": (0, 180),
          "planned_fireworks_start_hour": (0, 24), "planned_program_minutes": (0, 360),
          "forecast_temperature_c": (-60, 60), "forecast_precip_probability": (0, 100),
          "forecast_wind_speed_m_s": (0, 150), "forecast_cloud_cover_pct": (0, 100)}


def aware_time(value, label):
    if value is None or pd.isna(value):
        raise ValueError(f"{label}: missing timestamp")
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError(f"{label}: explicit timezone required")
    return stamp.tz_convert("Asia/Taipei")


def attach_d1_features(frame, metadata, root, source_paths, features, cutoff_hour=18):
    """Return a joined frame, extended metadata and a public provenance audit.

    Evidence records may contain several announcement versions. Selection uses
    the latest eligible publication before cutoff, never the observed outcome.
    """
    if len(features) != len(set(features)) or not set(features) <= set(BOUNDS):
        raise ValueError("D1 feature list contains duplicates or unapproved names")
    if not isinstance(cutoff_hour, int) or not 0 <= cutoff_hour <= 23:
        raise ValueError("D1 cutoff hour must be an integer 0..23")
    root = Path(root).resolve()
    result, info = frame.copy(), copy.deepcopy(metadata)
    records, sources = [], []
    for name in source_paths:
        path = root / name
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload.get("records"), list):
            raise ValueError(f"D1 source requires records list: {name}")
        records.extend(payload["records"])
        sources.append(_manifest_entry(root, path))
    events = sorted(pd.to_datetime(frame.event_date).unique())
    accepted, rejected = {}, {}
    for record in records:
        feature = record.get("feature")
        if feature not in BOUNDS:
            raise ValueError(f"Unapproved D1 source feature: {feature}")
        if feature not in features:
            continue
        date = pd.Timestamp(record["event_date"])
        if date.tzinfo is not None or date != date.normalize():
            raise ValueError("Evidence event_date must be a local calendar date")
        if date not in events:
            continue
        hour = record.get("hour") if feature in WEATHER_FEATURES else None
        if feature in WEATHER_FEATURES and (hour is None or int(hour) != hour or not 18 <= hour <= 23):
            raise ValueError("Forecast evidence requires hour 18..23")
        key = (date, hour, feature)
        cutoff = date.tz_localize("Asia/Taipei") - pd.Timedelta(days=1) + pd.Timedelta(hours=cutoff_hour)
        if record.get("availability_status") != "verified_pre_cutoff" or record.get("value") is None:
            rejected[key] = "unknown_or_unverified_source_version"
            continue
        published = aware_time(record.get("published_at"), "published_at")
        # A post-cutoff update cannot certify that the current text was available
        # earlier. A genuinely archived older version must have its own record.
        if record.get("updated_at"):
            published = max(published, aware_time(record["updated_at"], "updated_at"))
        if published > cutoff:
            rejected[key] = "published_after_cutoff"
            continue
        if not record.get("source_url") or not record.get("evidence_text"):
            raise ValueError("Verified evidence requires source_url and evidence_text")
        value = float(record["value"])
        low, high = BOUNDS[feature]
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"D1 feature outside plausible units/range: {feature}")
        if feature == "planned_drone" and value not in (0., 1.):
            raise ValueError("planned_drone must be 0, 1, or unknown")
        if feature in WEATHER_FEATURES:
            issued = aware_time(record.get("forecast_issued_at"), "forecast_issued_at")
            valid = aware_time(record.get("forecast_valid_at"), "forecast_valid_at")
            expected = date.tz_localize("Asia/Taipei") + pd.Timedelta(hours=hour)
            if issued > published or issued > cutoff or valid != expected:
                raise ValueError("Forecast issue/publication/valid time is inconsistent")
        candidate = {**record, "value": value, "effective_available_at": published.isoformat()}
        old = accepted.get(key)
        if old and published == pd.Timestamp(old["effective_available_at"]) and value != old["value"]:
            raise ValueError("Conflicting values in same published source version")
        if old is None or published > pd.Timestamp(old["effective_available_at"]):
            accepted[key] = candidate
    audit = []
    for feature in features:
        result[feature] = np.nan
        for date in events:
            date = pd.Timestamp(date)
            hours = sorted(frame.hour.unique()) if feature in WEATHER_FEATURES else [None]
            for hour in hours:
                key = (date, hour, feature)
                record = accepted.get(key)
                mask = result.event_date.eq(date)
                if hour is not None:
                    mask &= result.hour.eq(hour)
                if record:
                    result.loc[mask, feature] = record["value"]
                audit.append({"event_date": date.strftime("%Y-%m-%d"), "hour": hour,
                              "feature": feature, "value": record["value"] if record else np.nan,
                              "eligible": record is not None,
                              "reason": "verified_pre_cutoff" if record else rejected.get(key, "no_verified_record"),
                              "cutoff": (date.tz_localize("Asia/Taipei") - pd.Timedelta(days=1)
                                         + pd.Timedelta(hours=cutoff_hour)).isoformat(),
                              "effective_available_at": record["effective_available_at"] if record else "",
                              "source_url": record["source_url"] if record else ""})
        n = int(result[feature].notna().sum())
        info["feature_eligibility"][feature] = {
            "use": "verified_D1_new_covariate" if n else "all_unknown_no_learnable_values",
            "reason": "Only verified source versions before D-1 cutoff; train-only imputation, never actual outcomes.",
            "known_rows": n, "unknown_rows": int(len(result)-n),
        }
        info["missing_summary"][feature] = int(result[feature].isna().sum())
    info["manifest"].extend(sources)
    info["audit"]["new_d1_features"] = list(features)
    info["audit"]["new_d1_source_files"] = sources
    info["audit"]["new_d1_cutoff"] = f"D-1 {cutoff_hour:02d}:00 Asia/Taipei"
    info["audit"]["model_feature_whitelist"].extend(features)
    # Do not promote a partly audited retrospective dataset to D1 deployable.
    return result, info, pd.DataFrame(audit)
