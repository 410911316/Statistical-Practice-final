"""Data audits, with unavailable publication evidence kept separate from models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

KEYS = ["event_date", "station", "hour"]
TYPE_COLUMNS = ["planned_finale", "planned_enhanced_nonfinale"]


def event_types(path):
    document = json.loads(Path(path).read_text("utf-8"))
    rows = []
    for record in document["records"]:
        cutoff = pd.Timestamp(record["cutoff_at"])
        day = pd.Timestamp(record["event_date"], tz="Asia/Taipei")
        if cutoff != day - pd.Timedelta(hours=6):
            raise ValueError("Wrong D-1 18:00 cutoff")
        out = {"event_date": day.tz_localize(None)}
        for column in TYPE_COLUMNS:
            value = record[column]
            published = record.get("enhancement_published_at", record["published_at"]) if column == TYPE_COLUMNS[1] else record["published_at"]
            if value is not None:
                if value not in (0, 1) or record["availability_status"] != "verified_pre_cutoff":
                    raise ValueError("Unverified or nonbinary event type")
                if pd.Timestamp(published) > cutoff:
                    raise ValueError("Event-type announcement after cutoff")
            out[column] = np.nan if value is None else float(value)
        rows.append(out)
    frame = pd.DataFrame(rows)
    if frame.event_date.duplicated().any():
        raise ValueError("Duplicate event types")
    return frame


def read_google(path):
    raw = pd.read_csv(path, encoding="cp950", dtype=str)
    raw = raw.loc[raw.iloc[:, 0].notna()].copy()
    periods = pd.to_datetime(raw.iloc[:, 0], format="%b-%y", errors="raise").dt.to_period("M")
    text = raw.iloc[:, 1].str.strip()
    low = pd.to_numeric(text.replace("<1", "0"), errors="raise")
    high = pd.to_numeric(text.replace("<1", "1"), errors="raise")
    frame = pd.DataFrame({"month": periods, "raw_value": text.to_numpy(),
                          "lower": low.to_numpy(), "upper": high.to_numpy(),
                          "censored": text.eq("<1").to_numpy()})
    frame["midpoint"] = (frame.lower + frame.upper) / 2
    if frame.month.duplicated().any() or not frame.lower.between(0, 100).all() or not frame.upper.between(0, 100).all():
        raise ValueError("Invalid Google monthly index")
    if len(pd.period_range(frame.month.min(), frame.month.max(), freq="M")) != len(frame):
        raise ValueError("Missing Google months")
    return frame.reset_index(drop=True)


def baseline_audit(root, frame):
    # Recompute from original aggregated counts and calendar-selected days,
    # rather than trusting the already materialized model column.
    from analysis_v1.data import load_analysis_data
    original, metadata = load_analysis_data(root)
    paired = frame[KEYS + ["baseline_hourly_56d"]].merge(
        original[KEYS + ["baseline_hourly_56d"]], on=KEYS, validate="one_to_one",
        suffixes=("_snapshot", "_rebuilt"))
    if len(paired) != len(frame) or not np.allclose(paired.baseline_hourly_56d_snapshot,
                                                   paired.baseline_hourly_56d_rebuilt, rtol=0, atol=1e-8):
        raise ValueError("Baseline does not match independently rebuilt source")
    candidates = pd.read_csv(root / "data/processed/calendar_baseline/candidate_audit.csv", parse_dates=["event_date", "candidate_date"])
    selected = candidates.loc[candidates.selected.eq(True) & candidates.event_date.isin(frame.event_date)].copy()
    if (selected.candidate_date >= selected.event_date).any():
        raise ValueError("Nonpast baseline day")
    selected["same_event_month"] = selected.candidate_date.dt.to_period("M") == selected.event_date.dt.to_period("M")
    audit = selected.groupby("event_date").agg(
        distinct_days=("candidate_date", "nunique"),
        latest_observation=("candidate_date", "max"),
        same_month_station_day_pairs=("same_event_month", "sum")).reset_index()
    audit["publication_verified"] = False
    audit["status"] = "retrospective_sensitivity_only"
    return audit, metadata


def load(root):
    frame = pd.read_csv(root / ".local/analysis_v8/both/analysis_input.csv", parse_dates=["event_date"]).sort_values(KEYS).reset_index(drop=True)
    typ = event_types(root / "docs/d1_event_type_sources.json")
    frame = frame.merge(typ, on="event_date", validate="many_to_one")
    if len(frame) != 504 or frame[KEYS].duplicated().any():
        raise ValueError("Unexpected analysis panel")
    google = read_google(root / "data/original_data/google_網路聲量.csv")
    previous = frame.event_date.dt.to_period("M") - 1
    frame["unverified_google_prev_month"] = previous.map(google.set_index("month").midpoint)
    if frame.unverified_google_prev_month.isna().any():
        raise ValueError("Missing previous calendar month")
    audit, metadata = baseline_audit(root, frame)
    frame["retrospective_baseline_56d"] = frame.baseline_hourly_56d
    return frame, google, audit, metadata


def check_variant(name, groups, config):
    unsafe = {"retrospective_baseline_56d", "unverified_google_prev_month"} & set(groups)
    if unsafe and name not in config["sensitivity_only"]:
        raise ValueError("Publication-unverified feature in primary ablation")
    if set(groups) & {"y", "observed_has_drone_show", "observed_fireworks_seconds", "cloud_at_report_time_tenths"}:
        raise ValueError("Post-event data cannot be predictors")
