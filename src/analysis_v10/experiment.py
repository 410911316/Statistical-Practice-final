"""Temporal meta-validation and error complementarity, never outer-label selection."""
from itertools import combinations, product

import numpy as np
import pandas as pd

from analysis_v5.models import solve_stack

MODELS = ("ridge", "gam", "xgboost")
KEYS = ["event_date", "station", "hour"]


def checked_predictions(source, cutoff_year=None):
    required = [*KEYS, "year", "y", *MODELS, "base_train_end"]
    if not set(required) <= set(source) or source.empty:
        raise ValueError("Missing temporal predictions")
    frame = source.copy()
    for col in ("event_date", "base_train_end"):
        frame[col] = pd.to_datetime(frame[col], errors="raise")
    if frame[required].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Missing or duplicated temporal rows; subset outer fold before analysis")
    if not np.isfinite(frame[["y", *MODELS]].to_numpy(float)).all() or (frame[["y", *MODELS]] < 0).any().any():
        raise ValueError("Predictions and counts must be finite and nonnegative")
    if not frame.year.eq(frame.event_date.dt.year).all() or not frame.base_train_end.dt.year.lt(frame.year).all():
        raise ValueError("Base predictions contain training-year or future leakage")
    if cutoff_year is not None and not frame.year.lt(cutoff_year).all():
        raise ValueError("Outer labels cannot enter meta fitting or selection")
    stations = set(frame.station)
    if "北門" not in stations:
        raise ValueError("Beimen target station missing")
    panel = set(product(stations, range(18, 24)))
    for _, event in frame.groupby("event_date"):
        if len(event) != len(panel) or set(zip(event.station, event.hour)) != panel:
            raise ValueError("Incomplete event panel")
        if event.base_train_end.nunique() != 1:
            raise ValueError("Inconsistent base training cutoff")
    return frame.sort_values(KEYS).reset_index(drop=True)


def scope_rows(frame, scope):
    if scope not in ("all", "beimen"):
        raise ValueError("Unknown weight fitting scope")
    return frame if scope == "all" else frame.loc[frame.station.eq("北門")]


def fit_weights(frame, scope):
    part = scope_rows(frame, scope)
    w, audit = solve_stack(part[list(MODELS)], part.y, part.event_date)
    return w, audit


def gate_from_past(meta_source, fit_scope, outer_year, config):
    """Evaluate the fusion procedure on years later than its weight-training rows.

    All candidate base predictions are already time-OOF. An additional rolling
    layer fits weights on earlier OOF years and checks fusion vs XGB on later
    OOF years. Only after this gate decision are final weights refitted on all
    available past OOF. No test frame/truth is an argument to this function.
    """
    meta = checked_predictions(meta_source, cutoff_year=outer_year)
    min_train = int(config["gate_min_weight_training_events"])
    min_events = int(config["gate_min_validation_events"])
    min_years = int(config["gate_min_validation_years"])
    tol = float(config["gate_numerical_tolerance"])
    if min(min_train, min_events, min_years) < 1 or not np.isfinite(tol) or tol < 0:
        raise ValueError("Invalid prespecified gate limits")
    events, folds = [], []
    for year in sorted(meta.year.unique()):
        train, valid = meta.loc[meta.year.lt(year)], meta.loc[meta.year.eq(year) & meta.station.eq("北門")]
        if train.event_date.nunique() < min_train:
            continue
        w, _ = fit_weights(train, fit_scope)
        part = valid[KEYS + ["y", "year", "xgboost"]].copy()
        part["fusion"] = valid[list(MODELS)].to_numpy() @ w
        for date, event in part.groupby("event_date"):
            events.append({"event_date": date, "year": int(year),
                           "xgb_mae": float(abs(event.xgboost - event.y).mean()),
                           "fusion_mae": float(abs(event.fusion - event.y).mean())})
        folds.append({"validation_year": int(year), "weight_train_end": str(train.event_date.max().date()),
                      "weight_train_events": int(train.event_date.nunique()),
                      "validation_start": str(valid.event_date.min().date()),
                      "validation_events": int(valid.event_date.nunique()),
                      "weights": dict(zip(MODELS, map(float, w)))})
    event_frame = pd.DataFrame(events, columns=["event_date", "year", "xgb_mae", "fusion_mae"])
    sufficient = len(events) >= min_events and event_frame.year.nunique() >= min_years
    xgb_loss = float(event_frame.xgb_mae.mean()) if len(events) else None
    fusion_loss = float(event_frame.fusion_mae.mean()) if len(events) else None
    choose_fusion = bool(sufficient and fusion_loss < xgb_loss - tol)
    # Fit final weights exclusively on all past OOF, regardless of gate outcome.
    final, final_audit = fit_weights(meta, fit_scope)
    decision = {"outer_year": int(outer_year), "fit_scope": fit_scope, "evaluation_station": "北門",
                "chosen": "fusion" if choose_fusion else "xgboost",
                "reason": "lower_validation_mae" if choose_fusion else
                          ("insufficient_meta_validation" if not sufficient else "xgboost_better_or_tied"),
                "validation_events": len(events), "validation_years": int(event_frame.year.nunique()),
                "xgb_validation_mae": xgb_loss, "fusion_validation_mae": fusion_loss,
                "oof_events": int(meta.event_date.nunique()), "oof_end": str(meta.event_date.max().date()),
                "final_weights": dict(zip(MODELS, map(float, final))), "final_oof_mae": final_audit["oof_mae"],
                "validation_folds": folds}
    return decision, event_frame


def apply_decision(base_predictions, decision):
    """Application only reads three prediction columns, never truths or hours."""
    values = base_predictions[list(MODELS)].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid base prediction matrix")
    if decision["chosen"] == "xgboost":
        return values[:, 2].copy()
    if decision["chosen"] != "fusion":
        raise ValueError("Unknown fusion adoption decision")
    w = np.array([decision["final_weights"][m] for m in MODELS])
    if not np.isfinite(w).all() or w.min() < -1e-9 or abs(w.sum()-1) > 1e-8:
        raise ValueError("Invalid saved simplex weights")
    return values @ w


def evaluate_beimen(frame, methods):
    rows = []
    if not frame.station.eq("北門").all() or frame.duplicated(KEYS).any():
        raise ValueError("Scoring requires unique Beimen rows")
    if not np.isfinite(frame[["y", *methods]].to_numpy(float)).all():
        raise ValueError("Missing paired predictions")
    for date, event in frame.groupby("event_date"):
        if set(event.hour) != set(range(18,24)) or len(event) != 6:
            raise ValueError("Scoring requires all six event hours")
        event = event.sort_values("hour")
        peak = event.loc[event.y.idxmax()]
        for model in methods:
            error = event[model] - event.y
            signed = float(peak.y - peak[model])
            rows.append({"event_date": date, "year": int(pd.Timestamp(date).year), "model": model,
                         "mae": float(abs(error).mean()), "mse": float((error**2).mean()), "bias": float(error.mean()),
                         "peak_underprediction": max(0, signed), "peak_absolute_error": abs(signed),
                         "peak_hour_mae": abs(float(peak.hour - event.loc[event[model].idxmax(), "hour"]))})
    events = pd.DataFrame(rows)
    return summarize(events), events


def summarize(events, groups=("model",)):
    metrics = ["mae", "mse", "bias", "peak_underprediction", "peak_absolute_error", "peak_hour_mae"]
    result = events.groupby(list(groups), sort=False)[metrics].mean().reset_index()
    result["rmse"] = np.sqrt(result.mse)
    result["events"] = events.groupby(list(groups), sort=False).size().to_numpy()
    return result


def complementary_errors(source, scope="beimen"):
    frame = scope_rows(checked_predictions(source), scope).copy()
    counts = frame.event_date.value_counts()
    weights = 1/(len(counts)*frame.event_date.map(counts).to_numpy(float))
    errors = {m: frame[m].to_numpy(float) - frame.y.to_numpy(float) for m in MODELS}
    centered = {m: frame.assign(e=errors[m]).groupby("event_date").e.transform(lambda x: x-x.mean()).to_numpy() for m in MODELS}

    def correlation(x, y):
        u, v = x - weights @ x, y - weights @ y
        den = np.sqrt((weights @ (u*u))*(weights @ (v*v)))
        return float(weights @ (u*v)/den) if den > 1e-12 else np.nan

    pairs = []
    for a, b in combinations(MODELS, 2):
        u, v = errors[a], errors[b]
        pairs.append({"model_a": a, "model_b": b, "events": len(counts), "rows": len(frame),
                      "error_correlation": correlation(u, v), "within_event_error_correlation": correlation(centered[a], centered[b]),
                      "opposite_sign_fraction": float(weights @ (u*v < 0)),
                      "both_under_fraction": float(weights @ ((u < 0) & (v < 0))),
                      "both_over_fraction": float(weights @ ((u > 0) & (v > 0))),
                      "pair_equal_mae": float(weights @ abs((u+v)/2))})
    help_rows = []
    xgb = errors["xgboost"]
    under = xgb < 0
    mass = float(weights @ under)
    for m in ("ridge", "gam"):
        e = errors[m]
        blended = (e+xgb)/2
        help_rows.append({"candidate": m, "events": len(counts), "xgb_under_rows": int(under.sum()),
                          "xgb_under_events": int(frame.loc[under, "event_date"].nunique()),
                          "candidate_closer_when_xgb_under": float(weights @ (under & (abs(e) < abs(xgb)))/mass) if mass else np.nan,
                          "pair_blend_better_when_xgb_under": float(weights @ (under & (abs(blended) < abs(xgb)))/mass) if mass else np.nan,
                          "pair_blend_mae_minus_xgb": float(weights @ (abs(blended)-abs(xgb))),
                          "candidate_mae": float(weights @ abs(e)), "xgb_mae": float(weights @ abs(xgb))})
    return pd.DataFrame(pairs), pd.DataFrame(help_rows)
