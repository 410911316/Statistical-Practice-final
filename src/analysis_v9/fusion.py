"""Event-balanced simplex stacking; outer targets never enter weight fitting."""
from itertools import combinations, product

import numpy as np
import pandas as pd

from analysis_v5.models import solve_stack
from analysis_v6.reporting import HOURS, KEYS, STATIONS, validate_panel, validate_fusion_predictions

MODELS = ("ridge", "gam", "xgboost")
DISPLAY_MODELS = (*MODELS, "gam_equal", "mae_fusion", "mse_fusion")


def event_weights(predictions, truth, event_dates):
    p, y = np.asarray(predictions, float), np.asarray(truth, float)
    dates = pd.Series(np.asarray(event_dates))
    if (p.ndim != 2 or y.ndim != 1 or not len(y) or p.shape != (len(y), 3)
            or not np.isfinite(p).all() or not np.isfinite(y).all()
            or (p < 0).any() or (y < 0).any()
            or len(dates) != len(y) or dates.isna().any()):
        raise ValueError("Three finite nonnegative predictions, truth and matching event dates required")
    counts = dates.value_counts()
    return p, y, 1 / (len(counts) * dates.map(counts).to_numpy(float))


def solve_mse(predictions, truth, event_dates):
    """Global convex optimum by enumerating all seven faces of a 3-simplex.

    Each face uses constrained-sum weighted least squares. Boundary optima are
    covered by smaller faces. A KKT dual-gap certificate catches numerical or
    rank-degeneracy failures instead of silently returning a feasible fit.
    """
    p, y, a = event_weights(predictions, truth, event_dates)
    scale = max(1., float(p.max()), float(y.max()))
    q, z = p / scale, y / scale
    candidates = []
    for size in range(1, 4):
        for face in combinations(range(3), size):
            w = np.zeros(3)
            if size == 1:
                w[face[0]] = 1.
            else:
                last = face[-1]
                design = q[:, face[:-1]] - q[:, [last]]
                fit = np.linalg.lstsq(design * np.sqrt(a[:, None]),
                                     (z - q[:, last]) * np.sqrt(a), rcond=None)[0]
                w[list(face[:-1])] = fit
                w[last] = 1 - fit.sum()
            if w.min() < -1e-9:
                continue
            w = np.maximum(w, 0)
            w /= w.sum()
            candidates.append((float(a @ (q @ w - z)**2), w))
    _, w = min(candidates, key=lambda item: item[0])
    gradient = 2 * q.T @ (a * (q @ w - z))
    gap = float(gradient @ w - gradient.min())
    if not -1e-10 <= gap <= 1e-8:
        raise RuntimeError(f"MSE simplex KKT certificate failed: {gap}")
    return w, {"solver": "seven_simplex_faces_weighted_lstsq", "scaled_kkt_dual_gap": gap,
               "oof_mse": float(a @ (p @ w - y)**2), "scale": scale,
               "scope": "meta_training_loss_not_outer_test_score"}


def fit_losses(meta):
    """Only chronological meta-training rows are accepted by this fit API."""
    p, y, a = event_weights(meta[list(MODELS)], meta.y, meta.event_date)
    mae_w, mae_audit = solve_stack(p, y, meta.event_date)
    mse_w, mse_audit = solve_mse(p, y, meta.event_date)
    weights, audit = {"mae": mae_w, "mse": mse_w}, {"mae": mae_audit, "mse": mse_audit}
    for loss, w in weights.items():
        error = p @ w - y
        audit[loss].update(oof_mae=float(a @ abs(error)), oof_mse=float(a @ error**2),
                           oof_rmse=float(np.sqrt(a @ error**2)))
    if (audit["mae"]["oof_mae"] > audit["mse"]["oof_mae"] + 1e-6
            or audit["mse"]["oof_mse"] > audit["mae"]["oof_mse"] + 1e-4):
        raise RuntimeError("Own-objective dominance check failed")
    return weights, audit


def validate_meta(source, outer, folds, analysis_input):
    """Validate whole-event OOF panels, historical cutoffs and source labels."""
    required = {"event_date", "year", "station", "hour", "y", "base_train_end", "outer_year", *MODELS}
    if not required <= set(source) or source.empty:
        raise ValueError("Missing meta columns or rows")
    meta = source.copy()
    for col in ("event_date", "base_train_end"):
        meta[col] = pd.to_datetime(meta[col], errors="raise")
    if meta[list(required)].isna().any().any() or meta.duplicated(["outer_year", *KEYS]).any():
        raise ValueError("Missing or duplicate meta keys")
    event_weights(meta[list(MODELS)], meta.y, meta.event_date)
    if not (meta.base_train_end < meta.event_date).all() or not (meta.base_train_end.dt.year < meta.year).all():
        raise ValueError("OOF base model was trained on validation year or later")
    if not meta.year.eq(meta.event_date.dt.year).all():
        raise ValueError("Meta year/date mismatch")
    if set(meta.outer_year) != set(outer.outer_year):
        raise ValueError("Meta/outer fold mismatch")
    expected = set(product(STATIONS, HOURS))
    for _, part in meta.groupby(["outer_year", "event_date"]):
        if len(part) != len(expected) or set(zip(part.station, part.hour)) != expected:
            raise ValueError("Incomplete OOF station/hour panel")
        if part.base_train_end.nunique() != 1:
            raise ValueError("Inconsistent OOF base training cutoff")
    for fold in folds:
        year = fold["outer_year"]
        part = meta.loc[meta.outer_year.eq(year)]
        if (not part.year.lt(year).all() or not part.event_date.lt(pd.Timestamp(fold["test_start"])).all()
                or not part.event_date.le(pd.Timestamp(fold["train_end"])).all()):
            raise ValueError("Future outer-year labels entered fusion training")
        audit = fold["fusion_audit"]
        if (len(part) != audit["oof_rows"] or part.event_date.nunique() != audit["oof_events"]
                or part.event_date.min() != pd.Timestamp(audit["oof_start"])
                or part.event_date.max() != pd.Timestamp(audit["oof_end"])):
            raise ValueError("Meta cache does not match original fusion audit")
    inputs = analysis_input[KEYS + ["y"]].copy()
    inputs["event_date"] = pd.to_datetime(inputs.event_date)
    if inputs.duplicated(KEYS).any():
        raise ValueError("Duplicate input labels")
    for name, frame in (("OOF", meta), ("outer", outer)):
        merged = frame.merge(inputs, on=KEYS, how="left", validate="many_to_one", suffixes=("", "_input"))
        if not np.array_equal(merged.y.to_numpy(), merged.y_input.to_numpy()):
            raise ValueError(f"{name} labels do not match input snapshot")
    return meta.sort_values(["outer_year", *KEYS]).reset_index(drop=True)


def compare_variant(outer_source, meta_source, folds, analysis_input):
    outer = validate_panel(outer_source)
    validate_fusion_predictions(outer, folds)
    meta = validate_meta(meta_source, outer, folds, analysis_input)
    output, weight_rows, audits = outer.copy(), [], []
    for fold in folds:
        year = fold["outer_year"]
        training = meta.loc[meta.outer_year.eq(year)]
        weights, checks = fit_losses(training)
        recorded = np.array([fold["fusion_weights"]["gam_stack"][m] for m in MODELS])
        if not np.allclose(weights["mae"], recorded, rtol=0, atol=1e-7):
            raise ValueError("Refitted MAE weights differ from v8")
        mask = output.outer_year.eq(year)
        for loss, w in weights.items():
            output.loc[mask, f"{loss}_fusion"] = output.loc[mask, list(MODELS)].to_numpy() @ w
            weight_rows.append({"outer_year": year, "loss": loss,
                                **dict(zip(MODELS, map(float, w))),
                                "oof_events": int(training.event_date.nunique()),
                                "oof_start": str(training.event_date.min().date()),
                                "oof_end": str(training.event_date.max().date()), **checks[loss]})
        diff = float(abs(output.loc[mask, "mae_fusion"] - output.loc[mask, "gam_stack"]).max())
        if diff > 1e-5:
            raise ValueError("MAE predictions do not reproduce v8")
        audits.append({"outer_year": year, "mae_prediction_max_abs_difference": diff,
                       "mse_kkt_gap": checks["mse"]["scaled_kkt_dual_gap"]})
    return output, pd.DataFrame(weight_rows), audits


def evaluate(frame):
    """Event-balanced scores. RMSE is sqrt(mean(event MSE)), not mean RMSE."""
    rows = []
    for scope, block in (("all", frame), ("beimen", frame.loc[frame.station.eq("北門")])):
        for date, event in block.groupby("event_date"):
            for model in DISPLAY_MODELS:
                error = event[model].to_numpy(float) - event.y.to_numpy(float)
                peaks = []
                for _, station in event.groupby("station"):
                    station = station.sort_values("hour")
                    actual = station.loc[station.y.idxmax()]
                    predicted = station.loc[station[model].idxmax()]
                    signed = float(actual.y - actual[model])
                    peaks.append([max(0, signed), abs(signed), signed, abs(actual.hour - predicted.hour)])
                peak = np.asarray(peaks).mean(axis=0)
                rows.append({"scope": scope, "event_date": date, "year": pd.Timestamp(date).year,
                             "model": model, "rows": len(event), "mae": float(abs(error).mean()),
                             "mse": float((error**2).mean()), "bias": float(error.mean()),
                             "peak_underprediction": peak[0], "peak_absolute_error": peak[1],
                             "peak_signed_underprediction": peak[2], "peak_hour_mae": peak[3]})
    events = pd.DataFrame(rows)
    return aggregate_metrics(events), events


def aggregate_metrics(events, groups=("scope", "model")):
    measures = ["mae", "mse", "bias", "peak_underprediction", "peak_absolute_error",
                "peak_signed_underprediction", "peak_hour_mae"]
    aggregate = events.groupby(list(groups), sort=False)[measures].mean().reset_index()
    counts = events.groupby(list(groups), sort=False).agg(events=("event_date", "nunique"), rows=("rows", "sum")).reset_index()
    aggregate = aggregate.merge(counts, on=list(groups), validate="one_to_one")
    aggregate["rmse"] = np.sqrt(aggregate.mse)
    return aggregate
