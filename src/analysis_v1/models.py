"""Chronological nested selection. No test labels enter fit, tuning or stacking."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

BASE_MODELS = ("ols", "ridge", "xgboost")
ALL_MODELS = ("baseline", *BASE_MODELS, "equal", "stack")
CORE_NUMERIC = ("baseline_hourly_56d",)
ALLOWED_CANDIDATES = {"is_weekend", "day_of_year", "year_index"}


def event_mae(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    values = np.abs(frame.y.to_numpy(float) - np.asarray(prediction))
    return float(pd.Series(values).groupby(frame.event_date.to_numpy()).mean().mean())


def temporal_splits(frame: pd.DataFrame, min_events: int):
    """Whole calendar years; all rows of an event stay together."""
    for year in sorted(frame.year.unique()):
        train = frame.loc[frame.year < year].copy()
        valid = frame.loc[frame.year == year].copy()
        if train.event_date.nunique() >= min_events and len(valid):
            assert train.event_date.max() < valid.event_date.min()
            yield int(year), train, valid


class Encoder:
    """Fit numeric scaling, reference levels and constants on this train only."""

    def fit(self, frame: pd.DataFrame, groups: tuple[str, ...], station_hour_interaction: bool = False):
        if not set(groups) <= ALLOWED_CANDIDATES:
            raise ValueError("Unapproved or post-event feature in model")
        self.groups = groups
        self.station_hour_interaction = station_hour_interaction
        self.numeric = list(CORE_NUMERIC) + list(groups)
        self.mean = frame[self.numeric].mean()
        self.scale = frame[self.numeric].std(ddof=0).replace(0, 1)
        # Dummy/binary columns stay on their 0/1 scale, as specified in the PDF.
        if "is_weekend" in self.numeric:
            self.mean["is_weekend"], self.scale["is_weekend"] = 0.0, 1.0
        self.categories = {c: sorted(frame[c].unique()) for c in ("station", "hour")}
        self.all_names = self.numeric + [f"{c}={v}" for c, cats in self.categories.items() for v in cats[1:]]
        if self.station_hour_interaction:
            # Treatment coding retains both main effects. The reference station
            # and hour are learned from this training subset, never the holdout.
            self.all_names += [f"station={s}:hour={h}"
                               for s in self.categories["station"][1:]
                               for h in self.categories["hour"][1:]]
        raw = self._raw(frame)
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite model input; missing values are not zero")
        self.keep = np.ptp(raw, axis=0) > 1e-12
        self.names = [n for n, keep in zip(self.all_names, self.keep) if keep]
        self.dropped_constants = [n for n, keep in zip(self.all_names, self.keep) if not keep]
        return self

    def _raw(self, frame):
        arrays = [((frame[self.numeric] - self.mean) / self.scale).to_numpy(float)]
        for col, cats in self.categories.items():
            arrays.extend([(frame[col].to_numpy() == val).astype(float)[:, None] for val in cats[1:]])
        # Previously saved v1 encoders do not have this attribute.
        if getattr(self, "station_hour_interaction", False):
            arrays.extend([((frame.station.to_numpy() == station) &
                            (frame.hour.to_numpy() == hour)).astype(float)[:, None]
                           for station in self.categories["station"][1:]
                           for hour in self.categories["hour"][1:]])
        return np.column_stack(arrays)

    def transform(self, frame):
        # Stations/hours are structural; unknown structural levels are invalid.
        for col, cats in self.categories.items():
            if not set(frame[col].unique()) <= set(cats):
                raise ValueError(f"Unseen {col} category; cannot silently forecast a new station/hour")
        x = self._raw(frame)[:, self.keep]
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite prediction features")
        return x


@dataclass
class Fitted:
    name: str
    encoder: Encoder
    estimator: object
    params: dict
    rank: int
    columns: int
    fallback: str | None = None

    def predict(self, frame):
        if self.fallback:
            return np.maximum(0, frame.baseline_hourly_56d.to_numpy(float))
        x = self.encoder.transform(frame)
        if self.name == "ols":
            pred = np.column_stack([np.ones(len(x)), x]) @ self.estimator
        else:
            pred = self.estimator.predict(x)
        return np.maximum(0, np.asarray(pred, dtype=float))

    def info(self):
        return {"model": self.name, "groups": list(self.encoder.groups), "columns": self.encoder.names,
                "dropped_constants": self.encoder.dropped_constants, "params": self.params,
                "rank": self.rank, "n_columns_with_intercept": self.columns, "fallback": self.fallback}


def fit_one(frame, name, groups, params, config):
    encoder = Encoder().fit(frame, tuple(groups), station_hour_interaction=(
        name == "ridge" and config.get("ridge_station_hour_interaction", False)))
    x = encoder.transform(frame)
    design = np.column_stack([np.ones(len(x)), x])
    rank = int(np.linalg.matrix_rank(design))
    y = frame.y.to_numpy(float)
    fallback = None
    if name == "ols":
        if rank < design.shape[1]:
            fallback, estimator = "rank_deficient_use_historical_baseline", None
        else:
            estimator = np.linalg.lstsq(design, y, rcond=None)[0]
    elif name == "ridge":
        # sklearn minimizes SSE + alpha * ||beta||^2, PDF uses mean SSE.
        alpha = len(frame) * float(params["lambda"])
        estimator = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(x, y)
        params = {**params, "sklearn_alpha": alpha, "n_train_rows": len(frame)}
    elif name == "xgboost":
        # Grid candidates may deliberately vary a parameter formerly held fixed.
        # Merge before constructing the estimator so candidate values take effect.
        xgb_params = {"objective": "reg:squarederror", "tree_method": "hist", "n_jobs": 1,
                      "random_state": config["seed"], **config["xgb_fixed"], **params}
        estimator = XGBRegressor(**xgb_params).fit(x, y)
    else:
        raise ValueError(name)
    return Fitted(name, encoder, estimator, params, rank, design.shape[1], fallback)


def solve_stack(predictions, truth, event_dates):
    """Simplex MAE linear program with equal weight per event; sparse constraints."""
    p, y = np.asarray(predictions, float), np.asarray(truth, float)
    if p.shape != (len(y), 3) or not len(y) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Stack needs finite common OOF predictions from exactly three models")
    if (p < 0).any() or (y < 0).any():
        raise ValueError("Stack inputs must be nonnegative person counts")
    dates = pd.Series(event_dates)
    if len(dates) != len(y) or dates.isna().any():
        raise ValueError("Stack event dates must match prediction rows and contain no missing values")
    counts = dates.value_counts()
    a = (1 / (dates.map(counts).to_numpy(float) * len(counts)))
    n = len(y)
    # p*w - u <= y ; -p*w - u <= -y.
    inequalities = vstack([hstack([csr_matrix(p), -eye(n)]), hstack([-csr_matrix(p), -eye(n)])], format="csr")
    equality = csr_matrix(([1., 1., 1.], ([0, 0, 0], [0, 1, 2])), shape=(1, n + 3))
    answer = linprog(np.r_[np.zeros(3), a], A_ub=inequalities, b_ub=np.r_[y, -y],
                     A_eq=equality, b_eq=[1.], bounds=[(0, None)] * (n + 3), method="highs")
    if not answer.success:
        raise RuntimeError(f"Stack LP did not converge: {answer.message}")
    w = answer.x[:3]
    if w.min() < -1e-8 or abs(w.sum() - 1) > 1e-8:
        raise RuntimeError("Invalid simplex solution")
    w = np.maximum(w, 0); w /= w.sum()
    meta_loss = float(np.dot(a, np.abs(y - p @ w)))
    if meta_loss > min(float(np.dot(a, abs(y - p[:, m]))) for m in range(3)) + 1e-5:
        raise RuntimeError("Stack meta loss exceeds best feasible vertex")
    return w, {"oof_mae": meta_loss, "oof_events": len(counts), "solver": "scipy.optimize.linprog(highs)",
               "scope": "meta_training_loss_not_outer_test_score"}


class Trainer:
    def __init__(self, config):
        self.config, self.cache = config, {}
        self.selection_log, self.fold_log = [], []

    def fit_bundle(self, frame):
        columns = ["event_date", "year", "station", "hour", "y", *CORE_NUMERIC, *self.config["candidate_groups"]]
        ordered = frame[columns].sort_values(["event_date", "station", "hour"])
        digest = hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes())
        digest.update(json.dumps(self.config, sort_keys=True).encode())
        key = digest.hexdigest()
        if key in self.cache:
            return self.cache[key]
        if frame.event_date.nunique() < self.config["min_base_train_events"]:
            raise ValueError("Insufficient historical events for tuned base models")
        folds = list(temporal_splits(frame, self.config["min_inner_train_events"]))
        if len(folds) < self.config["min_inner_folds"]:
            raise ValueError("Insufficient inner chronological folds")
        for yr, train, valid in folds:
            self.fold_log.append({"layer": "inner", "parent_train_end": str(frame.event_date.max().date()),
                                  "validation_year": yr, "train_end": str(train.event_date.max().date()),
                                  "validation_start": str(valid.event_date.min().date()),
                                  "train_events": train.event_date.nunique(), "validation_events": valid.event_date.nunique()})

        def score(name, groups, params):
            parts = []
            for _, train, valid in folds:
                fitted = fit_one(train, name, groups, params, self.config)
                part = valid[["event_date", "y"]].copy()
                part["pred"] = fitted.predict(valid)
                parts.append(part)
            out = pd.concat(parts, ignore_index=True)
            loss = event_mae(out, out.pred.to_numpy())
            self.selection_log.append({"train_end": str(frame.event_date.max().date()), "model": name,
                                       "groups": list(groups), "params": params, "inner_event_mae": loss,
                                       "inner_events": out.event_date.nunique()})
            return loss

        groups: list[str] = []
        current = score("ols", groups, {})
        for _ in range(self.config["max_forward_groups"]):
            candidates = []
            for candidate in self.config["candidate_groups"]:
                if candidate in groups or frame[candidate].nunique() < 2:
                    continue
                trial = groups + [candidate]
                full = fit_one(frame, "ols", trial, {}, self.config)
                if full.fallback:
                    continue
                candidates.append((score("ols", trial, {}), candidate))
            if not candidates:
                break
            best, candidate = min(candidates)
            if current - best <= self.config["min_forward_improvement"]:
                break
            groups.append(candidate); current = best
        all_groups = self.config["candidate_groups"]
        ridge_scores = [(score("ridge", all_groups, {"lambda": lam}), lam) for lam in self.config["ridge_lambdas"]]
        ridge_lambda = min(ridge_scores)[1]
        xgb_scores = [(score("xgboost", all_groups, params), ix) for ix, params in enumerate(self.config["xgb_grid"])]
        xgb_params = self.config["xgb_grid"][min(xgb_scores)[1]]
        bundle = {"ols": fit_one(frame, "ols", groups, {}, self.config),
                  "ridge": fit_one(frame, "ridge", all_groups, {"lambda": ridge_lambda}, self.config),
                  "xgboost": fit_one(frame, "xgboost", all_groups, xgb_params, self.config)}
        self.cache[key] = bundle
        return bundle

    def build_meta(self, development):
        parts = []
        for year, train, valid in temporal_splits(development, self.config["min_base_train_events"]):
            bundle = self.fit_bundle(train)
            part = valid[["event_date", "year", "station", "hour", "y"]].copy()
            for model in BASE_MODELS:
                part[model] = bundle[model].predict(valid)
            part["base_train_end"] = str(train.event_date.max().date())
            parts.append(part)
            self.fold_log.append({"layer": "middle", "parent_train_end": str(development.event_date.max().date()),
                                  "validation_year": year, "train_end": str(train.event_date.max().date()),
                                  "validation_start": str(valid.event_date.min().date()),
                                  "train_events": train.event_date.nunique(), "validation_events": valid.event_date.nunique()})
        if not parts:
            raise ValueError("No valid temporal OOF rows for stacking")
        return pd.concat(parts, ignore_index=True)

    def backtest(self, frame, progress=print):
        predictions, metadata, meta_frames, fitted_folds = [], [], [], {}
        for year in self.config["outer_years"]:
            train, test = frame.loc[frame.year < year], frame.loc[frame.year == year]
            if test.empty:
                raise ValueError(f"Missing prespecified outer year {year}")
            progress(f"Outer {year}: {train.event_date.nunique()} training events, {test.event_date.nunique()} held-out events")
            meta = self.build_meta(train)
            weights, lp_info = solve_stack(meta[list(BASE_MODELS)], meta.y, meta.event_date)
            bundle = self.fit_bundle(train)
            part = test[["event_date", "year", "station", "hour", "y"]].copy()
            part["baseline"] = test.baseline_hourly_56d.to_numpy(float)
            for model in BASE_MODELS:
                part[model] = bundle[model].predict(test)
            part["equal"] = part[list(BASE_MODELS)].mean(axis=1)
            part["stack"] = part[list(BASE_MODELS)].to_numpy() @ weights
            part["outer_year"] = year
            part["train_end"] = str(train.event_date.max().date())
            predictions.append(part)
            meta["outer_year"] = year; meta_frames.append(meta)
            metadata.append({"outer_year": year, "train_events": train.event_date.nunique(),
                             "test_events": test.event_date.nunique(), "train_end": str(train.event_date.max().date()),
                             "test_start": str(test.event_date.min().date()), "weights": dict(zip(BASE_MODELS, weights)),
                             "lp": lp_info, "base_models": {m: f.info() for m, f in bundle.items()}})
            fitted_folds[year] = bundle
        return pd.concat(predictions, ignore_index=True), metadata, pd.concat(meta_frames, ignore_index=True), fitted_folds


def metrics_table(predictions):
    records = []
    for model in ALL_MODELS:
        df = predictions.copy()
        error = df[model] - df.y
        df["ae"], df["se"], df["err"] = abs(error), error**2, error
        per_event = df.groupby("event_date")[["ae", "se", "err"]].mean()
        peak_under = []
        peak_hour_error = []
        for _, block in df.groupby(["event_date", "station"]):
            block = block.sort_values("hour")
            true_peak = block.loc[block.y.idxmax()]
            pred_peak = block.loc[block[model].idxmax()]
            peak_under.append(max(0., true_peak.y - true_peak[model]))
            peak_hour_error.append(abs(float(true_peak.hour) - float(pred_peak.hour)))
        records.append({"model": model, "events": len(per_event), "rows": len(df),
                        "mae": per_event.ae.mean(), "rmse": np.sqrt(per_event.se.mean()),
                        "bias_pred_minus_actual": per_event.err.mean(),
                        "peak_underprediction_mean": np.mean(peak_under), "peak_hour_mae": np.mean(peak_hour_error)})
    return pd.DataFrame(records).sort_values(["mae", "model"]).reset_index(drop=True)


def vif_table(fitted, train, outer_year):
    """Training design diagnostic; no iid p-values are claimed."""
    x = fitted.encoder.transform(train)
    out = []
    for j, name in enumerate(fitted.encoder.names):
        target = x[:, j]
        others = np.column_stack([np.ones(len(x)), np.delete(x, j, axis=1)])
        residual = target - others @ np.linalg.lstsq(others, target, rcond=None)[0]
        tss, sse = np.sum((target - target.mean())**2), np.sum(residual**2)
        value = np.inf if sse < 1e-12 else tss / sse
        out.append({"outer_year": outer_year, "model": fitted.name, "column": name, "vif": value,
                    "scope": "training_design_only_not_selection_p_value"})
    return pd.DataFrame(out)
