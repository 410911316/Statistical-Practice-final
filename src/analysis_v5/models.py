"""V5: identical approved inputs, training-only selection, and held-out fusion.

All four estimators receive the same encoded columns within a training fold.
Removing an exactly aliased column is a common identifiability restriction, not
outcome-based variable selection. Prior versions remain separately reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor

from analysis_v1.models import Encoder as OriginalEncoder
from analysis_v1.models import event_mae, temporal_splits, vif_table

BASE_MODELS = ("ols", "ridge", "lasso", "xgboost")
FUSION_MODELS = ("equal", "stack", "select_best", "select_stack")
ALL_MODELS = (*BASE_MODELS, *FUSION_MODELS)


class Encoder(OriginalEncoder):
    """One shared, identifiable design learned without holdout features/labels.

The intercept is protected. Station/hour main effects have priority, followed
by their products, approved numerical fields, then missingness indicators.
Only numerical rank dependencies (machine-precision tolerance) are removed;
merely correlated predictors are retained. All estimators use this restriction.
"""

    def fit(self, frame, groups, station_hour_interaction=True,
            include_historical_baseline=False):
        if include_historical_baseline:
            raise ValueError("V5 excludes the publication-unverified historical baseline")
        if not station_hour_interaction:
            raise ValueError("V5 requires a common station-by-hour interaction design")
        if len(set(groups)) != len(groups):
            raise ValueError("Candidate groups must be unique")
        super().fit(frame, tuple(groups), station_hour_interaction=True,
                    include_historical_baseline=False)
        raw = self._raw(frame)
        constants = set(self.dropped_constants)

        def priority(index):
            name = self.all_names[index]
            if name.startswith(("station=", "hour=")):
                return (1 if ":hour=" in name else 0, index)
            return (3 if name.endswith(":missing") else 2, index)

        candidates = sorted((i for i, name in enumerate(self.all_names)
                             if name not in constants), key=priority)
        # Unit-norm columns make the numerical tolerance independent of units.
        current = np.ones((len(frame), 1), dtype=float) / np.sqrt(len(frame))
        self.rank_tolerance = float(10 * np.finfo(float).eps * max(raw.shape[0], raw.shape[1] + 1))
        self.selected_indices, self.dropped_dependencies = [], []
        for index in candidates:
            column = raw[:, index]
            scaled = column / np.linalg.norm(column)
            trial = np.column_stack([current, scaled])
            if np.linalg.matrix_rank(trial, tol=self.rank_tolerance) == trial.shape[1]:
                self.selected_indices.append(index)
                current = trial
            else:
                self.dropped_dependencies.append({
                    "column": self.all_names[index],
                    "reason": "exact_linear_dependency_with_intercept_and_prior_training_columns",
                    "scope": "training_design_only_same_restriction_for_all_four_models",
                })
        self.names = [self.all_names[i] for i in self.selected_indices]
        self.keep = np.array([i in self.selected_indices for i in range(len(self.all_names))])
        self.rank = int(current.shape[1])
        self.column_selection_audit = {
            "priority": ["intercept", "station_and_hour_main_effects", "station_hour_products",
                         "approved_numeric_in_candidate_order", "missing_indicators"],
            "rank_tolerance_on_unit_norm_columns": self.rank_tolerance,
            "dropped_constants": self.dropped_constants,
            "dropped_dependencies": self.dropped_dependencies,
            "uses_y": False,
        }
        return self

    def transform(self, frame):
        for col, cats in self.categories.items():
            if not set(frame[col].unique()) <= set(cats):
                raise ValueError(f"Unseen {col} category; cannot silently forecast a new station/hour")
        x = self._raw(frame)[:, self.selected_indices]
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite prediction features")
        return x


CommonEncoder = Encoder


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
        x = self.encoder.transform(frame)
        if self.name == "ols":
            pred = np.column_stack([np.ones(len(x)), x]) @ self.estimator
        else:
            pred = self.estimator.predict(x)
        pred = np.asarray(pred, dtype=float)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"Nonfinite {self.name} prediction")
        return np.maximum(0, pred)

    def info(self):
        result = {
            "model": self.name, "groups": list(self.encoder.groups), "columns": self.encoder.names,
            "dropped_constants": self.encoder.dropped_constants,
            "dropped_dependencies": self.encoder.dropped_dependencies,
            "column_selection": self.encoder.column_selection_audit,
            "params": self.params, "rank": self.rank, "n_columns_with_intercept": self.columns,
            "fallback": None, "includes_historical_baseline": False,
            "common_station_hour_interactions": True,
            "imputation": self.encoder.imputation_audit,
            "all_training_values_unknown": self.encoder.all_missing_numeric,
            "missing_indicator_candidates": self.encoder.missing_indicator_numeric,
        }
        if self.name in ("ols", "ridge", "lasso"):
            coef = self.estimator[1:] if self.name == "ols" else self.estimator.coef_
            intercept = self.estimator[0] if self.name == "ols" else self.estimator.intercept_
            result["coefficients"] = {name: float(value) for name, value in zip(self.encoder.names, coef)}
            result["intercept"] = float(intercept)
            result["coefficient_scale"] = "continuous_training_standardized_dummy_and_binary_0_1"
        if self.name == "lasso":
            result["lasso_zero_columns"] = [name for name, value in result["coefficients"].items() if value == 0.]
            result["lasso_nonzero_columns"] = [name for name, value in result["coefficients"].items() if value != 0.]
            result["n_iter"] = int(self.estimator.n_iter_)
            result["dual_gap"] = float(self.estimator.dual_gap_)
            result["optimization_certificate"] = self.estimator.optimization_certificate_
        return result


def certify_lasso(estimator, x, y, alpha, tol):
    """Check the original L1 objective via feasible dual bound and KKT residuals.

    The intercept is unpenalized, so the dual vector must have zero sum. This
    certificate is independent of solver stopping status and rejects a short
    or numerically failed path instead of silently scoring its coefficients.
    """
    coefficient = np.asarray(estimator.coef_, dtype=float)
    residual = y - estimator.predict(x)
    if not np.isfinite(coefficient).all() or not np.isfinite(residual).all():
        raise RuntimeError("Lasso returned nonfinite coefficients or residuals")
    n = len(y)
    gradient = x.T @ residual / n
    active = coefficient != 0.
    active_error = float(np.max(abs(gradient[active] - alpha * np.sign(coefficient[active])))) if active.any() else 0.
    inactive_excess = float(np.max(np.maximum(0., abs(gradient[~active]) - alpha))) if (~active).any() else 0.
    intercept_error = abs(float(residual.mean()))
    theta = (residual - residual.mean()) / n
    dual_norm = float(np.max(abs(x.T @ theta))) if x.shape[1] else 0.
    # A slight inward margin avoids roundoff placing the dual just outside its
    # feasible set. This still gives a valid, conservative primal-dual gap.
    if dual_norm:
        theta *= min(1., alpha / dual_norm) * (1. - 10 * np.finfo(float).eps)
    primal = float(residual @ residual / (2 * n) + alpha * np.sum(abs(coefficient)))
    dual = float(y @ theta - n * (theta @ theta) / 2)
    gap = primal - dual
    gap_limit = float(tol * max(1., np.var(y)))
    kkt_limit = float(tol * max(1., alpha))
    if (not np.isfinite(gap) or gap < -gap_limit or gap > gap_limit
            or max(active_error, inactive_excess, intercept_error) > kkt_limit):
        raise RuntimeError(f"Lasso optimality certificate failed: alpha={alpha}, gap={gap:.6g}, "
                           f"gap_limit={gap_limit:.6g}, KKT={max(active_error, inactive_excess, intercept_error):.6g}")
    solver_reported_gap = float(estimator.dual_gap_)
    estimator.dual_gap_ = max(0., gap)
    estimator.optimization_certificate_ = {
        "solver": "sklearn.linear_model.Lasso", "certificate_method": "independently_computed_original_objective_KKT_and_feasible_dual",
        "primal_objective": primal,
        "feasible_dual_objective": dual, "primal_dual_gap": max(0., gap),
        "gap_tolerance": gap_limit, "active_kkt_error": active_error,
        "inactive_kkt_excess": inactive_excess, "intercept_kkt_error": intercept_error,
        "kkt_tolerance": kkt_limit, "dual_constraint_max_abs": float(np.max(abs(x.T @ theta))),
        "dual_zero_sum_error": abs(float(theta.sum())), "solver_reported_dual_gap": solver_reported_gap,
        "verified": True,
    }


def fit_one(frame, name, groups, params, config, encoder=None):
    if config.get("include_historical_baseline", False):
        raise ValueError("V5 requires include_historical_baseline=false")
    if name not in BASE_MODELS:
        raise ValueError(f"Unknown base model {name}")
    encoder = Encoder().fit(frame, tuple(groups)) if encoder is None else encoder
    if tuple(groups) != encoder.groups:
        raise ValueError("Shared encoder groups do not match model groups")
    x = encoder.transform(frame)
    design = np.column_stack([np.ones(len(x)), x])
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise RuntimeError("Common training design remains rank deficient")
    y = frame.y.to_numpy(float)
    if not len(y) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Training labels must be finite nonnegative person counts")
    params = dict(params)
    if name == "ols":
        estimator = np.linalg.lstsq(design, y, rcond=None)[0]
    elif name == "ridge":
        lam = float(params["lambda"])
        if not np.isfinite(lam) or lam < 0:
            raise ValueError("Ridge lambda must be finite and nonnegative")
        alpha = len(frame) * lam
        estimator = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(x, y)
        params.update(sklearn_alpha=alpha, n_train_rows=len(frame))
    elif name == "lasso":
        alpha = float(params["alpha"])
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Lasso alpha must be finite and strictly positive")
        # Weak penalties can need >100,000 iterations on the small, correlated
        # early-weather design. Use the same precise solver for every fit and
        # certify the original objective independently before any scoring.
        estimator = Lasso(alpha=alpha, fit_intercept=True, selection="cyclic",
                          max_iter=config.get("lasso_max_iter", 1000000),
                          tol=config.get("lasso_tol", 1e-10))
        tol = float(config.get("lasso_certificate_tol", 1e-6))
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("Lasso numerical tolerance must be finite and positive")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                estimator.fit(x, y)
            except ConvergenceWarning as error:
                raise RuntimeError(f"Lasso did not converge for alpha={alpha}; results are not accepted") from error
        certify_lasso(estimator, x, y, alpha, tol)
        params.update(n_train_rows=len(frame), objective="SSE/(2*n)+alpha*L1",
                      solver="sklearn.linear_model.Lasso", max_iter=estimator.max_iter,
                      solver_tolerance=estimator.tol, optimality_tolerance=tol)
    else:
        xgb_params = {"objective": "reg:squarederror", "tree_method": "hist", "n_jobs": 1,
                      "random_state": config["seed"], **config["xgb_fixed"], **params}
        estimator = XGBRegressor(**xgb_params).fit(x, y)
    return Fitted(name, encoder, estimator, params, rank, design.shape[1])


def solve_stack(predictions, truth, event_dates):
    """Equal-event MAE simplex LP for any positive number of base models."""
    p, y = np.asarray(predictions, dtype=float), np.asarray(truth, dtype=float)
    if (p.ndim != 2 or y.ndim != 1 or not len(y) or p.shape[0] != len(y)
            or p.shape[1] < 1 or not np.isfinite(p).all() or not np.isfinite(y).all()):
        raise ValueError("Stack needs finite common OOF rows and at least one model")
    if (p < 0).any() or (y < 0).any():
        raise ValueError("Stack inputs must be nonnegative person counts")
    dates = pd.Series(np.asarray(event_dates))
    if len(dates) != len(y) or dates.isna().any():
        raise ValueError("Stack event dates must match prediction rows and contain no missing values")
    counts = dates.value_counts()
    a = 1 / (dates.map(counts).to_numpy(float) * len(counts))
    n, k = p.shape
    inequalities = vstack([hstack([csr_matrix(p), -eye(n)]),
                           hstack([-csr_matrix(p), -eye(n)])], format="csr")
    equality = csr_matrix((np.ones(k), (np.zeros(k, dtype=int), np.arange(k))), shape=(1, n + k))
    answer = linprog(np.r_[np.zeros(k), a], A_ub=inequalities, b_ub=np.r_[y, -y],
                     A_eq=equality, b_eq=[1.], bounds=[(0, None)] * (n + k), method="highs")
    if not answer.success:
        raise RuntimeError(f"Stack LP did not converge: {answer.message}")
    weights = answer.x[:k]
    if weights.min() < -1e-8 or abs(weights.sum() - 1) > 1e-8:
        raise RuntimeError("Invalid simplex solution")
    weights = np.maximum(weights, 0)
    weights /= weights.sum()
    loss = float(np.dot(a, np.abs(y - p @ weights)))
    if loss > min(float(np.dot(a, abs(y - p[:, m]))) for m in range(k)) + 1e-5:
        raise RuntimeError("Stack meta loss exceeds best feasible vertex")
    return weights, {"oof_mae": loss, "oof_events": len(counts), "n_models": k,
                     "solver": "scipy.optimize.linprog(highs)",
                     "scope": "meta_training_loss_not_outer_test_score"}


def fit_fusion(meta, top_k=2):
    """Select/train exclusively on chronological OOF rows inside development.

The same OOF rows select members and fit their weights. This is meta-training,
not an honest estimate of ensemble performance: only a later outer year scores
the complete selection-plus-fusion procedure. Top-k is prespecified, not tuned.
"""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(BASE_MODELS):
        raise ValueError("fusion_top_k must be a prespecified integer between 1 and 4")
    required = {"event_date", "y", *BASE_MODELS}
    if not required <= set(meta.columns):
        raise ValueError(f"Fusion meta data missing columns: {sorted(required - set(meta.columns))}")
    # The full stack validates finite/nonnegative values and aligned dates first.
    full_weights, full_info = solve_stack(meta[list(BASE_MODELS)], meta.y, meta.event_date)
    scores = {name: event_mae(meta, meta[name].to_numpy(float)) for name in BASE_MODELS}
    ranked = sorted(BASE_MODELS, key=lambda name: (scores[name], BASE_MODELS.index(name)))
    selected = ranked[:top_k]
    selected_weights, selected_info = solve_stack(meta[selected], meta.y, meta.event_date)
    weights = {
        "equal": {name: 1 / len(BASE_MODELS) for name in BASE_MODELS},
        "stack": {name: float(value) for name, value in zip(BASE_MODELS, full_weights)},
        "select_best": {name: float(name == ranked[0]) for name in BASE_MODELS},
        "select_stack": {name: 0. for name in BASE_MODELS},
    }
    weights["select_stack"].update({name: float(value) for name, value in zip(selected, selected_weights)})
    audit = {
        "oof_event_mae": scores, "ranked_models": ranked, "selected_models": selected,
        "selected_best_model": ranked[0], "top_k": top_k,
        "tie_break": "BASE_MODELS_order_ols_ridge_lasso_xgboost",
        "lp": {"stack": full_info, "select_stack": selected_info},
        "oof_events": int(meta.event_date.nunique()), "oof_rows": int(len(meta)),
        "oof_start": str(pd.to_datetime(meta.event_date).min().date()),
        "oof_end": str(pd.to_datetime(meta.event_date).max().date()),
        "scope": "meta_training_selection_and_weight_fit_only_not_outer_test_score",
    }
    return weights, audit


class Trainer:
    def __init__(self, config):
        if config.get("include_historical_baseline", False):
            raise ValueError("V5 requires include_historical_baseline=false")
        self.config, self.cache = config, {}
        self.selection_log, self.fold_log = [], []

    def fit_bundle(self, frame):
        groups = tuple(self.config["candidate_groups"])
        columns = ["event_date", "year", "station", "hour", "y", *groups]
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
        encoders = {year: Encoder().fit(train, groups) for year, train, _ in folds}
        for year, train, valid in folds:
            self.fold_log.append({
                "layer": "inner", "parent_train_end": str(frame.event_date.max().date()),
                "validation_year": year, "train_end": str(train.event_date.max().date()),
                "validation_start": str(valid.event_date.min().date()),
                "train_events": int(train.event_date.nunique()), "validation_events": int(valid.event_date.nunique()),
            })

        def score(name, params):
            parts, numerical_diagnostics = [], []
            for year, train, valid in folds:
                fitted = fit_one(train, name, groups, params, self.config, encoder=encoders[year])
                if name == "lasso":
                    certificate = fitted.estimator.optimization_certificate_
                    numerical_diagnostics.append({
                        "n_iter": fitted.estimator.n_iter_, "gap": certificate["primal_dual_gap"],
                        "gap_ratio": certificate["primal_dual_gap"] / certificate["gap_tolerance"],
                        "kkt_ratio": max(certificate["active_kkt_error"], certificate["inactive_kkt_excess"],
                                         certificate["intercept_kkt_error"]) / certificate["kkt_tolerance"],
                    })
                part = valid[["event_date", "y"]].copy()
                part["pred"] = fitted.predict(valid)
                parts.append(part)
            out = pd.concat(parts, ignore_index=True)
            loss = event_mae(out, out.pred.to_numpy())
            selection_record = {
                "train_end": str(frame.event_date.max().date()), "model": name,
                "groups": list(groups), "params": params, "inner_event_mae": loss,
                "inner_events": int(out.event_date.nunique()),
                "scope": "inner_training_selection_only_not_outer_test_score",
            }
            if numerical_diagnostics:
                selection_record.update(
                    solver="sklearn.linear_model.Lasso", numerical_fits=len(numerical_diagnostics),
                    numerical_all_verified=True, max_n_iter=int(max(item["n_iter"] for item in numerical_diagnostics)),
                    max_primal_dual_gap=float(max(item["gap"] for item in numerical_diagnostics)),
                    max_gap_to_tolerance=float(max(item["gap_ratio"] for item in numerical_diagnostics)),
                    max_kkt_to_tolerance=float(max(item["kkt_ratio"] for item in numerical_diagnostics)),
                )
            self.selection_log.append(selection_record)
            return loss

        score("ols", {})
        grids = {
            "ridge": [{"lambda": value} for value in self.config["ridge_lambdas"]],
            "lasso": [{"alpha": value} for value in self.config.get("lasso_alphas", [.1, 1, 5, 10, 25, 50, 100, 250])],
            "xgboost": self.config["xgb_grid"],
        }
        selected = {"ols": {}}
        for model, grid in grids.items():
            if not grid:
                raise ValueError(f"Empty {model} hyperparameter grid")
            losses = [score(model, params) for params in grid]
            selected[model] = grid[min(range(len(grid)), key=lambda index: (losses[index], index))]
        common = Encoder().fit(frame, groups)
        bundle = {model: fit_one(frame, model, groups, selected[model], self.config, encoder=common)
                  for model in BASE_MODELS}
        if len({tuple(fitted.encoder.names) for fitted in bundle.values()}) != 1:
            raise RuntimeError("Base models do not have an identical training design")
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
            self.fold_log.append({
                "layer": "middle", "parent_train_end": str(development.event_date.max().date()),
                "validation_year": year, "train_end": str(train.event_date.max().date()),
                "validation_start": str(valid.event_date.min().date()),
                "train_events": int(train.event_date.nunique()), "validation_events": int(valid.event_date.nunique()),
            })
        if not parts:
            raise ValueError("No valid temporal OOF rows for selection/fusion")
        return pd.concat(parts, ignore_index=True)

    def backtest(self, frame, progress=print):
        predictions, metadata, meta_frames, fitted_folds = [], [], [], {}
        for year in self.config["outer_years"]:
            train, test = frame.loc[frame.year < year], frame.loc[frame.year == year]
            if test.empty:
                raise ValueError(f"Missing prespecified outer year {year}")
            progress(f"Outer {year}: {train.event_date.nunique()} training events, {test.event_date.nunique()} held-out events")
            meta = self.build_meta(train)
            weights, fusion_audit = fit_fusion(meta, top_k=self.config.get("fusion_top_k", 2))
            bundle = self.fit_bundle(train)
            part = test[["event_date", "year", "station", "hour", "y"]].copy()
            for model in BASE_MODELS:
                part[model] = bundle[model].predict(test)
            for method in FUSION_MODELS:
                part[method] = part[list(BASE_MODELS)].to_numpy() @ np.array([weights[method][model] for model in BASE_MODELS])
            part["outer_year"] = year
            part["train_end"] = str(train.event_date.max().date())
            predictions.append(part)
            meta["outer_year"] = year
            meta_frames.append(meta)
            metadata.append({
                "outer_year": year, "train_events": int(train.event_date.nunique()),
                "test_events": int(test.event_date.nunique()), "train_end": str(train.event_date.max().date()),
                "test_start": str(test.event_date.min().date()), "weights": weights["stack"],
                "fusion_weights": weights, "fusion_audit": fusion_audit, "lp": fusion_audit["lp"]["stack"],
                "base_models": {model: fitted.info() for model, fitted in bundle.items()},
            })
            fitted_folds[year] = bundle
        return (pd.concat(predictions, ignore_index=True), metadata,
                pd.concat(meta_frames, ignore_index=True), fitted_folds)


def metrics_table(predictions):
    records = []
    for model in ALL_MODELS:
        frame = predictions.copy()
        error = frame[model] - frame.y
        frame["ae"], frame["se"], frame["err"] = abs(error), error**2, error
        per_event = frame.groupby("event_date")[["ae", "se", "err"]].mean()
        peak_under, peak_signed, peak_absolute, peak_hour_error = [], [], [], []
        for _, block in frame.groupby(["event_date", "station"]):
            block = block.sort_values("hour")
            true_peak = block.loc[block.y.idxmax()]
            pred_peak = block.loc[block[model].idxmax()]
            signed = float(true_peak.y - true_peak[model])
            peak_under.append(max(0., signed))
            peak_signed.append(signed)
            peak_absolute.append(abs(signed))
            peak_hour_error.append(abs(float(true_peak.hour) - float(pred_peak.hour)))
        records.append({
            "model": model, "events": len(per_event), "rows": len(frame), "mae": float(per_event.ae.mean()),
            "rmse": float(np.sqrt(per_event.se.mean())), "bias_pred_minus_actual": float(per_event.err.mean()),
            "peak_underprediction_mean": float(np.mean(peak_under)),
            "peak_signed_underprediction_mean": float(np.mean(peak_signed)),
            "peak_absolute_error_mean": float(np.mean(peak_absolute)),
            "peak_hour_mae": float(np.mean(peak_hour_error)),
        })
    return pd.DataFrame(records).sort_values(["mae", "model"]).reset_index(drop=True)
