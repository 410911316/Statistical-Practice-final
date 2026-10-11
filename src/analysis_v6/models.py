"""A conservative additive spline model and the prespecified R/G/X fusion.

V5 base-model fitting and four old ensembles are reused without alteration.
The new GAM uses the same training-only raw design, replacing only supported
continuous columns with cubic P-splines. Its Gaussian loss has a second-order
coefficient-difference penalty; it is not an exact integrated-curvature penalty.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from pygam import LinearGAM, l, s

from analysis_v5 import models as v5
from analysis_v5.models import Encoder, event_mae, solve_stack, temporal_splits, vif_table

OLD_BASE_MODELS = v5.BASE_MODELS
OLD_ALL_MODELS = v5.ALL_MODELS
BASE_MODELS = (*OLD_BASE_MODELS, "gam")
GAM_FUSION_BASE_MODELS = ("ridge", "gam", "xgboost")
FUSION_MODELS = (*v5.FUSION_MODELS, "gam_equal", "gam_stack")
ALL_MODELS = (*OLD_ALL_MODELS, "gam", "gam_equal", "gam_stack")
SMOOTH_CANDIDATES = frozenset({
    "day_of_year", "planned_fireworks_minutes", "planned_fireworks_start_hour",
    "forecast_temperature_c", "forecast_wind_speed_m_s", "forecast_cloud_cover_pct",
})
CommonEncoder = Encoder


def make_terms(frame, encoder, lam, config):
    """Learn term kinds, support gates and spline boundary knots on train only."""
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("GAM lambda must be finite and positive")
    n_splines = int(config.get("gam_n_splines", 5))
    degree = int(config.get("gam_spline_order", 3))
    min_events = int(config.get("gam_min_known_events", 6))
    min_distinct = int(config.get("gam_min_unique", 8))
    if (n_splines, degree, min_events, min_distinct) != (5, 3, 6, 8):
        raise ValueError("V6 fixes GAM support/complexity before outer evaluation: 5/3/6/8")
    candidates = frozenset(config.get("gam_smooth_candidates", SMOOTH_CANDIDATES))
    if candidates != SMOOTH_CANDIDATES:
        raise ValueError("V6 GAM smooth candidate set is prespecified before outer evaluation")
    x = encoder.transform(frame)
    terms, audit = None, []
    for index, name in enumerate(encoder.names):
        lo, hi = float(x[:, index].min()), float(x[:, index].max())
        known_events, distinct, raw_lo, raw_hi = None, None, None, None
        smooth = False
        reason = "structural_binary_missing_or_year_fixed_linear"
        if name in candidates:
            values = pd.to_numeric(frame[name], errors="raise")
            known = values.notna()
            known_events = int(frame.loc[known, "event_date"].nunique())
            distinct = int(values[known].nunique())
            raw_lo = float(values[known].min()) if known.any() else None
            raw_hi = float(values[known].max()) if known.any() else None
            smooth = known_events >= min_events and distinct >= min_distinct
            reason = ("training_support_passed" if smooth else
                      "insufficient_training_known_events_or_distinct_values_keep_linear")
        if smooth:
            term = s(index, n_splines=n_splines, spline_order=degree,
                     lam=lam, penalties="derivative", basis="ps", edge_knots=[lo, hi])
        else:
            term = l(index, lam=0., penalties=None)
        terms = term if terms is None else terms + term
        audit.append({
            "column": name, "feature_index": index, "kind": "smooth" if smooth else "linear",
            "reason": reason, "known_training_events": known_events,
            "distinct_known_training_values": distinct, "training_min_raw": raw_lo,
            "training_max_raw": raw_hi, "training_min_encoded": lo,
            "training_max_encoded": hi, "n_splines": n_splines if smooth else None,
            "spline_order": degree if smooth else None, "lambda": lam if smooth else 0.,
            "penalty": "second_order_coefficient_difference" if smooth else "none",
            "edge_knots_encoded": [lo, hi] if smooth else None,
            "uses_y_for_term_or_knot_selection": False,
        })
    if terms is None:
        raise ValueError("GAM requires at least one nonconstant approved design column")
    return terms, audit


def certify_gam(estimator, x, y, tol=1e-6):
    """Independently check penalized Gaussian normal equations and solver stop."""
    coefficients = np.asarray(estimator.coef_, dtype=float)
    pred = np.asarray(estimator.predict(x), dtype=float)
    if not np.isfinite(coefficients).all() or not np.isfinite(pred).all():
        raise RuntimeError("GAM returned nonfinite fitted coefficients/predictions")
    diffs = estimator.logs_.get("diffs", [])
    if not diffs or not np.isfinite(diffs[-1]) or diffs[-1] >= estimator.tol:
        raise RuntimeError("GAM did not converge; candidate is not scored")
    matrix = estimator._modelmat(x)
    penalty = estimator._P()
    residual = pred - y
    gradient = np.asarray(matrix.T @ residual + penalty @ coefficients).ravel()
    reference = max(1., float(np.max(np.abs(np.asarray(matrix.T @ y).ravel()))))
    relative = float(np.max(np.abs(gradient))) / reference
    edof = float(estimator.statistics_["edof"])
    if not np.isfinite(edof) or not 0 < edof <= min(len(y), len(coefficients)) + 1e-6:
        raise RuntimeError("GAM effective degrees of freedom are invalid")
    if not np.isfinite(relative) or relative > tol:
        raise RuntimeError(f"GAM normal-equation certificate failed: relative residual={relative:.6g}")
    return {
        "verified": True, "solver": "pygam.LinearGAM penalized PIRLS/SVD",
        "iterations": len(diffs), "last_relative_coefficient_change": float(diffs[-1]),
        "stopping_tolerance": float(estimator.tol), "max_absolute_normal_equation_residual": float(np.max(abs(gradient))),
        "relative_normal_equation_residual": relative, "certificate_tolerance": tol,
        "sum_squared_error": float(residual @ residual),
        "coefficient_difference_penalty": float(coefficients @ (penalty @ coefficients)),
        "effective_degrees_of_freedom": edof,
        "objective": "sum_squared_error + sum(lambda * squared_second_coefficient_differences)",
        "numerical_note": "pyGAM adds a machine-scale diagonal for numerical stability; checked against stated objective",
    }


@dataclass
class GAMFitted(v5.Fitted):
    terms_audit: list | None = None
    certificate: dict | None = None

    def info(self):
        result = super().info()
        result.update(
            family="Gaussian", link="identity", clipping="same_nonnegative_prediction_clipping_as_v5",
            terms=self.terms_audit, smoothing_penalty="second_order_P_spline_coefficient_differences_not_exact_integrated_curvature",
            smooth_columns=[item["column"] for item in self.terms_audit if item["kind"] == "smooth"],
            linear_columns=[item["column"] for item in self.terms_audit if item["kind"] == "linear"],
            no_eligible_smooth_terms=not any(item["kind"] == "smooth" for item in self.terms_audit),
            n_basis_coefficients=int(len(self.estimator.coef_)),
            numerical_certificate=self.certificate,
            effective_degrees_of_freedom=float(self.estimator.statistics_["edof"]),
            term_effective_degrees_of_freedom=[
                {"column": item["column"], "edf": float(np.sum(
                    self.estimator.statistics_["edof_per_coef"][self.estimator.terms.get_coef_indices(i)]))}
                for i, item in enumerate(self.terms_audit)
            ],
            spline_boundary_behavior="pyGAM linear extrapolation outside training edge knots",
            curve_interpretation="conditional predictive component, not causal effect; no spline p-value inference",
        )
        return result

    def extrapolation_audit(self, frame):
        """Evaluation features are inspected only after training for reporting."""
        x = self.encoder.transform(frame)
        records = []
        for item in self.terms_audit:
            if item["kind"] != "smooth":
                continue
            values = x[:, item["feature_index"]]
            below = values < item["training_min_encoded"]
            above = values > item["training_max_encoded"]
            records.append({"column": item["column"], "rows": len(frame),
                            "below_training_range_rows": int(below.sum()),
                            "above_training_range_rows": int(above.sum()),
                            "outside_training_range_events": int(frame.loc[below | above, "event_date"].nunique()),
                            "behavior": "linear_extrapolation_not_clipping_or_refitting_knots"})
        return records

    def effect_curves(self, train, points=80):
        """Training-range component curves centered at empirical train effects.

        Centering defines a readable reference, not a causal contrast or a
        unique partition of contributions among correlated predictors.
        """
        x = self.encoder.transform(train)
        records = []
        for term_index, item in enumerate(self.terms_audit):
            if item["kind"] != "smooth":
                continue
            index, name = item["feature_index"], item["column"]
            grid = np.linspace(item["training_min_encoded"], item["training_max_encoded"], points)
            evaluation = np.tile(np.median(x, axis=0), (points, 1))
            evaluation[:, index] = grid
            mean_effect = float(np.mean(self.estimator.partial_dependence(term=term_index, X=x)))
            effects = np.asarray(self.estimator.partial_dependence(term=term_index, X=evaluation)) - mean_effect
            raw_grid = grid * float(self.encoder.scale[name]) + float(self.encoder.mean[name])
            records.extend({"column": name, "x_raw": float(value),
                            "centered_partial_effect": float(effect),
                            "training_mean_partial_effect": mean_effect,
                            "reference": "training_empirical_mean_component",
                            "scope": "training_range_predictive_component_not_causal_no_CI"}
                           for value, effect in zip(raw_grid, effects))
        return pd.DataFrame(records, columns=["column", "x_raw", "centered_partial_effect",
                                             "training_mean_partial_effect", "reference", "scope"])


def fit_one(frame, name, groups, params, config, encoder=None):
    if name != "gam":
        return v5.fit_one(frame, name, groups, params, config, encoder=encoder)
    if config.get("include_historical_baseline", False):
        raise ValueError("V6 excludes the publication-unverified baseline")
    encoder = Encoder().fit(frame, tuple(groups)) if encoder is None else encoder
    if tuple(groups) != encoder.groups:
        raise ValueError("Shared encoder groups do not match GAM groups")
    x, y = encoder.transform(frame), frame.y.to_numpy(float)
    if not len(y) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("GAM training labels must be finite nonnegative person counts")
    lam = float(params["lambda"])
    terms, audit = make_terms(frame, encoder, lam, config)
    estimator = LinearGAM(terms, fit_intercept=True,
                          max_iter=int(config.get("gam_max_iter", 1000)),
                          tol=float(config.get("gam_tol", 1e-6))).fit(x, y)
    certificate = certify_gam(estimator, x, y)
    design = np.column_stack([np.ones(len(x)), x])
    return GAMFitted("gam", encoder, estimator,
                     {**params, "n_splines": 5, "spline_order": 3,
                      "lambda_scale": "unnormalized_sum_SSE_no_n_multiplier",
                      "max_iter": estimator.max_iter, "tol": estimator.tol},
                     int(np.linalg.matrix_rank(design)), design.shape[1],
                     terms_audit=audit, certificate=certificate)


def fit_fusion(meta, top_k=2):
    """Retain four legacy ensembles and add the prespecified R/G/X triplet."""
    weights, audit = v5.fit_fusion(meta, top_k=top_k)
    if not set(GAM_FUSION_BASE_MODELS) <= set(meta.columns):
        raise ValueError("GAM fusion needs aligned Ridge/GAM/XGBoost chronological OOF rows")
    fitted, info = solve_stack(meta[list(GAM_FUSION_BASE_MODELS)], meta.y, meta.event_date)
    weights.update(
        gam_equal={model: 1 / 3 for model in GAM_FUSION_BASE_MODELS},
        gam_stack={model: float(value) for model, value in zip(GAM_FUSION_BASE_MODELS, fitted)},
    )
    audit["gam_triplet"] = {
        "prespecified_models": list(GAM_FUSION_BASE_MODELS), "selection_by_outer_results": False,
        "oof_event_mae": {model: event_mae(meta, meta[model].to_numpy(float)) for model in GAM_FUSION_BASE_MODELS},
        "lp": info, "weight_constraints": "nonnegative_sum_to_one_one_global_weight_vector",
        "scope": "meta_training_weight_fit_not_outer_test_score",
    }
    return weights, audit


class Trainer:
    """Composition keeps the old estimators and old fusion numerically intact."""
    def __init__(self, config):
        self.config = config
        self.legacy = v5.Trainer(config)
        self.gam_cache, self.gam_selection_log, self.gam_fold_log = {}, [], []

    @property
    def selection_log(self):
        return [*self.legacy.selection_log, *self.gam_selection_log]

    @property
    def fold_log(self):
        return [*self.legacy.fold_log, *self.gam_fold_log]

    def fit_bundle(self, frame):
        old = self.legacy.fit_bundle(frame)
        groups = tuple(self.config["candidate_groups"])
        ordered = frame[["event_date", "year", "station", "hour", "y", *groups]].sort_values(
            ["event_date", "station", "hour"])
        digest = hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes())
        digest.update(json.dumps(self.config, sort_keys=True).encode())
        key = digest.hexdigest()
        if key in self.gam_cache:
            return {**old, "gam": self.gam_cache[key]}
        folds = list(temporal_splits(frame, self.config["min_inner_train_events"]))
        encoders = {year: Encoder().fit(train, groups) for year, train, _ in folds}
        lambdas = list(self.config.get("gam_lambdas", [.1, 1., 10., 100., 1000.]))
        if not lambdas or any(not np.isfinite(value) or value <= 0 for value in lambdas):
            raise ValueError("GAM lambda grid must contain finite positive values")
        for year, train, valid in folds:
            self.gam_fold_log.append({
                "layer": "inner_gam", "parent_train_end": str(frame.event_date.max().date()),
                "validation_year": year, "train_end": str(train.event_date.max().date()),
                "validation_start": str(valid.event_date.min().date()),
                "train_events": int(train.event_date.nunique()), "validation_events": int(valid.event_date.nunique()),
            })
        losses = []
        for lam in lambdas:
            pieces, fits = [], []
            for year, train, valid in folds:
                fitted = fit_one(train, "gam", groups, {"lambda": lam}, self.config, encoder=encoders[year])
                piece = valid[["event_date", "y"]].copy()
                piece["pred"] = fitted.predict(valid)
                pieces.append(piece)
                fits.append(fitted)
            out = pd.concat(pieces, ignore_index=True)
            loss = event_mae(out, out.pred.to_numpy())
            losses.append(loss)
            self.gam_selection_log.append({
                "train_end": str(frame.event_date.max().date()), "model": "gam", "groups": list(groups),
                "params": {"lambda": lam}, "inner_event_mae": loss,
                "inner_events": int(out.event_date.nunique()), "numerical_all_verified": True,
                "max_normal_equation_relative_residual": max(fit.certificate["relative_normal_equation_residual"] for fit in fits),
                "scope": "inner_training_selection_only_not_outer_test_score",
                "selection_metric": "chronological_event_balanced_MAE_not_GCV",
            })
        selected = lambdas[min(range(len(lambdas)), key=lambda i: (losses[i], i))]
        fitted = fit_one(frame, "gam", groups, {"lambda": selected}, self.config,
                         encoder=old["ridge"].encoder)
        self.gam_cache[key] = fitted
        return {**old, "gam": fitted}

    def build_meta(self, development):
        pieces = []
        for year, train, valid in temporal_splits(development, self.config["min_base_train_events"]):
            bundle = self.fit_bundle(train)
            piece = valid[["event_date", "year", "station", "hour", "y"]].copy()
            for model in BASE_MODELS:
                piece[model] = bundle[model].predict(valid)
            piece["base_train_end"] = str(train.event_date.max().date())
            pieces.append(piece)
            self.gam_fold_log.append({
                "layer": "middle", "parent_train_end": str(development.event_date.max().date()),
                "validation_year": year, "train_end": str(train.event_date.max().date()),
                "validation_start": str(valid.event_date.min().date()),
                "train_events": int(train.event_date.nunique()), "validation_events": int(valid.event_date.nunique()),
            })
        if not pieces:
            raise ValueError("No chronological OOF rows for GAM fusion")
        return pd.concat(pieces, ignore_index=True)

    def backtest(self, frame, progress=print):
        predictions, metadata, meta_frames, fitted_folds = [], [], [], {}
        for year in self.config["outer_years"]:
            train, test = frame.loc[frame.year < year], frame.loc[frame.year == year]
            if test.empty:
                raise ValueError(f"Missing prespecified outer year {year}")
            progress(f"Outer {year}: {train.event_date.nunique()} training events, {test.event_date.nunique()} held-out events")
            meta = self.build_meta(train)
            weights, audit = fit_fusion(meta, top_k=self.config.get("fusion_top_k", 2))
            bundle = self.fit_bundle(train)
            part = test[["event_date", "year", "station", "hour", "y"]].copy()
            for model in BASE_MODELS:
                part[model] = bundle[model].predict(test)
            for method in FUSION_MODELS:
                names = list(weights[method])
                part[method] = part[names].to_numpy() @ np.array([weights[method][name] for name in names])
            part["outer_year"] = year
            part["train_end"] = str(train.event_date.max().date())
            predictions.append(part)
            meta["outer_year"] = year
            meta_frames.append(meta)
            metadata.append({
                "outer_year": year, "train_events": int(train.event_date.nunique()),
                "test_events": int(test.event_date.nunique()), "train_end": str(train.event_date.max().date()),
                "test_start": str(test.event_date.min().date()), "weights": weights["stack"],
                "fusion_weights": weights, "fusion_audit": audit, "lp": audit["lp"]["stack"],
                "gam_extrapolation": bundle["gam"].extrapolation_audit(test),
                "base_models": {model: fitted.info() for model, fitted in bundle.items()},
            })
            fitted_folds[year] = bundle
        return (pd.concat(predictions, ignore_index=True), metadata,
                pd.concat(meta_frames, ignore_index=True), fitted_folds)


def metrics_table(predictions):
    """Use the exact old metric implementation for both old and new methods."""
    required = set(ALL_MODELS)
    if not required <= set(predictions):
        raise ValueError(f"Missing model predictions: {sorted(required - set(predictions))}")
    legacy = v5.metrics_table(predictions)
    new_rows = []
    for model in ("gam", "gam_equal", "gam_stack"):
        copy = predictions.copy()
        copy["ols"] = copy[model]
        row = v5.metrics_table(copy).loc[lambda d: d.model.eq("ols")].copy()
        row["model"] = model
        new_rows.append(row)
    return pd.concat([legacy, *new_rows], ignore_index=True).sort_values(["mae", "model"]).reset_index(drop=True)
