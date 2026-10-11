"""V7: train-only event support and full-coefficient ridge stabilize GAM.

Legacy eight estimators/ensembles are delegated without alteration. GAM uses
an explicitly audited subset of the same D-1 candidate information. Numerical
variables supported by fewer than six distinct known training events are
omitted, not estimated as an unpenalized linear fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from pygam import LinearGAM, l, s

from analysis_v5 import models as v5
from analysis_v6 import models as v6
from analysis_v5.models import event_mae, solve_stack, temporal_splits, vif_table
from analysis_v6.models import (OLD_BASE_MODELS, OLD_ALL_MODELS, BASE_MODELS,
    GAM_FUSION_BASE_MODELS, FUSION_MODELS, ALL_MODELS, SMOOTH_CANDIDATES,
    fit_fusion, metrics_table)


class Encoder(v5.Encoder):
    """GAM's support gate precedes median/scaling/rank fitting on train only."""
    def fit(self, frame, groups, config=None):
        config = config or {}
        if int(config.get("gam_min_known_events", 6)) != 6:
            raise ValueError("V7 fixes a minimum of six known training events")
        binary = tuple(config.get("gam_binary_groups", ["is_weekend"]))
        if binary != ("is_weekend",):
            raise ValueError("V7 prespecifies is_weekend as its support-gate exemption")
        groups = tuple(groups)
        self.requested_groups = groups
        self.continuous_audit, effective = [], []
        for name in groups:
            if name in binary:
                effective.append(name)
                continue
            values = pd.to_numeric(frame[name], errors="raise")
            known = values.notna()
            if not np.isfinite(values.loc[known].to_numpy(float)).all():
                raise ValueError(f"Nonfinite GAM candidate {name}")
            n_events = int(frame.loc[known, "event_date"].nunique())
            accepted = n_events >= 6
            if accepted:
                effective.append(name)
            self.continuous_audit.append({
                "column": name, "known_training_events": n_events,
                "known_training_rows": int(known.sum()),
                "distinct_known_training_values": int(values.loc[known].nunique()),
                "training_min_raw": float(values.loc[known].min()) if known.any() else None,
                "training_max_raw": float(values.loc[known].max()) if known.any() else None,
                "support_gate_passed": accepted,
                "omission_reason": None if accepted else "fewer_than_six_known_training_events",
                "gate_uses_y": False,
            })
        super().fit(frame, tuple(effective))
        dependencies = {item["column"] for item in self.dropped_dependencies}
        for item in self.continuous_audit:
            name = item["column"]
            item["active"] = name in self.names
            if item["support_gate_passed"] and not item["active"]:
                item["omission_reason"] = ("training_constant" if name in self.dropped_constants else
                    "training_exact_rank_dependency" if name in dependencies else "not_in_encoded_design")
            item["missing_indicator_active"] = f"{name}:missing" in self.names
        self.column_selection_audit["scope"] = "GAM_specific_train_only_subset; legacy_design_unchanged"
        self.column_selection_audit["support_gate"] = self.continuous_audit
        return self


CommonEncoder = Encoder


def make_terms(frame, encoder, lam, config, ridge_alpha=.1):
    """Build D2 plus coefficient-ridge penalties without penalizing intercept."""
    if not np.isfinite(lam) or lam <= 0 or not np.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError("GAM penalties must be finite and positive")
    values = (int(config.get("gam_n_splines", 5)), int(config.get("gam_spline_order", 3)),
              int(config.get("gam_min_known_events", 6)), int(config.get("gam_min_unique", 8)))
    if values != (5, 3, 6, 8):
        raise ValueError("V7 fixes GAM support/complexity before evaluation: 5/3/6/8")
    if frozenset(config.get("gam_smooth_candidates", SMOOTH_CANDIDATES)) != SMOOTH_CANDIDATES:
        raise ValueError("V7 smooth candidate whitelist is prespecified")
    if not isinstance(encoder, Encoder):
        raise ValueError("GAM requires its own training-support encoder")
    x, audit, terms = encoder.transform(frame), [], None
    numeric_audit = {item["column"]: item for item in encoder.continuous_audit}
    ridge_penalty = len(frame) * ridge_alpha
    for index, name in enumerate(encoder.names):
        support = numeric_audit.get(name, {})
        known, distinct = support.get("known_training_events"), support.get("distinct_known_training_values")
        smooth = name in SMOOTH_CANDIDATES and known >= 6 and distinct >= 8
        lo, hi = float(x[:, index].min()), float(x[:, index].max())
        if smooth:
            term = s(index, n_splines=5, spline_order=3, lam=[lam, ridge_penalty],
                     penalties=["derivative", "l2"], basis="ps", edge_knots=[lo, hi])
        else:
            term = l(index, lam=ridge_penalty, penalties="l2")
        terms = term if terms is None else terms + term
        audit.append({
            "column": name, "feature_index": index, "kind": "smooth" if smooth else "linear",
            "reason": "training_support_passed_for_smoothing" if smooth else
                      "supported_linear_or_structural_term_with_ridge",
            "known_training_events": known, "distinct_known_training_values": distinct,
            "training_min_raw": support.get("training_min_raw"),
            "training_max_raw": support.get("training_max_raw"),
            "training_min_encoded": lo, "training_max_encoded": hi,
            "n_splines": 5 if smooth else None, "spline_order": 3 if smooth else None,
            "lambda": lam if smooth else 0., "ridge_alpha": ridge_alpha,
            "ridge_penalty": ridge_penalty,
            "penalty": "second_order_coefficient_difference_plus_full_coefficient_l2" if smooth else "coefficient_l2",
            "edge_knots_encoded": [lo, hi] if smooth else None,
            "uses_y_for_term_or_knot_selection": False,
        })
    if terms is None:
        raise ValueError("GAM requires a nonconstant approved structural or numerical term")
    return terms, audit


def penalty_components(estimator, audit):
    """Construct both matrices independently from the stated coefficient loss."""
    n = len(estimator.coef_)
    d2, ridge = np.zeros((n, n)), np.zeros((n, n))
    for i, item in enumerate(audit):
        indices = estimator.terms.get_coef_indices(i)
        ridge[np.ix_(indices, indices)] = np.eye(len(indices)) * item["ridge_penalty"]
        if item["kind"] == "smooth":
            difference = np.diff(np.eye(len(indices)), n=2, axis=0)
            d2[np.ix_(indices, indices)] = item["lambda"] * difference.T @ difference
    return d2, ridge


def certify_gam(estimator, x, y, audit=None, tol=1e-6):
    audit = audit if audit is not None else estimator.v7_terms_audit_
    # V6 checks finite results, actual stopping, EDF and penalized normal equations.
    certificate = v6.certify_gam(estimator, x, y, tol=tol)
    d2, ridge = penalty_components(estimator, audit)
    actual = estimator._P().toarray()
    error = float(np.max(np.abs(actual - d2 - ridge)))
    limit = 1e-10 * max(1., float(np.max(abs(actual))))
    intercept = estimator.terms.get_coef_indices(len(estimator.terms) - 1)
    minimum_eigenvalue = float(np.linalg.eigvalsh(d2 + ridge).min())
    nonintercept = np.ones(len(actual), dtype=bool)
    nonintercept[intercept] = False
    minimum_ridge = float(np.diag(ridge)[nonintercept].min())
    if (not estimator.terms[-1].isintercept or np.any(actual[intercept, :]) or error > limit
            or minimum_eigenvalue < -limit or minimum_ridge <= 0):
        raise RuntimeError("GAM penalty decomposition/intercept certificate failed")
    coefficient = np.asarray(estimator.coef_)
    difference_value = float(coefficient @ d2 @ coefficient)
    ridge_value = float(coefficient @ ridge @ coefficient)
    certificate.update(
        coefficient_difference_penalty=difference_value,
        coefficient_l2_penalty=ridge_value,
        total_penalty=difference_value + ridge_value,
        penalty_matrix_max_absolute_discrepancy=error,
        penalty_matrix_certificate_tolerance=limit,
        penalty_matrix_min_eigenvalue=minimum_eigenvalue,
        minimum_nonintercept_l2_diagonal=minimum_ridge,
        intercept_unpenalized=True,
        objective="sum_squared_error + lambda_s*sum(D2*spline_coefficients)^2 + n*alpha*sum(nonintercept_coefficients^2)",
    )
    return certificate


@dataclass
class GAMFitted(v6.GAMFitted):
    def info(self):
        result = super().info()
        smooth = bool(result["smooth_columns"])
        result.update(
            requested_candidate_groups=list(self.encoder.requested_groups),
            effective_candidate_groups=list(self.encoder.groups),
            continuous_support_audit=self.encoder.continuous_audit,
            smoothing_penalty="D2_P_spline_differences_plus_L2_on_all_nonintercept_coefficients",
            effective_parameters=["lambda", "ridge_alpha"] if smooth else ["ridge_alpha"],
            smooth_lambda_identifiable=smooth,
            support_gate="omit_numeric_group_and_missing_indicator_when_known_training_events_below_six",
            coefficient_penalty="n*ridge_alpha on all linear and spline coefficients; intercept unpenalized",
        )
        return result

    def extrapolation_audit(self, frame):
        """Report every original continuous candidate, including omitted ones."""
        records = []
        kinds = {item["column"]: item["kind"] for item in self.terms_audit}
        for item in self.encoder.continuous_audit:
            values = pd.to_numeric(frame[item["column"]], errors="raise")
            known = values.notna()
            if not np.isfinite(values.loc[known].to_numpy(float)).all():
                raise ValueError("Nonfinite evaluation value in GAM audit")
            if item["training_min_raw"] is None:
                below = above = pd.Series(False, index=frame.index)
            else:
                below = values.lt(item["training_min_raw"])
                above = values.gt(item["training_max_raw"])
            records.append({**item, "rows": len(frame), "evaluation_missing_rows": int((~known).sum()),
                "evaluation_known_rows": int(known.sum()), "range_comparison_available": item["training_min_raw"] is not None,
                "below_training_range_rows": int(below.sum()), "above_training_range_rows": int(above.sum()),
                "outside_training_range_events": int(frame.loc[below | above, "event_date"].nunique()),
                "kind": kinds.get(item["column"], "omitted"),
                "behavior": "linear_extrapolation_not_clipping_or_refitting_knots" if item["active"] else
                            "value_column_omitted_missing_indicator_active" if item["missing_indicator_active"] else
                            "omitted_candidate_has_no_direct_prediction_effect",
            })
        return records


def fit_one(frame, name, groups, params, config, encoder=None):
    if name != "gam":
        return v5.fit_one(frame, name, groups, params, config, encoder=encoder)
    if config.get("include_historical_baseline", False):
        raise ValueError("V7 excludes the publication-unverified baseline")
    encoder = Encoder().fit(frame, groups, config) if encoder is None else encoder
    if not isinstance(encoder, Encoder) or tuple(groups) != encoder.requested_groups:
        raise ValueError("GAM requires a matching GAM-specific training-support encoder")
    x, y = encoder.transform(frame), frame.y.to_numpy(float)
    if not len(y) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("GAM training labels must be finite nonnegative person counts")
    lam, alpha = float(params["lambda"]), float(params["ridge_alpha"])
    terms, audit = make_terms(frame, encoder, lam, config, ridge_alpha=alpha)
    estimator = LinearGAM(terms, fit_intercept=True, max_iter=int(config.get("gam_max_iter", 1000)),
                          tol=float(config.get("gam_tol", 1e-6))).fit(x, y)
    estimator.v7_terms_audit_ = audit
    certificate = certify_gam(estimator, x, y, audit)
    design = np.column_stack([np.ones(len(x)), x])
    return GAMFitted("gam", encoder, estimator,
        {**params, "n_splines": 5, "spline_order": 3, "ridge_penalty": len(frame) * alpha,
         "n_train_rows": len(frame), "lambda_scale": "unnormalized_sum_SSE_no_n_multiplier",
         "ridge_alpha_scale": "n_rows_times_alpha_coefficient_L2", "max_iter": estimator.max_iter, "tol": estimator.tol},
        int(np.linalg.matrix_rank(design)), design.shape[1], terms_audit=audit, certificate=certificate)


class Trainer(v6.Trainer):
    """Old-model delegation plus audited, protected GAM temporal tuning."""
    def __init__(self, config):
        super().__init__(config)
        self.diagnostic_log, self.extrapolation_log = [], []

    def record_evaluation(self, fitted, train, valid, layer, parent_end, validation_year):
        prediction = fitted.predict(valid)
        raw = np.asarray(fitted.estimator.predict(fitted.encoder.transform(valid)))
        common = {"layer": layer, "parent_train_end": str(pd.Timestamp(parent_end).date()),
            "train_start": str(train.event_date.min().date()), "train_end": str(train.event_date.max().date()),
            "validation_start": str(valid.event_date.min().date()), "validation_end": str(valid.event_date.max().date()),
            "validation_year": int(validation_year), "train_events": int(train.event_date.nunique()),
            "validation_events": int(valid.event_date.nunique()), "lambda": fitted.params["lambda"],
            "ridge_alpha": fitted.params["ridge_alpha"], "ridge_penalty": fitted.params["ridge_penalty"]}
        self.diagnostic_log.append({**common, "event_mae": event_mae(valid, prediction),
            "max_raw_prediction": float(raw.max()), "min_raw_prediction": float(raw.min()),
            "max_clipped_prediction": float(prediction.max()), "n_prediction_rows": len(valid),
            "max_truth": float(valid.y.max()), "smooth_term_count": sum(t["kind"] == "smooth" for t in fitted.terms_audit),
            "effective_degrees_of_freedom": fitted.certificate["effective_degrees_of_freedom"],
            "normal_equation_relative_residual": fitted.certificate["relative_normal_equation_residual"],
            "numerical_verified": fitted.certificate["verified"]})
        self.extrapolation_log.extend({**common, **row} for row in fitted.extrapolation_audit(valid))
        return prediction

    def fit_bundle(self, frame):
        old = self.legacy.fit_bundle(frame)
        groups = tuple(self.config["candidate_groups"])
        ordered = frame[["event_date", "year", "station", "hour", "y", *groups]].sort_values(["event_date", "station", "hour"])
        digest = hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes())
        digest.update(json.dumps(self.config, sort_keys=True).encode())
        key = digest.hexdigest()
        if key in self.gam_cache:
            return {**old, "gam": self.gam_cache[key]}
        folds = list(temporal_splits(frame, self.config["min_inner_train_events"]))
        if not folds:
            raise ValueError("GAM requires chronological validation folds")
        encoders = {year: Encoder().fit(train, groups, self.config) for year, train, _ in folds}
        lambdas = list(self.config.get("gam_lambdas", [.1, 1., 10., 100., 1000.]))
        alphas = list(self.config.get("gam_ridge_alphas", [.001, .01, .1, 1.]))
        if not lambdas or not alphas or any(not np.isfinite(v) or v <= 0 for v in [*lambdas, *alphas]):
            raise ValueError("GAM penalty grids must contain finite positive values")
        for year, train, valid in folds:
            self.gam_fold_log.append({"layer": "inner_gam", "parent_train_end": str(frame.event_date.max().date()),
                "validation_year": year, "train_end": str(train.event_date.max().date()),
                "validation_start": str(valid.event_date.min().date()),
                "train_events": int(train.event_date.nunique()), "validation_events": int(valid.event_date.nunique())})
        grid, losses = [(lam, alpha) for lam in lambdas for alpha in alphas], []
        for lam, alpha in grid:
            pieces, fits = [], []
            for year, train, valid in folds:
                fitted = fit_one(train, "gam", groups, {"lambda": lam, "ridge_alpha": alpha}, self.config, encoders[year])
                piece = valid[["event_date", "y"]].copy()
                piece["pred"] = self.record_evaluation(fitted, train, valid, "inner_grid", frame.event_date.max(), year)
                pieces.append(piece)
                fits.append(fitted)
            out = pd.concat(pieces, ignore_index=True)
            loss = event_mae(out, out.pred.to_numpy())
            losses.append(loss)
            self.gam_selection_log.append({"train_end": str(frame.event_date.max().date()), "model": "gam", "groups": list(groups),
                "params": {"lambda": lam, "ridge_alpha": alpha}, "inner_event_mae": loss,
                "inner_events": int(out.event_date.nunique()), "numerical_all_verified": True,
                "inner_fits_without_smooth_terms": sum(fit.info()["no_eligible_smooth_terms"] for fit in fits),
                "tie_break": "prespecified_lambda_then_ridge_alpha_grid_order",
                "max_normal_equation_relative_residual": max(fit.certificate["relative_normal_equation_residual"] for fit in fits),
                "scope": "inner_training_selection_only_not_outer_test_score", "selection_metric": "chronological_event_balanced_MAE_not_GCV"})
        selected = grid[min(range(len(grid)), key=lambda i: (losses[i], i))]
        fitted = fit_one(frame, "gam", groups, {"lambda": selected[0], "ridge_alpha": selected[1]}, self.config)
        self.gam_cache[key] = fitted
        return {**old, "gam": fitted}

    def build_meta(self, development):
        pieces = []
        for year, train, valid in temporal_splits(development, self.config["min_base_train_events"]):
            bundle = self.fit_bundle(train)
            piece = valid[["event_date", "year", "station", "hour", "y"]].copy()
            for model in OLD_BASE_MODELS:
                piece[model] = bundle[model].predict(valid)
            piece["gam"] = self.record_evaluation(bundle["gam"], train, valid, "middle_oof", development.event_date.max(), year)
            piece["base_train_end"] = str(train.event_date.max().date())
            pieces.append(piece)
            self.gam_fold_log.append({"layer": "middle", "parent_train_end": str(development.event_date.max().date()),
                "validation_year": year, "train_end": str(train.event_date.max().date()),
                "validation_start": str(valid.event_date.min().date()), "train_events": int(train.event_date.nunique()),
                "validation_events": int(valid.event_date.nunique())})
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
            for model in OLD_BASE_MODELS:
                part[model] = bundle[model].predict(test)
            part["gam"] = self.record_evaluation(bundle["gam"], train, test, "outer", train.event_date.max(), year)
            for method in FUSION_MODELS:
                names = list(weights[method])
                part[method] = part[names].to_numpy() @ np.array([weights[method][name] for name in names])
            part["outer_year"], part["train_end"] = year, str(train.event_date.max().date())
            predictions.append(part)
            meta["outer_year"] = year
            meta_frames.append(meta)
            metadata.append({"outer_year": year, "train_events": int(train.event_date.nunique()),
                "test_events": int(test.event_date.nunique()), "train_end": str(train.event_date.max().date()),
                "test_start": str(test.event_date.min().date()), "weights": weights["stack"],
                "fusion_weights": weights, "fusion_audit": audit, "lp": audit["lp"]["stack"],
                "gam_extrapolation": bundle["gam"].extrapolation_audit(test),
                "base_models": {model: fitted.info() for model, fitted in bundle.items()}})
            fitted_folds[year] = bundle
        return pd.concat(predictions, ignore_index=True), metadata, pd.concat(meta_frames, ignore_index=True), fitted_folds
