"""Gaussian GAM fit diagnostics and interpretable components."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from . import linear as linear_models

SMOOTH_CANDIDATES = frozenset({
    "day_of_year", "planned_fireworks_minutes", "planned_fireworks_start_hour",
    "forecast_temperature_c", "forecast_wind_speed_m_s", "forecast_cloud_cover_pct",
})


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
class GAMFitted(linear_models.Fitted):
    terms_audit: list | None = None
    certificate: dict | None = None

    def info(self):
        result = super().info()
        result.update(
            family="Gaussian", link="identity", clipping="nonnegative_prediction_clipping",
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
