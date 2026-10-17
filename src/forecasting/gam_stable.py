"""Support-gated GAM and explicit penalty verification."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from . import linear as linear_models
from . import gam_base as base_gam
from .gam_base import SMOOTH_CANDIDATES

class Encoder(linear_models.Encoder):
    """GAM's support gate precedes median/scaling/rank fitting on train only."""
    def fit(self, frame, groups, config=None):
        config = config or {}
        if int(config.get("gam_min_known_events", 6)) != 6:
            raise ValueError("GAM requires at least six known training events")
        binary = tuple(config.get("gam_binary_groups", ["is_weekend"]))
        if binary != ("is_weekend",):
            raise ValueError("Only is_weekend is exempt from the continuous support gate")
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
    audit = audit if audit is not None else estimator.terms_audit_
    # Check finite results, actual stopping, EDF and penalized normal equations.
    certificate = base_gam.certify_gam(estimator, x, y, tol=tol)
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
class GAMFitted(base_gam.GAMFitted):
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
