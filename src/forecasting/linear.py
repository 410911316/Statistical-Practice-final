"""Identifiable design and regularized linear estimators."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from .encoding_base import Encoder as OriginalEncoder

BASE_MODELS = ("ridge", "xgboost")


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
            raise ValueError("The base encoder excludes unverified historical baseline; use the approved feature encoder")
        if not station_hour_interaction:
            raise ValueError("A common station-by-hour interaction design is required")
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
        if self.name == "ridge":
            coef = self.estimator.coef_
            intercept = self.estimator.intercept_
            result["coefficients"] = {name: float(value) for name, value in zip(self.encoder.names, coef)}
            result["intercept"] = float(intercept)
            result["coefficient_scale"] = "continuous_training_standardized_dummy_and_binary_0_1"
        return result


def fit_one(frame, name, groups, params, config, encoder=None):
    if config.get("include_historical_baseline", False):
        raise ValueError("Legacy unverified baseline flag must remain false; approved baseline is a named feature group")
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
    if name == "ridge":
        lam = float(params["lambda"])
        if not np.isfinite(lam) or lam < 0:
            raise ValueError("Ridge lambda must be finite and nonnegative")
        alpha = len(frame) * lam
        estimator = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(x, y)
        params.update(sklearn_alpha=alpha, n_train_rows=len(frame))
    else:
        xgb_params = {"objective": "reg:squarederror", "tree_method": "hist", "n_jobs": 1,
                      "random_state": config["seed"], **config["xgb_fixed"], **params}
        estimator = XGBRegressor(**xgb_params).fit(x, y)
    return Fitted(name, encoder, estimator, params, rank, design.shape[1])
