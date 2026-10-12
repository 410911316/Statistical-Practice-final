"""Numerics, event weighting, chronology and reporting of fixed-cache stacking."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis_v9.fusion import (MODELS, DISPLAY_MODELS, aggregate_metrics, evaluate,
                                event_weights, fit_losses, solve_mse, validate_meta)
from analysis_v6.reporting import STATIONS, HOURS


class MSEStackTests(unittest.TestCase):
    def test_recovers_known_interior_weights(self):
        p = np.array([[1, 4, 2], [2, 1, 5], [4, 2, 1], [0, 2, 5]], float)
        w = np.array([.2, .3, .5])
        result, audit = solve_mse(p, p @ w, [1, 1, 2, 2])
        np.testing.assert_allclose(result, w, atol=1e-12)
        self.assertLess(audit["scaled_kkt_dual_gap"], 1e-12)

    def test_boundary_vertex(self):
        p = np.array([[1, 2, 3], [2, 3, 4]], float)
        w, _ = solve_mse(p, [0, 0], [1, 2])
        np.testing.assert_allclose(w, [1, 0, 0])

    def test_rank_deficient_columns(self):
        p = np.array([[1, 1, 3], [2, 2, 4], [3, 3, 5]], float)
        w, _ = solve_mse(p, [2, 3, 4], [1, 1, 2])
        np.testing.assert_allclose(p @ w, [2, 3, 4], atol=1e-12)
        p = np.ones((3, 3))
        w, _ = solve_mse(p, [0, 2, 4], [1, 1, 2])
        self.assertAlmostEqual(w.sum(), 1)

    def test_independent_slsqp_agrees_on_random_cases(self):
        rng = np.random.default_rng(419)
        for _ in range(20):
            p = rng.uniform(0, 3, (40, 3))
            y = rng.uniform(0, 4, 40)
            dates = np.repeat(range(4), 10)
            w, _ = solve_mse(p, y, dates)
            result = minimize(lambda v: np.mean((p @ v-y)**2), np.ones(3)/3,
                              jac=lambda v: 2 * p.T @ (p @ v-y)/40,
                              constraints={"type": "eq", "fun": lambda v: v.sum()-1,
                                           "jac": lambda v: np.ones(3)},
                              bounds=[(0, 1)]*3, method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000})
            self.assertTrue(result.success)
            self.assertAlmostEqual(np.mean((p @ w-y)**2), result.fun, places=9)

    def test_duplication_within_event_does_not_increase_its_weight(self):
        p = np.array([[1, 3, 6], [2, 5, 1], [2, 0, 5], [4, 2, 0]], float)
        y = np.array([2, 4, 3, 3], float)
        w, _ = solve_mse(p, y, [1, 1, 2, 2])
        indices = [0, 1, 0, 1, 0, 1, 2, 3]
        other, _ = solve_mse(p[indices], y[indices], [1, 1, 1, 1, 1, 1, 2, 2])
        np.testing.assert_allclose(w, other, atol=1e-12)

    def test_mae_and_mse_are_distinct_objectives(self):
        meta = pd.DataFrame({"ridge": [0]*5, "gam": [10]*5, "xgboost": [5]*5,
                             "y": [0, 0, 0, 0, 10], "event_date": [1]*5})
        w, _ = fit_losses(meta)
        p = meta[list(MODELS)].to_numpy()
        np.testing.assert_allclose(p @ w["mae"], 0, atol=1e-9)
        np.testing.assert_allclose(p @ w["mse"], 2, atol=1e-9)

    def test_invalid_inputs_rejected(self):
        for p, y, dates in ((np.ones((2, 2)), [1, 2], [1, 2]),
                            (np.ones((2, 3)), [1, np.nan], [1, 2]),
                            (np.ones((2, 3)), [1, -2], [1, 2]),
                            (np.ones((2, 3)), [1, 2], [1, None]),
                            (np.ones((2, 3)), [1, 2], [1])):
            with self.assertRaises(ValueError):
                event_weights(p, y, dates)


class ChronologyTests(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame([{"event_date": "2023-07-01", "year": 2023, "station": s, "hour": h,
                                   "y": 10., "ridge": 8., "gam": 11., "xgboost": 13.,
                                   "base_train_end": "2022-07-01", "outer_year": 2024}
                                  for s in STATIONS for h in HOURS])
        self.outer = pd.DataFrame([{"event_date": pd.Timestamp("2024-07-01"), "year": 2024,
                                    "station": s, "hour": h, "y": 14., "outer_year": 2024}
                                   for s in STATIONS for h in HOURS])
        self.inputs = pd.concat([self.meta, self.outer], ignore_index=True)
        self.folds = [{"outer_year": 2024, "train_end": "2023-07-01", "test_start": "2024-07-01",
                       "fusion_audit": {"oof_rows": 24, "oof_events": 1,
                                        "oof_start": "2023-07-01", "oof_end": "2023-07-01"}}]

    def test_valid_chronology(self):
        self.assertEqual(len(validate_meta(self.meta, self.outer, self.folds, self.inputs)), 24)

    def test_base_training_future_or_same_year_rejected(self):
        for date in ("2023-07-01", "2023-01-01", "2025-01-01"):
            meta = self.meta.assign(base_train_end=date)
            with self.assertRaisesRegex(ValueError, "OOF base"):
                validate_meta(meta, self.outer, self.folds, self.inputs)

    def test_outer_year_labels_cannot_enter_meta(self):
        meta = self.meta.assign(event_date="2024-01-01", year=2024)
        with self.assertRaisesRegex(ValueError, "Future outer"):
            validate_meta(meta, self.outer, self.folds, self.inputs)

    def test_missing_panel_or_duplicate_rows_rejected(self):
        for meta in (self.meta.iloc[:-1], pd.concat([self.meta, self.meta.iloc[:1]])):
            with self.assertRaises(ValueError):
                validate_meta(meta, self.outer, self.folds, self.inputs)

    def test_changed_source_labels_rejected(self):
        bad = self.inputs.copy()
        bad.loc[0, "y"] = 999
        with self.assertRaisesRegex(ValueError, "labels do not match"):
            validate_meta(self.meta, self.outer, self.folds, bad)

    def test_outer_truth_is_not_fitting_input(self):
        valid = validate_meta(self.meta, self.outer, self.folds, self.inputs)
        before, _ = fit_losses(valid)
        new_outer = self.outer.assign(y=99999.)
        new_inputs = pd.concat([self.meta, new_outer])
        changed = validate_meta(self.meta, new_outer, self.folds, new_inputs)
        after, _ = fit_losses(changed)
        for loss in before:
            np.testing.assert_array_equal(before[loss], after[loss])


class MetricTests(unittest.TestCase):
    def test_event_rmse_not_averaged_directly(self):
        frame = pd.DataFrame({"event_date": pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-02"]),
                              "station": ["北門"]*3, "hour": [18, 18, 19], "y": [0., 0., 0.]})
        for model in DISPLAY_MODELS:
            frame[model] = [1., 3., 3.]
        summary, _ = evaluate(frame)
        self.assertAlmostEqual(summary.iloc[0].rmse, np.sqrt(5))
        self.assertAlmostEqual(summary.iloc[0].mae, 2)

    def test_peak_compares_same_hour_not_two_maxima(self):
        frame = pd.DataFrame({"event_date": pd.to_datetime(["2024-07-01"]*2),
                              "station": ["北門"]*2, "hour": [18, 19], "y": [10., 5.]})
        for model in DISPLAY_MODELS:
            frame[model] = [4., 9.]
        summary, _ = evaluate(frame)
        self.assertEqual(summary.iloc[0].peak_underprediction, 6)
        self.assertEqual(summary.iloc[0].peak_hour_mae, 1)


if __name__ == "__main__":
    unittest.main()
