"""Common-design fairness, sparse regression, and temporal selection/fusion."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoLars

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_v5.models import (  # noqa: E402
    ALL_MODELS, BASE_MODELS, Encoder, Trainer, event_mae, fit_fusion, fit_one,
    metrics_table, solve_stack, certify_lasso,
)
from test_analysis_v1_models import small_config, synthetic_events  # noqa: E402


def config():
    result = small_config()
    result.update(include_historical_baseline=False, lasso_alphas=[.1, 10.], fusion_top_k=2)
    return result


class CommonDesignTest(unittest.TestCase):
    def test_all_four_models_have_identical_columns_and_products(self):
        frame = synthetic_events(2021)
        bundle = Trainer(config()).fit_bundle(frame)
        names = bundle["ols"].encoder.names
        self.assertEqual(len([name for name in names if ":hour=" in name]), 15)
        for model, fitted in bundle.items():
            with self.subTest(model=model):
                self.assertEqual(fitted.encoder.names, names)
                self.assertEqual(fitted.rank, fitted.columns)
                self.assertFalse(fitted.fallback)
                self.assertNotIn("baseline_hourly_56d", names)
                np.testing.assert_array_equal(fitted.encoder.transform(frame), bundle["ols"].encoder.transform(frame))
                json.dumps(fitted.info(), allow_nan=False)
        self.assertEqual(bundle["ols"].encoder.groups, tuple(config()["candidate_groups"]))

    def test_exact_alias_removal_is_common_structural_and_training_only(self):
        train = synthetic_events(2020)
        # This approved field exactly duplicates a structural main effect.
        train["planned_fireworks_minutes"] = np.where(train.station.eq("大橋頭"), 10., 5.)
        train["forecast_temperature_c"] = np.nan
        encoder = Encoder().fit(train, ("planned_fireworks_minutes", "forecast_temperature_c"))
        self.assertIn("station=大橋頭", encoder.names)
        self.assertNotIn("planned_fireworks_minutes", encoder.names)
        self.assertTrue(any(item["column"] == "planned_fireworks_minutes" for item in encoder.dropped_dependencies))
        before = json.dumps(encoder.column_selection_audit, sort_keys=True)
        future = train.copy()
        future["planned_fireworks_minutes"] = np.arange(len(future)) + 20000
        future["forecast_temperature_c"] = np.arange(len(future)) + 10000
        future["y"] *= 1000
        np.testing.assert_array_equal(encoder.transform(train), encoder.transform(future))
        self.assertEqual(before, json.dumps(encoder.column_selection_audit, sort_keys=True))
        self.assertNotIn("forecast_temperature_c", encoder.names)
        self.assertEqual(encoder.imputation_values["planned_fireworks_minutes"], 5.)

    def test_high_correlation_is_not_a_rank_dependency(self):
        frame = synthetic_events(2021)
        frame["forecast_temperature_c"] = frame.year_index + 1e-4 * np.sin(np.arange(len(frame)))
        encoder = Encoder().fit(frame, ("year_index", "forecast_temperature_c"))
        self.assertIn("year_index", encoder.names)
        self.assertIn("forecast_temperature_c", encoder.names)
        self.assertFalse(encoder.dropped_dependencies)

    def test_identical_missing_flags_drop_only_redundant_training_column(self):
        frame = synthetic_events(2021)
        frame["forecast_temperature_c"] = 20 + np.sin(np.arange(len(frame)))
        frame["forecast_cloud_cover_pct"] = 40 + np.cos(np.arange(len(frame)))
        frame.loc[frame.year.eq(2017), ["forecast_temperature_c", "forecast_cloud_cover_pct"]] = np.nan
        encoder = Encoder().fit(frame, ("forecast_temperature_c", "forecast_cloud_cover_pct"))
        self.assertIn("forecast_temperature_c:missing", encoder.names)
        self.assertNotIn("forecast_cloud_cover_pct:missing", encoder.names)
        self.assertEqual([item["column"] for item in encoder.dropped_dependencies], ["forecast_cloud_cover_pct:missing"])

    def test_baseline_cannot_be_accidentally_enabled_or_change_predictions(self):
        frame = synthetic_events(2021)
        fitted = fit_one(frame, "ols", config()["candidate_groups"], {}, config())
        changed = frame.copy()
        changed["baseline_hourly_56d"] = np.nan
        np.testing.assert_array_equal(fitted.predict(frame), fitted.predict(changed))
        with self.assertRaisesRegex(ValueError, "baseline"):
            Encoder().fit(frame, (), include_historical_baseline=True)


class LassoTest(unittest.TestCase):
    def test_lasso_high_penalty_selects_intercept_only_and_preserves_candidate_design(self):
        frame = synthetic_events(2021)
        low = fit_one(frame, "lasso", config()["candidate_groups"], {"alpha": .1}, config())
        high = fit_one(frame, "lasso", config()["candidate_groups"], {"alpha": 1e6}, config())
        self.assertEqual(low.encoder.names, high.encoder.names)
        self.assertGreater(np.count_nonzero(low.estimator.coef_), 0)
        self.assertEqual(np.count_nonzero(high.estimator.coef_), 0)
        np.testing.assert_allclose(high.predict(frame), frame.y.mean())
        self.assertEqual(high.info()["lasso_zero_columns"], high.encoder.names)
        self.assertEqual(high.params["objective"], "SSE/(2*n)+alpha*L1")
        self.assertTrue(np.isfinite(low.info()["dual_gap"]))

    def test_nonconvergence_is_an_error_not_a_silently_scored_candidate(self):
        def failed_fit(*args, **kwargs):
            warnings.warn("intentional convergence failure", ConvergenceWarning)
        with patch("analysis_v5.models.Lasso.fit", failed_fit):
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                fit_one(synthetic_events(2021), "lasso", config()["candidate_groups"], {"alpha": .1}, config())

    def test_truncated_iterations_are_rejected(self):
        short = config()
        short["lasso_max_iter"] = 1
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            fit_one(synthetic_events(2021), "lasso", short["candidate_groups"], {"alpha": .1}, short)

    def test_inaccurate_coefficients_fail_independent_certificate(self):
        frame = synthetic_events(2021)
        fitted = fit_one(frame, "lasso", config()["candidate_groups"], {"alpha": .1}, config())
        fitted.estimator.coef_[0] += 100
        with self.assertRaisesRegex(RuntimeError, "optimality certificate failed"):
            certify_lasso(fitted.estimator, fitted.encoder.transform(frame), frame.y.to_numpy(), .1, 1e-6)

    def test_lasso_mean_loss_is_invariant_to_replicating_rows(self):
        frame = synthetic_events(2020)
        double = pd.concat([frame, frame], ignore_index=True)
        first = fit_one(frame, "lasso", config()["candidate_groups"], {"alpha": 1.}, config())
        second = fit_one(double, "lasso", config()["candidate_groups"], {"alpha": 1.}, config())
        np.testing.assert_allclose(first.predict(frame), second.predict(frame), rtol=1e-6, atol=1e-6)

    def test_lasso_matches_lars_for_same_objective(self):
        frame = synthetic_events(2021)
        fitted = fit_one(frame, "lasso", config()["candidate_groups"], {"alpha": .1}, config())
        x = fitted.encoder.transform(frame)
        reference = LassoLars(alpha=.1, fit_intercept=True, max_iter=100000).fit(x, frame.y)
        np.testing.assert_allclose(fitted.predict(frame), reference.predict(x), rtol=1e-6, atol=1e-5)
        certificate = fitted.info()["optimization_certificate"]
        self.assertTrue(certificate["verified"])
        self.assertLessEqual(certificate["primal_dual_gap"], certificate["gap_tolerance"])
        self.assertLessEqual(certificate["active_kkt_error"], certificate["kkt_tolerance"])


class FusionTest(unittest.TestCase):
    def test_four_model_lp_recovers_known_mixture_and_one_model_is_valid(self):
        predictions = np.array([[10, 30, 50, 90], [20, 90, 30, 40], [60, 15, 80, 30], [90, 40, 20, 10]], float)
        truth = predictions @ np.array([.1, .2, .3, .4])
        weights, info = solve_stack(predictions, truth, ["a", "b", "c", "d"])
        np.testing.assert_allclose(predictions @ weights, truth, atol=1e-6)
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertTrue((weights >= 0).all())
        self.assertEqual(info["n_models"], 4)
        weight, _ = solve_stack(predictions[:, :1], truth, ["a", "b", "c", "d"])
        np.testing.assert_array_equal(weight, [1.])

    def test_top_two_excludes_other_models_and_reports_meta_training_scores(self):
        meta = pd.DataFrame({"event_date": pd.to_datetime(["2020-07-15", "2020-07-15", "2021-07-15"]), "y": [10., 20., 30.],
                             "ols": [8., 18., 28.], "ridge": [12., 22., 32.],
                             "lasso": [0., 0., 0.], "xgboost": [100., 100., 100.]})
        weights, audit = fit_fusion(meta)
        self.assertEqual(audit["selected_models"], ["ols", "ridge"])
        self.assertEqual(audit["selected_best_model"], "ols")
        self.assertEqual(weights["select_stack"]["lasso"], 0.)
        self.assertEqual(weights["select_stack"]["xgboost"], 0.)
        p = meta[list(BASE_MODELS)].to_numpy()
        mixed = p @ np.array([weights["select_stack"][model] for model in BASE_MODELS])
        np.testing.assert_allclose(mixed, meta.y, atol=1e-7)
        np.testing.assert_allclose(p @ np.array(list(weights["equal"].values())), meta[list(BASE_MODELS)].mean(axis=1))
        self.assertEqual(audit["oof_event_mae"]["ols"], 2.)
        self.assertIn("meta_training", audit["scope"])
        for method in weights:
            self.assertAlmostEqual(sum(weights[method].values()), 1.)

    def test_event_balance_and_invalid_inputs(self):
        truth = np.array([0.] * 9 + [20., 20.])
        dates = ["a"] * 9 + ["b", "c"]
        predictions = np.tile([0., 10., 20., 30.], (len(truth), 1))
        weights, info = solve_stack(predictions, truth, dates)
        np.testing.assert_allclose(predictions @ weights, 20.)
        self.assertAlmostEqual(info["oof_mae"], 20 / 3)
        for invalid in [np.ones((2, 0)), np.ones(2), np.array([[1., np.nan], [2., 3.]]), np.array([[-1.], [2.]])]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                solve_stack(invalid, [1., 2.], ["a", "b"])


class TemporalSelectionTest(unittest.TestCase):
    def test_outer_truth_cannot_change_base_fit_selection_or_fusion(self):
        frame = synthetic_events(2024)
        changed = frame.copy()
        changed.loc[changed.year.eq(2024), "y"] += 500000
        first = Trainer(config()).backtest(frame, progress=lambda _: None)
        second = Trainer(config()).backtest(changed, progress=lambda _: None)
        self.assertEqual(first[1], second[1])
        np.testing.assert_array_equal(first[0][list(ALL_MODELS)], second[0][list(ALL_MODELS)])
        self.assertFalse(np.array_equal(first[0].y, second[0].y))
        self.assertTrue((pd.to_datetime(first[2].base_train_end) < first[2].event_date).all())
        self.assertEqual(set(metrics_table(first[0]).model), set(ALL_MODELS))
        for fold in first[1]:
            self.assertLess(pd.Timestamp(fold["fusion_audit"]["oof_end"]), pd.Timestamp(fold["test_start"]))
            self.assertEqual(fold["fusion_audit"]["top_k"], 2)

    def test_future_oof_labels_cannot_change_earlier_base_predictions(self):
        frame = synthetic_events(2023)
        changed = frame.copy()
        changed.loc[changed.year.ge(2022), "y"] += 10000
        before = Trainer(config()).build_meta(frame)
        after = Trainer(config()).build_meta(changed)
        early = before.year.le(2022)
        np.testing.assert_array_equal(before.loc[early, list(BASE_MODELS)], after.loc[early, list(BASE_MODELS)])

    def test_bundle_cache_includes_training_labels_and_config(self):
        frame = synthetic_events(2020)
        changed = frame.copy()
        changed.y += 100
        trainer = Trainer(config())
        original = trainer.fit_bundle(frame)["ols"].predict(frame)
        revised = trainer.fit_bundle(changed)["ols"].predict(frame)
        np.testing.assert_allclose(revised - original, 100.)
        self.assertEqual(len(trainer.cache), 2)
        trainer.config["lasso_alphas"] = [250.]
        trainer.fit_bundle(frame)
        self.assertEqual(len(trainer.cache), 3)


if __name__ == "__main__":
    unittest.main()
