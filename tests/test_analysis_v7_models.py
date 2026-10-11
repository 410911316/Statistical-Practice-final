"""Regression tests for rare-event numerical predictors and penalized GAM."""
from pathlib import Path
import json
import sys
import unittest

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
from analysis_v6 import models as v6
from analysis_v7 import models as v7
from analysis_v5 import models as v5
from sklearn.linear_model import Ridge
from test_analysis_v6_models import config, curved_events
from test_analysis_v1_models import synthetic_events


def robust_config():
    result = config()
    result.update(gam_ridge_alphas=[.01, .1], gam_binary_groups=["is_weekend"])
    return result


def rare_weather():
    frame = synthetic_events(2021)
    frame["forecast_temperature_c"] = np.nan
    known = frame.year.eq(2021)
    z = np.sin(np.arange(known.sum()) * 1.7)
    frame.loc[known, "forecast_temperature_c"] = 20 + .01 * z
    frame.loc[known, "y"] += 20 * z
    future = frame.loc[known].copy()
    future["forecast_temperature_c"] = 35.
    return frame, future


class RobustGAMRegressionTest(unittest.TestCase):
    def test_single_known_event_cannot_drive_extrapolated_predictions(self):
        train, future = rare_weather()
        groups = ("is_weekend", "year_index", "forecast_temperature_c")
        old = v6.fit_one(train, "gam", groups, {"lambda": 1.}, robust_config())
        self.assertGreater(old.predict(future).max(), 10000.)
        fixed = v7.fit_one(train, "gam", groups, {"lambda": 1., "ridge_alpha": .1}, robust_config())
        changed = future.copy()
        changed["forecast_temperature_c"] = 100000.
        np.testing.assert_array_equal(fixed.predict(future), fixed.predict(changed))
        self.assertNotIn("forecast_temperature_c", fixed.encoder.groups)
        self.assertNotIn("forecast_temperature_c:missing", fixed.encoder.names)
        self.assertLess(fixed.predict(future).max(), 1000.)

    def test_gate_counts_known_events_instead_of_duplicated_rows(self):
        train, _ = rare_weather()
        many_rows = pd.concat([train] * 30, ignore_index=True)
        encoder = v7.Encoder().fit(many_rows, ("year_index", "forecast_temperature_c"), robust_config())
        audit = {item["column"]: item for item in encoder.continuous_audit}
        self.assertEqual(audit["forecast_temperature_c"]["known_training_events"], 1)
        self.assertEqual(audit["year_index"]["known_training_events"], 5)
        self.assertEqual(encoder.groups, ())
        self.assertFalse(any("forecast_temperature_c" in item for item in encoder.names))
        enough = synthetic_events(2022)
        enough["forecast_temperature_c"] = 20 + np.sin(np.arange(len(enough)))
        supported = v7.Encoder().fit(enough, ("year_index", "forecast_temperature_c"), robust_config())
        self.assertEqual(supported.groups, ("year_index", "forecast_temperature_c"))

    def test_no_smooth_is_ridge_with_unpenalized_intercept_and_n_times_alpha(self):
        frame = synthetic_events(2024)
        fitted = v7.fit_one(frame, "gam", ("year_index", "is_weekend"),
                            {"lambda": 1., "ridge_alpha": .1}, robust_config())
        self.assertTrue(fitted.info()["no_eligible_smooth_terms"])
        self.assertEqual(fitted.info()["effective_parameters"], ["ridge_alpha"])
        same = v7.fit_one(frame, "gam", ("year_index", "is_weekend"),
                          {"lambda": 1000., "ridge_alpha": .1}, robust_config())
        np.testing.assert_array_equal(same.predict(frame), fitted.predict(frame))
        stronger = v7.fit_one(frame, "gam", ("year_index", "is_weekend"),
                              {"lambda": 1., "ridge_alpha": 1.}, robust_config())
        self.assertGreater(np.max(abs(stronger.predict(frame) - fitted.predict(frame))), 1.)
        x = fitted.encoder.transform(frame)
        sklearn = Ridge(alpha=len(frame) * .1, solver="svd").fit(x, frame.y)
        np.testing.assert_allclose(fitted.estimator.predict(x), sklearn.predict(x), atol=2e-6, rtol=1e-7)
        shifted = frame.copy()
        shifted["y"] += 10000.
        other = v7.fit_one(shifted, "gam", ("year_index", "is_weekend"),
                           {"lambda": 1000., "ridge_alpha": .1}, robust_config())
        np.testing.assert_allclose(other.predict(frame) - fitted.predict(frame), 10000., atol=5e-6)
        self.assertAlmostEqual(fitted.certificate["coefficient_difference_penalty"], 0.)
        self.assertGreater(fitted.certificate["coefficient_l2_penalty"], 0.)

    def test_two_penalties_are_verified_and_smoothing_still_adds_curve(self):
        frame = curved_events()
        fitted = v7.fit_one(frame, "gam", ("forecast_temperature_c",),
                            {"lambda": .1, "ridge_alpha": .001}, robust_config())
        self.assertEqual(fitted.info()["smooth_columns"], ["forecast_temperature_c"])
        d2, ridge = v7.penalty_components(fitted.estimator, fitted.terms_audit)
        np.testing.assert_allclose(fitted.estimator._P().toarray(), d2 + ridge, atol=1e-12)
        self.assertTrue(fitted.certificate["intercept_unpenalized"])
        self.assertGreater(fitted.certificate["coefficient_difference_penalty"], 0.)
        self.assertGreater(fitted.certificate["coefficient_l2_penalty"], 0.)
        self.assertTrue(fitted.certificate["verified"])
        self.assertGreater(fitted.certificate["minimum_nonintercept_l2_diagonal"], 0.)
        self.assertGreaterEqual(fitted.certificate["penalty_matrix_min_eigenvalue"], -1e-12)
        indices = fitted.estimator.terms.get_coef_indices(len(fitted.estimator.terms) - 1)
        self.assertEqual(np.sum(abs((d2 + ridge)[indices, :])), 0.)
        x = fitted.encoder.transform(frame)
        fitted.estimator.coef_[0] += 100.
        with self.assertRaisesRegex(RuntimeError, "normal-equation"):
            v7.certify_gam(fitted.estimator, x, frame.y.to_numpy())

    def test_all_original_numeric_candidates_are_audited_even_if_omitted_or_aliased(self):
        frame = synthetic_events(2024)
        frame["planned_fireworks_minutes"] = frame.year_index + 10.
        frame["planned_program_minutes"] = np.nan
        frame["forecast_temperature_c"] = 25.
        frame["forecast_cloud_cover_pct"] = np.nan
        frame.loc[frame.year.eq(2024), "forecast_cloud_cover_pct"] = np.arange(24.)
        groups = ("year_index", "planned_fireworks_minutes", "planned_program_minutes",
                  "forecast_temperature_c", "forecast_cloud_cover_pct")
        fitted = v7.fit_one(frame, "gam", groups, {"lambda": 1., "ridge_alpha": .1}, robust_config())
        future = frame.iloc[:4].copy()
        for name in groups:
            future[name] = 1000.
        future.loc[future.index[0], "forecast_temperature_c"] = np.nan
        audits = {item["column"]: item for item in fitted.extrapolation_audit(future)}
        self.assertEqual(set(audits), set(groups))
        self.assertTrue(audits["year_index"]["active"])
        self.assertEqual(audits["year_index"]["above_training_range_rows"], 4)
        self.assertEqual(audits["planned_fireworks_minutes"]["omission_reason"], "training_exact_rank_dependency")
        self.assertEqual(audits["forecast_temperature_c"]["omission_reason"], "training_constant")
        self.assertEqual(audits["forecast_temperature_c"]["evaluation_missing_rows"], 1)
        self.assertFalse(audits["planned_program_minutes"]["range_comparison_available"])
        self.assertEqual(audits["forecast_cloud_cover_pct"]["above_training_range_rows"], 4)
        self.assertEqual(audits["forecast_cloud_cover_pct"]["known_training_events"], 1)
        json.dumps(audits, allow_nan=False)

    def test_test_geometry_and_labels_do_not_refit_scaling_knots_or_support(self):
        frame = curved_events()
        fitted = v7.fit_one(frame, "gam", ("forecast_temperature_c",),
                            {"lambda": 1., "ridge_alpha": .01}, robust_config())
        before = json.dumps(fitted.info(), sort_keys=True, allow_nan=False)
        initial = fitted.predict(frame)
        test = frame.copy()
        test["forecast_temperature_c"] += 100.
        test["y"] += 50000.
        fitted.predict(test)
        report = fitted.extrapolation_audit(test)
        self.assertEqual(report[0]["above_training_range_rows"], len(test))
        np.testing.assert_array_equal(initial, fitted.predict(frame))
        self.assertEqual(before, json.dumps(fitted.info(), sort_keys=True, allow_nan=False))
        curve = fitted.effect_curves(frame, points=17)
        self.assertEqual(len(curve), 17)
        self.assertAlmostEqual(curve.x_raw.min(), frame.forecast_temperature_c.min())
        self.assertAlmostEqual(curve.x_raw.max(), frame.forecast_temperature_c.max())

    def test_legacy_encoder_cannot_bypass_the_gam_gate(self):
        train, _ = rare_weather()
        groups = ("year_index", "forecast_temperature_c")
        with self.assertRaisesRegex(ValueError, "GAM-specific"):
            v7.fit_one(train, "gam", groups, {"lambda": 1., "ridge_alpha": .1},
                       robust_config(), encoder=v5.Encoder().fit(train, groups))


class RobustGAMTemporalTest(unittest.TestCase):
    def test_old_eight_predictions_bitexact_and_future_labels_do_not_change_model_or_fusion(self):
        frame = synthetic_events(2024)
        cfg = robust_config()
        cfg.update(outer_years=[2024], gam_lambdas=[.1], gam_ridge_alphas=[.01, .1])
        old, old_folds, _, _ = v5.Trainer(cfg).backtest(frame, progress=lambda _: None)
        trainer = v7.Trainer(cfg)
        new, folds, meta, _ = trainer.backtest(frame, progress=lambda _: None)
        for model in v5.ALL_MODELS:
            np.testing.assert_array_equal(new[model], old[model])
        for method, weights in old_folds[0]["fusion_weights"].items():
            self.assertEqual(folds[0]["fusion_weights"][method], weights)
        changed = frame.copy()
        changed.loc[changed.year.eq(2024), "y"] += 100000.
        after, changed_folds, _, _ = v7.Trainer(cfg).backtest(changed, progress=lambda _: None)
        for model in v7.ALL_MODELS:
            np.testing.assert_array_equal(new[model], after[model])
        self.assertEqual(folds[0]["fusion_weights"], changed_folds[0]["fusion_weights"])
        self.assertTrue((pd.to_datetime(meta.base_train_end) < meta.event_date).all())
        self.assertEqual(set(pd.DataFrame(trainer.diagnostic_log).layer), {"inner_grid", "middle_oof", "outer"})
        for item in trainer.diagnostic_log:
            self.assertLess(pd.Timestamp(item["train_end"]), pd.Timestamp(item["validation_start"]))
            self.assertTrue(item["numerical_verified"])
        self.assertEqual(set(pd.DataFrame(trainer.extrapolation_log).layer), {"inner_grid", "middle_oof", "outer"})
        self.assertEqual(set(folds[0]["fusion_weights"]["gam_stack"]), {"ridge", "gam", "xgboost"})
        self.assertAlmostEqual(sum(folds[0]["fusion_weights"]["gam_stack"].values()), 1.)


if __name__ == "__main__":
    unittest.main()
