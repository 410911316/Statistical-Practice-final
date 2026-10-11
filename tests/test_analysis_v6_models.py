"""GAM objective/support, preserved comparisons and chronological fusion."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
from analysis_v5 import models as v5
from analysis_v6.models import (ALL_MODELS, BASE_MODELS, Encoder, Trainer, certify_gam,
                                fit_fusion, fit_one, make_terms, metrics_table)
from test_analysis_v1_models import synthetic_events, small_config


def config():
    value = small_config()
    value.update(include_historical_baseline=False, lasso_alphas=[1.],
                 ridge_lambdas=[.1], xgb_grid=[value["xgb_grid"][0]],
                 gam_lambdas=[.1, 10.], fusion_top_k=2)
    return value


def curved_events():
    frame = synthetic_events(2024)
    random = np.random.default_rng(1005)
    temperature = random.uniform(20, 35, len(frame))
    frame["forecast_temperature_c"] = temperature
    frame["y"] = 100. + 2. * (temperature - 27.) ** 2 + 5. * (frame.hour - 18.)
    return frame


class GAMDesignTest(unittest.TestCase):
    def test_support_gate_uses_events_and_known_values_not_replicated_rows(self):
        frame = curved_events()
        frame["planned_fireworks_minutes"] = np.resize([5., 10., 15., 20., 25.], len(frame))
        names = ("year_index", "planned_fireworks_minutes", "forecast_temperature_c")
        encoder = Encoder().fit(frame, names)
        _, audit = make_terms(frame, encoder, 1., config())
        terms = {row["column"]: row for row in audit}
        self.assertEqual(terms["forecast_temperature_c"]["kind"], "smooth")
        self.assertEqual(terms["planned_fireworks_minutes"]["kind"], "linear")
        self.assertEqual(terms["year_index"]["kind"], "linear")
        early = frame.loc[frame.year <= 2020]
        doubled = pd.concat([early] * 10, ignore_index=True)
        _, early_audit = make_terms(doubled, Encoder().fit(doubled, names), 1., config())
        early_terms = {row["column"]: row for row in early_audit}
        self.assertEqual(early_terms["forecast_temperature_c"]["known_training_events"], 4)
        self.assertEqual(early_terms["forecast_temperature_c"]["kind"], "linear")
        missing = frame.copy()
        missing.loc[missing.year > 2020, "forecast_temperature_c"] = np.nan
        _, missing_audit = make_terms(missing, Encoder().fit(missing, names), 1., config())
        missing_terms = {row["column"]: row for row in missing_audit}
        self.assertEqual(missing_terms["forecast_temperature_c"]["kind"], "linear")

    def test_training_only_terms_knots_scaling_and_missing_gate(self):
        frame = curved_events()
        fitted = fit_one(frame, "gam", ("forecast_temperature_c",), {"lambda": 1.}, config())
        before = json.dumps(fitted.info(), sort_keys=True, allow_nan=False)
        old_train = fitted.predict(frame)
        future = frame.copy()
        future["forecast_temperature_c"] += 10000
        future["y"] += 100000
        fitted.predict(future)
        fitted.extrapolation_audit(future)
        np.testing.assert_array_equal(old_train, fitted.predict(frame))
        self.assertEqual(before, json.dumps(fitted.info(), sort_keys=True, allow_nan=False))
        self.assertEqual(fitted.info()["smooth_columns"], ["forecast_temperature_c"])
        self.assertFalse(any(item["uses_y_for_term_or_knot_selection"] for item in fitted.terms_audit))

    def test_curve_fit_uses_gaussian_difference_penalty_and_high_lambda_reduces_edf(self):
        frame = curved_events()
        groups = ("forecast_temperature_c",)
        low = fit_one(frame, "gam", groups, {"lambda": .1}, config())
        high = fit_one(frame, "gam", groups, {"lambda": 1000.}, config())
        linear = fit_one(frame, "ols", groups, {}, config())
        self.assertEqual(low.estimator.distribution._name, "normal")
        self.assertEqual(low.estimator.link._name, "identity")
        self.assertLess(np.mean((low.predict(frame) - frame.y) ** 2),
                        np.mean((linear.predict(frame) - frame.y) ** 2) / 4)
        self.assertLess(high.info()["effective_degrees_of_freedom"], low.info()["effective_degrees_of_freedom"])
        self.assertGreater(low.certificate["coefficient_difference_penalty"], 0.)
        self.assertTrue(low.certificate["verified"])
        self.assertEqual(low.info()["terms"][-1]["penalty"], "second_order_coefficient_difference")

    def test_no_supported_smooth_means_unpenalized_linear_not_forced_curve(self):
        frame = synthetic_events(2021)
        gam = fit_one(frame, "gam", config()["candidate_groups"], {"lambda": 1.}, config())
        ols = fit_one(frame, "ols", config()["candidate_groups"], {}, config())
        self.assertTrue(gam.info()["no_eligible_smooth_terms"])
        np.testing.assert_allclose(gam.predict(frame), ols.predict(frame), atol=2e-5, rtol=1e-7)
        self.assertEqual(gam.effect_curves(frame).shape[0], 0)

    def test_boundary_is_linear_extrapolation_and_reported_not_refitted(self):
        train = curved_events()
        fitted = fit_one(train, "gam", ("forecast_temperature_c",), {"lambda": .1}, config())
        outside = pd.concat([train.iloc[[0]]] * 3, ignore_index=True)
        outside["forecast_temperature_c"] = train.forecast_temperature_c.max() + np.array([1., 2., 3.])
        pred = fitted.predict(outside)
        self.assertAlmostEqual(float(np.diff(pred, n=2)[0]), 0., places=6)
        report = fitted.extrapolation_audit(outside)[0]
        self.assertEqual(report["above_training_range_rows"], 3)
        self.assertEqual(report["outside_training_range_events"], 1)
        curve = fitted.effect_curves(train, points=17)
        self.assertEqual(len(curve), 17)
        self.assertAlmostEqual(curve.x_raw.min(), train.forecast_temperature_c.min())
        self.assertAlmostEqual(curve.x_raw.max(), train.forecast_temperature_c.max())
        self.assertEqual(curve.reference.unique().tolist(), ["training_empirical_mean_component"])
        self.assertFalse(any("pvalue" in column or "CI" in column for column in curve.columns))

    def test_invalid_coefficients_or_failed_stop_are_not_scored(self):
        frame = curved_events()
        fitted = fit_one(frame, "gam", ("forecast_temperature_c",), {"lambda": 1.}, config())
        x, y = fitted.encoder.transform(frame), frame.y.to_numpy()
        fitted.estimator.coef_[0] += 100
        with self.assertRaisesRegex(RuntimeError, "normal-equation"):
            certify_gam(fitted.estimator, x, y)
        fitted.estimator.logs_["diffs"][-1] = 10.
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            certify_gam(fitted.estimator, x, y)


class GAMFusionTest(unittest.TestCase):
    def test_triplet_has_prespecified_members_simplex_weights_and_unchanged_old_fusion(self):
        meta = pd.DataFrame({"event_date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
                             "y": [15., 22., 33.], "ols": [1., 2., 3.], "ridge": [10., 20., 30.],
                             "lasso": [2., 3., 4.], "xgboost": [20., 21., 31.], "gam": [16., 23., 34.]})
        old_weights, _ = v5.fit_fusion(meta)
        weights, audit = fit_fusion(meta)
        for name in old_weights:
            self.assertEqual(weights[name], old_weights[name])
        self.assertEqual(weights["gam_equal"], dict(ridge=1/3, gam=1/3, xgboost=1/3))
        self.assertEqual(set(weights["gam_stack"]), {"ridge", "gam", "xgboost"})
        self.assertAlmostEqual(sum(weights["gam_stack"].values()), 1.)
        self.assertTrue(all(value >= 0 for value in weights["gam_stack"].values()))
        self.assertFalse(audit["gam_triplet"]["selection_by_outer_results"])

    def test_old_predictions_unchanged_and_outer_labels_cannot_change_own_predictions(self):
        frame = synthetic_events(2024)
        old, old_folds, _, _ = v5.Trainer(config()).backtest(frame, progress=lambda _: None)
        current_trainer = Trainer(config())
        current, folds, meta, bundle = current_trainer.backtest(frame, progress=lambda _: None)
        for model in v5.ALL_MODELS:
            np.testing.assert_array_equal(current[model], old[model])
        for index, fold in enumerate(folds):
            for method, weights in old_folds[index]["fusion_weights"].items():
                self.assertEqual(fold["fusion_weights"][method], weights)
            self.assertEqual(bundle[fold["outer_year"]]["gam"].encoder.names,
                             bundle[fold["outer_year"]]["ridge"].encoder.names)
        changed = frame.copy()
        changed.loc[changed.year.eq(2024), "y"] += 50000
        after, altered_folds, _, _ = Trainer(config()).backtest(changed, progress=lambda _: None)
        for model in ALL_MODELS:
            np.testing.assert_array_equal(current[model], after[model])
        self.assertEqual(folds[-1]["fusion_weights"], altered_folds[-1]["fusion_weights"])
        self.assertTrue((pd.to_datetime(meta.base_train_end) < meta.event_date).all())
        self.assertTrue(all(pd.Timestamp(row["train_end"]) < pd.Timestamp(row["validation_start"])
                            for row in current_trainer.fold_log))
        self.assertTrue(set(BASE_MODELS) <= set(current))
        self.assertEqual(set(metrics_table(current).model), set(ALL_MODELS))
        self.assertTrue(all(row["selection_metric"] == "chronological_event_balanced_MAE_not_GCV"
                            for row in current_trainer.gam_selection_log))


if __name__ == "__main__":
    unittest.main()
