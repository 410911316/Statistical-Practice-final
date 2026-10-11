"""D-1 unknown inputs remain distinct from zero and are fit within each fold."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_v1.models import BASE_MODELS, Encoder, Trainer, fit_one  # noqa: E402
from test_analysis_v1_models import small_config, synthetic_events  # noqa: E402


def optional_events(last_year=2024):
    frame = synthetic_events(last_year)
    frame["planned_drone"] = frame.year.mod(2).astype(float)
    frame.loc[frame.year.eq(2017), "planned_drone"] = np.nan
    frame["planned_fireworks_minutes"] = 5. + 3. * frame.year.mod(3)
    frame.loc[frame.year.eq(2018), "planned_fireworks_minutes"] = np.nan
    frame["forecast_temperature_c"] = np.nan
    return frame


def optional_config():
    config = small_config()
    config["candidate_groups"] += ["planned_drone", "planned_fireworks_minutes",
                                    "forecast_temperature_c"]
    config["ridge_station_hour_interaction"] = True
    return config


class OptionalD1EncoderTest(unittest.TestCase):
    def test_excluded_baseline_cannot_change_any_fitted_prediction(self):
        frame = optional_events(2022)
        config = optional_config()
        config["include_historical_baseline"] = False
        changed = frame.copy()
        changed["baseline_hourly_56d"] = np.nan
        for model, params in (("ols", {}), ("ridge", {"lambda": .1}),
                              ("xgboost", {"max_depth": 2, "n_estimators": 5})):
            fit = fit_one(frame, model, config["candidate_groups"], params, config)
            self.assertNotIn("baseline_hourly_56d", fit.encoder.names)
            np.testing.assert_array_equal(fit.predict(frame), fit.predict(changed))

    def test_train_only_median_mode_and_missing_indicators(self):
        frame = synthetic_events(2017).iloc[:4].copy()
        frame["planned_fireworks_minutes"] = [5., 15., np.nan, 100.]
        frame["planned_drone"] = [0., 0., 1., np.nan]
        encoder = Encoder().fit(frame, ("planned_fireworks_minutes", "planned_drone"))
        self.assertEqual(encoder.imputation_values["planned_fireworks_minutes"], 15.)
        self.assertEqual(encoder.imputation_values["planned_drone"], 0.)
        self.assertIn("planned_fireworks_minutes:missing", encoder.names)
        self.assertIn("planned_drone:missing", encoder.names)
        before = json.dumps(encoder.imputation_audit, sort_keys=True)
        original_mean, original_scale = encoder.mean.copy(), encoder.scale.copy()
        later = frame.copy()
        later["planned_fireworks_minutes"] = [999999., np.nan, 800000., 500000.]
        self.assertTrue(np.isfinite(encoder.transform(later)).all())
        self.assertEqual(json.dumps(encoder.imputation_audit, sort_keys=True), before)
        pd.testing.assert_series_equal(encoder.mean, original_mean)
        pd.testing.assert_series_equal(encoder.scale, original_scale)

    def test_entirely_unknown_training_feature_cannot_activate_in_future(self):
        train = synthetic_events(2019)
        train["forecast_temperature_c"] = np.nan
        train["planned_drone"] = np.nan
        encoder = Encoder().fit(train, ("forecast_temperature_c", "planned_drone"))
        self.assertEqual(encoder.all_missing_numeric, ["forecast_temperature_c", "planned_drone"])
        self.assertTrue(all(name not in encoder.names for name in (
            "forecast_temperature_c", "forecast_temperature_c:missing",
            "planned_drone", "planned_drone:missing")))
        future = train.copy()
        future["forecast_temperature_c"] = np.arange(len(future), dtype=float)
        future["planned_drone"] = np.arange(len(future)) % 2
        np.testing.assert_array_equal(encoder.transform(train), encoder.transform(future))

    def test_unknown_drone_is_distinguishable_from_explicit_no_drone(self):
        frame = synthetic_events(2018)
        frame["planned_drone"] = 0.
        frame.loc[frame.year.eq(2018), "planned_drone"] = np.nan
        encoder = Encoder().fit(frame, ("planned_drone",))
        self.assertNotIn("planned_drone", encoder.names)  # Known zero is constant.
        self.assertIn("planned_drone:missing", encoder.names)
        missing = encoder.transform(frame)[:, encoder.names.index("planned_drone:missing")]
        np.testing.assert_array_equal(missing, frame.planned_drone.isna().astype(float))
        self.assertEqual(encoder.imputation_audit["planned_drone"]["training_unknown_rows"],
                         int(frame.planned_drone.isna().sum()))

    def test_unapproved_missing_and_invalid_optional_values_are_rejected(self):
        for name in ("baseline_hourly_56d", "is_weekend"):
            frame = synthetic_events(2018)
            frame.loc[frame.index[0], name] = np.nan
            groups = ("is_weekend",) if name == "is_weekend" else ()
            with self.subTest(column=name), self.assertRaisesRegex(ValueError, "Nonfinite"):
                Encoder().fit(frame, groups)
        for name, value, message in (("planned_drone", 2., "planned_drone"),
                                      ("forecast_temperature_c", np.inf, "infinity")):
            frame = synthetic_events(2018)
            frame[name] = value
            with self.subTest(column=name), self.assertRaisesRegex(ValueError, message):
                Encoder().fit(frame, (name,))
        frame = synthetic_events(2018)
        frame["observed_has_drone_show"] = 1.
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            Encoder().fit(frame, ("observed_has_drone_show",))

    def test_info_preserves_json_safe_imputation_audit_and_legacy_compatibility(self):
        frame = optional_events(2019)
        fitted = fit_one(frame, "ridge", ["planned_drone", "forecast_temperature_c"],
                         {"lambda": .1}, optional_config())
        info = fitted.info()
        json.dumps(info, allow_nan=False)
        self.assertEqual(info["all_training_values_unknown"], ["forecast_temperature_c"])
        self.assertIn("forecast_temperature_c:missing", info["dropped_constants"])
        legacy = Encoder().fit(frame, ("year_index",))
        expected = legacy.transform(frame)
        for attr in ("imputation_values", "imputation_audit", "missing_indicator_numeric",
                     "all_missing_numeric", "station_hour_interaction"):
            delattr(legacy, attr)
        np.testing.assert_array_equal(legacy.transform(frame), expected)


class OptionalD1TemporalTest(unittest.TestCase):
    def test_future_values_cannot_change_earlier_oof_imputation_or_predictions(self):
        frame = optional_events(2023)
        changed = frame.copy()
        changed.loc[changed.year.ge(2022), "planned_fireworks_minutes"] += 10000
        changed.loc[changed.year.ge(2022), "y"] += 100000
        before = Trainer(optional_config()).build_meta(frame)
        after = Trainer(optional_config()).build_meta(changed)
        early = before.year.lt(2022)
        self.assertTrue(early.any())
        np.testing.assert_array_equal(before.loc[early, list(BASE_MODELS)],
                                      after.loc[early, list(BASE_MODELS)])

    def test_outer_truth_cannot_change_selected_parameters_or_imputation(self):
        frame = optional_events(2024)
        changed = frame.copy()
        changed.loc[changed.year.eq(2024), "y"] += 500000
        first = Trainer(optional_config()).backtest(frame, progress=lambda _: None)
        second = Trainer(optional_config()).backtest(changed, progress=lambda _: None)
        self.assertEqual(first[1], second[1])
        np.testing.assert_array_equal(first[0][["baseline", *BASE_MODELS, "equal", "stack"]],
                                      second[0][["baseline", *BASE_MODELS, "equal", "stack"]])
        self.assertFalse(np.array_equal(first[0].y, second[0].y))


if __name__ == "__main__":
    unittest.main()
