"""Contracts for station-hour Ridge and train-only XGBoost tuning.

Synthetic data make differing station profiles identifiable without relying on
private snapshots or asserting that a change improves the observed outer test.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_v1.models import BASE_MODELS, Encoder, Trainer, fit_one  # noqa: E402
from test_analysis_v1_models import small_config, synthetic_events  # noqa: E402


def interaction_config():
    config = small_config()
    config["ridge_station_hour_interaction"] = True
    config["xgb_grid"] = [
        {"max_depth": 1, "n_estimators": 3, "min_child_weight": 1, "reg_lambda": 1.0},
        {"max_depth": 2, "n_estimators": 3, "min_child_weight": 2, "reg_lambda": 3.0},
    ]
    return config


class StationHourRidgeTest(unittest.TestCase):
    def test_interaction_fits_opposing_station_profiles_main_effects_cannot(self):
        rows = []
        for year in range(2017, 2023):
            for station in ("A", "B"):
                for hour in (20, 21):
                    y = 1100. if (station == "A") == (hour == 20) else 100.
                    rows.append({"event_date": pd.Timestamp(year, 8, 1), "year": year,
                                 "station": station, "hour": hour,
                                 "baseline_hourly_56d": 50., "y": y})
        train = pd.DataFrame(rows)
        additive = fit_one(train, "ridge", [], {"lambda": 1e-8}, small_config())
        interaction = fit_one(train, "ridge", [], {"lambda": 1e-8}, interaction_config())
        self.assertGreater(np.mean(abs(additive.predict(train) - train.y)), 499.)
        self.assertLess(np.mean(abs(interaction.predict(train) - train.y)), .01)
        self.assertEqual(interaction.encoder.names, ["station=B", "hour=21", "station=B:hour=21"])
        # Main effects and product form a full-rank treatment-coded design.
        self.assertEqual(interaction.rank, interaction.columns)

    def test_interaction_categories_and_scaling_are_training_only(self):
        frame = synthetic_events(2021)
        train = frame.loc[(frame.year < 2021) & frame.hour.isin([20, 21])].copy()
        valid = frame.loc[(frame.year == 2021) & frame.hour.isin([20, 21])].copy()
        encoder = Encoder().fit(train, ("year_index",), station_hour_interaction=True)
        mean = encoder.mean.copy()
        categories = {key: list(value) for key, value in encoder.categories.items()}
        valid["baseline_hourly_56d"] += 1e6
        encoder.transform(valid)
        pd.testing.assert_series_equal(mean, encoder.mean)
        self.assertEqual(categories, encoder.categories)
        self.assertEqual(encoder.categories["hour"], [20, 21])
        for column, value in [("station", "unknown"), ("hour", 22)]:
            unseen = valid.iloc[:1].copy()
            unseen[column] = value
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, f"Unseen {column}"):
                    encoder.transform(unseen)

    def test_default_design_and_legacy_saved_encoder_are_unchanged(self):
        frame = synthetic_events(2020)
        default = Encoder().fit(frame, ("year_index",))
        explicit = Encoder().fit(frame, ("year_index",), station_hour_interaction=False)
        expected_names = ["baseline_hourly_56d", "year_index"]
        for column in ("station", "hour"):
            expected_names.extend(f"{column}={v}" for v in sorted(frame[column].unique())[1:])
        self.assertEqual(default.names, expected_names)
        expected = default.transform(frame).copy()
        np.testing.assert_array_equal(explicit.transform(frame), expected)
        # joblib restores an old object's state without calling the new fit().
        del default.station_hour_interaction
        np.testing.assert_array_equal(default.transform(frame), expected)

    def test_flag_only_changes_ridge_design(self):
        frame = synthetic_events(2020)
        cfg = interaction_config()
        params = {"ols": {}, "ridge": {"lambda": .01}, "xgboost": cfg["xgb_grid"][0]}
        for name in BASE_MODELS:
            fitted = fit_one(frame, name, [], params[name], cfg)
            with self.subTest(model=name):
                self.assertEqual(any(":hour=" in col for col in fitted.encoder.names), name == "ridge")


class TunedXGBoostTemporalTest(unittest.TestCase):
    def test_candidate_parameters_override_fixed_values_in_actual_estimator(self):
        frame = synthetic_events(2020)
        config = interaction_config()
        candidate = config["xgb_grid"][0]
        fitted = fit_one(frame, "xgboost", [], candidate, config)
        actual = fitted.estimator.get_params()
        self.assertEqual(actual["min_child_weight"], 1)
        self.assertEqual(actual["reg_lambda"], 1.0)
        self.assertEqual(actual["learning_rate"], config["xgb_fixed"]["learning_rate"])
        self.assertEqual(config["xgb_fixed"]["min_child_weight"], 4)
        self.assertEqual(config["xgb_fixed"]["reg_lambda"], 10.0)

    def test_new_configuration_future_labels_cannot_change_earlier_oof(self):
        frame = synthetic_events(2023)
        changed = frame.copy()
        changed.loc[changed.year >= 2022, "y"] += 10000
        before = Trainer(interaction_config()).build_meta(frame)
        after = Trainer(interaction_config()).build_meta(changed)
        keys = ["event_date", "station", "hour"]
        early_before = before.loc[before.year <= 2022].sort_values(keys).reset_index(drop=True)
        early_after = after.loc[after.year <= 2022].sort_values(keys).reset_index(drop=True)
        self.assertGreater(len(early_before), 0)
        np.testing.assert_allclose(early_before[list(BASE_MODELS)], early_after[list(BASE_MODELS)],
                                   rtol=0, atol=1e-9)
        self.assertFalse(np.allclose(early_before.y, early_after.y))
        self.assertTrue((pd.to_datetime(before.base_train_end) < before.event_date).all())

    def test_new_configuration_outer_truth_cannot_change_model_or_stack(self):
        frame = synthetic_events(2024)
        changed = frame.copy()
        changed.loc[changed.year == 2024, "y"] += 500000
        pred1, metadata1, _, _ = Trainer(interaction_config()).backtest(frame, progress=lambda _: None)
        pred2, metadata2, _, _ = Trainer(interaction_config()).backtest(changed, progress=lambda _: None)
        self.assertEqual(metadata1, metadata2)
        np.testing.assert_allclose(pred1[["baseline", *BASE_MODELS, "equal", "stack"]],
                                   pred2[["baseline", *BASE_MODELS, "equal", "stack"]],
                                   rtol=0, atol=1e-9)
        self.assertFalse(np.allclose(pred1.y, pred2.y))


if __name__ == "__main__":
    unittest.main()
