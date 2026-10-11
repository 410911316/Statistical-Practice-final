"""Numerical/reporting contracts for paired models and auditable fusion."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis_v5.reporting import (
    ALL_MODELS, BASE_MODELS, CALENDAR, FEATURES, FIT_CODE, HOURS, STATIONS, VARIANTS,
    align_panels, mean_event_counts, metrics_table, selection_table, validate_manifests, validate_panel,
)


def frames():
    rows = []
    for date, truth in (("2031-07-01", 100.), ("2032-07-01", 300.)):
        for station, hour in product(STATIONS, HOURS):
            rows.append({"event_date": pd.Timestamp(date), "year": int(date[:4]), "outer_year": int(date[:4]),
                         "station": station, "hour": hour, "train_end": f"{int(date[:4]) - 1}-12-31",
                         "y": truth, **{model: truth * .8 for model in ALL_MODELS}})
    frame = pd.DataFrame(rows)
    return {variant: frame.copy(deep=True) for variant in VARIANTS}


def manifests():
    values = {}
    for variant in VARIANTS:
        sources = [{"path": "data/processed/source.csv", "sha256": "base"}]
        if variant in ("fireworks", "both"):
            sources.append({"path": "docs/d1_program_sources.json", "sha256": "program"})
        if variant in ("weather", "both"):
            sources.append({"path": "docs/d1_weather_sources.json", "sha256": "weather"})
        values[variant] = {"config": {"version": variant, "include_historical_baseline": False,
                                     "d1_cutoff_hour": 18, "station_hour_interaction": True,
                                     "linear_design": "full_shared_with_training_rank_filter",
                                     "candidate_groups": CALENDAR + FEATURES[variant],
                                     "d1_features": FEATURES[variant].copy(), "seed": 7,
                                     "fusion_top_k": 2, "lasso_alphas": [1., 5., 10.]},
                           "data": {"manifest": sources},
                           "code": [{"path": path, "sha256": "code"} for path in sorted(FIT_CODE)]}
    return values


def folds():
    scores = {"ols": 20., "ridge": 18., "lasso": 14., "xgboost": 12.}
    weights = {"equal": {m: .25 for m in BASE_MODELS},
               "stack": {"ols": .1, "ridge": .1, "lasso": .3, "xgboost": .5},
               "select_best": {"ols": 0., "ridge": 0., "lasso": 0., "xgboost": 1.},
               "select_stack": {"ols": 0., "ridge": 0., "lasso": .4, "xgboost": .6}}
    fold = {"outer_year": 2031, "train_end": "2030-12-31", "test_start": "2031-07-01",
            "fusion_weights": weights, "fusion_audit": {"oof_event_mae": scores,
            "ranked_models": ["xgboost", "lasso", "ridge", "ols"], "selected_models": ["xgboost", "lasso"],
            "selected_best_model": "xgboost", "top_k": 2},
            "base_models": {m: {"columns": ["year", "station=北門:hour=21"], "groups": ["year"]} for m in BASE_MODELS}}
    return {variant: [deepcopy(fold)] for variant in VARIANTS}


class PairedCountsTests(unittest.TestCase):
    def test_activity_totals_then_mean_preserves_common_sample(self):
        table = mean_event_counts(frames())
        for scope, expected in (("all", 4800.), ("beimen", 1200.)):
            part = table.loc[table.variant.eq("both") & table.scope.eq(scope)].set_index("model")
            self.assertEqual(part.loc["y", "mean_event_origin_trips"], expected)
            self.assertAlmostEqual(part.loc["lasso", "mean_event_origin_trips"], expected * .8)
            self.assertAlmostEqual(part.loc["select_stack", "mean_bias_pct"], -20.)
            self.assertTrue(part.events.eq(2).all())
        self.assertEqual(len(table), 4 * 2 * 9)

    def test_rejects_common_missing_row_or_duplicate_row(self):
        incomplete = {name: frame.iloc[:-1] for name, frame in frames().items()}
        with self.assertRaisesRegex(ValueError, "incomplete station/hour"):
            mean_event_counts(incomplete)
        frame = frames()["core"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_panel(pd.concat([frame, frame.iloc[:1]]))

    def test_rejects_changed_truth_and_training_cutoff(self):
        left, right = frames()["core"], frames()["both"]
        right.loc[0, "y"] += 1
        with self.assertRaisesRegex(ValueError, "different truths"):
            align_panels(left, right)
        right = frames()["both"]
        right["train_end"] = pd.to_datetime(right.train_end) - pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "different training cutoffs"):
            align_panels(left, right)

    def test_future_cutoff_or_missing_date_cannot_be_silently_scored(self):
        frame = frames()["core"]
        frame["train_end"] = frame.event_date
        with self.assertRaisesRegex(ValueError, "Training must end"):
            metrics_table(frame)
        with self.assertRaisesRegex(ValueError, "Missing prediction columns"):
            metrics_table(frames()["core"].drop(columns="train_end"))

    def test_event_balanced_loss_and_peak_ties(self):
        frame = frames()["core"]
        standard = metrics_table(frame).query("scope == 'all' and model == 'lasso'").iloc[0]
        self.assertAlmostEqual(standard.mae, 40.)
        self.assertAlmostEqual(standard.rmse, np.sqrt((20**2 + 60**2) / 2))
        self.assertAlmostEqual(standard.bias_pred_minus_actual, -40.)
        # Actual hours tie, so use hour 18; prediction peaks at 19.
        frame.loc[frame.station.eq("北門") & frame.hour.eq(18), "lasso"] = frame.y * .5
        frame.loc[frame.station.eq("北門") & frame.hour.eq(19), "lasso"] = frame.y * 2
        peak = metrics_table(frame).query("scope == 'beimen' and model == 'lasso'").iloc[0]
        self.assertAlmostEqual(peak.peak_absolute_error_mean, 100.)
        self.assertAlmostEqual(peak.peak_underprediction_mean, 100.)
        self.assertAlmostEqual(peak.peak_hour_mae, 1.)

    def test_zero_truth_never_produces_infinite_percentages(self):
        source = frames()
        for frame in source.values():
            frame["y"] = 0.
        table = mean_event_counts(source)
        self.assertTrue(table.mean_bias_pct.isna().all())
        self.assertTrue(np.isfinite(table.mean_event_origin_trips).all())


class FairDesignTests(unittest.TestCase):
    def test_feature_and_source_provenance_variations_are_allowed(self):
        checks = validate_manifests(manifests())
        self.assertTrue(checks["same_settings_except_feature_groups"])
        self.assertEqual(len(checks["d1_source_hashes"]), 2)

    def test_rejects_changed_source_code_or_training_rule(self):
        for alter in (
            lambda m: m["both"]["data"]["manifest"][0].update(sha256="changed"),
            lambda m: m["both"]["code"][0].update(sha256="changed"),
            lambda m: m["both"]["config"].update(lasso_alphas=[1., 3.]),
        ):
            data = manifests()
            alter(data)
            with self.assertRaisesRegex(ValueError, "comparison changes"):
                validate_manifests(data)
        data = manifests()
        data["both"]["data"]["manifest"][1]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "shared D1 source hashes differ"):
            validate_manifests(data)

    def test_rejects_hidden_baseline_post_event_feature_or_different_design(self):
        for updates in ({"include_historical_baseline": True}, {"d1_cutoff_hour": 23},
                        {"station_hour_interaction": False}, {"linear_design": "forward_selection"},
                        {"d1_features": FEATURES["both"] + ["observed_temperature"]}, {"fusion_top_k": 3}):
            data = manifests()
            data["both"]["config"].update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_manifests(data)


class FusionAuditTests(unittest.TestCase):
    def test_reports_recorded_earlier_oof_choices_not_outer_truth(self):
        table = selection_table(folds())
        self.assertEqual(len(table), 4 * 4 * 4)
        selected = table.query("variant == 'both' and fusion == 'select_stack'").set_index("base_model")
        self.assertAlmostEqual(selected.loc["lasso", "weight"], .4)
        self.assertEqual(selected.loc["xgboost", "oof_rank"], 1)
        self.assertFalse(selected.loc["ols", "selected_top2"])

    def test_rejects_outer_year_training_and_mismatched_design(self):
        data = folds()
        data["both"][0]["train_end"] = "2031-07-01"
        with self.assertRaisesRegex(ValueError, "cutoff must precede"):
            selection_table(data)
        data = folds()
        data["both"][0]["base_models"]["ols"]["columns"] = ["different"]
        with self.assertRaisesRegex(ValueError, "same encoded design"):
            selection_table(data)

    def test_rejects_selection_not_based_on_recorded_oof_order(self):
        data = folds()
        data["both"][0]["fusion_audit"]["selected_models"] = ["ridge", "lasso"]
        with self.assertRaisesRegex(ValueError, "Selected models disagree"):
            selection_table(data)
        data = folds()
        data["both"][0]["fusion_audit"]["ranked_models"] = ["ols", "lasso", "ridge", "xgboost"]
        with self.assertRaisesRegex(ValueError, "ranking disagrees"):
            selection_table(data)

    def test_rejects_invalid_or_unselected_fusion_weights(self):
        data = folds()
        data["both"][0]["fusion_weights"]["select_stack"].update(ols=.1, xgboost=.5)
        with self.assertRaisesRegex(ValueError, "Unselected model"):
            selection_table(data)
        data = folds()
        data["both"][0]["fusion_weights"]["stack"]["lasso"] = -.1
        with self.assertRaisesRegex(ValueError, "nonnegative and sum"):
            selection_table(data)


if __name__ == "__main__":
    unittest.main()
