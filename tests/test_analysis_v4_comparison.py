"""Ablation contracts: comparable experiments and complete, paired mean totals."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis_v1.models import ALL_MODELS
from compare_analysis_v4 import (
    CALENDAR, FEATURES, FIT_CODE, HOURS, STATIONS, VARIANTS,
    ablation_metrics, mean_event_counts, validate_manifests, validate_panel,
)


def complete_frames():
    rows = []
    for date, truth in (("2031-07-01", 100.), ("2032-07-01", 300.)):
        for station, hour in product(STATIONS, HOURS):
            rows.append({"event_date": pd.Timestamp(date), "station": station, "hour": hour,
                         "train_end": f"{int(date[:4]) - 1}-12-31", "y": truth,
                         "year": int(date[:4]), "outer_year": int(date[:4]),
                         **{model: truth * .8 for model in ALL_MODELS}})
    frame = pd.DataFrame(rows)
    return {variant: frame.copy(deep=True) for variant in VARIANTS}


def fair_manifests():
    grid = [dict(zip(("max_depth", "n_estimators", "min_child_weight", "reg_lambda"), values))
            for values in product((1, 2, 3, 4, 5), (50, 100, 250), (1, 6, 24), (1., 10.))]
    manifests = {}
    for variant in VARIANTS:
        sources = [{"path": "data/processed/example.csv", "sha256": "base", "bytes": 42}]
        if variant in ("fireworks", "both"):
            sources.append({"path": "docs/d1_program_sources.json", "sha256": "program", "bytes": 42})
        if variant in ("weather", "both"):
            sources.append({"path": "docs/d1_weather_sources.json", "sha256": "weather", "bytes": 42})
        manifests[variant] = {"config": {"version": variant, "include_historical_baseline": False,
                                           "d1_cutoff_hour": 18, "candidate_groups": CALENDAR + FEATURES[variant],
                                           "d1_features": FEATURES[variant].copy(), "xgb_grid": deepcopy(grid),
                                           "seed": 7, "outer_years": [2031, 2032], "ridge_lambdas": [.1, 1.]},
                              "data": {"manifest": sources},
                              "code": [{"path": path, "sha256": "code"} for path in sorted(FIT_CODE)]}
    return manifests


class MeanCountsTests(unittest.TestCase):
    def test_mean_is_per_activity_sum_not_mean_hour_and_uses_same_events(self):
        means = mean_event_counts(complete_frames())
        for scope, expected in (("all", 4800.), ("beimen", 1200.)):
            block = means.loc[means.variant.eq("both") & means.scope.eq(scope)].set_index("model")
            self.assertEqual(block.loc["y", "mean_event_origin_trips"], expected)
            self.assertAlmostEqual(block.loc["ridge", "mean_event_origin_trips"], expected * .8)
            self.assertAlmostEqual(block.loc["stack", "mean_bias_pct"], -20.)
            self.assertTrue(block.events.eq(2).all())
        self.assertEqual(len(means), 4 * 2 * 5)

    def test_rejects_incomplete_panels_even_if_all_variants_missing_same_row(self):
        frames = {name: frame.iloc[:-1] for name, frame in complete_frames().items()}
        with self.assertRaisesRegex(ValueError, "incomplete station/hour"):
            mean_event_counts(frames)
        frames = complete_frames()
        for name in frames:
            frames[name] = frames[name].loc[frames[name].station.ne("雙連")]
        with self.assertRaisesRegex(ValueError, "incomplete station/hour"):
            mean_event_counts(frames)

    def test_rejects_changed_targets_or_missing_event_across_variants(self):
        frames = complete_frames()
        frames["weather"].loc[0, "y"] += 1
        with self.assertRaisesRegex(ValueError, "target values differ"):
            mean_event_counts(frames)
        frames = complete_frames()
        frames["weather"] = frames["weather"].iloc[:24]
        with self.assertRaisesRegex(ValueError, "keys differ"):
            mean_event_counts(frames)

    def test_rejects_absent_future_or_invalid_training_cutoff(self):
        frame = complete_frames()["core"]
        with self.assertRaisesRegex(ValueError, "train_end is required"):
            validate_panel(frame.drop(columns="train_end"))
        frame["train_end"] = frame.event_date
        with self.assertRaisesRegex(ValueError, "training must end"):
            validate_panel(frame)

    def test_fixed_actual_peak_earliest_tie_does_not_use_predicted_peak(self):
        frames = complete_frames()
        # All actual hours tie. Earliest hour 18 must be used although forecast peaks at 19.
        for frame in frames.values():
            frame.loc[frame.station.eq("北門") & frame.hour.eq(18), "ridge"] = frame.y * .5
            frame.loc[frame.station.eq("北門") & frame.hour.eq(19), "ridge"] = frame.y * 2
        summary, annual = ablation_metrics(frames)
        row = summary.loc[summary.variant.eq("core") & summary.scope.eq("beimen") & summary.model.eq("ridge")].iloc[0]
        self.assertAlmostEqual(row.peak_absolute_error_mean, 100.)
        self.assertAlmostEqual(row.peak_underprediction_mean, 100.)
        self.assertAlmostEqual(row.peak_hour_mae, 1.)
        self.assertEqual(set(annual.year), {2031, 2032})
        self.assertEqual(len(annual), 4 * 2 * 2 * 4)

    def test_zero_actual_mean_does_not_produce_infinite_percentage(self):
        frames = complete_frames()
        for frame in frames.values():
            frame["y"] = 0.
        table = mean_event_counts(frames)
        self.assertTrue(table.mean_bias_pct.isna().all())
        self.assertTrue(np.isfinite(table.mean_event_origin_trips).all())


class FairAblationTests(unittest.TestCase):
    def test_feature_differences_and_additional_source_manifest_are_allowed(self):
        checks = validate_manifests(fair_manifests())
        self.assertEqual(checks["candidate_count"], 90)
        self.assertEqual(len(checks["d1_source_hashes"]), 2)

    def test_rejects_changed_base_or_shared_source_hash(self):
        for index, message in ((0, "base source input hashes differ"), (1, "shared D1 source hashes differ")):
            manifests = fair_manifests()
            manifests["both"]["data"]["manifest"][index]["sha256"] = "changed"
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, message):
                validate_manifests(manifests)

    def test_rejects_changed_model_code_seed_grid_or_hidden_baseline(self):
        for change, message in (
            (lambda m: m["both"]["code"][0].update(sha256="changed"), "modeling code differs"),
            (lambda m: m["both"]["config"].update(seed=999), "settings differ"),
            (lambda m: m["both"]["config"]["xgb_grid"].pop(), "90-candidate"),
            (lambda m: m["both"]["config"].update(include_historical_baseline=True), "baseline must be excluded"),
            (lambda m: m["both"]["config"].update(d1_cutoff_hour=23), "cutoff must be 18:00"),
        ):
            manifests = fair_manifests()
            change(manifests)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_manifests(manifests)

    def test_rejects_extra_post_event_feature_or_missing_source_provenance(self):
        manifests = fair_manifests()
        manifests["both"]["config"]["d1_features"].append("observed_weather")
        with self.assertRaisesRegex(ValueError, "incorrect ablation features"):
            validate_manifests(manifests)
        manifests = fair_manifests()
        manifests["weather"]["data"]["manifest"].pop()
        with self.assertRaisesRegex(ValueError, "D1 source hashes missing"):
            validate_manifests(manifests)


if __name__ == "__main__":
    unittest.main()
