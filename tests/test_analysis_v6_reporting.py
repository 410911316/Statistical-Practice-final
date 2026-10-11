"""Paired, provenance-checked GAM and historical comparator reporting."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis_v6.reporting import (
    ALL_MODELS, OLD_BASE_MODELS, OLD_MODELS, FUSION_MODELS, CALENDAR, FEATURES,
    FIT_CODE, HOURS, STATIONS, VARIANTS, align_panels, effect_figure, fusion_table,
    mean_event_counts, metrics_table, validate_fusion_predictions, validate_manifests,
    validate_old_reproduction, validate_panel,
)


def frames():
    rows = []
    for date, truth in (("2031-07-01", 100.), ("2032-07-01", 300.)):
        for station, hour in product(STATIONS, HOURS):
            rows.append({"event_date": pd.Timestamp(date), "year": int(date[:4]), "outer_year": int(date[:4]),
                         "station": station, "hour": hour, "train_end": f"{int(date[:4]) - 1}-12-31",
                         "y": truth, **{m: truth * .8 for m in ALL_MODELS}})
    frame = pd.DataFrame(rows)
    return {variant: frame.copy(deep=True) for variant in VARIANTS}


def manifests():
    result = {}
    for variant in VARIANTS:
        sources = [{"path": "data/processed/source.csv", "sha256": "source"}]
        if variant in ("fireworks", "both"):
            sources.append({"path": "docs/d1_program_sources.json", "sha256": "program"})
        if variant in ("weather", "both"):
            sources.append({"path": "docs/d1_weather_sources.json", "sha256": "weather"})
        result[variant] = {"config": {"version": variant, "include_historical_baseline": False,
                                    "d1_cutoff_hour": 18, "station_hour_interaction": True,
                                    "linear_design": "full_shared_with_training_rank_filter",
                                    "candidate_groups": CALENDAR + FEATURES[variant], "d1_features": FEATURES[variant].copy(),
                                    "seed": 7, "fusion_top_k": 2, "gam_lambdas": [.1, 1, 10],
                                    "gam_fusion_models": list(FUSION_MODELS)},
                           "data": {"manifest": sources},
                           "code": [{"path": path, "sha256": "code"} for path in sorted(FIT_CODE)]}
    return result


def old_manifest():
    old = deepcopy(manifests()["both"])
    old["config"] = {k: v for k, v in old["config"].items() if not k.startswith("gam_")}
    old["config"]["version"] = "old"
    return old


def folds():
    scores = {"ols": 20., "ridge": 18., "lasso": 14., "xgboost": 12.}
    weights = {"equal": {m: .25 for m in OLD_BASE_MODELS},
               "stack": {"ols": .1, "ridge": .1, "lasso": .3, "xgboost": .5},
               "select_best": {"ols": 0., "ridge": 0., "lasso": 0., "xgboost": 1.},
               "select_stack": {"ols": 0., "ridge": 0., "lasso": .4, "xgboost": .6},
               "gam_equal": {m: 1 / 3 for m in FUSION_MODELS},
               "gam_stack": {"ridge": .1, "gam": .3, "xgboost": .6}}
    result = []
    for year in (2031, 2032):
        result.append({"outer_year": year, "train_end": f"{year - 1}-12-31", "test_start": f"{year}-07-01",
                       "fusion_weights": deepcopy(weights), "fusion_audit": {"oof_event_mae": scores.copy(),
                       "ranked_models": ["xgboost", "lasso", "ridge", "ols"], "selected_models": ["xgboost", "lasso"],
                       "selected_best_model": "xgboost", "top_k": 2},
                       "base_models": {m: {"columns": ["year", "station=北門:hour=21"], "groups": ["year"]} for m in OLD_BASE_MODELS}})
    return {variant: deepcopy(result) for variant in VARIANTS}


class PairedCountsTests(unittest.TestCase):
    def test_event_totals_then_mean_uses_same_sample_and_units(self):
        table = mean_event_counts(frames())
        for scope, actual in (("all", 4800.), ("beimen", 1200.)):
            part = table.query("variant == 'both' and scope == @scope").set_index("model")
            self.assertEqual(part.loc["y", "mean_event_origin_trips"], actual)
            self.assertAlmostEqual(part.loc["gam_stack", "mean_event_origin_trips"], actual * .8)
            self.assertAlmostEqual(part.loc["gam", "mean_bias_pct"], -20.)
            self.assertTrue(part.events.eq(2).all())
        self.assertEqual(len(table), 4 * 2 * 12)

    def test_missing_common_panel_or_new_model_never_silently_compared(self):
        with self.assertRaisesRegex(ValueError, "incomplete station/hour"):
            mean_event_counts({v: f.iloc[:-1] for v, f in frames().items()})
        with self.assertRaisesRegex(ValueError, "Missing prediction columns"):
            validate_panel(frames()["core"].drop(columns="gam"))
        f = frames()["core"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_panel(pd.concat([f, f.iloc[:1]]))

    def test_changed_truth_and_cutoff_are_not_comparable(self):
        left, right = frames()["core"], frames()["both"]
        right.loc[0, "y"] += 1
        with self.assertRaisesRegex(ValueError, "different truths"):
            align_panels(left, right)
        right = frames()["both"]
        right["train_end"] = pd.to_datetime(right.train_end) - pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "different training cutoffs"):
            align_panels(left, right)

    def test_loss_and_true_peak_error_use_new_methods(self):
        frame = frames()["core"]
        summary = metrics_table(frame).query("scope == 'all' and model == 'gam'").iloc[0]
        self.assertAlmostEqual(summary.mae, 40.)
        self.assertAlmostEqual(summary.rmse, np.sqrt((20**2 + 60**2) / 2))
        frame.loc[frame.station.eq("北門") & frame.hour.eq(18), "gam"] = frame.y * .5
        frame.loc[frame.station.eq("北門") & frame.hour.eq(19), "gam"] = frame.y * 2
        peak = metrics_table(frame).query("scope == 'beimen' and model == 'gam'").iloc[0]
        self.assertAlmostEqual(peak.peak_absolute_error_mean, 100.)
        self.assertAlmostEqual(peak.peak_hour_mae, 1.)


class ManifestTests(unittest.TestCase):
    def test_feature_ablations_keep_same_sources_and_methods(self):
        checks = validate_manifests(manifests())
        self.assertTrue(checks["same_settings_except_feature_groups"])
        self.assertEqual(len(checks["d1_source_hashes"]), 2)

    def test_changed_smoothing_grid_code_or_cutoff_rejected(self):
        for change in (
            lambda m: m["both"]["config"].update(gam_lambdas=[1, 100]),
            lambda m: m["both"]["code"][0].update(sha256="changed"),
            lambda m: m["both"]["data"]["manifest"][0].update(sha256="changed"),
            lambda m: m["both"]["config"].update(d1_cutoff_hour=23),
        ):
            data = manifests()
            change(data)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_manifests(data)

    def test_historical_eight_reproduce_before_new_results_published(self):
        source = frames()["both"]
        audit = validate_old_reproduction(source, source, manifests()["both"], old_manifest())
        self.assertEqual(set(audit["max_abs_difference"]), set(OLD_MODELS))
        self.assertTrue(all(v == 0 for v in audit["max_abs_difference"].values()))
        source.loc[0, "ridge"] += .001
        with self.assertRaisesRegex(ValueError, "ridge predictions failed reproduction"):
            validate_old_reproduction(source, frames()["both"], manifests()["both"], old_manifest())

    def test_old_setting_source_or_code_change_cannot_masquerade_as_gam_effect(self):
        for change in (
            lambda m: m["config"].update(seed=9),
            lambda m: m["config"].update(unknown_extra_rule=1),
            lambda m: m["data"]["manifest"][0].update(sha256="changed"),
            lambda m: next(c for c in m["code"] if c["path"] == "src/analysis_v5/models.py").update(sha256="changed"),
        ):
            data = manifests()["both"]
            change(data)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_old_reproduction(frames()["both"], frames()["both"], data, old_manifest())


class FusionTests(unittest.TestCase):
    def test_three_equal_and_learned_weights_are_distinct_from_old_four(self):
        table = fusion_table(folds())
        self.assertEqual(len(table), 4 * 2 * (4 * 4 + 2 * 3))
        part = table.query("variant == 'both' and fusion == 'gam_stack'")
        self.assertEqual(set(part.base_model), set(FUSION_MODELS))
        self.assertTrue(part.loc[part.base_model.eq("gam"), "weight"].eq(.3).all())
        validate_fusion_predictions(validate_panel(frames()["core"]), folds()["core"])

    def test_three_equal_weight_formula_checked_against_actual_predictions(self):
        frame = validate_panel(frames()["core"])
        frame.loc[0, "gam_equal"] += 1
        with self.assertRaisesRegex(ValueError, "predictions do not match"):
            validate_fusion_predictions(frame, folds()["core"])
        source = folds()
        source["both"][0]["fusion_weights"]["gam_equal"].update(ridge=.5, gam=.2, xgboost=.3)
        with self.assertRaisesRegex(ValueError, "each be one third"):
            fusion_table(source)

    def test_invalid_weights_members_or_future_training_blocked(self):
        for change in (
            lambda f: f["fusion_weights"]["gam_stack"].update(gam=-.1, ridge=.5),
            lambda f: f["fusion_weights"]["gam_stack"].update(lasso=0),
            lambda f: f.update(train_end=f["test_start"]),
        ):
            source = folds()
            change(source["both"][0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                fusion_table(source)

    def test_effect_display_rejects_extrapolation_and_missing_provenance(self):
        curves = pd.DataFrame({"outer_year": [2026, 2026], "feature": ["forecast_temperature_c"] * 2,
                               "value": [10., 30.], "effect": [-100, 100], "training_min": [15, 15],
                               "training_max": [30, 30], "training_events": [12, 12]})
        with self.assertRaisesRegex(ValueError, "outside training support"):
            effect_figure(curves, Path("unused"))
        with self.assertRaisesRegex(ValueError, "Missing GAM effect curve provenance"):
            effect_figure(curves.drop(columns="training_events"), Path("unused"))


if __name__ == "__main__":
    unittest.main()
