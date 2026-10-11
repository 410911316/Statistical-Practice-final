"""Numerical-repair comparisons cannot hide source changes or repeat OOF rows."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis_v6 import reporting as old
from analysis_v7 import reporting as report


def frame():
    rows = []
    for date, truth in (("2031-07-01", 100.), ("2032-07-01", 300.)):
        for station, hour in product(report.STATIONS, report.HOURS):
            rows.append({"event_date": date, "year": int(date[:4]), "outer_year": int(date[:4]),
                         "station": station, "hour": hour, "train_end": f"{int(date[:4]) - 1}-12-31",
                         "y": truth, **{model: truth * .8 for model in report.ALL_MODELS}})
    return pd.DataFrame(rows)


def manifests(version="v7"):
    result = {}
    for variant in report.VARIANTS:
        sources = [{"path": "data/processed/source.csv", "sha256": "source"}]
        if variant in ("fireworks", "both"):
            sources.append({"path": "docs/d1_program_sources.json", "sha256": "program"})
        if variant in ("weather", "both"):
            sources.append({"path": "docs/d1_weather_sources.json", "sha256": "weather"})
        result[variant] = {"config": {"version": version + variant, "include_historical_baseline": False,
                                    "d1_cutoff_hour": 18, "station_hour_interaction": True,
                                    "linear_design": "full_shared_with_training_rank_filter",
                                    "candidate_groups": old.CALENDAR + old.FEATURES[variant],
                                    "d1_features": old.FEATURES[variant].copy(), "seed": 7, "fusion_top_k": 2,
                                    "gam_lambdas": [.1, 1, 10], "gam_fusion_models": list(report.FUSION_MODELS)},
                           "packages": {"numpy": "2.2.6", "pygam": "0.11.0"}, "data": {"manifest": sources},
                           "code": [{"path": path, "sha256": "code"} for path in sorted(report.FIT_CODE if version == "v7" else old.FIT_CODE)]}
        if version == "v7":
            result[variant]["config"].update(gam_ridge_alphas=[.001, .01, .1, 1], gam_robustness_policy="repair")
    return result


def oof():
    source = frame().rename(columns={"train_end": "base_train_end"})
    source["outer_year"] = 2034
    repeat = source.copy()
    repeat["outer_year"] = 2035
    return pd.concat([source, repeat], ignore_index=True)


def diagnostics():
    config = manifests()["core"]["config"]
    config.update(gam_lambdas=[.1, 1], gam_ridge_alphas=[.01, .1], gam_binary_groups=["is_weekend"], gam_min_known_events=6)
    records = []
    for lam, alpha in product(config["gam_lambdas"], config["gam_ridge_alphas"]):
        records.append({"layer": "inner_grid", "parent_train_end": "2030-12-31", "train_end": "2029-12-31",
                        "validation_start": "2030-07-01", "validation_end": "2030-07-01", "validation_year": 2030,
                        "lambda": lam, "ridge_alpha": alpha, "train_events": 4, "validation_events": 1,
                        "event_mae": 20., "max_raw_prediction": 80., "min_raw_prediction": 80.,
                        "max_clipped_prediction": 80., "max_truth": 100., "n_prediction_rows": 24,
                        "normal_equation_relative_residual": 1e-9, "numerical_verified": True})
    records += [{**records[0], "layer": "middle_oof", "parent_train_end": "2032-12-31", "train_end": "2030-12-31",
                 "validation_start": "2031-07-01", "validation_end": "2031-07-01", "validation_year": 2031},
                {**records[0], "layer": "outer", "parent_train_end": "2031-12-31", "train_end": "2031-12-31",
                 "validation_start": "2032-07-01", "validation_end": "2032-07-01", "validation_year": 2032,
                 "train_events": 7, "event_mae": 60., "max_raw_prediction": 240., "min_raw_prediction": 240.,
                 "max_clipped_prediction": 240., "max_truth": 300.}]
    extra = []
    for row in records:
        for column in ("day_of_year", "year_index"):
            extra.append({**row, "column": column, "known_training_events": row["train_events"],
                          "active": row["train_events"] >= 6, "missing_indicator_active": False,
                          "support_gate_passed": row["train_events"] >= 6, "gate_uses_y": False})
    return pd.DataFrame(records), pd.DataFrame(extra), config


class RepairInvariantTests(unittest.TestCase):
    def test_repair_allows_only_gam_controls_and_preserves_old_eight(self):
        data = frame()
        data["gam"] *= 1.1
        checks = report.validate_repair_pair(data, frame(), manifests()["both"], manifests("v6")["both"])
        self.assertEqual(set(checks["legacy_max_abs_difference"]), set(report.OLD_MODELS))
        self.assertTrue(all(value == 0 for value in checks["legacy_max_abs_difference"].values()))
        self.assertIn("gam_ridge_alphas", checks["changed_gam_controls"])

    def test_changed_legacy_predictions_not_accepted_as_gam_repair(self):
        data = frame()
        data.loc[0, "xgboost"] += .01
        with self.assertRaisesRegex(ValueError, "Legacy xgboost"):
            report.validate_repair_pair(data, frame(), manifests()["both"], manifests("v6")["both"])

    def test_changed_data_settings_package_or_preprocessor_rejected(self):
        for change in (
            lambda m: m["config"].update(seed=10),
            lambda m: m["config"].update(new_unrelated_rule=True),
            lambda m: m["packages"].update(pygam="different"),
            lambda m: m["data"]["manifest"][0].update(sha256="different"),
            lambda m: next(item for item in m["code"] if item["path"] == "src/analysis_v1/data.py").update(sha256="different"),
        ):
            new = manifests()["both"]
            change(new)
            with self.subTest(change=change), self.assertRaises(ValueError):
                report.validate_repair_pair(frame(), frame(), new, manifests("v6")["both"])

    def test_truth_panel_or_cutoff_changes_rejected(self):
        for change in (
            lambda f: f.loc.__setitem__((0, "y"), 500),
            lambda f: f.__setitem__("train_end", "2030-01-01"),
        ):
            new = frame()
            change(new)
            with self.subTest(change=change), self.assertRaises(ValueError):
                report.validate_repair_pair(new, frame(), manifests()["both"], manifests("v6")["both"])

    def test_all_four_variants_require_identical_repair_and_eligible_information(self):
        self.assertTrue(report.validate_manifests(manifests())["same_settings_except_feature_groups"])
        for change in (
            lambda m: m["both"]["config"].update(gam_ridge_alphas=[1]),
            lambda m: m["weather"]["config"].update(d1_cutoff_hour=23),
            lambda m: m["core"]["config"].update(candidate_groups=["future_actual_weather"]),
            lambda m: m["fireworks"]["code"][0].update(sha256="changed"),
            lambda m: m["both"]["data"]["manifest"][1].update(sha256="changed"),
        ):
            data = manifests()
            change(data)
            with self.subTest(change=change), self.assertRaises(ValueError):
                report.validate_manifests(data)


class OOFHealthTests(unittest.TestCase):
    def test_outer_reuse_deduplicated_before_yearly_scores(self):
        unique = report.deduplicate_oof(oof())
        self.assertEqual(len(unique), 48)
        self.assertNotIn("outer_year", unique)
        table = report.oof_health_table(oof())
        row = table.query("oof_year == 2032 and scope == 'all' and model == 'gam'").iloc[0]
        self.assertEqual(row.events, 1)
        self.assertEqual(row.rows, 24)
        self.assertAlmostEqual(row.event_mae, 60.)
        self.assertAlmostEqual(row.max_prediction, 240.)

    def test_duplicate_within_outer_fit_is_an_error(self):
        data = oof()
        with self.assertRaisesRegex(ValueError, "Duplicate OOF activity"):
            report.deduplicate_oof(pd.concat([data, data.iloc[:1]]))

    def test_disagreeing_reused_predictions_are_not_silently_dropped(self):
        for column, value in (("gam", 900.), ("y", 900.), ("base_train_end", "2029-12-31")):
            data = oof()
            data.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "Repeated OOF forecast"):
                report.deduplicate_oof(data)

    def test_oof_uses_past_forecasts_and_complete_panels(self):
        for change in (
            lambda f: f.__setitem__("base_train_end", f.event_date),
            lambda f: f.__setitem__("outer_year", f.year),
            lambda f: f.loc.__setitem__((0, "gam"), np.inf),
        ):
            data = oof()
            change(data)
            with self.subTest(change=change), self.assertRaises(ValueError):
                report.deduplicate_oof(data)
        with self.assertRaisesRegex(ValueError, "Incomplete OOF"):
            report.deduplicate_oof(oof().loc[lambda f: ~f.station.eq("北門")])

    def test_paired_oof_checks_old_predictions_but_allows_repaired_gam(self):
        repaired = oof()
        repaired["gam"] *= 1.2
        audit = report.validate_oof_pair(repaired, oof())
        self.assertEqual(audit["events"], 2)
        repaired["ridge"] *= 1.01
        with self.assertRaisesRegex(ValueError, "OOF legacy ridge"):
            report.validate_oof_pair(repaired, oof())


class PeakAndDeltaTests(unittest.TestCase):
    def test_peak_error_is_at_actual_peak_and_reports_hour_miss(self):
        data = frame()
        mask = data.station.eq("北門")
        data.loc[mask & data.hour.eq(20), "y"] = 1000.
        data.loc[mask & data.hour.eq(20), "gam"] = 600.
        data.loc[mask & data.hour.eq(22), "gam"] = 1200.
        table = report.peak_table(data).query("model == 'gam'")
        self.assertTrue(table.actual_peak_hour.eq(20).all())
        self.assertTrue(table.predicted_peak_hour.eq(22).all())
        self.assertTrue(table.positive_underprediction.eq(400.).all())
        self.assertTrue(table.peak_hour_absolute_error.eq(2).all())

    def test_paired_metric_deltas_match_same_keys(self):
        data = frame()
        before = old.metrics_table(data).assign(version="v6", variant="both")
        data["gam"] *= 1.1
        after = old.metrics_table(data).assign(version="v7", variant="both")
        table = report.paired_changes(pd.concat([before, after], ignore_index=True))
        row = table.query("scope == 'beimen' and model == 'gam'").iloc[0]
        self.assertAlmostEqual(row.mae_change_v7_minus_v6, -16.)
        with self.assertRaisesRegex(ValueError, "Incomplete before/after"):
            report.paired_changes(pd.concat([before, after.iloc[1:]], ignore_index=True))
        with self.assertRaisesRegex(ValueError, "Duplicate paired"):
            report.paired_changes(pd.concat([before, after, after.iloc[:1]], ignore_index=True))


class AllLayerDiagnosticTests(unittest.TestCase):
    def test_every_grid_and_selected_fit_is_certified_and_inputs_audited(self):
        diagnostic, extra, config = diagnostics()
        checks = report.validate_diagnostics(diagnostic, extra, config, old.validate_panel(frame()), oof())
        self.assertEqual(checks["evaluations"], 6)
        self.assertEqual(checks["candidate_feature_audits"], 12)
        self.assertTrue(checks["unsupported_inputs_and_indicators_omitted"])

    def test_failed_fit_or_removed_candidate_cannot_be_hidden(self):
        for mutate in (
            lambda d: d.loc.__setitem__((0, "numerical_verified"), False),
            lambda d: d.loc.__setitem__((0, "normal_equation_relative_residual"), .01),
            lambda d: d.loc.__setitem__((0, "parent_train_end"), "2030-01-01"),
        ):
            diagnostic, extra, config = diagnostics()
            mutate(diagnostic)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                report.validate_diagnostics(diagnostic, extra, config, old.validate_panel(frame()), oof())
        diagnostic, extra, config = diagnostics()
        with self.assertRaisesRegex(ValueError, "grid audit omitted"):
            report.validate_diagnostics(diagnostic.iloc[1:], extra, config, old.validate_panel(frame()), oof())

    def test_unsupported_feature_or_indicator_must_be_absent(self):
        for column in ("active", "missing_indicator_active", "gate_uses_y"):
            diagnostic, extra, config = diagnostics()
            extra.loc[0, column] = True
            with self.subTest(column=column), self.assertRaises(ValueError):
                report.validate_diagnostics(diagnostic, extra, config, old.validate_panel(frame()), oof())
        diagnostic, extra, config = diagnostics()
        with self.assertRaisesRegex(ValueError, "every evaluation"):
            report.validate_diagnostics(diagnostic, extra.iloc[1:], config, old.validate_panel(frame()), oof())

    def test_selected_fit_diagnostic_must_match_prediction_file(self):
        diagnostic, extra, config = diagnostics()
        diagnostic.loc[diagnostic.layer.eq("outer"), "event_mae"] = 10.
        with self.assertRaisesRegex(ValueError, "disagree with saved"):
            report.validate_diagnostics(diagnostic, extra, config, old.validate_panel(frame()), oof())


if __name__ == "__main__":
    unittest.main()
