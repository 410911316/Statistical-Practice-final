"""A new predictor cannot silently change samples, provenance or evaluation."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis_v6 import reporting as common
from analysis_v7 import reporting as repair
from analysis_v8 import reporting as report


def predictions():
    rows = []
    for date, truth in (("2031-07-01", 100.), ("2032-07-01", 300.)):
        for station, hour in product(report.STATIONS, report.HOURS):
            rows.append({"event_date": date, "year": int(date[:4]), "outer_year": int(date[:4]),
                         "station": station, "hour": hour, "train_end": f"{int(date[:4]) - 1}-12-31",
                         "y": truth, **{model: truth * .8 for model in report.ALL_MODELS}})
    return pd.DataFrame(rows)


def manifests(version="v8"):
    result = {}
    for variant in report.VARIANTS:
        config = {"version": version + variant, "include_historical_baseline": False, "d1_cutoff_hour": 18,
                  "station_hour_interaction": True, "linear_design": "full_shared_with_training_rank_filter",
                  "candidate_groups": common.CALENDAR + common.FEATURES[variant],
                  "d1_features": common.FEATURES[variant].copy(), "seed": 7, "fusion_top_k": 2,
                  "gam_lambdas": [.1, 1, 10], "gam_ridge_alphas": [.001, .01, .1, 1],
                  "gam_fusion_models": list(report.FUSION_MODELS)}
        sources = [{"path": "data/processed/source.csv", "sha256": "source"}]
        if variant in ("fireworks", "both") or version == "v8":
            sources.append({"path": "docs/d1_program_sources.json", "sha256": "program"})
        if variant in ("weather", "both"):
            sources.append({"path": "docs/d1_weather_sources.json", "sha256": "weather"})
        if version == "v8":
            config.update(candidate_groups=config["candidate_groups"] + list(report.END_FEATURES),
                          baseline_variant=variant, end_source_file=report.END_SOURCE, end_signal_policy=report.END_POLICY)
            sources.append({"path": report.END_SOURCE, "sha256": "end"})
        result[variant] = {"config": config, "packages": {"numpy": "2.2.6", "pygam": "0.11.0"},
                           "data": {"manifest": sources}, "code": [{"path": path, "sha256": "code"}
                           for path in sorted(report.FIT_CODE if version == "v8" else repair.FIT_CODE)]}
    return result


def inputs_and_audit():
    old = predictions()[["event_date", "year", "station", "hour", "y"]].copy()
    old["old_feature"] = 1.
    new = old.copy()
    new[list(report.END_FEATURES)] = np.nan
    known = new.event_date.eq("2031-07-01")
    for code, station in report.STATION_CODES.items():
        for start, end in ((0, 60), (60, 120)):
            column = f"planned_end_{start}_{end}_{code}"
            overlap = np.maximum(0., np.minimum((new.hour + 1) * 60, 1220 + end) - np.maximum(new.hour * 60, 1220 + start))
            new.loc[known, column] = (overlap * new.station.eq(station) / 60.).loc[known]
    audit = pd.DataFrame([
        {"event_date": "2031-07-01", "eligible": True, "scope": "fireworks_only", "planned_end_hour": 20 + 1 / 3,
         "cutoff_at": "2031-06-30T18:00:00+08:00", "effective_available_at": "2031-06-28T12:00:00+08:00"},
        {"event_date": "2032-07-01", "eligible": False, "scope": "unknown", "planned_end_hour": np.nan,
         "cutoff_at": "2032-06-30T18:00:00+08:00", "effective_available_at": "2032-07-02T12:00:00+08:00"},
    ])
    return new, old, audit


def oof():
    data = predictions().rename(columns={"train_end": "base_train_end"})
    data["outer_year"] = 2034
    return pd.concat([data, data.assign(outer_year=2035)], ignore_index=True)


class PairedDesignTests(unittest.TestCase):
    def test_all_models_may_change_when_shared_predictors_change(self):
        new = predictions()
        new[list(report.ALL_MODELS)] *= 1.2
        result = report.validate_pair(new, predictions(), manifests()["both"], manifests("v7")["both"])
        self.assertTrue(result["all_models_refit_with_new_inputs"])
        self.assertEqual(set(result["prediction_changes_allowed"]), set(report.ALL_MODELS))

    def test_extra_parent_provenance_is_allowed_for_core(self):
        report.validate_pair(predictions(), predictions(), manifests()["core"], manifests("v7")["core"])

    def test_reordered_candidates_or_changed_grid_source_code_packages_rejected(self):
        for mutate in (
            lambda m: m["config"]["candidate_groups"].reverse(),
            lambda m: m["config"].update(gam_ridge_alphas=[100]),
            lambda m: m["config"].update(unexpected_setting=True),
            lambda m: m["data"]["manifest"][0].update(sha256="changed"),
            lambda m: m["packages"].update(pygam="different"),
            lambda m: next(row for row in m["code"] if row["path"] == "src/analysis_v7/models.py").update(sha256="changed"),
        ):
            new = manifests()["both"]
            mutate(new)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                report.validate_pair(predictions(), predictions(), new, manifests("v7")["both"])

    def test_new_data_file_cannot_be_hidden_as_end_source(self):
        new = manifests()["both"]
        new["data"]["manifest"].append({"path": "future_observed_egress.csv", "sha256": "new"})
        with self.assertRaisesRegex(ValueError, "Existing data"):
            report.validate_pair(predictions(), predictions(), new, manifests("v7")["both"])

    def test_all_variants_must_share_sources_and_tuning(self):
        self.assertTrue(report.validate_manifests(manifests())["same_settings_except_candidate_groups"])
        for mutate in (
            lambda m: m["weather"]["config"].update(seed=8),
            lambda m: m["core"]["config"].update(baseline_variant="both"),
            lambda m: m["both"]["data"]["manifest"][-1].update(sha256="different"),
            lambda m: m.pop("core"),
        ):
            new = manifests()
            mutate(new)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                report.validate_manifests(new)

    def test_input_values_and_all_activity_rows_preserved(self):
        new, old, _ = inputs_and_audit()
        check = report.validate_input_pair(new, old)
        self.assertEqual((check["events"], check["rows"]), (2, 48))
        for mutate in (
            lambda f: f.loc.__setitem__((0, "old_feature"), 2.),
            lambda f: f.loc.__setitem__((0, "y"), 1000.),
        ):
            changed = new.copy()
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, "Existing input values"):
                report.validate_input_pair(changed, old)
        with self.assertRaisesRegex(ValueError, "Existing input values"):
            report.validate_input_pair(new.iloc[:-1], old)


class EndFeatureAuditTests(unittest.TestCase):
    def test_minutes_use_overlap_not_end_hour_or_observed_peak(self):
        new, _, audit = inputs_and_audit()
        check = report.validate_end_audit(new, audit, ["2031-07-01"])
        self.assertEqual(check["known_events"], 1)
        self.assertEqual(check["unknown_events"], 1)
        self.assertEqual(check["outer_known_events"], 1)
        bm = new.loc[new.event_date.eq("2031-07-01") & new.station.eq("北門")].set_index("hour")
        self.assertAlmostEqual(bm.loc[20, "planned_end_0_60_beimen"], 2 / 3)
        self.assertAlmostEqual(bm.loc[21, "planned_end_0_60_beimen"], 1 / 3)
        self.assertAlmostEqual(bm.loc[21, "planned_end_60_120_beimen"], 2 / 3)
        self.assertAlmostEqual(bm.loc[22, "planned_end_60_120_beimen"], 1 / 3)

    def test_future_announcement_or_wrong_cutoff_cannot_be_used(self):
        for column, value in (("effective_available_at", "2031-07-01T00:00:00+08:00"),
                              ("cutoff_at", "2031-06-30T23:00:00+08:00"),
                              ("effective_available_at", None)):
            new, _, audit = inputs_and_audit()
            audit.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                report.validate_end_audit(new, audit)

    def test_unknown_not_filled_zero_or_removed(self):
        new, _, audit = inputs_and_audit()
        new.loc[new.event_date.eq("2032-07-01"), report.END_FEATURES[0]] = 0.
        with self.assertRaisesRegex(ValueError, "Unknown show end"):
            report.validate_end_audit(new, audit)
        new, _, audit = inputs_and_audit()
        with self.assertRaisesRegex(ValueError, "every training event"):
            report.validate_end_audit(new, audit.iloc[:1])

    def test_wrong_station_or_bin_overlap_rejected(self):
        new, _, audit = inputs_and_audit()
        row = new.index[new.event_date.eq("2031-07-01") & new.hour.eq(20) & new.station.eq("雙連")][0]
        new.loc[row, "planned_end_0_60_beimen"] = 2 / 3
        with self.assertRaisesRegex(ValueError, "overlap predictor"):
            report.validate_end_audit(new, audit)

    def test_scope_cannot_be_relabelled_as_measured_egress(self):
        new, _, audit = inputs_and_audit()
        audit.loc[0, "scope"] = "actual_egress"
        with self.assertRaisesRegex(ValueError, "scope"):
            report.validate_end_audit(new, audit)


class OOFAndMetricsTests(unittest.TestCase):
    def test_past_oof_predictions_can_change_but_samples_cannot(self):
        new = oof()
        new[list(repair.OOF_MODELS)] *= 1.2
        self.assertEqual(report.validate_oof_pair(new, oof())["unique_events"], 2)
        new["base_train_end"] = "2029-01-01"
        with self.assertRaisesRegex(ValueError, "Past OOF"):
            report.validate_oof_pair(new, oof())

    def test_signed_positive_absolute_peaks_are_at_actual_peak(self):
        data = predictions()
        mask = data.station.eq("北門")
        data.loc[mask & data.hour.eq(20), "y"] = 1000.
        data.loc[mask & data.hour.eq(20), "gam"] = 1200.
        data.loc[mask & data.hour.eq(22), "gam"] = 1500.
        table = report.peak_table(data).query("model == 'gam'")
        self.assertTrue(table.actual_peak_hour.eq(20).all())
        self.assertTrue(table.predicted_peak_hour.eq(22).all())
        self.assertTrue(table.signed_underprediction.eq(-200.).all())
        self.assertTrue(table.positive_underprediction.eq(0.).all())
        self.assertTrue(table.peak_absolute_error.eq(200.).all())

    def test_paired_changes_include_signed_and_absolute_peak_error(self):
        original = predictions()
        before = common.metrics_table(original).assign(version="v7", variant="both")
        original["gam"] *= 1.1
        after = common.metrics_table(original).assign(version="v8", variant="both")
        changes = report.paired_changes(pd.concat([before, after], ignore_index=True))
        row = changes.query("scope == 'beimen' and model == 'gam'").iloc[0]
        self.assertAlmostEqual(row.mae_change_v8_minus_v7, -16.)
        self.assertIn("peak_signed_underprediction_mean_change_v8_minus_v7", changes)
        with self.assertRaisesRegex(ValueError, "Incomplete paired"):
            report.paired_changes(pd.concat([before, after.iloc[1:]], ignore_index=True))


if __name__ == "__main__":
    unittest.main()
