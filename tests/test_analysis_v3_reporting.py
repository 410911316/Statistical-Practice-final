"""Tuning plots use inner searches, and availability plots do not recode unknowns."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis_v1.d1_reporting import build_tuning_figures, coverage_table, tuning_tables  # noqa: E402


def tuning_fixture():
    rows, folds = [], []
    for year, scale in ((2024, 100.), (2025, 200.)):
        end = f"{year-1}-08-20"
        for depth in (2, 3):
            for trees in (50, 100):
                for penalty in (1., 10.):
                    params = {"max_depth": depth, "n_estimators": trees,
                              "min_child_weight": 1, "reg_lambda": penalty}
                    rows.append({"model": "xgboost", "train_end": end,
                                 "params": json.dumps(params), "groups": "[]",
                                 "inner_event_mae": scale + (3-depth)*20 + (100-trees)/5 + (10-penalty)})
        folds.append({"outer_year": year, "train_end": end,
                      "base_models": {"xgboost": {"params": {
                          "max_depth": 3, "n_estimators": 100,
                          "min_child_weight": 1, "reg_lambda": 10.}}}})
    # A later research fit has smaller MAE; it must never supply the 2024/25 plot.
    rows.append({"model": "xgboost", "train_end": "2025-08-20", "groups": "[]",
                 "params": rows[-1]["params"], "inner_event_mae": 1.})
    return pd.DataFrame(rows), folds


def coverage_fixture():
    rows = []
    for date in ("2024-07-17", "2024-07-31"):
        rows.append({"event_date": date, "hour": None, "feature": "planned_drone",
                     "value": 0. if date.endswith("17") else np.nan,
                     "eligible": date.endswith("17")})
        for hour in range(18, 24):
            rows.append({"event_date": date, "hour": hour, "feature": "forecast_temperature_c",
                         "value": 30. if hour == 18 else np.nan, "eligible": hour == 18})
    return pd.DataFrame(rows)


class D1ReportingTest(unittest.TestCase):
    def test_tuning_surface_matches_parent_training_end_and_profiles_other_parameters(self):
        choices, folds = tuning_fixture()
        surface, selected = tuning_tables(choices, folds)
        self.assertEqual(len(surface), 8)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected.inner_event_mae.tolist(), [100., 200.])
        self.assertEqual(selected.train_end.tolist(), ["2023-08-20", "2024-08-20"])
        self.assertTrue(surface.best_reg_lambda.eq(10.).all())
        self.assertTrue(surface.profiled_candidate_count.eq(2).all())
        self.assertTrue(surface.selection_scope.str.contains("not_outer_test").all())
        self.assertEqual(int(surface.selected_cell.sum()), 2)

    def test_reporting_refuses_to_explain_a_nonwinning_parameter_as_selected(self):
        choices, folds = tuning_fixture()
        folds[0]["base_models"]["xgboost"]["params"]["max_depth"] = 2
        with self.assertRaisesRegex(ValueError, "minimize"):
            tuning_tables(choices, folds)
        choices, folds = tuning_fixture()
        folds[0]["train_end"] = "2024-08-20"
        with self.assertRaisesRegex(ValueError, "precede"):
            tuning_tables(choices, folds)

    def test_coverage_counts_zero_as_known_and_six_hour_weather_denominator(self):
        coverage = coverage_table(coverage_fixture()).set_index(["event_date", "feature"])
        self.assertEqual(coverage.loc[("2024-07-17", "planned_drone"), "available_fraction"], 1.)
        self.assertEqual(coverage.loc[("2024-07-31", "planned_drone"), "available_fraction"], 0.)
        weather = coverage.loc[("2024-07-17", "forecast_temperature_c")]
        self.assertEqual(weather.expected_slots, 6)
        self.assertAlmostEqual(weather.available_fraction, 1/6)
        with self.assertRaisesRegex(ValueError, "6 weather"):
            coverage_table(coverage_fixture().iloc[:-1])

    def test_complete_public_figures_and_tables_are_written(self):
        choices, folds = tuning_fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            build_tuning_figures(choices, folds, output, coverage_fixture())
            for name in ("figures/xgb_tuning_heatmap.png", "figures/d1_feature_coverage.png",
                         "tables/xgb_tuning_surface.csv", "tables/xgb_selected_params.csv",
                         "tables/d1_feature_coverage.csv"):
                self.assertGreater((output/name).stat().st_size, 100)
            selected = pd.read_csv(output/"tables/xgb_selected_params.csv")
            self.assertEqual(selected.inner_event_mae.tolist(), [100., 200.])


if __name__ == "__main__":
    unittest.main()
