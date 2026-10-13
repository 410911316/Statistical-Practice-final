"""Leakage, gate generalization, complementarity, and paired Beimen scoring."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis_v10.experiment import (
    MODELS, checked_predictions, fit_weights, gate_from_past, apply_decision,
    complementary_errors, evaluate_beimen,
)

CONFIG = json.loads((ROOT / "configs/analysis_v10.json").read_text(encoding="utf-8"))


def panel(year_counts=((2021, 4), (2022, 3)), stations=("北門",)):
    rows = []
    for year, count in year_counts:
        for day in range(1, count + 1):
            for station in stations:
                for hour in range(18, 24):
                    rows.append(dict(event_date=pd.Timestamp(year, 7, day), year=year,
                                     station=station, hour=hour, y=100.,
                                     ridge=100., gam=140., xgboost=120.,
                                     base_train_end=pd.Timestamp(year-1, 8, 1)))
    return pd.DataFrame(rows)


class ChronologyTests(unittest.TestCase):
    def test_past_complete_panel_accepted(self):
        data = checked_predictions(panel(), cutoff_year=2023)
        self.assertEqual(len(data), 42)

    def test_outer_year_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, "Outer labels"):
            checked_predictions(panel(), cutoff_year=2022)

    def test_same_year_base_training_rejected(self):
        data = panel()
        data["base_train_end"] = data.event_date - pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "leakage"):
            checked_predictions(data)

    def test_duplicate_incomplete_and_missing_rejected(self):
        data = panel()
        missing = data.copy()
        missing.loc[0, "gam"] = np.nan
        for invalid in (pd.concat([data, data.iloc[:1]]), data.iloc[1:], missing):
            with self.assertRaises(ValueError):
                checked_predictions(invalid)

    def test_bad_year_and_nonfinite_rejected(self):
        for column, value in (("year", 2001), ("ridge", np.inf), ("gam", -1)):
            data = panel()
            data.loc[0, column] = value
            with self.assertRaises(ValueError):
                checked_predictions(data)

    def test_event_requires_common_base_cutoff(self):
        data = panel()
        data.loc[0, "base_train_end"] = pd.Timestamp("2019-01-01")
        with self.assertRaisesRegex(ValueError, "cutoff"):
            checked_predictions(data)


class GateTests(unittest.TestCase):
    def test_insufficient_history_falls_back_even_if_weights_fit_perfectly(self):
        decision, valid = gate_from_past(panel(((2021, 2), (2022, 3))), "beimen", 2023, CONFIG)
        self.assertEqual(decision["chosen"], "xgboost")
        self.assertEqual(decision["reason"], "insufficient_meta_validation")
        self.assertTrue(valid.empty)
        self.assertAlmostEqual(decision["final_oof_mae"], 0)

    def test_heldout_improvement_adopts_fusion(self):
        decision, valid = gate_from_past(panel(), "beimen", 2023, CONFIG)
        self.assertEqual(decision["chosen"], "fusion")
        self.assertEqual(decision["validation_events"], 3)
        self.assertAlmostEqual(valid.fusion_mae.mean(), 0)
        self.assertAlmostEqual(valid.xgb_mae.mean(), 20)
        fold = decision["validation_folds"][0]
        self.assertEqual(fold["weight_train_events"], 4)
        self.assertLess(pd.Timestamp(fold["weight_train_end"]), pd.Timestamp(fold["validation_start"]))

    def test_fitted_weights_can_fail_on_later_validation(self):
        data = panel()
        data.loc[data.year.eq(2022), "ridge"] = 160.
        decision, valid = gate_from_past(data, "beimen", 2023, CONFIG)
        self.assertEqual(decision["chosen"], "xgboost")
        self.assertAlmostEqual(valid.fusion_mae.mean(), 60)

    def test_tie_keeps_xgboost(self):
        data = panel()
        data.loc[data.year.eq(2022), "ridge"] = 120.
        decision, _ = gate_from_past(data, "beimen", 2023, CONFIG)
        self.assertEqual(decision["chosen"], "xgboost")
        self.assertEqual(decision["reason"], "xgboost_better_or_tied")

    def test_not_enough_validation_events_keeps_xgboost(self):
        decision, _ = gate_from_past(panel(((2021,4),(2022,2))), "beimen", 2023, CONFIG)
        self.assertEqual(decision["reason"], "insufficient_meta_validation")

    def test_validation_truth_does_not_change_earlier_weight_fit(self):
        data = panel()
        before, _ = gate_from_past(data, "beimen", 2023, CONFIG)
        data.loc[data.year.eq(2022), "y"] = 140.
        after, _ = gate_from_past(data, "beimen", 2023, CONFIG)
        self.assertEqual(before["validation_folds"], after["validation_folds"])
        self.assertEqual(before["chosen"], "fusion")
        self.assertEqual(after["chosen"], "xgboost")

    def test_prediction_application_ignores_outer_truth(self):
        decision, _ = gate_from_past(panel(), "beimen", 2023, CONFIG)
        outer = panel(((2023,1),))
        expected = apply_decision(outer, decision)
        outer["y"] = 999999.
        np.testing.assert_array_equal(apply_decision(outer, decision), expected)
        np.testing.assert_array_equal(apply_decision(outer.drop(columns="y"), decision), expected)

    def test_beimen_weights_ignore_other_station_labels(self):
        data = panel(stations=("北門", "大橋頭站", "雙連", "民權西路"))
        first, _ = fit_weights(data, "beimen")
        data.loc[data.station.ne("北門"), "y"] = 140.
        second, _ = fit_weights(data, "beimen")
        pooled, _ = fit_weights(data, "all")
        np.testing.assert_allclose(first, second, atol=1e-10)
        self.assertGreater(np.linalg.norm(first-pooled), .5)

    def test_invalid_limits_scope_and_decisions_rejected(self):
        for key in ("gate_min_weight_training_events", "gate_min_validation_events", "gate_min_validation_years"):
            with self.assertRaises(ValueError):
                gate_from_past(panel(), "beimen", 2023, {**CONFIG, key:0})
        with self.assertRaises(ValueError):
            fit_weights(panel(), "unknown")
        for decision in ({"chosen":"oracle"}, {"chosen":"fusion","final_weights":dict(zip(MODELS, [.5,.5,.5]))}):
            with self.assertRaises(ValueError):
                apply_decision(panel(), decision)


class ComplementarityTests(unittest.TestCase):
    def fixture(self):
        frame = panel(((2021,1),))
        error = np.array([-30,-20,-10,10,20,30.])
        frame["ridge"] = 100-error
        frame["gam"] = 100+2*error
        frame["xgboost"] = 100+error
        return frame

    def test_opposite_and_same_direction_known_values(self):
        pairs, _ = complementary_errors(self.fixture())
        opposite = pairs.loc[pairs.model_a.eq("ridge") & pairs.model_b.eq("xgboost")].iloc[0]
        same = pairs.loc[pairs.model_a.eq("gam") & pairs.model_b.eq("xgboost")].iloc[0]
        self.assertAlmostEqual(opposite.error_correlation, -1)
        self.assertAlmostEqual(opposite.opposite_sign_fraction, 1)
        self.assertAlmostEqual(opposite.pair_equal_mae, 0)
        self.assertAlmostEqual(same.error_correlation, 1)
        self.assertAlmostEqual(same.opposite_sign_fraction, 0)
        self.assertAlmostEqual(same.pair_equal_mae, 30)

    def test_opposite_sign_not_same_as_individually_more_accurate(self):
        _, help_rows = complementary_errors(self.fixture())
        ridge = help_rows.set_index("candidate").loc["ridge"]
        self.assertEqual(ridge.xgb_under_rows, 3)
        self.assertAlmostEqual(ridge.candidate_closer_when_xgb_under, 0)
        self.assertAlmostEqual(ridge.pair_blend_better_when_xgb_under, 1)
        self.assertAlmostEqual(ridge.pair_blend_mae_minus_xgb, -20)

    def test_event_centering_removes_shared_event_offsets(self):
        frame = panel(((2021,2),))
        error = np.tile([-3,-2,-1,1,2,3], 2)
        offset = np.repeat([-30,30], 6)
        frame["ridge"] = 100+error+offset
        frame["gam"] = 100-error+offset
        pairs, _ = complementary_errors(frame)
        pair = pairs.iloc[0]
        self.assertGreater(pair.error_correlation, .9)
        self.assertAlmostEqual(pair.within_event_error_correlation, -1)

    def test_constant_errors_and_no_underestimation_are_unknown_not_zero(self):
        pairs, help_rows = complementary_errors(panel(((2021,1),)))
        self.assertTrue(pairs.error_correlation.isna().all())
        self.assertTrue(help_rows.candidate_closer_when_xgb_under.isna().all())


class ScoringTests(unittest.TestCase):
    def test_rmse_is_root_of_mean_mse_and_peak_is_same_actual_hour(self):
        frame = panel(((2024,2),))
        frame["pred"] = frame.y + np.repeat([0.,10.],6)
        scores, _ = evaluate_beimen(frame, ["pred"])
        self.assertAlmostEqual(scores.iloc[0].rmse, np.sqrt(50))
        frame = panel(((2024,1),))
        frame["y"] = [1,10,2,3,4,5]
        frame["pred"] = [1,4,9,3,4,5]
        scores, _ = evaluate_beimen(frame, ["pred"])
        self.assertEqual(scores.iloc[0].peak_underprediction, 6)
        self.assertEqual(scores.iloc[0].peak_hour_mae, 1)

    def test_peak_overestimate_has_zero_underestimate_but_positive_absolute_error(self):
        frame = panel(((2024,1),))
        frame["pred"] = 130.
        scores, _ = evaluate_beimen(frame, ["pred"])
        self.assertEqual(scores.iloc[0].peak_underprediction, 0)
        self.assertEqual(scores.iloc[0].peak_absolute_error, 30)

    def test_nonpaired_scores_rejected(self):
        data = panel()
        for invalid in (data.iloc[1:], pd.concat([data, data.iloc[:1]])):
            with self.assertRaises(ValueError):
                evaluate_beimen(invalid, ["xgboost"])


if __name__ == "__main__":
    unittest.main()

