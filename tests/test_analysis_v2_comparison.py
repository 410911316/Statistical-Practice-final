"""Comparison contracts: unchanged targets, paired true peaks, event balance."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis_v1.comparison import align_predictions, comparison_tables, write_comparison
from analysis_v1.models import ALL_MODELS


def paired_frames():
    rows = []
    for date, truths in (("2031-07-01", [40., 100., 100.]),
                         ("2032-07-01", [80., 60., 40.])):
        for station in ("北門", "其他站"):
            for hour, truth in zip((18, 19, 20), truths):
                rows.append({"event_date": pd.Timestamp(date), "station": station,
                             "hour": hour, "y": truth,
                             **{model: truth - 20. for model in ALL_MODELS}})
    old = pd.DataFrame(rows)
    new = old.copy()
    new["ridge"] = new.y - 5.
    new["xgboost"] = new.y + 30.
    return old, new


class PairedComparisonTests(unittest.TestCase):
    def test_accepts_shuffled_identical_keys_but_does_not_modify_inputs(self):
        old, new = paired_frames()
        original = new.copy(deep=True)
        shuffled = new.sample(frac=1, random_state=9)
        shuffled["event_date"] = shuffled.event_date.dt.strftime("%Y-%m-%d")
        left, right = align_predictions(old, shuffled)
        pd.testing.assert_frame_equal(left[["event_date", "station", "hour", "y"]],
                                      right[["event_date", "station", "hour", "y"]])
        pd.testing.assert_frame_equal(new, original)

    def test_rejects_changed_keys_targets_and_duplicates(self):
        old, new = paired_frames()
        changed_key = new.copy()
        changed_key.loc[0, "hour"] = 17
        changed_target = new.copy()
        changed_target.loc[0, "y"] += .01
        for frame, message in ((new.iloc[:-1], "keys differ"),
                               (changed_key, "keys differ"),
                               (changed_target, "target values differ"),
                               (pd.concat([new, new.iloc[[0]]]), "duplicate")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                align_predictions(old, frame)

    def test_rejects_nonfinite_negative_or_nonchronological_predictions(self):
        old, new = paired_frames()
        for value, message in ((np.nan, "nonfinite"), (-1., "nonnegative")):
            frame = new.copy()
            frame.loc[0, "ridge"] = value
            with self.assertRaisesRegex(ValueError, message):
                align_predictions(old, frame)
        new["train_end"] = new.event_date
        with self.assertRaisesRegex(ValueError, "training must end"):
            align_predictions(old, new)

    def test_actual_peak_same_hour_earliest_tie_and_signed_overprediction(self):
        old, new = paired_frames()
        # New maximum is deliberately at a different hour than the true peak.
        mask = new.station.eq("北門") & new.event_date.eq(pd.Timestamp("2031-07-01"))
        new.loc[mask, "ridge"] = [0., 80., 150.]
        tables = comparison_tables(old.iloc[::-1], new.sample(frac=1, random_state=2))
        peaks = tables["beimen_event_peaks"]
        ridge = peaks.loc[peaks.model.eq("ridge") & peaks.event_date.eq(pd.Timestamp("2031-07-01"))].iloc[0]
        self.assertEqual(ridge.actual_peak_hour, 19)
        self.assertEqual(ridge.prediction_at_actual_peak_after, 80.)
        self.assertEqual(ridge.predicted_peak_value_after, 150.)
        self.assertEqual(ridge.predicted_peak_hour_after, 20)
        self.assertEqual(ridge.peak_hour_error_after, 1)
        xgb = peaks.loc[peaks.model.eq("xgboost")].iloc[0]
        self.assertEqual(xgb.signed_underprediction_after, -30.)
        self.assertEqual(xgb.positive_underprediction_after, 0.)
        self.assertEqual(xgb.signed_underprediction_pct_after, -30.)
        summary = tables["beimen_peak_summary"].set_index("model").loc["xgboost"]
        # Less positive underprediction can coexist with worse absolute error.
        self.assertEqual(summary.events_positive_under_reduced, 2)
        self.assertEqual(summary.events_absolute_peak_error_reduced, 0)

    def test_metrics_equal_weight_events_not_number_of_rows(self):
        old, new = paired_frames()
        # Retain six rows in the easy event and one in the harder event.
        keep = old.event_date.eq(pd.Timestamp("2031-07-01")) | (
            old.station.eq("北門") & old.hour.eq(18))
        old, new = old.loc[keep].copy(), new.loc[keep].copy()
        old["ridge"] = old.y + np.where(old.event_date.dt.year.eq(2031), 1., 10.)
        new["ridge"] = new.y + 2.
        change = comparison_tables(old, new)["metrics_change"]
        row = change.loc[change.scope.eq("all_stations") & change.model.eq("ridge")].iloc[0]
        self.assertAlmostEqual(row.mae_before, 5.5)
        self.assertAlmostEqual(row.mae_after, 2.)
        self.assertAlmostEqual(row.mae_change, -3.5)
        self.assertAlmostEqual(row.mae_reduction_pct, 100 * (1 - 2 / 5.5))
        self.assertAlmostEqual(row.rmse_before, np.sqrt(50.5))
        self.assertEqual(row.events, 2)

    def test_writer_generates_only_aggregate_outputs_and_honest_report(self):
        old, new = paired_frames()
        saved = []

        def capture(figure, path):
            saved.append(path.name)
            plt.close(figure)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("analysis_v1.comparison.style"), \
                 patch("analysis_v1.comparison.save", side_effect=capture):
                tables = write_comparison(old, new, output)
            self.assertEqual(len(list((output / "tables").glob("*.csv"))), 4)
            self.assertEqual(set(saved), {"beimen_profiles_ridge.png", "beimen_profiles_xgboost.png",
                                          "beimen_peak_underprediction.png", "metrics_before_after.png"})
            self.assertEqual(len(tables["beimen_event_peaks"]), 2 * len(ALL_MODELS))
            text = (output / "report.md").read_text(encoding="utf-8")
            for expected in ("2031、2032 年、2 場、12 列", "回顧式模型修訂比較",
                             "Ridge**：北門 MAE 20.00 → 5.00（下降）",
                             "XGBoost**：北門 MAE 20.00 → 30.00（上升）",
                             "2/2 場正向低估減少，0/2 場尖峰絕對誤差減少",
                             "來源時段起站人次"):
                self.assertIn(expected, text)
            self.assertFalse((output / "outer_predictions.csv").exists())


if __name__ == "__main__":
    unittest.main()
