"""Report metadata and held-out scoring contracts, using synthetic data only."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis_v1.models import ALL_MODELS, metrics_table  # noqa: E402
from analysis_v1.reporting import build_model_figures, write_report  # noqa: E402


def report_inputs(years=(2031, 2032)):
    """Deliberately differ from the real 2024--2026, nine-event run."""
    events = 2 * len(years)
    metrics = pd.DataFrame([
        {"model": model, "events": events, "rows": events * 24,
         "mae": {"baseline": 10., "equal": 1.}.get(model, 2.),
         "rmse": 2. if model == "ols" else 3.,
         "peak_underprediction_mean": .5 if model == "xgboost" else 2.,
         "peak_hour_mae": .25}
        for model in ALL_MODELS
    ]).sort_values(["mae", "model"]).reset_index(drop=True)
    folds = [
        {"outer_year": year, "train_events": i + 3, "test_events": 2,
         "base_models": {"ols": {"groups": ["is_weekend"] if i else []},
                         "ridge": {"params": {"lambda": .37}}},
         "weights": {"ols": .1, "ridge": .2, "xgboost": .7}}
        for i, year in enumerate(years)
    ]
    info = {"activity_counts": {
        "by_year": {str(min(years) - 1): 1, **{str(year): 2 for year in years}},
        "held_dates": events + 1, "rows": (events + 1) * 24,
        "context_dates": events + 3,
    }}
    queue = {"event_date": f"{max(years)}-11-02", "station": "雙連",
             "forecast_model": "ridge", "directions": {"北向": .75, "南向": .25},
             "walk_delay_minutes": 9,
             "interpretation": "合成情境：可用容量 173 人、班距 7 分鐘。"}
    return metrics, folds, info, queue


class DynamicReportTest(unittest.TestCase):
    def test_small_nonzero_ridge_penalty_is_not_reported_as_zero(self):
        metrics, folds, info, queue = report_inputs((2031,))
        folds[0]["base_models"]["ridge"]["params"]["lambda"] = .001
        with tempfile.TemporaryDirectory() as directory:
            write_report(Path(directory), metrics, folds, info, queue)
            text = (Path(directory) / "report.md").read_text(encoding="utf-8")
        self.assertIn("| 2031 | 3 | 2 | 無 | 0.001 |", text)

    def test_report_uses_supplied_years_counts_queue_and_winners(self):
        metrics, folds, info, queue = report_inputs()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_report(output, metrics, folds, info, queue)
            text = (output / "report.md").read_text(encoding="utf-8")

        for expected in (
            "2030–2032 年 **5 場**", "合計 **120 列**", "共 7 個日期",
            "**2031、2032 年 4 場、96 列**", "測試僅 2 個年度",
            "等權平均：1.00", "Forward OLS：2.00", "XGBoost：0.50 人次",
            "相較降低 90.0%", "## 4. 2032 年全部外層預測",
            "figures/heldout_profiles_latest.png", "2032-11-02 雙連 外層 ridge",
            "北向 75%／南向 25%", "步行 9 分鐘", queue["interpretation"],
            "| 2031 | 3 | 2 | 無 | 0.37 | 0.10 | 0.20 | 0.70 |",
            "| 2032 | 4 | 2 | is_weekend | 0.37 | 0.10 | 0.20 | 0.70 |",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, text)
        for obsolete in ("2026", "**21 場**", "**504 列**", "9 場、216 列",
                         "60%", "40%", "步行 5 分鐘", "heldout_profiles_2026"):
            with self.subTest(obsolete=obsolete):
                self.assertNotIn(obsolete, text)
        # These qualifications must survive regenerating the actual numbers.
        self.assertIn("尚未核實真實 D−1 可用性", text)
        self.assertIn("不能保證未來活動也最佳", text)
        self.assertIn("不是已估計的真實候車或最佳列車時刻表", text)

    def test_one_test_year_and_absent_optional_interpretation(self):
        metrics, folds, info, queue = report_inputs((1998,))
        queue.pop("interpretation")
        queue.update(event_date="1998-11-02", station="民權西路",
                     forecast_model="equal", walk_delay_minutes=0,
                     directions={"A": .2, "B": .8})
        with tempfile.TemporaryDirectory() as directory:
            write_report(Path(directory), metrics, folds, info, queue)
            text = (Path(directory) / "report.md").read_text(encoding="utf-8")
        for expected in ("1997–1998 年 **3 場**", "合計 **72 列**",
                         "**1998 年 2 場、48 列**", "測試僅 1 個年度",
                         "## 4. 1998 年全部外層預測",
                         "1998-11-02 民權西路 外層 equal", "A 20%／B 80%",
                         "步行 0 分鐘"):
            with self.subTest(expected=expected):
                self.assertIn(expected, text)
        self.assertNotIn("2026", text)
        self.assertNotIn("合成情境", text)

    def test_figure_titles_and_latest_profile_name_use_prediction_years(self):
        records = []
        for year in (2031, 2032):
            for station in ("北門", "大橋頭站", "雙連", "民權西路"):
                for hour in range(18, 24):
                    y = float(100 + (hour - 18) * 5)
                    records.append({"event_date": pd.Timestamp(year, 7, 15),
                                    "year": year, "station": station, "hour": hour,
                                    "y": y, **{m: y + 2. for m in ALL_MODELS}})
        predictions = pd.DataFrame(records)
        metrics = metrics_table(predictions)
        _, folds, _, _ = report_inputs()
        vif = pd.DataFrame([{"outer_year": 2032, "model": "ridge",
                             "column": "baseline_hourly_56d", "vif": 2.}])
        saved = {}

        def capture_save(figure, path):
            saved[path.name] = {
                "title": figure._suptitle.get_text() if figure._suptitle else "",
                "panels": [axis.get_title() for axis in figure.axes],
            }
            plt.close(figure)

        with tempfile.TemporaryDirectory() as directory:
            with patch("analysis_v1.reporting.style"), \
                 patch("analysis_v1.reporting.save", side_effect=capture_save):
                build_model_figures(predictions, metrics, folds, vif, Path(directory))
        self.assertEqual(saved["model_comparison.png"]["title"],
                         "共同外層測試：2031、2032 年 2 場活動（回顧式驗證）")
        latest = saved["heldout_profiles_latest.png"]
        self.assertEqual(latest["title"], "2032 年全部 1 場外層預測｜未將當年真值交給訓練程序")
        self.assertEqual(len(latest["panels"]), 4)
        self.assertTrue(all("2032-07-15" in title for title in latest["panels"]))
        self.assertNotIn("heldout_profiles_2026.png", saved)
        self.assertNotIn("2026", str(saved))


class HeldOutScoringTest(unittest.TestCase):
    def test_mae_rmse_and_bias_give_each_activity_equal_weight(self):
        # Nine easy rows versus one hard row distinguish event from row means.
        dates = [pd.Timestamp("2031-07-01")] * 9 + [pd.Timestamp("2032-07-01")]
        frame = pd.DataFrame({"event_date": dates, "station": "A",
                              "hour": list(range(9, 18)) + [18], "y": 20.})
        for model in ALL_MODELS:
            frame[model] = [21.] * 9 + [10.]
        scores = metrics_table(frame)
        self.assertTrue(scores.events.eq(2).all())
        self.assertTrue(scores.rows.eq(10).all())
        np.testing.assert_allclose(scores.mae, (1. + 10.) / 2)
        np.testing.assert_allclose(scores.rmse, np.sqrt((1. + 100.) / 2))
        np.testing.assert_allclose(scores.bias_pred_minus_actual, (1. - 10.) / 2)
        self.assertFalse(np.isclose(scores.iloc[0].mae, (9. + 10.) / 10))

    def test_peak_underprediction_is_positive_part_at_actual_peak(self):
        # An overprediction at B's actual peak contributes zero, not -20.
        frame = pd.DataFrame({
            "event_date": pd.to_datetime(["2031-07-01"] * 4 + ["2032-07-01"] * 3),
            "station": "A", "hour": [18, 19, 20, 21, 18, 19, 20],
            "y": [40., 100., 100., 50., 80., 60., 40.],
        })
        for model in ALL_MODELS:
            frame[model] = [60., 80., 120., 20., 100., 30., 50.]
        scores = metrics_table(frame)
        np.testing.assert_allclose(scores.peak_underprediction_mean, 10.)
        np.testing.assert_allclose(scores.peak_hour_mae, .5)

    def test_tied_peaks_choose_earliest_hour_independent_of_input_order(self):
        frame = pd.DataFrame({
            "event_date": pd.Timestamp("2032-07-01"),
            "station": ["A"] * 3 + ["B"] * 3, "hour": [18, 19, 20] * 2,
            "y": [100., 100., 20., 100., 30., 20.],
        })
        for model in ALL_MODELS:
            frame[model] = [0., 80., 80., 20., 80., 80.]
        ordered = metrics_table(frame)
        reversed_rows = metrics_table(frame.iloc[::-1].reset_index(drop=True))
        np.testing.assert_allclose(ordered.peak_underprediction_mean, 90.)
        np.testing.assert_allclose(ordered.peak_hour_mae, 1.)
        pd.testing.assert_frame_equal(ordered, reversed_rows)


if __name__ == "__main__":
    unittest.main()
