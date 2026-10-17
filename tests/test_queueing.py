from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from queueing import (
    compare_headway_scenarios,
    disaggregate_hourly,
    mmc_metrics,
    simulate_bulk_queue,
)


class QueueingTest(unittest.TestCase):
    def test_mmc_reference_example_and_instability(self):
        result = mmc_metrics(30, 20, 2)
        self.assertAlmostEqual(result["empty_probability"], 1 / 7)
        self.assertAlmostEqual(result["wait_probability"], 9 / 14)
        self.assertAlmostEqual(result["mean_wait_minutes"], 9 / 140)
        self.assertAlmostEqual(result["mean_queue"], 27 / 14)
        self.assertFalse(mmc_metrics(40, 20, 2)["stable"])
        self.assertIsNone(mmc_metrics(40, 20, 2)["mean_wait_minutes"])
        self.assertEqual(mmc_metrics(0, 20, 2)["mean_wait_minutes"], 0)

    def test_direction_and_walking_delay_preserve_every_hour(self):
        result = disaggregate_hourly([600, 1200], {"east": 0.75, "west": 0.25}, walk_delay_minutes=5)
        self.assertEqual(len(result["east"]), 125)
        self.assertEqual(result["east"][:5], [0] * 5)
        self.assertAlmostEqual(math.fsum(result["east"]), 1350)
        self.assertAlmostEqual(math.fsum(result["west"]), 450)
        self.assertAlmostEqual(sum(sum(v[5:65]) for v in result.values()), 600)
        self.assertAlmostEqual(sum(sum(v[65:125]) for v in result.values()), 1200)

    def test_custom_profile_not_silently_normalized(self):
        profile = [0.0] * 59 + [1.0]
        result = disaggregate_hourly([120], {"only": 1}, minute_profile=profile)
        self.assertEqual(result["only"][-1], 120)
        self.assertEqual(sum(result["only"][:-1]), 0)
        with self.assertRaises(ValueError):
            disaggregate_hourly([120], {"only": 0.5})
        with self.assertRaises(ValueError):
            disaggregate_hourly([120], {"only": 1}, minute_profile=[1] * 60)

    def test_pdf_example_area_capacity_carryover_and_conservation(self):
        result = simulate_bulk_queue([450, 250, 100], [4, 8, 12], [300, 300, 300], interval_minutes=4)
        self.assertEqual([r["left_behind"] for r in result["train_records"]], [150, 100, 0])
        self.assertEqual([r["boarded"] for r in result["train_records"]], [300, 300, 200])
        self.assertAlmostEqual(result["waiting_person_minutes"], 2600)
        self.assertAlmostEqual(result["mean_wait_minutes"], 3.25)
        self.assertEqual(result["total_boarded"], 800)
        self.assertEqual(result["conservation_error"], 0)
        self.assertTrue(result["cleared"])

    def test_uniform_minute_expansion_matches_coarse_exact_integral(self):
        arrivals = [450 / 4] * 4 + [250 / 4] * 4 + [100 / 4] * 4
        result = simulate_bulk_queue(arrivals, [4, 8, 12], [300] * 3)
        self.assertAlmostEqual(result["waiting_person_minutes"], 2600)
        self.assertAlmostEqual(result["mean_wait_minutes"], 3.25)

    def test_fractional_train_time_splits_arrival_bin_exactly(self):
        result = simulate_bulk_queue([100], [0.5, 1], [100, 100])
        self.assertEqual(result["total_boarded"], 100)
        self.assertAlmostEqual(result["waiting_person_minutes"], 25)
        self.assertAlmostEqual(result["mean_wait_minutes"], 0.25)

    def test_waiting_space_checks_preboarding_not_just_residual(self):
        result = simulate_bulk_queue([450], [4], [300], interval_minutes=4, waiting_space_limit=300)
        self.assertEqual(result["residual_queue"], 150)
        self.assertEqual(result["max_queue"], 450)
        self.assertTrue(result["waiting_space_exceeded"])
        self.assertIsNone(result["mean_wait_minutes"])

    def test_no_service_retains_censored_passengers_and_later_train_clears(self):
        result = simulate_bulk_queue([60], [], [])
        self.assertEqual(result["waiting_person_minutes"], 30)
        self.assertEqual(result["residual_queue"], 60)
        self.assertIsNone(result["mean_wait_minutes"])
        later = simulate_bulk_queue([60], [3], [60])
        self.assertEqual(later["waiting_person_minutes"], 150)
        self.assertAlmostEqual(later["mean_wait_minutes"], 2.5)
        self.assertEqual(later["horizon_minutes"], 3)

    def test_initial_queue_does_not_claim_complete_historical_wait(self):
        result = simulate_bulk_queue([], [2], [15], initial_queue=15)
        self.assertEqual(result["waiting_person_minutes"], 30)
        self.assertTrue(result["cleared"])
        self.assertIsNone(result["mean_wait_minutes"])
        self.assertEqual(result["mean_observed_window_wait_minutes"], 2)

    def test_negative_nonfinite_and_unsorted_inputs_rejected(self):
        for invalid in (-1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    simulate_bulk_queue([invalid], [], [])
                with self.assertRaises(ValueError):
                    disaggregate_hourly([invalid], {"only": 1})
        with self.assertRaises(ValueError):
            simulate_bulk_queue([1], [1, 0], [1, 1])
        with self.assertRaises(ValueError):
            simulate_bulk_queue([1], [1, 1], [1, 1])
        with self.assertRaises(ValueError):
            simulate_bulk_queue([1], [1], [])
        with self.assertRaises(ValueError):
            mmc_metrics(1, 0, 2)

    def test_explicit_scenarios_keep_comparison_horizon_and_assumptions(self):
        scenario = {
            "name": "hypothetical", "headway_minutes": 2, "available_capacity": 0,
            "first_train_minutes": 2, "clearance_minutes": 2,
            "assumptions": ["Teaching scenario only; not Taipei Metro capacity."],
        }
        result = compare_headway_scenarios([60], [scenario])[0]
        self.assertEqual(result["scenario_name"], "hypothetical")
        self.assertEqual(result["metrics"]["horizon_minutes"], 3)
        self.assertEqual(result["metrics"]["waiting_person_minutes"], 150)
        self.assertIsNone(result["metrics"]["mean_wait_minutes"])
        self.assertEqual(result["assumptions"], scenario["assumptions"])
        with self.assertRaises(ValueError):
            compare_headway_scenarios([60], [dict(scenario, assumptions=[])])
        with self.assertRaises(ValueError):
            compare_headway_scenarios([60], [{"name": "missing_capacity"}])


if __name__ == "__main__":
    unittest.main()
