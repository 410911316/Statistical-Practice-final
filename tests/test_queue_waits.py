from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from queue_waits import fifo_wait_distribution, summarize_wait_pieces
from queueing import simulate_bulk_queue


class FifoWaitDistributionTest(unittest.TestCase):
    def test_regular_headway_ample_capacity_matches_half_headway(self):
        # Uniform arrivals, headway H, never full: wait is Uniform(0, H).
        result = fifo_wait_distribution([10] * 60, list(range(6, 61, 6)), [1000] * 10,
                                        thresholds_minutes=[3])
        self.assertAlmostEqual(result["mean_wait_minutes"], 3)
        self.assertAlmostEqual(result["quantiles"][0.5], 3)
        self.assertAlmostEqual(result["quantiles"][0.95], 5.7)
        self.assertAlmostEqual(result["max_wait_minutes"], 6)
        self.assertAlmostEqual(result["share_waiting_longer_than"][3], 0.5)

    def test_irregular_headway_matches_random_incidence_formula(self):
        # Alternating 2/8-minute headways: E[H]/2 * (1 + CV^2) = 2.5 * 1.36.
        trains = sorted([10 * k + 2 for k in range(6)] + [10 * k + 10 for k in range(6)])
        result = fifo_wait_distribution([10] * 60, trains, [1000] * len(trains))
        self.assertAlmostEqual(result["mean_wait_minutes"], 3.4)

    def test_left_behind_passengers_wait_longer_than_one_headway(self):
        # Hand-checked overload example from the bulk-service slide.
        result = fifo_wait_distribution([450, 250, 100], [4, 8, 12], [300] * 3,
                                        interval_minutes=4, thresholds_minutes=[4])
        self.assertAlmostEqual(result["mean_wait_minutes"], 3.25)
        self.assertAlmostEqual(result["max_wait_minutes"], 5.6)
        self.assertAlmostEqual(result["share_waiting_longer_than"][4], 250 / 800)

    def test_cumulative_curve_mean_equals_littles_law_area(self):
        # Time-integral form of L = lambda * W: area / arrivals.
        arrivals = [0, 30, 120, 300, 80, 0, 45, 10]
        trains, capacity = [2.5, 4, 5.5, 7, 9], [150, 150, 150, 150, 150]
        waits = fifo_wait_distribution(arrivals, trains, capacity)
        queue = simulate_bulk_queue(arrivals, trains, capacity)
        self.assertAlmostEqual(waits["mean_wait_minutes"], queue["mean_wait_minutes"])
        self.assertAlmostEqual(waits["mean_wait_minutes"], waits["queue_mean_wait_minutes"])

    def test_zero_width_piece_counts_as_point_mass(self):
        pieces = np.array([[1.0, 2.0, 2.0], [1.0, 0.0, 4.0]])
        result = summarize_wait_pieces(pieces, quantiles=[0.5], thresholds_minutes=[2])
        self.assertAlmostEqual(result["share_waiting_longer_than"][2], 0.25)
        self.assertAlmostEqual(result["quantiles"][0.5], 2)
        self.assertAlmostEqual(result["mean_wait_minutes"], 2)

    def test_censored_run_reports_no_quantiles(self):
        result = fifo_wait_distribution([450], [4], [300], interval_minutes=4,
                                        thresholds_minutes=[4])
        self.assertFalse(result["cleared"])
        self.assertEqual(result["censored_passengers"], 150)
        self.assertIsNone(result["mean_wait_minutes"])
        self.assertIsNone(result["quantiles"][0.95])
        self.assertIsNone(result["share_waiting_longer_than"][4])
        with self.assertRaises(ValueError):
            fifo_wait_distribution([1], [1], [1], quantiles=[1])


if __name__ == "__main__":
    unittest.main()
