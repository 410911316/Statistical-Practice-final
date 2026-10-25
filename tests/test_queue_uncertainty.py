import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from queue_scenarios import shaped_arrivals
from queueing import simulate_bulk_queue
from run_queue_uncertainty import NORTH,SOUTH,donor_ratios,simulate_draw

TRAINS={NORTH:list(range(4,400,6)),SOUTH:list(range(2,400,6))}


class UncertaintyTests(unittest.TestCase):
    def test_donor_is_never_the_event_itself(self):
        ratios=pd.DataFrame(np.arange(18.).reshape(3,6),index=['a','b','c'],columns=range(18,24))
        rng=np.random.default_rng(0)
        for _ in range(200):
            donor,ratio=donor_ratios(ratios,'b',rng)
            self.assertNotEqual(donor,'b')
            np.testing.assert_array_equal(ratio,ratios.loc[donor].to_numpy())

    def test_deterministic_draw_matches_v1_direction_sum(self):
        hourly=np.array([1200.,2500,6000,3000,800,300])
        row=simulate_draw(hourly,'concentrated',5,30,.4,700,TRAINS,5)
        rate=shaped_arrivals(hourly,'concentrated',5,30,5)
        area=demand=0.
        for direction,share in [(NORTH,.4),(SOUTH,.6)]:
            q=simulate_bulk_queue(rate*share,TRAINS[direction],[700]*len(TRAINS[direction]))
            area+=q['waiting_person_minutes'];demand+=q['total_arrivals']
        self.assertTrue(row['cleared'])
        self.assertAlmostEqual(row['mean_wait_minutes'],area/demand)
        self.assertGreaterEqual(row['p95_wait_minutes'],row['p50_wait_minutes'])

    def test_same_seed_same_draw_and_poisson_keeps_scale(self):
        hourly=np.array([1200.,2500,6000,3000,800,300])
        a=simulate_draw(hourly,'broad',20,15,.5,1110,TRAINS,5,np.random.default_rng(7))
        b=simulate_draw(hourly,'broad',20,15,.5,1110,TRAINS,5,np.random.default_rng(7))
        self.assertEqual(a,b)
        self.assertLess(abs(a['demand']-hourly.sum()),5*np.sqrt(hourly.sum()))


if __name__=='__main__':
    unittest.main()
