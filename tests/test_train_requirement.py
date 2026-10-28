import sys
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dispatch_optimization import DirectionQueue
from train_requirement import required_headways,schedule_from_headways,trains_per_slot


class TrainRequirementTests(unittest.TestCase):
    def test_schedule_steps_by_the_current_slot_headway(self):
        times=schedule_from_headways([10,3,10],15,45)
        self.assertEqual(times,[0,10,20,23,26,29,32,42])
        np.testing.assert_array_equal(trains_per_slot(times,15,3),[2,4,2])
        with self.assertRaises(ValueError): schedule_from_headways([2.5],15,30)

    def test_burst_slot_gets_shorter_headway_then_relaxes(self):
        base=np.r_[np.full(30,5.),np.full(15,120.),np.full(45,5.)]
        queue=DirectionQueue(np.tile(base,(10,1)),400,100)
        headways,shares,feasible=required_headways(queue,6,15,90,range(2,11),.05)
        self.assertEqual(headways[0],10)
        self.assertLess(headways[2],10)
        self.assertEqual(headways[-1],10)
        self.assertTrue(all(feasible))
        self.assertTrue(all(s<=.05 for s in shares))

    def test_more_demand_never_needs_fewer_trains(self):
        rng=np.random.default_rng(1)
        low=rng.poisson(np.r_[np.full(20,20.),np.full(25,90.),np.full(15,20.)],size=(30,60))
        q_low=DirectionQueue(low,500,70);q_high=DirectionQueue(low*1.5,500,70)
        h_low,_,_=required_headways(q_low,4,15,60,range(2,11),.05)
        h_high,_,_=required_headways(q_high,4,15,60,range(2,11),.05)
        self.assertTrue(all(b<=a for a,b in zip(h_low,h_high)))

    def test_backtracks_when_a_sparse_slot_lets_a_burst_pile_up(self):
        # Burst at minutes 11-14: with a 10-minute first slot the next train is at 20
        # and meets ~600 riders, more than one 300-space train can carry.
        base=np.r_[np.full(11,2.),np.full(4,150.),np.full(30,2.)]
        queue=DirectionQueue(np.tile(base,(5,1)),300,60)
        headways,_,feasible=required_headways(queue,3,15,45,range(2,11),.05)
        self.assertTrue(all(feasible))
        self.assertLess(headways[0],10)
        _,_,greedy=required_headways(queue,3,15,45,range(2,11),.05,max_backtrack=0)
        self.assertFalse(all(greedy))

    def test_insufficient_capacity_is_flagged_not_hidden(self):
        queue=DirectionQueue(np.full((5,30),200.),100,40)
        headways,shares,feasible=required_headways(queue,2,15,30,range(2,11),.05)
        self.assertEqual(headways,[2,2])
        self.assertFalse(any(feasible))
        self.assertEqual(shares[0],1.)


if __name__=='__main__':
    unittest.main()
