import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dispatch_optimization import DirectionQueue,feasible_minutes,optimize_insertions,plan_cost
from queueing import simulate_bulk_queue


class DispatchOptimizationTests(unittest.TestCase):
    def test_vectorized_queue_matches_reference_simulator(self):
        rng=np.random.default_rng(3)
        arrivals=rng.poisson(rng.uniform(5,80,size=(6,1))*np.ones((6,40))).astype(float)
        arrivals[:,10:14]*=8
        capacity=rng.uniform(150,400,size=6)
        trains=[3,7,11,15,19,23,27,31,35,39,44]
        queue=DirectionQueue(arrivals,capacity,horizon=50)
        wait,residual,left=queue.evaluate(trains)
        for s in range(6):
            ref=simulate_bulk_queue(arrivals[s],trains,[capacity[s]]*len(trains))
            self.assertAlmostEqual(residual[s],ref['residual_queue'])
            self.assertEqual(left[s],sum(t['left_behind']>1e-7 for t in ref['train_records']))
            if ref['cleared']:
                self.assertAlmostEqual(wait[s],ref['waiting_person_minutes'],places=6)

    def test_feasible_minutes_respect_separation(self):
        minutes=feasible_minutes([10,16],2,5,20)
        self.assertTrue(all(min(abs(m-10),abs(m-16))>=2 for m in minutes))
        self.assertIn(13,minutes);self.assertNotIn(11,minutes);self.assertNotIn(15,minutes)

    def test_small_problem_matches_brute_force(self):
        rng=np.random.default_rng(5)
        base=np.r_[np.full(10,20.),np.full(6,260.),np.full(14,30.)]
        queues={'a':DirectionQueue(rng.poisson(base,size=(20,30)),500,40),
                'b':DirectionQueue(rng.poisson(base[::-1],size=(20,30)),500,40)}
        trains={'a':[0,6,12,18,24,30,36],'b':[3,9,15,21,27,33,39]}
        plans=optimize_insertions(queues,trains,2,2,(0,40))
        candidates=[('a',m) for m in feasible_minutes(trains['a'],2,0,40)]+[('b',m) for m in feasible_minutes(trains['b'],2,0,40)]
        brute=np.inf
        for pair in itertools.combinations(candidates,2):
            extras={'a':sorted(m for d,m in pair if d=='a'),'b':sorted(m for d,m in pair if d=='b')}
            if any(np.any(np.diff(sorted(trains[d]+extras[d]))<2) for d in extras): continue
            brute=min(brute,plan_cost(queues,trains,extras,60.))
        self.assertAlmostEqual(plan_cost(queues,trains,plans[2],60.),brute)
        self.assertEqual(len(plans),3)
        self.assertLessEqual(plan_cost(queues,trains,plans[2],60.),plan_cost(queues,trains,plans[1],60.))


if __name__=='__main__':
    unittest.main()
