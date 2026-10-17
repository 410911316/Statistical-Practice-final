from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from analysis_v19.experiment import fit, training_weights
from analysis_v14.experiment import training_weights as old_weights
from test_analysis_v14 import fixture, GROUPS

TWO=['北門', '大橋頭站']

class TwoStationComparison(unittest.TestCase):
    def test_weighted_four_matches_archived_formula(self):
        f=fixture()
        w,_=training_weights(f, sorted(f.station.unique()), 4)
        expected,_=old_weights(f, 4)
        np.testing.assert_array_equal(w, expected)

    def test_two_station_event_weights_average_one(self):
        f=fixture();f=f.loc[f.station.isin(TWO)].reset_index(drop=True)
        w,audit=training_weights(f,TWO,4)
        self.assertTrue(all(abs(a['weight_sum']-12)<1e-10 for a in audit))
        for _,g in f.groupby('event_date'):
            values=g.loc[g.station.eq('北門'),['planned_end_0_60_beimen','planned_end_60_120_beimen']].to_numpy(float)
            self.assertAlmostEqual(w[g.index].sum(),12)
            if np.isfinite(values).all() and values.sum()>0:
                self.assertGreater(w[g.loc[g.station.eq('北門')].index].sum(),6)

    def test_removed_stations_and_future_labels_do_not_affect_weighted_fit(self):
        f=fixture();params=dict(max_depth=2,n_estimators=10,learning_rate=.1,min_child_weight=1)
        a,train,_=fit(f,2021,TWO,'xgboost_w4',GROUPS,params,1)
        self.assertEqual(set(train.station),set(TWO))
        self.assertTrue(train.year.lt(2021).all())
        self.assertEqual(set(a.encoder.categories['station']),set(TWO))
        changed=f.copy()
        changed.loc[~changed.station.isin(TWO)|changed.year.ge(2021),'y']=1e9
        changed.loc[~changed.station.isin(TWO),'retrospective_baseline_56d']=1e9
        b,_,_=fit(changed,2021,TWO,'xgboost_w4',GROUPS,params,1)
        test=f.loc[f.year.eq(2021)&f.station.isin(TWO)]
        np.testing.assert_array_equal(a.predict(test),b.predict(test))

    def test_ridge_preserves_average_loss_penalty_and_train_scaling(self):
        f=fixture();a,train,_=fit(f,2021,TWO,'ridge',GROUPS,{'lambda':.1},1)
        self.assertAlmostEqual(a.estimator.alpha,.1*len(train))
        self.assertAlmostEqual(a.encoder.base.mean['retrospective_baseline_56d'],train.retrospective_baseline_56d.mean())
        self.assertEqual(set(a.encoder.base.categories['station']),set(TWO))

    def test_partial_planned_end_is_rejected(self):
        f=fixture();f=f.loc[f.station.isin(TWO)].reset_index(drop=True)
        i=f.loc[f.station.eq('北門')].index[0]
        f.loc[i,'planned_end_0_60_beimen']=np.nan
        with self.assertRaises(ValueError):training_weights(f,TWO,4)

if __name__=='__main__':unittest.main()
