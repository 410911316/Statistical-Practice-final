from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from analysis_v16.experiment import check_panel, select_train, fit_cohort, common_metrics, ALL
from test_analysis_v14 import fixture,GROUPS

THREE=['北門','大橋頭站','雙連']


class Cohorts(unittest.TestCase):
    def test_three_station_complete_same_events(self):
        f=fixture();x=select_train(f,2021,THREE)
        self.assertEqual(set(x.station),set(THREE))
        self.assertEqual(x.event_date.nunique(),4)
        self.assertEqual(len(x),4*3*6)
        self.assertTrue(x.year.lt(2021).all())

    def test_four_keeps_exact_input_order(self):
        f=fixture()
        self.assertTrue(select_train(f,2021,list(ALL)).equals(f.loc[f.year.lt(2021)]))

    def test_duplicate_stations_rejected(self):
        with self.assertRaises(ValueError):select_train(fixture(),2021,['北門','北門'])

    def test_invalid_station_rejected(self):
        with self.assertRaises(ValueError):select_train(fixture(),2021,['北門','台北'])

    def test_north_required(self):
        with self.assertRaises(ValueError):select_train(fixture(),2021,['雙連'])

    def test_partial_event_rejected(self):
        f=fixture().drop(index=0)
        with self.assertRaises(ValueError):check_panel(f,list(ALL))

    def test_duplicate_target_rejected(self):
        import pandas as pd
        f=fixture();f=pd.concat([f,f.iloc[:1]])
        with self.assertRaises(ValueError):check_panel(f,list(ALL))

    def test_year_mismatch_rejected(self):
        f=fixture();f.loc[0,'year']=2050
        with self.assertRaises(ValueError):check_panel(f,list(ALL))

    def test_excluded_station_values_do_not_affect_model(self):
        f=fixture();before,_=fit_cohort(f,2021,THREE,'ridge','none',GROUPS,{'lambda':.1},1)
        f.loc[f.station.eq('民權西路'),'y']=1e9
        f.loc[f.station.eq('民權西路'),'retrospective_baseline_56d']=1e9
        after,_=fit_cohort(f,2021,THREE,'ridge','none',GROUPS,{'lambda':.1},1)
        test=f.loc[f.year.eq(2021)&f.station.eq('北門')]
        np.testing.assert_array_equal(before.predict(test),after.predict(test))
        self.assertNotIn('民權西路',before.encoder.base.categories['station'])

    def test_future_labels_do_not_affect_fit(self):
        f=fixture();before,_=fit_cohort(f,2021,THREE,'xgboost','none',GROUPS,
                                     {'max_depth':2,'n_estimators':5},1)
        f.loc[f.year.eq(2021),'y']=1e9
        after,_=fit_cohort(f,2021,THREE,'xgboost','none',GROUPS,
                          {'max_depth':2,'n_estimators':5},1)
        test=f.loc[f.year.eq(2021)&f.station.eq('北門')]
        np.testing.assert_array_equal(before.predict(test),after.predict(test))

    def test_scaling_and_ridge_mean_loss_penalty_use_selected_train(self):
        f=fixture();fitted,train=fit_cohort(f,2021,THREE,'ridge','none',GROUPS,{'lambda':.1},1)
        self.assertAlmostEqual(fitted.estimator.alpha,len(train)*.1)
        self.assertAlmostEqual(fitted.encoder.base.mean['retrospective_baseline_56d'],
                               train.retrospective_baseline_56d.mean())

    def test_common_scores_and_invalid_predictions(self):
        f=fixture().iloc[:3].copy();f.y=[1,3,5]
        score=common_metrics(f,[2,3,3])
        self.assertAlmostEqual(score['mae'],1.)
        self.assertAlmostEqual(score['rmse'],np.sqrt(5/3))
        with self.assertRaises(ValueError):common_metrics(f,[-1,0,0])


if __name__=='__main__':unittest.main()
