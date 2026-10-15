from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from analysis_v17.experiment import subset_plans, materialize, aggregate
from analysis_v15.experiment import fit
from test_analysis_v14 import fixture, GROUPS

STATIONS=["北門","大橋頭站","雙連"]


class EventSubsets(unittest.TestCase):
    def plans(self, frame=None):
        return subset_plans(fixture() if frame is None else frame,2021,STATIONS,[2,3,4],3,14)

    def test_complete_events_and_no_future(self):
        f=fixture()
        for p in self.plans(f):
            t=materialize(f,p,STATIONS)
            self.assertEqual(len(t),p["n_events"]*18)
            self.assertTrue(t.year.lt(2021).all())
            self.assertEqual(set(t.station),set(STATIONS))

    def test_nested_at_each_replicate(self):
        plans=self.plans()
        for r in range(3):
            small=[p for p in plans if p["protocol"]=="event_subsample" and p["replicate"]==r]
            self.assertTrue(set(small[0]["train_dates"])<set(small[1]["train_dates"]))

    def test_deterministic_and_label_blind(self):
        f=fixture();p=self.plans(f)
        f.y=np.arange(len(f))*999
        self.assertEqual(p,self.plans(f))
        self.assertEqual(p,self.plans())

    def test_full_fitted_once(self):
        plans=self.plans()
        self.assertEqual(sum(p["protocol"]=="full" for p in plans),1)
        self.assertEqual(sum(p["n_events"]==4 for p in plans),1)
        self.assertEqual(len(plans),11)

    def test_order_sensitivity_actual_dates(self):
        plans=self.plans()
        old=next(p for p in plans if p["protocol"]=="oldest_first" and p["n_events"]==2)
        recent=next(p for p in plans if p["protocol"]=="recent_first" and p["n_events"]==2)
        self.assertEqual(old["train_dates"],["2017-08-15","2018-08-15"])
        self.assertEqual(recent["train_dates"],["2019-08-15","2020-08-15"])

    def test_invalid_counts_and_repeats(self):
        for counts in [[],[1],[3,2],[2,2],[2.5]]:
            with self.assertRaises(ValueError):subset_plans(fixture(),2021,STATIONS,counts,3,14)
        with self.assertRaises(ValueError):subset_plans(fixture(),2021,STATIONS,[2],0,14)
        with self.assertRaises(ValueError):subset_plans(fixture(),2018,STATIONS,[2],3,14)

    def test_unknown_future_duplicate_dates_rejected(self):
        p=self.plans()[1]
        for dates in [["2017-08-15","2021-08-15"],["2017-08-15","2017-08-15"],["2017-08-15","2016-08-15"]]:
            with self.assertRaises(ValueError):materialize(fixture(),{**p,"train_dates":dates},STATIONS)

    def test_incomplete_event_rejected(self):
        with self.assertRaises(ValueError):self.plans(fixture().iloc[1:])

    def test_full_preserves_v16_order(self):
        f=fixture();t=materialize(f,self.plans()[0],STATIONS)
        self.assertTrue(t.equals(f.loc[f.year.lt(2021)&f.station.isin(STATIONS)]))

    def test_excluded_and_future_values_cannot_change_subset_fit(self):
        f=fixture();p=self.plans()[1]
        t=materialize(f,p,STATIONS);m=fit(t,2021,"ridge","none",GROUPS,{"lambda":.1},1)
        changed=f.copy()
        excluded=~changed.event_date.isin(pd.to_datetime(p["train_dates"]))|changed.station.eq("民權西路")
        changed.loc[excluded,["y","retrospective_baseline_56d"]]=1e8
        other=fit(materialize(changed,p,STATIONS),2021,"ridge","none",GROUPS,{"lambda":.1},1)
        test=f.loc[f.year.eq(2021)&f.station.eq("北門")]
        np.testing.assert_array_equal(m.predict(test),other.predict(test))
        self.assertAlmostEqual(m.estimator.alpha,len(t)*.1)
        self.assertAlmostEqual(m.encoder.base.mean["retrospective_baseline_56d"],t.retrospective_baseline_56d.mean())

    def test_validation_set_unchanged_across_sizes(self):
        f=fixture();keys=["event_date","station","hour"]
        validation=f.loc[f.year.eq(2021)&f.station.eq("北門"),keys]
        before=validation.copy(deep=True)
        for p in self.plans(f):materialize(f,p,STATIONS)
        pd.testing.assert_frame_equal(before,validation)
        self.assertEqual(len(validation),6)

    def test_summary_band_hand_check_and_full_endpoint_once(self):
        records=[]
        for protocol,n,v,rep in [("event_subsample",2,10.,0),("event_subsample",2,30.,1),("full",4,7.,-1)]:
            records.append(dict(year=2021,model="ridge",protocol=protocol,n_events=n,scope="validation",replicate=rep,
                                **{m:v for m in ["mae","rmse","peak_abs","peak_under","cost_2"]}))
        a=aggregate(pd.DataFrame(records))
        row=a.loc[a.n_events.eq(2)].iloc[0]
        self.assertEqual(row.mae_mean,20.)
        self.assertEqual(row.mae_p10,12.)
        self.assertEqual(row.mae_p90,28.)
        self.assertEqual(len(a.loc[a.n_events.eq(4)]),3)
        self.assertTrue(a.loc[a.n_events.eq(4),"n_fits"].eq(1).all())


if __name__=="__main__":unittest.main()
