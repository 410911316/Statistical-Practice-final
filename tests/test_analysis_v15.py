from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from analysis_v15.experiment import build_features, columns, Encoder, fit
from test_analysis_v14 import fixture, GROUPS


def source():
    date=pd.Timestamp('2024-08-10'); days=[date-pd.Timedelta(days=7),date]
    h=pd.DataFrame([dict(service_date=d,station='北門',source_hour=b,
        origin_count=b*10+(100 if d==date else 0),
        destination_count_same_source_bin=b*20+(50 if d==date else 0),
        origin_complete=True,destination_complete=True) for d in days for b in range(12,24)])
    events=pd.DataFrame([dict(event_date=date,station='北門',hour=b,y=999) for b in range(18,24)])
    ref=pd.DataFrame([dict(event_date=date,station='北門',candidate_date=days[0],selected=True)])
    return events,h,ref


class Snapshot(unittest.TestCase):
    def test_exact_window_values(self):
        f,a=build_features(*source()); r=f.iloc[0]
        self.assertEqual(r.lag_origin_last_w1,270)
        self.assertEqual(r.lag_origin_excess_w1,100)
        self.assertEqual(r.lag_origin_sum_w3,780)
        self.assertEqual(r.lag_origin_sum_w5,1250)
        self.assertEqual(r.lag_origin_slope_w5,10)
        self.assertEqual(r.lag_origin_excess_w5,500)
        self.assertEqual(r.lag_destination_sum_w5,1750)
        self.assertEqual(r.lag_destination_excess_w5,250)
        self.assertEqual(next(z for z in a if z['window']==5)['bins'],[13,14,15,16,17])

    def test_future_and_unused_hours_cannot_change_features(self):
        e,h,r=source(); expected,_=build_features(e,h,r)
        h.loc[~h.source_hour.between(13,17),'origin_count']=123456
        h.loc[~h.source_hour.between(13,17),'destination_count_same_source_bin']=987654
        got,_=build_features(e,h,r);pd.testing.assert_frame_equal(expected,got)

    def test_target_truth_cannot_change_features(self):
        e,h,r=source(); expected,_=build_features(e,h,r)
        e['y']=0;e['hour']=100
        got,_=build_features(e,h,r);pd.testing.assert_frame_equal(expected,got)

    def test_snapshot_frozen_across_all_future_targets(self):
        e,h,r=source(); f,_=build_features(e,h,r)
        joined=e.merge(f,on=['event_date','station'],validate='many_to_one')
        self.assertTrue(joined.filter(like='lag_').nunique().eq(1).all())

    def test_missing_bin_fails(self):
        e,h,r=source();h=h.loc[h.source_hour.ne(15)]
        with self.assertRaisesRegex(ValueError,'incomplete'): build_features(e,h,r)

    def test_incomplete_destination_fails(self):
        e,h,r=source();h.loc[h.source_hour.eq(17),'destination_complete']=False
        with self.assertRaisesRegex(ValueError,'incomplete'):build_features(e,h,r)

    def test_negative_count_fails(self):
        e,h,r=source();h.loc[h.source_hour.eq(17),'origin_count']=-1
        with self.assertRaisesRegex(ValueError,'Invalid'):build_features(e,h,r)

    def test_duplicate_bin_fails(self):
        e,h,r=source();h=pd.concat([h,h.loc[h.source_hour.eq(17)]])
        with self.assertRaisesRegex(ValueError,'Duplicate'):build_features(e,h,r)

    def test_future_reference_fails(self):
        e,h,r=source();r['candidate_date']=r.event_date
        with self.assertRaisesRegex(ValueError,'precede'):build_features(e,h,r)

    def test_empty_reference_fails(self):
        e,h,r=source();r['selected']=False
        with self.assertRaisesRegex(ValueError,'Reference'):build_features(e,h,r)

    def test_non_selected_future_reference_not_used(self):
        e,h,r=source();expected,_=build_features(e,h,r)
        extra=r.assign(candidate_date=r.event_date,selected=False)
        actual,_=build_features(e,h,pd.concat([r,extra]))
        pd.testing.assert_frame_equal(expected,actual)

    def test_disallow_other_clock_and_variant(self):
        with self.assertRaises(ValueError):build_features(*source(),cutoff=19)
        with self.assertRaises(ValueError):columns('future_y')
        self.assertEqual(len(columns('both_w5')),8)
        self.assertEqual(len(columns('origin_w1')),2)


class ModelGuards(unittest.TestCase):
    def training(self):
        f=fixture()
        f['lag_origin_last_w1']=f.year*3
        f['lag_origin_excess_w1']=f.hour+f.year
        return f

    def test_scaling_uses_training_only(self):
        f=self.training();e=Encoder().fit(f,GROUPS,'origin_w1')
        means=e.means.copy();scales=e.scales.copy()
        test=f.copy();test['lag_origin_last_w1']=1e9
        z=e.transform(test)
        np.testing.assert_array_equal(e.means,means);np.testing.assert_array_equal(e.scales,scales)
        self.assertGreater(z[:,-2].min(),1e6)

    def test_current_year_training_rejected(self):
        with self.assertRaisesRegex(ValueError,'earlier'):
            fit(self.training(),2021,'ridge','origin_w1',GROUPS,{'lambda':.1},1)

    def test_labels_do_not_determine_encoding(self):
        f=self.training();a=Encoder().fit(f,GROUPS,'origin_w1')
        f.y=1e9;b=Encoder().fit(f,GROUPS,'origin_w1')
        np.testing.assert_allclose(a.transform(f),b.transform(f))

    def test_missing_lag_at_prediction_rejected(self):
        f=self.training();e=Encoder().fit(f,GROUPS,'origin_w1')
        f.loc[0,'lag_origin_last_w1']=np.nan
        with self.assertRaisesRegex(ValueError,'Invalid'):e.transform(f)

    def test_ridge_handles_added_collinearity(self):
        f=self.training();f['lag_origin_excess_w1']=f.lag_origin_last_w1*2
        model=fit(f,2022,'ridge','origin_w1',GROUPS,{'lambda':.1},1)
        self.assertTrue(np.isfinite(model.predict(f)).all())

    def test_unknown_model_fails(self):
        with self.assertRaises(ValueError):fit(self.training(),2022,'future','none',GROUPS,{},1)


if __name__=='__main__':unittest.main()
