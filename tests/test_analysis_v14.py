from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from analysis_v14.experiment import training_weights,split,candidates,select_past,estimator,fit_final,mae_nonnegative,STATIONS


def fixture():
    rows=[]
    for year in range(2017,2022):
        for j,station in enumerate(sorted(STATIONS)):
            for h in range(18,24):
                rows.append(dict(event_date=pd.Timestamp(f'{year}-08-15'),year=year,station=station,hour=h,
                    y=100+(year-2017)*7+j*9+(h-18)*11+(h==21)*(year-2016)*30,
                    retrospective_baseline_56d=70+j*5+h,is_weekend=int(year%2==0),
                    planned_end_0_60_beimen=float(station=='北門' and h==20),
                    planned_end_60_120_beimen=float(station=='北門' and h==21)))
    return pd.DataFrame(rows)


def config():
    return dict(seed=1,max_depth=[2],learning_rate=[.1],min_child_weight=[1],
        fixed=dict(reg_lambda=1,reg_alpha=0,subsample=1,colsample_bytree=1),fixed_trees=[5,12],
        max_trees=12,patience=3,fallback_trees=5,min_base_events=2,min_inner_training_events=2)

GROUPS=['is_weekend','retrospective_baseline_56d']


class Weights(unittest.TestCase):
    def test_uniform_exactly_one(self):np.testing.assert_array_equal(training_weights(fixture(),1)[0],1)
    def test_preserve_event_mass_and_positive_other_stations(self):
        f=fixture();w,a=training_weights(f,4)
        self.assertTrue((w>0).all());np.testing.assert_allclose(f.assign(w=w).groupby('event_date').w.sum(),24)
        for _,g in f.assign(w=w).groupby('event_date'):
            north=g.loc[g.station.eq('北門')&g.hour.eq(20)].w.iloc[0]
            other=g.loc[g.station.ne('北門')].w.iloc[0];self.assertAlmostEqual(north/other,4)
    def test_unknown_does_not_fabricate_peak(self):
        f=fixture();f.loc[f.year.eq(2017),['planned_end_0_60_beimen','planned_end_60_120_beimen']]=np.nan
        w,a=training_weights(f,4);np.testing.assert_array_equal(w[f.year.eq(2017)],1)
        self.assertTrue(a[0]['unknown_end'])
    def test_partial_unknown_rejected(self):
        f=fixture();f.loc[f.station.eq('北門')&f.hour.eq(20),'planned_end_0_60_beimen']=np.nan
        with self.assertRaises(ValueError):training_weights(f,2)
    def test_fractional_overlap_preserved(self):
        f=fixture();f.loc[f.station.eq('北門')&f.hour.eq(20),'planned_end_0_60_beimen']=.5
        w,_=training_weights(f,4);g=f.assign(w=w).query('year==2017')
        self.assertAlmostEqual(g.query("station=='北門' and hour==20").w.iloc[0]/g.query("station!='北門'").w.iloc[0],2.5)
    def test_weights_never_follow_observed_peak(self):
        f=fixture();a=training_weights(f,4)[0];f['y']=np.arange(len(f))*12345+1
        np.testing.assert_array_equal(a,training_weights(f,4)[0])
    def test_duplicate_or_incomplete_rejected(self):
        f=fixture()
        for bad in [f.iloc[:-1],pd.concat([f,f.iloc[:1]])]:
            with self.assertRaises(ValueError):training_weights(bad,2)


class Chronology(unittest.TestCase):
    def test_stop_year_is_earlier_and_whole_year(self):
        train,valid,inner,stop,active=split(fixture(),2021,config())
        self.assertTrue(active);self.assertEqual(set(stop.year),{2020});self.assertTrue(inner.year.lt(2020).all())
        self.assertEqual(len(stop),6);self.assertTrue(train.year.lt(2021).all());self.assertEqual(len(valid),6)
    def test_too_few_events_falls_back(self):
        matrices,a=candidates(fixture(),2019,1,config(),GROUPS)
        self.assertFalse(a['stopping'][0]['stopping_active']);self.assertEqual(a['stopping'][0]['trees'],5)
    def test_held_out_labels_do_not_change_stopping_or_predictions(self):
        f=fixture();a,meta=candidates(f,2021,2,config(),GROUPS)
        f.loc[f.year.eq(2021),'y']=1e8
        b,other=candidates(f,2021,2,config(),GROUPS)
        for key in ['fixed','es']:np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(meta['stopping'],other['stopping'])
    def test_selection_rejects_present_year(self):
        f=fixture().query("station=='北門'")
        with self.assertRaises(ValueError):select_past(f,np.ones((len(f),2)),2021)
    def test_selection_depends_only_on_supplied_past(self):
        f=fixture().query("station=='北門' and year<2021")
        p=np.column_stack([f.y+10,f.y+1]);j,loss=select_past(f,p,2021)
        self.assertEqual(j,1);np.testing.assert_allclose(loss,[10,1])
    def test_final_fit_cannot_receive_current_year(self):
        with self.assertRaises(ValueError):fit_final(fixture(),2021,{'n_estimators':5,'max_depth':2},1,config(),GROUPS)


class Numerical(unittest.TestCase):
    def test_prefix_equals_actual_shorter_refit(self):
        f=fixture();cfg=config();a,meta=candidates(f,2021,2,cfg,GROUPS)
        params={**meta['stopping'][0]['params'],'n_estimators':5}
        fit=fit_final(f.loc[f.year.lt(2021)],2021,params,2,cfg,GROUPS)
        valid=f.loc[f.year.eq(2021)&f.station.eq('北門')].sort_values(['event_date','station','hour'])
        np.testing.assert_allclose(fit.predict(valid),a['fixed'][:,0],rtol=0,atol=1e-5)
    def test_stopping_score_uses_clipped_count(self):
        self.assertEqual(mae_nonnegative(np.array([0.,10.]),np.array([-100.,8.])),1)


if __name__=='__main__':unittest.main()
