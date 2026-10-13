"""Meaningful checks for signed residuals, availability and convex stacking."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from analysis_v13.data import require_features,eligible_training
from analysis_v13.models import fit_model
from analysis_v13.fusion import solve


def fixture():
    rows=[]
    for year in (2021,2022,2023):
        for station in ('北門','雙連'):
            for hour in range(18,24):
                rows.append(dict(event_date=pd.Timestamp(f'{year}-08-15'),year=year,station=station,hour=hour,
                    y=90.,available_baseline=100.,is_weekend=1,base_train_year=year-1,
                    baseline_version_at=f'{year}-08-01T00:00:00+08:00',cutoff_at=f'{year}-08-14T18:00:00+08:00',
                    label_version_at=f'{year}-09-15T00:00:00+08:00',
                    planned_end_0_60_beimen=float(hour==20),planned_end_60_120_beimen=float(hour==21)))
    return pd.DataFrame(rows)


class Availability(unittest.TestCase):
    def test_late_baseline_rejected(self):
        f=fixture();f.loc[0,'baseline_version_at']='2021-08-14T18:00:01+08:00'
        with self.assertRaisesRegex(ValueError,'Future'):require_features(f)
    def test_missing_baseline_rejected(self):
        f=fixture();f.loc[0,'available_baseline']=np.nan
        with self.assertRaises(ValueError):require_features(f)
    def test_late_training_labels_excluded(self):
        f=fixture();f.loc[f.year.eq(2022),'label_version_at']='2024-01-01T00:00:00+08:00'
        self.assertEqual(set(eligible_training(f,2023).year),{2021})
    def test_same_year_not_training(self):self.assertEqual(set(eligible_training(fixture(),2022).year),{2021})


class Residuals(unittest.TestCase):
    def test_negative_increments_allowed_all_models(self):
        f=fixture();cfg={'seed':1,'xgb_fixed':{},'gam_tol':1e-6}
        params={'ridge':{'lambda':.1},'gam':{'lambda':10,'ridge_alpha':.1,'n_splines':5},
                'xgboost':{'n_estimators':10,'max_depth':2,'learning_rate':.1}}
        for model,p in params.items():
            fit=fit_model(f,model,'increment',['is_weekend','available_baseline'],p,cfg)
            raw,total=fit.predict_components(f)
            np.testing.assert_allclose(raw,-10,atol=1e-5);np.testing.assert_allclose(total,90,atol=1e-5)
    def test_clip_only_final_total(self):
        f=fixture();fit=fit_model(f,'ridge','increment',['is_weekend','available_baseline'],{'lambda':.1},{'seed':1,'xgb_fixed':{}})
        class Negative:
            def predict(self,x):return np.full(len(x),-200.)
        fit.estimator=Negative();raw,total=fit.predict_components(f)
        np.testing.assert_array_equal(raw,-200);np.testing.assert_array_equal(total,0)
    def test_target_change_does_not_change_encoder(self):
        f=fixture();a=fit_model(f,'ridge','direct',['is_weekend','available_baseline'],{'lambda':.1},{})
        b=fit_model(f,'ridge','increment',['is_weekend','available_baseline'],{'lambda':.1},{})
        self.assertEqual(a.encoder.names,b.encoder.names)
        np.testing.assert_allclose(a.encoder.transform(f),b.encoder.transform(f))


class Fusion(unittest.TestCase):
    def frame(self):
        f=fixture().query("station=='北門'").copy().reset_index(drop=True)
        rng=np.random.default_rng(713);f['y']=rng.uniform(0,100,len(f))
        for m in ('ridge','gam','xgboost'):f[m]=rng.uniform(0,100,len(f))
        return f
    def test_no_future_or_in_sample_oof(self):
        f=self.frame()
        with self.assertRaisesRegex(ValueError,'Future'):solve(f,2023)
        f['base_train_year']=f.year
        with self.assertRaisesRegex(ValueError,'In-sample'):solve(f,2024)
    def test_unreleased_labels_rejected(self):
        f=self.frame();f.loc[0,'label_version_at']='2025-01-01T00:00:00+08:00'
        with self.assertRaisesRegex(ValueError,'Unreleased'):solve(f,2024)
    def test_lp_matches_grid_and_simplex(self):
        f=self.frame();w,a=solve(f,2024,2,False)
        self.assertAlmostEqual(w.sum(),1);self.assertTrue((w>=0).all())
        p=f[['ridge','gam','xgboost']].to_numpy();y=f.y.to_numpy()
        best=np.inf
        for i in range(101):
            for j in range(101-i):
                r=y-p@np.array([i,j,100-i-j])/100
                best=min(best,np.mean(2*np.maximum(r,0)+np.maximum(-r,0)))
        self.assertLessEqual(a['objective'],best+1e-6)
        self.assertLess(best-a['objective'],.2)
    def test_zero_weight_nondispatch_hours_do_not_change_solution_value(self):
        f=self.frame();w,a=solve(f,2024,2,True)
        f.loc[~f.hour.isin([20,21]),'y']=1e7
        other,b=solve(f,2024,2,True)
        self.assertAlmostEqual(a['objective'],b['objective'],places=7)
    def test_equal_event_mass_with_repeated_independent_event(self):
        f=self.frame();a=solve(f,2024,2,False)[1]['objective']
        g=f.copy();g.event_date=g.event_date+pd.Timedelta(days=1)
        b=solve(pd.concat([f,g],ignore_index=True),2024,2,False)[1]['objective']
        self.assertAlmostEqual(a,b,places=7)


if __name__=='__main__':unittest.main()
