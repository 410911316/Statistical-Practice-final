"""Numerical and chronology checks for dispatch-oriented tuning."""
from pathlib import Path
import json
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tests')]
from analysis_v11.evaluation import costs, window_weights, select_candidate, metrics, WINDOW
from analysis_v11.models import fit_one
from analysis_v7.models import fit_one as old_fit
from test_analysis_v7_models import robust_config, rare_weather
from test_analysis_v6_models import curved_events


def panel():
    f = pd.DataFrame({'event_date':pd.Timestamp('2022-08-01'), 'station':'北門',
                      'hour':range(18,24), 'y':100.})
    f[WINDOW[0]]=[0.,0.,1.,0.,0.,0.]
    f[WINDOW[1]]=[0.,0.,0.,1.,0.,0.]
    return f


class DispatchEvaluationTest(unittest.TestCase):
    def test_known_asymmetric_losses(self):
        f=panel()
        p=np.column_stack([np.full(6,90.),np.full(6,115.)])
        result=costs(f,p)
        np.testing.assert_array_equal(result[1],[10,15])
        np.testing.assert_array_equal(result[2],[20,15])
        np.testing.assert_array_equal(result[3],[30,15])
        self.assertEqual(select_candidate(f,p,2023,1,2)[0],0)
        self.assertEqual(select_candidate(f,p,2023,2,2)[0],1)

    def test_window_is_planned_not_observed_peak(self):
        f=panel();f['y']=[10000,1,1,1,1,1]
        a,_=window_weights(f)
        np.testing.assert_array_equal(a,[0,0,.5,.5,0,0])

    def test_partial_overlap_and_equal_events(self):
        f=panel();f[WINDOW[0]]=[0,0,.5,.5,0,0];f[WINDOW[1]]=[0,0,0,.5,.5,0]
        g=panel();g['event_date']=pd.Timestamp('2023-08-01');g[WINDOW]=np.nan
        a,audit=window_weights(pd.concat([f,g],ignore_index=True))
        np.testing.assert_allclose(a[:6],[0,0,.125,.25,.125,0])
        np.testing.assert_allclose(a[6:],np.full(6,1/12))
        self.assertIn('unknown',audit[1]['fallback'])

    def test_known_outside_window_is_explicit_fallback(self):
        f=panel();f[WINDOW]=0
        a,audit=window_weights(f)
        np.testing.assert_allclose(a,np.full(6,1/6))
        self.assertIn('outside',audit[0]['fallback'])

    def test_invalid_overlap_rejected(self):
        for value in [np.nan,-.1,1.1]:
            with self.subTest(value=value):
                f=panel();f.loc[0,WINDOW[0]]=value
                with self.assertRaises(ValueError):window_weights(f)

    def test_future_labels_rejected(self):
        f=panel()
        for year in [2021,2022]:
            with self.assertRaises(ValueError):select_candidate(f,np.full((6,2),100.),year,2,2)

    def test_original_grid_prefix_limits_selection(self):
        f=panel();p=np.column_stack([np.full(6,90.),np.full(6,100.)])
        self.assertEqual(select_candidate(f,p,2023,2,1)[0],0)
        self.assertEqual(select_candidate(f,p,2023,2,2)[0],1)

    def test_tie_is_first_candidate(self):
        self.assertEqual(select_candidate(panel(),np.full((6,2),100.),2023,2,2)[0],0)

    def test_duplicate_or_incomplete_or_wrong_station_rejected(self):
        f=panel();wrong=f.assign(station='其他')
        for bad in [f.iloc[:-1],pd.concat([f,f]),wrong]:
            with self.assertRaises(ValueError):window_weights(bad)

    def test_metrics_use_actual_peak_same_hour(self):
        f=panel();f['y']=[1,2,100,40,2,1]
        p=np.array([1,2,60,90,2,1])
        m=metrics(f,p)
        self.assertEqual(m['peak_under'],40)
        self.assertEqual(m['peak_hour'],1)
        self.assertAlmostEqual(m['rmse'],np.sqrt((40**2+50**2)/6))
        self.assertAlmostEqual(m['cost_2'],65)

    def test_zero_errors_and_prediction_validation(self):
        f=panel()
        self.assertEqual(metrics(f,f.y)['cost_3'],0)
        for p in [np.full(6,-1.),np.full(6,np.inf)]:
            with self.assertRaises(ValueError):costs(f,p)


class TunedGAMTest(unittest.TestCase):
    def test_five_basis_preserves_archived_implementation(self):
        f=curved_events();groups=('forecast_temperature_c',)
        original=old_fit(f,'gam',groups,{'lambda':1.,'ridge_alpha':.1},robust_config())
        new=fit_one(f,'gam',groups,{'lambda':1.,'ridge_alpha':.1,'n_splines':5},robust_config())
        np.testing.assert_allclose(original.predict(f),new.predict(f),rtol=0,atol=1e-9)

    def test_each_basis_size_changes_actual_spline_and_passes_certificate(self):
        f=curved_events()
        for k in [4,5,7]:
            fit=fit_one(f,'gam',('forecast_temperature_c',),{'lambda':1.,'ridge_alpha':.1,'n_splines':k},robust_config())
            smooth=[t for t in fit.terms_audit if t['kind']=='smooth']
            self.assertEqual(len(smooth),1)
            self.assertEqual(smooth[0]['n_splines'],k)
            self.assertLess(fit.certificate['relative_normal_equation_residual'],1e-7)

    def test_rare_event_predictor_remains_excluded(self):
        train,future=rare_weather()
        fit=fit_one(train,'gam',('forecast_temperature_c',),{'lambda':1.,'ridge_alpha':.1,'n_splines':7},robust_config())
        changed=future.copy();changed['forecast_temperature_c']=100000.;changed['y']=1000000.
        np.testing.assert_array_equal(fit.predict(future),fit.predict(changed))

    def test_invalid_basis_rejected(self):
        with self.assertRaises(ValueError):
            fit_one(curved_events(),'gam',('forecast_temperature_c',),{'lambda':1.,'ridge_alpha':.1,'n_splines':3},robust_config())

    def test_fixed_candidate_counts_and_uniqueness(self):
        cfg=json.loads((ROOT/'configs/analysis_v11.json').read_text('utf-8'))
        for name,n in [('ridge',25),('gam',60),('xgboost',180)]:
            self.assertEqual(len(cfg['grids'][name]),n)
            self.assertEqual(len({json.dumps(p,sort_keys=True) for p in cfg['grids'][name]}),n)
        self.assertEqual({p['n_splines'] for p in cfg['grids']['gam'][:20]},{5})
        self.assertEqual({p['learning_rate'] for p in cfg['grids']['xgboost'][:90]},{.05})
        self.assertEqual({p['learning_rate'] for p in cfg['grids']['xgboost'][90:]},{.01,.03,.1})


if __name__=='__main__':unittest.main()
