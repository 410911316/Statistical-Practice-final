from pathlib import Path
import json,sys,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from forecasting.features import Encoder,GAMEncoder,fit_one
from forecasting.weighted_tree import training_weights,fit_final,panel_check,STATIONS
from forecasting.evaluation import metrics
def fixture():
    rows=[]
    for year in range(2017,2027):
        for j,s in enumerate(sorted(STATIONS)):
            for h in range(18,24):
                rows.append(dict(event_date=pd.Timestamp(f'{year}-08-{10+year-2017:02d}'),
                    year=year,station=s,hour=h,is_weekend=year%2,day_of_year=220+year-2017,
                    year_index=year-2017,retrospective_baseline_56d=70+j*5+h+(year-2017)*2,
                    planned_end_0_60_beimen=float(s=='北門' and h==20),
                    planned_end_60_120_beimen=float(s=='北門' and h==21),
                    y=100+(year-2017)*7+j*9+(h-18)*11+(h==21)*(year-2016)*30))
    return pd.DataFrame(rows)
GROUPS=['is_weekend','day_of_year','year_index','retrospective_baseline_56d']
CFG=dict(seed=1,xgb_fixed={},gam_min_known_events=6,gam_binary_groups=['is_weekend'])
class ForecastRules(unittest.TestCase):
    def test_weights_preserve_event_mass_and_do_not_read_y(self):
        f=fixture();w,_=training_weights(f,4)
        np.testing.assert_allclose(f.assign(w=w).groupby('event_date').w.sum(),24)
        changed=f.copy();changed.y=1e6
        np.testing.assert_array_equal(w,training_weights(changed,4)[0])
        g=f.assign(w=w).query('year==2017')
        self.assertAlmostEqual(g.query("station=='北門' and hour==20").w.iloc[0]/g.query("station!='北門'").w.iloc[0],4)
    def test_unknown_window_no_fabricated_peak_and_partial_missing_rejected(self):
        f=fixture();cols=['planned_end_0_60_beimen','planned_end_60_120_beimen']
        f.loc[f.year.eq(2017),cols]=np.nan
        np.testing.assert_array_equal(training_weights(f,4)[0][f.year.eq(2017)],1)
        f.loc[f.year.eq(2018)&f.station.eq('北門')&f.hour.eq(20),cols[0]]=np.nan
        with self.assertRaises(ValueError):training_weights(f,4)
    def test_panel_rejects_duplicates_incomplete_negative(self):
        f=fixture()
        for bad in [f.iloc[:-1],pd.concat([f,f.iloc[:1]]),f.assign(y=-1)]:
            with self.assertRaises(ValueError):panel_check(bad)
    def test_encoder_does_not_use_y_or_future_scale(self):
        train=fixture().query('year<2026')
        a=Encoder().fit(train,tuple(GROUPS));b=Encoder().fit(train.assign(y=1e9),tuple(GROUPS))
        np.testing.assert_array_equal(a.transform(train),b.transform(train))
        self.assertAlmostEqual(a.mean['retrospective_baseline_56d'],train.retrospective_baseline_56d.mean())
        with self.assertRaises(ValueError):a.transform(train.assign(station='不存在'))
        with self.assertRaises(ValueError):Encoder().fit(train,('y',))
    def test_ridge_matches_centered_normal_equation(self):
        f=fixture().query('year<2026');m=fit_one(f,'ridge',tuple(GROUPS),{'lambda':.2},CFG)
        x=m.encoder.transform(f);xc=x-x.mean(axis=0);yc=f.y-f.y.mean()
        beta=np.linalg.solve(xc.T@xc+len(f)*.2*np.eye(x.shape[1]),xc.T@yc)
        np.testing.assert_allclose(beta,m.estimator.coef_,rtol=1e-8,atol=1e-8)
        self.assertAlmostEqual(m.estimator.intercept_,f.y.mean()-x.mean(axis=0)@beta)
    def test_tree_rejects_current_or_future_training(self):
        params=dict(n_estimators=5,max_depth=2,learning_rate=.1)
        with self.assertRaises(ValueError):fit_final(fixture(),2026,params,4,CFG,GROUPS)
        train=fixture().query('year<2026');test=fixture().query('year==2026')
        m=fit_final(train,2026,params,4,CFG,GROUPS)
        np.testing.assert_array_equal(m.predict(test),m.predict(test.assign(y=1e9)))
    def test_gam_convergence_penalty_and_rare_candidate_support(self):
        f=fixture().query('year<2026').copy()
        f['forecast_temperature_c']=np.nan;f.loc[f.year.eq(2025),'forecast_temperature_c']=25.
        groups=tuple(GROUPS+['forecast_temperature_c'])
        m=fit_one(f,'gam',groups,{'lambda':.1,'ridge_alpha':.001,'n_splines':5},CFG)
        self.assertNotIn('forecast_temperature_c',m.encoder.names)
        self.assertTrue(m.certificate['intercept_unpenalized'])
        self.assertLess(m.certificate['relative_normal_equation_residual'],1e-6)
        np.testing.assert_array_equal(m.predict(f),m.predict(f.assign(forecast_temperature_c=1e6)))
    def test_metrics_match_direct_mae_rmse(self):
        f=fixture().query("station=='北門' and year>=2024").reset_index(drop=True)
        p=f.y.to_numpy()+np.arange(len(f))-10
        a=metrics(f,p);e=p-f.y.to_numpy()
        self.assertAlmostEqual(a['mae'],abs(e).mean())
        self.assertAlmostEqual(a['rmse'],np.sqrt(np.mean(e*e)))
class FrozenOutputs(unittest.TestCase):
    @unittest.skipUnless((ROOT/'.local/forecast/all_predictions.csv').exists(),'Private reference required')
    def test_saved_equation_and_model_reload(self):
        import joblib
        f=pd.read_csv(ROOT/'.local/forecast/analysis_input.csv',parse_dates=['event_date'])
        p=pd.read_csv(ROOT/'.local/forecast/all_predictions.csv',parse_dates=['event_date'])
        eq=json.loads((ROOT/'result/forecast/ridge_equation_2026.json').read_text('utf8'))
        t=p.query('year==2026').copy()
        values={};raw=t.copy()
        for name,mean in eq['mean'].items():
            col=raw[name]
            if name in eq['imputation']:col=col.fillna(eq['imputation'][name])
            values[name]=(col-mean)/eq['scale'][name]
        for key in eq['coefficients']:
            if key.endswith(':missing'):values[key]=raw[key[:-8]].isna().astype(float)
            elif key.startswith('station='):
                station,*hour=key[8:].split(':hour=')
                values[key]=(raw.station==station).astype(float)
                if hour:values[key]*=(raw.hour==int(hour[0])).astype(float)
            elif key.startswith('hour='):values[key]=(raw.hour==int(key[5:])).astype(float)
        manual=eq['intercept']+sum(eq['coefficients'][k]*values[k] for k in eq['coefficients'])
        np.testing.assert_allclose(np.maximum(manual,0),t.ridge,rtol=0,atol=1e-8)
        for year in [2024,2025,2026]:
            test=p.query('year==@year')
            for name in ['ridge','gam','xgboost','xgboost_weighted']:
                model=joblib.load(ROOT/f'.local/forecast/models/{year}_{name}.joblib')
                np.testing.assert_allclose(model.predict(test),test[name],rtol=0,atol=1e-8)
if __name__=='__main__':unittest.main()
