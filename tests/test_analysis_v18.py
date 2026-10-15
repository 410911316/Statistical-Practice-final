from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from analysis_v18.experiment import scheduled_gate, calibration_data, fit_correction, WINDOW


def fixture(n=6, adjustment=.15):
    rows=[]
    for i in range(n):
        date=pd.Timestamp(2010+i,8,1)
        for hour in range(18,24):
            q=[float(hour==20),float(hour==21)]
            rows.append(dict(event_date=date,station="北門",hour=hour,y=100*(1+adjustment*sum(q)),
                base_prediction=100.,base_train_end=f"{date.year-1}-08-01",
                parameter_validation_end_year=date.year-1,**dict(zip(WINDOW,q))))
    return pd.DataFrame(rows)


class SmallCorrection(unittest.TestCase):
    def test_known_coefficients_recovered(self):
        model=fit_correction(fixture(),2024,penalty=0)
        np.testing.assert_allclose(model.theta,[.15,.15],atol=1e-8)

    def test_correction_can_be_negative(self):
        model=fit_correction(fixture(adjustment=-.15),2024,penalty=0)
        np.testing.assert_allclose(model.theta,[-.15,-.15],atol=1e-8)

    def test_cap_and_nonnegative_predictions(self):
        for sign in [1,-1]:
            f=fixture(adjustment=sign*.7);m=fit_correction(f,2024,penalty=0)
            np.testing.assert_allclose(m.theta,[sign*.25]*2)
            p=m.predict(f,f.base_prediction)
            self.assertTrue((p>=75).all() and (p<=125).all())

    def test_penalty_shrinks_to_zero(self):
        m=fit_correction(fixture(),2024,penalty=100)
        np.testing.assert_allclose(m.theta,[0,0],atol=1e-9)

    def test_small_support_falls_back(self):
        m=fit_correction(fixture(5),2024)
        self.assertEqual(m.status,"insufficient_known_egress_events")
        self.assertEqual(m.theta,[0.,0.])

    def test_unknown_event_not_calibration_support(self):
        f=fixture();f.loc[f.event_date.eq(f.event_date.iloc[0]),WINDOW]=np.nan
        m=fit_correction(f,2024)
        self.assertEqual(len(m.eligible_dates),5)
        self.assertEqual(m.theta,[0.,0.])

    def test_unknown_and_outside_window_unchanged(self):
        m=fit_correction(fixture(),2024,penalty=0)
        f=fixture();f.loc[f.event_date.eq(f.event_date.iloc[0]),WINDOW]=np.nan
        p=m.predict(f,f.base_prediction)
        mask=scheduled_gate(f).sum(axis=1)==0
        np.testing.assert_array_equal(p[mask],f.base_prediction[mask])

    def test_other_station_unchanged(self):
        m=fit_correction(fixture(),2024,penalty=0)
        f=fixture();f.station="雙連"
        np.testing.assert_array_equal(m.predict(f,f.base_prediction),f.base_prediction)

    def test_gate_does_not_require_or_use_truth(self):
        f=fixture();q=scheduled_gate(f)
        np.testing.assert_array_equal(q,scheduled_gate(f.drop(columns=["y"])))
        f.y=1e8
        np.testing.assert_array_equal(q,scheduled_gate(f))

    def test_partial_unknown_rejected(self):
        f=fixture();f.loc[0,WINDOW[0]]=np.nan
        with self.assertRaises(ValueError):scheduled_gate(f)

    def test_overlapping_hours_invalid(self):
        f=fixture();f.loc[0,WINDOW]=[.8,.8]
        with self.assertRaises(ValueError):scheduled_gate(f)

    def test_current_year_labels_rejected(self):
        with self.assertRaises(ValueError):fit_correction(fixture(),2015)

    def test_in_sample_residual_rejected(self):
        f=fixture();f.loc[0,"base_train_end"]="2010-08-01"
        with self.assertRaises(ValueError):fit_correction(f,2024)

    def test_hyperparameter_leakage_rejected(self):
        f=fixture();f.loc[0,"parameter_validation_end_year"]=2010
        with self.assertRaises(ValueError):fit_correction(f,2024)

    def test_duplicate_or_missing_hour_rejected(self):
        for f in [pd.concat([fixture(),fixture().iloc[:1]]),fixture().iloc[1:]]:
            with self.assertRaises(ValueError):fit_correction(f,2024)

    def test_each_event_equal_weight(self):
        f=fixture();f.loc[f.event_date.eq(f.event_date.iloc[0]),WINDOW]=0.
        f.loc[2,WINDOW[0]]=.2
        _,_,_,w,dates=calibration_data(f,2024)
        self.assertEqual(len(dates),6)
        for _,idx in f.groupby("event_date").groups.items():
            self.assertAlmostEqual(w[idx].sum(),1/6)

    def test_ratio_two_targets_higher_quantile(self):
        f=fixture(10,adjustment=-.2)
        active=f.hour.isin([20,21]) & f.event_date.dt.year.ge(2016)
        f.loc[active,"y"]=120
        a=fit_correction(f,2024,ratio=1,penalty=0)
        b=fit_correction(f,2024,ratio=2,penalty=0)
        np.testing.assert_allclose(a.theta,[-.2,-.2])
        np.testing.assert_allclose(b.theta,[.2,.2])

    def test_half_overlap_is_half_adjustment(self):
        m=fit_correction(fixture(),2024,penalty=0);f=fixture()
        f.loc[2,WINDOW]=[.5,0]
        self.assertAlmostEqual(m.predict(f,f.base_prediction)[2],107.5)

    def test_invalid_parameters_and_predictions(self):
        for kw in [dict(ratio=0),dict(bound=1),dict(penalty=-1),dict(min_events=0)]:
            with self.assertRaises(ValueError):fit_correction(fixture(),2024,**kw)
        m=fit_correction(fixture(),2024)
        with self.assertRaises(ValueError):m.predict(fixture(),np.full(36,-1.))

    def test_zero_base_and_row_order(self):
        f=fixture();f.base_prediction=0
        m=fit_correction(f,2024)
        np.testing.assert_array_equal(m.predict(f,f.base_prediction),np.zeros(len(f)))
        f=fixture();a=fit_correction(f,2024,penalty=0)
        b=fit_correction(f.sample(frac=1,random_state=5),2024,penalty=0)
        np.testing.assert_allclose(a.theta,b.theta)


if __name__=="__main__":unittest.main()
