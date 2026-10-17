import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from metro_operations import parse_timetable,parse_links,dispatch_paths,minute_of_service_day,sha
from queue_scenarios import shaped_arrivals,remaining_capacity
from queueing import simulate_bulk_queue
from run_queue_scenarios import prepare_forecasts


class ScenarioTests(unittest.TestCase):
    def test_hourly_totals_and_walking_tail(self):
        for shape,sigma in [('uniform',None),('broad',20),('concentrated',5)]:
            arr=shaped_arrivals([120,600,240],shape,sigma,delay=5)
            np.testing.assert_allclose(arr[:5],0)
            np.testing.assert_allclose(arr[5:].reshape(-1,60).sum(axis=1),[120,600,240])
            self.assertEqual(len(arr),185)

    def test_only_forecast_peak_hour_changes(self):
        a=shaped_arrivals([120,600,240],'concentrated',5,delay=0).reshape(-1,60)
        np.testing.assert_allclose(a[0],2);np.testing.assert_allclose(a[2],4)
        self.assertGreater(a[1].max(),10)

    def test_reject_invalid_inputs(self):
        for vals in [[1,-1],[float('nan')],[]]:
            with self.assertRaises(ValueError):shaped_arrivals(vals,'uniform',None)
        with self.assertRaises(ValueError):shaped_arrivals([1],'unknown',5)
        with self.assertRaises(ValueError):shaped_arrivals([1],'broad',0)

    def test_capacity_after_alighting(self):
        self.assertEqual(remaining_capacity(2220,1900,200),520)
        self.assertEqual(remaining_capacity(2220,2300,0),0)
        with self.assertRaises(ValueError):remaining_capacity(2220,100,101)

    def test_midnight_stays_in_service_day(self):
        self.assertEqual(minute_of_service_day(0,14),1454)
        self.assertEqual(minute_of_service_day(23,59),1439)
        with self.assertRaises(ValueError):minute_of_service_day(5,0)
        with self.assertRaises(ValueError):minute_of_service_day(23,60)

    def test_independent_minute_integral_and_censoring(self):
        a=shaped_arrivals([300,1200,100],'concentrated',5,delay=5)
        train_times=list(range(3,210,7));capacity=50
        q=0.;area=0.;peak=0.
        for t in range(max(len(a),max(train_times))):
            if t in train_times:q=max(0,q-capacity)
            increase=a[t] if t<len(a) else 0
            area+=q+increase/2;q+=increase;peak=max(peak,q)
        if max(len(a),max(train_times)) in train_times:q=max(0,q-capacity)
        actual=simulate_bulk_queue(a,train_times,[capacity]*len(train_times))
        self.assertAlmostEqual(area,actual['waiting_person_minutes'],places=8)
        self.assertAlmostEqual(peak,actual['max_queue'],places=8)
        self.assertAlmostEqual(q,actual['residual_queue'],places=8)
        censored=simulate_bulk_queue([100],[1],[20])
        self.assertEqual(censored['residual_queue'],80)
        self.assertIsNone(censored['mean_wait_minutes'])

    def test_frozen_predictions_refuse_changed_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'p.csv';p.write_text('changed',encoding='utf-8')
            with self.assertRaises(ValueError):prepare_forecasts(p,'0'*64,'frozen_w4')


@unittest.skipUnless((ROOT/'.local/transit_reference/20261015/normalized.json').exists(),'Run official snapshot collection first')
class OfficialSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=ROOT/'.local/transit_reference/20261015'
        cls.tables=json.loads((cls.folder/'normalized.json').read_text(encoding='utf-8'))

    def test_north_weekend_times_visually_checked_against_pdf(self):
        effective,rows=parse_timetable(self.folder/'105b.pdf',['台電大樓','新店'])
        self.assertEqual(effective,'2026-08-30')
        mins=[r['departure_minute']-21*60 for r in rows if r['day_type']=='saturday' and 21*60<=r['departure_minute']<22*60]
        self.assertEqual(mins,[3,10,17,24,32,39,46,53])
        late=[r['departure_minute'] for r in rows if r['day_type']=='saturday' and r['departure_minute']>=1440]
        self.assertEqual(late,[1441,1453,1461,1474,1488])

    def test_underlines_are_not_discarded(self):
        _,rows=parse_timetable(self.folder/'105b.pdf',['台電大樓','新店'])
        marked=[r for r in rows if r['day_type']=='weekday' and r['departure_minute']==7*60+39]
        self.assertEqual(marked[0]['terminal'],'台電大樓')
        _,rows=parse_timetable(self.folder/'128a.pdf',['蘆洲','迴龍'])
        self.assertEqual(sum(r['terminal']=='蘆洲' for r in rows),373)
        self.assertEqual(sum(r['terminal']=='迴龍' for r in rows),373)

    def test_source_dates_not_backfilled_into_events(self):
        dates={s['local_file']:s['effective_from'] for s in self.tables['metro_source_snapshot']}
        self.assertEqual(dates['128b.pdf'],'2026-10-01')
        self.assertEqual(dates['105a.pdf'],'2026-08-30')

    def test_dispatch_origin_dwell_excluded_and_reverse_explicit(self):
        paths=dispatch_paths(parse_links(self.folder/'link_times.csv'))
        row=next(r for r in paths if r['origin']=='台電大樓' and r['station_label']=='北門')
        self.assertEqual(row['travel_seconds'],549)
        self.assertEqual(row['reverse_assumed'],1)
        self.assertIsNone(row['total_dispatch_seconds'])
        self.assertFalse(any('轉乘' in r['path_text'] for r in paths))

    def test_duplicate_midnight_keys_and_totals(self):
        rows=self.tables['metro_departure']
        self.assertEqual(len(rows),7118)
        keys={(r['source_id'],r['day_type'],r['departure_minute']) for r in rows}
        self.assertEqual(len(keys),len(rows))

    def test_capacity_sensitivity_and_run_count(self):
        p=ROOT/'result/queue_dispatch_v1/scenarios.csv'
        if not p.exists():self.skipTest('Run scenarios first')
        data=pd.read_csv(p);self.assertEqual(len(data),1134)
        for _,g in data.groupby(['event_date','shape','center_minute','northbound_share','direction']):
            g=g.sort_values('remaining_fraction')
            self.assertTrue(np.all(np.diff(g.mean_wait_minutes)<=1e-8))
            self.assertTrue(np.all(np.diff(g.max_queue)<=1e-8))
        self.assertTrue(data.cleared.all())


if __name__=='__main__':unittest.main()
