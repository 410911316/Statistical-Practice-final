import csv
from datetime import date, datetime, timedelta
from decimal import Decimal
import io
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from build_calendar_baseline import choose_candidates
from import_calendar import parse_calendar
from import_event_timing import timing_row
from verify_weather_source import check_row, report_hour_end


class CalendarTests(unittest.TestCase):
    def content(self, mutate=None):
        rows=[]
        for i in range(365):
            d=date(2023,1,1)+timedelta(days=i)
            rows.append([d.strftime("%Y%m%d"),"一二三四五六日"[d.weekday()],0 if d==date(2023,2,18) or d.weekday()<5 else 2,""])
        if mutate:mutate(rows)
        stream=io.StringIO();writer=csv.writer(stream)
        writer.writerow(["西元日期","星期","是否放假","備註"]);writer.writerows(rows)
        return stream.getvalue().encode("utf-8-sig")

    def test_makeup_saturday_not_weekend_holiday(self):
        rows=parse_calendar(self.content(),2023,"https://example.test/calendar.csv")
        makeup=next(r for r in rows if r["calendar_date"]=="2023-02-18")
        self.assertEqual((makeup["is_weekend"],makeup["is_holiday"],makeup["is_makeup_workday"]),(1,0,1))

    def test_duplicate_day_rejected(self):
        with self.assertRaisesRegex(ValueError,"incomplete/duplicated"):
            parse_calendar(self.content(lambda rows:rows.__setitem__(1,rows[0])),2023,"u")

    def test_weekday_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,"Weekday mismatch"):
            parse_calendar(self.content(lambda rows:rows[0].__setitem__(1,"一")),2023,"u")


class BaselineTests(unittest.TestCase):
    def test_event_holiday_partial_bins_excluded_before_nearest_four_selection(self):
        event="2023-08-16"
        rows=[dict(event_date=event,station="北門",candidate_date=(date(2023,8,16)-timedelta(days=i*7)).isoformat(),lag_days=i*7,
                   complete_hours=6,observed_hours=6,candidate_origin_total=99999 if i==1 else i) for i in range(1,9)]
        cal={d:dict(is_holiday=0,is_makeup_workday=0,day_note="") for d in [event]+[r["candidate_date"] for r in rows]}
        cal[rows[1]["candidate_date"]]["is_holiday"]=1
        rows[2]["complete_hours"]=5
        selected=choose_candidates(pd.DataFrame(rows),{event,rows[0]["candidate_date"]},cal)
        self.assertEqual(selected.loc[selected.selected,"lag_days"].tolist(),[28,35,42,49])
        self.assertEqual(selected.reason.tolist(),["event_date_excluded","calendar_class_mismatch","missing_or_incomplete_hours","selected","selected","selected","selected","beyond_nearest_four"])


class WeatherTests(unittest.TestCase):
    def test_hour_24_keeps_next_day_boundary_not_23_hour(self):
        self.assertEqual(report_hour_end(datetime(2025,8,6,23,59)),datetime(2025,8,7))
        self.assertEqual(report_hour_end(datetime(2025,8,6,23)),datetime(2025,8,6,23))
        with self.assertRaises(ValueError):report_hour_end(datetime(2025,8,6,20,30))

    def example(self):
        local=dict(air_temperature_c=Decimal("30"),mean_wind_speed_m_s=Decimal("0.3"),precipitation_raw_value="-9.8",
                   temperature_quality_flag=None,wind_speed_quality_flag=None,precipitation_quality_flag=None,
                   total_cloud_amount_tenths=Decimal("0"),cloud_measurement_type="satellite_retrieved")
        remote=dict(AirTemperature={"Instantaneous":30},WindSpeed={"Mean":0.3,"TenMinutelyMaximum":1.1},
                    Precipitation={"Accumulation":-9.8},TotalCloudAmount={"Instantaneous":None,"SatRetrieved":0})
        return local,remote

    def test_historical_satellite_and_trace_match_exact_source_fields(self):
        local,remote=self.example();self.assertEqual(check_row(local,remote),([],True))
        remote["Precipitation"]["Accumulation"]=0
        self.assertIn("precipitation_raw_value",check_row(local,remote)[0])

    def test_manual_cloud_cannot_be_substituted_for_satellite(self):
        local,remote=self.example();local["cloud_measurement_type"]="manual_instantaneous"
        self.assertFalse(check_row(local,remote)[1])


class TimingTests(unittest.TestCase):
    def event(self, **kw):
        return dict(event_date="2026-07-25",event_status="held",announced_show_start_time="20:00:00",
                    fireworks_duration_seconds=480,combined_show_duration_seconds=1200,official_source_url="https://example.test",**kw)

    def test_drone_combined_duration_not_firework_duration(self):
        r=timing_row(self.event(),{},"0"*64)
        self.assertEqual(r["nominal_show_end"],"2026-07-25 20:20:00")
        self.assertIsNone(r["actual_show_end"])

    def test_cancelled_date_has_no_simulated_end(self):
        e=self.event();e["event_status"]="cancelled"
        r=timing_row(e,{},"0"*64)
        self.assertIsNone(r["nominal_show_end"])
        self.assertEqual(r["nominal_end_basis"],"not_held")

    def test_2018_window_is_not_exact_start(self):
        e=self.event();e.update(event_date="2018-08-18",announced_show_start_time=None,combined_show_duration_seconds=380)
        r=timing_row(e,{},"0"*64)
        self.assertIsNone(r["nominal_show_end"])
        self.assertEqual(r["announced_show_window_end"],"20:30:00")


if __name__=="__main__":unittest.main()
