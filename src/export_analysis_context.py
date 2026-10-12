"""Export four-station hourly retrospective context; preserve weather measurement semantics."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from build_calendar_baseline import POLICY, STATIONS
from import_event_finance import sql_value
from load_data import mysql
from project_paths import PROCESSED

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--login-path",default="codex-local")
    args=parser.parse_args()
    # Hour-ending weather is for retrospective explanation of the bin, not information available at its start.
    sql=f"""SELECT JSON_OBJECT(
      'event_date',CAST(h.event_date AS CHAR),'hour_start',CAST(h.hour_start AS CHAR),'station',h.station,
      'event_status',e.event_status,'origin_count',h.origin_count,
      'destination_count_same_source_bin',h.destination_count_same_source_bin,
      'baseline_origin_mean_28d',h.baseline_origin_mean_28d,'baseline_n_days_28d',h.baseline_n_days,
      'is_holiday',c.is_holiday,'is_makeup_workday',c.is_makeup_workday,'calendar_note',c.day_note,
      'nominal_show_end',CAST(t.nominal_show_end AS CHAR),'nominal_end_basis',t.nominal_end_basis,
      'scheduled_stage_end',CAST(t.scheduled_stage_end AS CHAR),'stage_scope',t.stage_scope,
      'actual_show_end',CAST(t.actual_show_end AS CHAR),'actual_stage_end',CAST(t.actual_stage_end AS CHAR),
      'show_source_url',t.show_source_url,'additional_timing_source_url',t.additional_source_url,
      'actual_time_source_url',t.actual_time_source_url,
      'weather_station_id',w.station_id,'weather_raw_label',CAST(w.observed_at AS CHAR),
      'weather_report_hour_end',CAST(w.official_report_hour_end AS CHAR),
      'temperature_at_report_time_c',w.air_temperature_c,'wind_preceding_10min_mean_m_s',w.mean_wind_speed_m_s,
      'precipitation_hour_ending_mm',w.hourly_precipitation_mm,'precipitation_raw',w.precipitation_raw_value,
      'precipitation_status',w.precipitation_status,'cloud_at_report_time_tenths',w.total_cloud_amount_tenths,
      'cloud_measurement_type',w.cloud_measurement_type,'cloud_method_verified',w.cloud_method_verified,
      'baseline_window_mean_56d',b.baseline_origin_mean,'baseline_n_days_56d',b.baseline_n_days,
      'baseline_dates_56d',b.baseline_dates,'baseline_56d_eligible',b.eligible)
    FROM taipei_metro_analysis.analysis_event_station_hourly h
    JOIN taipei_metro_analysis.dadaocheng_fireworks_events e USING(event_date)
    JOIN taipei_metro_analysis.calendar_day c ON c.calendar_date=h.event_date
    LEFT JOIN taipei_metro_analysis.event_timing_context t USING(event_date)
    LEFT JOIN taipei_metro_analysis.event_weather_station ew ON ew.event_date=h.event_date AND ew.station_role='primary'
    LEFT JOIN taipei_metro_analysis.v_weather_verified w ON w.station_id=ew.station_id
       AND w.official_report_hour_end=DATE_ADD(h.hour_start,INTERVAL 1 HOUR) AND w.values_match=1
    LEFT JOIN taipei_metro_analysis.analysis_calendar_baseline b ON b.event_date=h.event_date AND b.station=h.station
       AND b.policy_name={sql_value(POLICY)} AND b.window_start_hour=18 AND b.window_end_hour_exclusive=24
    WHERE HOUR(h.hour_start)>=18 AND HOUR(h.hour_start)<24 AND DATE(h.hour_start)=h.event_date
      AND h.station IN ({','.join(sql_value(s) for s in STATIONS)})
    ORDER BY h.event_date,h.station,h.hour_start;"""
    rows=[json.loads(line) for line in mysql(sql,args.login_path).splitlines() if line]
    data=pd.DataFrame(rows)
    expected=int(mysql("SELECT COUNT(*)*24 FROM taipei_metro_analysis.dadaocheng_fireworks_events;",args.login_path))
    if len(data)!=expected or data.duplicated(["event_date","hour_start","station"]).any():
        raise RuntimeError("Context join changed the number of event/station/hour rows")
    data["baseline_dates_56d"]=data.baseline_dates_56d.map(json.dumps)
    data.to_csv(PROCESSED/"event_station_hourly_context.csv",index=False,encoding="utf-8-sig")
    print(json.dumps(dict(rows=len(data),weather_matched=int(data.weather_station_id.notna().sum()),
                          actual_show_end_available=int(data.actual_show_end.notna().sum())),ensure_ascii=False))

if __name__=="__main__":main()
