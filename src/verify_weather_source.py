"""Verify the existing weather CSV against public CODiS responses; never rewrite raw values."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json

import pandas as pd
import requests

from import_weather import DEFAULT_CSV, parse_csv, sql_value
from load_data import mysql
from project_paths import ROOT

API = "https://codis.cwa.gov.tw/api/station"
CONFIG = "https://codis.cwa.gov.tw/Pages/StationData/stn-tool/stn-config.js"
README = "https://codis.cwa.gov.tw/pdf_data/Readme.pdf"
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS taipei_metro_analysis.weather_source_verification (
 station_id VARCHAR(10) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
 observed_at DATETIME NOT NULL,
 source_file_sha256 CHAR(64) NOT NULL,
 official_report_hour_end DATETIME NOT NULL COMMENT '23:59 interpreted as report hour 24; not all variables are hourly means',
 values_match BOOLEAN NOT NULL,
 cloud_method_verified BOOLEAN NOT NULL,
 response_sha256 CHAR(64) NOT NULL,
 request_parameters JSON NOT NULL,
 verified_at_utc DATETIME NOT NULL,
 PRIMARY KEY(station_id,observed_at),
 FOREIGN KEY(station_id,observed_at) REFERENCES taipei_metro_analysis.weather_observation(station_id,observed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE OR REPLACE VIEW taipei_metro_analysis.v_weather_verified AS
SELECT w.*,v.official_report_hour_end,v.values_match,v.cloud_method_verified,
 CASE WHEN v.cloud_method_verified=1 THEN 0 ELSE w.cloud_method_review_required END AS effective_cloud_review_required,
 v.response_sha256,v.verified_at_utc
FROM taipei_metro_analysis.weather_observation w
LEFT JOIN taipei_metro_analysis.weather_source_verification v
 ON v.station_id=w.station_id AND v.observed_at=w.observed_at
 AND v.source_file_sha256=w.source_file_sha256;
"""


def report_hour_end(stamp: datetime) -> datetime:
    if stamp.hour==23 and stamp.minute==59:
        return stamp.replace(hour=0,minute=0,second=0)+timedelta(days=1)
    if stamp.minute or stamp.second or stamp.hour==0:
        raise ValueError(f"Unexpected CODiS hourly label: {stamp}")
    return stamp


def equal_number(a, b):
    if a in (None, "") or b in (None, ""):
        return a in (None, "") and b in (None, "")
    return Decimal(str(a)) == Decimal(str(b))


def check_row(local, remote):
    pairs = [("air_temperature_c", "AirTemperature", "Instantaneous"),
             ("mean_wind_speed_m_s", "WindSpeed", "Mean"),
             ("precipitation_raw_value", "Precipitation", "Accumulation")]
    errors=[field for field, group, key in pairs if not equal_number(local[field],remote.get(group,{}).get(key))]
    for field,group,key in [("temperature_quality_flag","AirTemperature","Instantaneousf"),
                            ("wind_speed_quality_flag","WindSpeed","Meanf"),
                            ("precipitation_quality_flag","Precipitation","Accumulationf")]:
        a,b=local[field],remote.get(group,{}).get(key)
        if ("" if a is None else str(a)) != ("" if b is None else str(b)):
            errors.append(field)
    clouds=remote.get("TotalCloudAmount",{})
    method=local["cloud_measurement_type"]
    field={"manual_instantaneous":"Instantaneous","satellite_retrieved":"SatRetrieved"}.get(method)
    if field:
        cloud_ok=equal_number(local["total_cloud_amount_tenths"],clouds.get(field))
    else:
        cloud_ok=local["total_cloud_amount_tenths"] is None and all(clouds.get(k) is None for k in ("Instantaneous","SatRetrieved"))
    if not cloud_ok:errors.append("cloud_value_or_method")
    return errors, cloud_ok


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mysql",action="store_true")
    parser.add_argument("--refresh",action="store_true",help="Re-download cached public station-day responses")
    parser.add_argument("--login-path",default="codex-local")
    args=parser.parse_args()
    tables,source_sha=parse_csv(DEFAULT_CSV)
    observations=tables["weather_observation"]
    cache=ROOT/"data/original_data/weather_verification"
    cache.mkdir(parents=True,exist_ok=True)
    config_path=cache/"stn-config.js"
    if not config_path.exists() or args.refresh:
        r=requests.get(CONFIG,timeout=(10,45));r.raise_for_status();config_path.write_bytes(r.content)
    config=config_path.read_text(encoding="utf-8-sig")
    # The official table renders :59 as hour 24; the chart advances 23:59 by one minute.
    if 'o.DataTime.substr(11, 5) === "23:59"' not in config or 't += 60000' not in config:
        raise RuntimeError("CODiS time conversion changed; review the official JavaScript")
    groups=sorted({(r["station_id"],r["observed_at"].strftime("%Y-%m-%d")) for r in observations})
    def fetch(group):
        station,day=group
        parameters=dict(date=day,type="report_date",stn_ID=station,
                        stn_type="auto_C0" if station.startswith("C0") else "cwb",more="",
                        start=day+"T00:00:00",end=day+"T23:59:59",item="")
        path=cache/f"{station}_{day}.json"
        if not path.exists() or args.refresh:
            response=requests.post(API,data=parameters,timeout=(10,60));response.raise_for_status()
            data=response.json()
            if data.get("code") != 200:raise RuntimeError(f"CODiS unavailable for {station}/{day}")
            path.write_bytes(response.content)
        content=path.read_bytes();data=json.loads(content)
        if data.get("code")!=200 or data.get("metadata",{}).get("timezone")!="+08:00":
            raise ValueError("Unexpected CODiS response")
        station_data=data.get("data",[])
        if len(station_data)!=1 or station_data[0]["StationID"]!=station:
            raise ValueError(f"Missing station-day: {station}/{day}")
        remote={r["DataTime"]:r for r in station_data[0]["dts"]}
        if len(remote)!=24 or len(station_data[0]["dts"])!=24:
            raise ValueError(f"Incomplete/duplicate station-day: {station}/{day}")
        print(f"Verified response {station}/{day}",flush=True)
        return group,(remote,hashlib.sha256(content).hexdigest(),parameters)
    with ThreadPoolExecutor(max_workers=3) as pool:
        fetched=dict(pool.map(fetch,groups))
    rows,issues=[],[]
    now=datetime.now(timezone.utc).replace(tzinfo=None,microsecond=0)
    for local in observations:
        stamp=local["observed_at"];day=stamp.strftime("%Y-%m-%d")
        remote,digest,params=fetched[(local["station_id"],day)]
        key=stamp.isoformat(timespec="seconds")
        if key not in remote:raise ValueError(f"Timestamp absent in official response: {key}")
        errors,cloud_ok=check_row(local,remote[key])
        if errors:issues.append(dict(station_id=local["station_id"],observed_at=key,fields=errors))
        rows.append(dict(station_id=local["station_id"],observed_at=stamp,source_file_sha256=source_sha,
                         official_report_hour_end=report_hour_end(stamp),values_match=int(not errors),
                         cloud_method_verified=int(cloud_ok),response_sha256=digest,
                         request_parameters=json.dumps(params),verified_at_utc=now))
    output=ROOT/"results/weather";output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output/"source_verification.csv",encoding="utf-8-sig",index=False)
    summary=dict(rows=len(rows),station_days=len(groups),values_matched=sum(r["values_match"] for r in rows),
                 cloud_methods_verified=sum(r["cloud_method_verified"] for r in rows),
                 hour_24_labels=sum(r["observed_at"].minute==59 for r in rows),issues=issues,
                 historical_satellite_verified=sum(r["cloud_method_review_required"] for r in observations) if not issues else None,
                 source_file_sha256=source_sha,api=API,official_time_conversion=CONFIG,
                 config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),element_definitions=README,
                 interpretation="Report hour end is not an hourly mean for all elements. Rain is accumulation with special codes retained; wind is previous ten-minute mean; temperature/cloud are reported observation values.")
    (output/"source_verification_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    if args.mysql:
        if issues:raise RuntimeError(f"{len(issues)} mismatches; audit saved, no verification rows published")
        mysql(SCHEMA_SQL,args.login_path)
        cols=list(rows[0]);statements=["START TRANSACTION;"]
        for i in range(0,len(rows),200):
            values=",".join("("+",".join(sql_value(r[c]) for c in cols)+")" for r in rows[i:i+200])
            updates=",".join(f"{c}=incoming.{c}" for c in cols if c not in {"station_id","observed_at"})
            statements.append(f"INSERT INTO taipei_metro_analysis.weather_source_verification ({','.join(cols)}) VALUES {values} AS incoming ON DUPLICATE KEY UPDATE {updates};")
        statements.append("COMMIT;");mysql("\n".join(statements),args.login_path)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
