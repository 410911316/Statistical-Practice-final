"""Download official OD versions and extract the four-station hourly panel.

Current Azure object modification times are conservative version cutoffs, not
proof of historical public ACLs or the first publication of an earlier version.
Never infer release dates from month names or local filesystem timestamps.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import base64
import hashlib
import json
import io
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / ".local/analysis_v13"
INDEX = "https://data.taipei/api/frontstage/tpeod/dataset/resource.download?rid=eb481f58-1238-4cff-8caa-fa7bb20cb4f4"
STATIONS = ["北門", "大橋頭站", "雙連", "民權西路"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(record):
    path = PRIVATE / "od" / (record["month"] + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = path.with_suffix(".json")
    if path.exists() and stamp.exists():
        saved = json.loads(stamp.read_text("utf-8"))
        if saved.get("transactional_range_md5_verified") and saved["etag"] == record["ETag"] and digest(path) == saved["sha256"]:
            return saved
    md5, sha = hashlib.md5(), hashlib.sha256()
    temporary = path.with_suffix(".part")
    size=int(record['Content-Length']);step=4*1024*1024
    def get_range(start):
        end=min(start+step,size)-1
        response=requests.get(record['url'],headers={'If-Match':record['ETag'],'Range':f'bytes={start}-{end}',
                             'x-ms-range-get-content-md5':'true','x-ms-version':'2023-11-03'},timeout=(20,90))
        response.raise_for_status()
        if (response.status_code!=206 or response.headers.get('ETag','').strip('"')!=record['ETag'].strip('"') or len(response.content)!=end-start+1
            or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}'
            or response.headers.get('Content-MD5')!=base64.b64encode(hashlib.md5(response.content).digest()).decode()):
            raise ValueError('OD range or version changed during download')
        return response.content
    with temporary.open('wb') as stream, ThreadPoolExecutor(max_workers=6) as pool:
        # bounded batches avoid retaining a whole large month in memory
        starts=list(range(0,size,step))
        for first in range(0,len(starts),6):
            for block in pool.map(get_range,starts[first:first+6]):
                stream.write(block);md5.update(block);sha.update(block)
    if temporary.stat().st_size != int(record["Content-Length"]):
        raise ValueError("Incomplete official CSV")
    temporary.replace(path)
    saved = dict(month=record["month"], url=record["url"], etag=record["ETag"],
                 version_at=pd.Timestamp(record["Last-Modified"]).isoformat(),
                 sha256=sha.hexdigest(), md5=base64.b64encode(md5.digest()).decode(),
                 stored_whole_md5=record['Content-MD5'],
                 stored_md5_matches=(base64.b64encode(md5.digest()).decode()==record['Content-MD5']) if record['Content-MD5'] else None,
                 transactional_range_md5_verified=True,
                 bytes=path.stat().st_size, retrieved_at=pd.Timestamp.now(tz="UTC").isoformat())
    stamp.write_text(json.dumps(saved, ensure_ascii=False, indent=2), "utf-8")
    print("downloaded", record["month"], flush=True)
    return saved


def extract(record, dates):
    path = PRIVATE / "od" / (record["month"] + ".csv")
    pieces = []
    for chunk in pd.read_csv(path, encoding="utf-8-sig", dtype={"日期":str,"進站":str,"出站":str}, chunksize=500000):
        chunk["station"] = chunk["進站"].replace({"大橋頭":"大橋頭站"})
        chosen = chunk.loc[chunk["日期"].isin(dates) & chunk.station.isin(STATIONS) & chunk["時段"].between(18,23)].copy()
        if len(chosen): pieces.append(chosen)
    frame = pd.concat(pieces, ignore_index=True)
    if frame.duplicated(["日期","時段","station","出站"]).any():
        raise ValueError("Duplicate OD rows")
    if (frame["人次"] < 0).any() or not frame["人次"].eq(frame["人次"].round()).all():
        raise ValueError("Non-count OD labels")
    result = frame.groupby(["日期","station","時段"], as_index=False).agg(y=("人次","sum"),destinations=("出站","nunique"))
    # The official destination schema changes within May 2023. Require the
    # same destination set across all selected station-hours of EACH date,
    # rather than incorrectly assuming a fixed network across the month.
    sets=frame.groupby(['日期','station','時段'])['出站'].agg(frozenset)
    if (sets.groupby(level=0).nunique().ne(1).any() or result.destinations.min() < 100):
        raise ValueError("Incomplete destination coverage")
    result = result.rename(columns={"日期":"date","時段":"hour"})
    for day, part in result.groupby("date"):
        if len(part) != 24: raise ValueError(f"Missing four-station hours: {day}")
    if set(result.date) != set(dates): raise ValueError("Missing selected date")
    result["month"] = record["month"]
    print("extracted",record["month"],len(result),flush=True)
    return result


def main():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    response=requests.get(INDEX,timeout=45);response.raise_for_status()
    (PRIVATE/'official_od_index.csv').write_bytes(response.content)
    index=pd.read_csv(io.BytesIO(response.content),encoding='utf-8-sig')
    def head(row):
        url=row['URL'].replace('http:','https:');h=requests.head(url,timeout=30);h.raise_for_status()
        return dict(month=f"{int(row['西元年']):04}{int(row['月']):02}",url=url,status=h.status_code,
                    **{k:h.headers.get(k) for k in ['Last-Modified','Content-Length','Content-MD5','ETag']})
    with ThreadPoolExecutor(max_workers=8) as pool: probe=list(pool.map(head,index.to_dict('records')))
    (PRIVATE/'od_head_probe.json').write_text(json.dumps(probe,indent=2),encoding='utf-8')
    dates=pd.read_csv(ROOT/'.local/forecast/analysis_input.csv',usecols=['event_date']).event_date.unique()
    plan=[dict(event_date=d) for d in sorted(dates)]
    # Exclude days already listed as activities, including cancelled scheduled days.
    context = pd.read_csv(ROOT/"data/processed/event_station_hourly_context.csv",usecols=["event_date"])
    excluded = set(context.event_date)
    records = {r["month"]:r for r in probe if r["status"]==200}
    days = {}
    for row in plan:
        day = pd.Timestamp(row["event_date"])
        cutoff = day.tz_localize("Asia/Taipei") - pd.Timedelta(hours=6)
        candidates=[]
        for lag in range(7,57,7):
            d=day-pd.Timedelta(days=lag); key=d.strftime("%Y%m"); record=records.get(key)
            if record and pd.Timestamp(record["Last-Modified"])<=cutoff and str(d.date()) not in excluded:
                candidates.append(str(d.date()))
        row["days"],row["eligible"] = candidates[:4],len(candidates)>=2
        row["cutoff_at"] = cutoff.isoformat()
        if not row["eligible"]: continue
        for d in [*row["days"], row["event_date"]]:
            days.setdefault(d[:7].replace("-",""),set()).add(d)
    (PRIVATE/"baseline_day_plan.json").write_text(json.dumps(plan,indent=2),"utf-8")
    with ThreadPoolExecutor(max_workers=3) as pool:
        ledgers=list(pool.map(fetch,[records[m] for m in sorted(days)]))
    frames=[extract(records[m],sorted(days[m])) for m in sorted(days)]
    panel=pd.concat(frames,ignore_index=True)
    panel.to_csv(PRIVATE/"official_hourly.csv",index=False,encoding="utf-8-sig")
    output=ROOT/"docs/d1_od_version_sources.json"
    output.write_text(json.dumps(dict(index_url=INDEX,
        policy="Use current official blob bytes only if Last-Modified <= D-1 18:00; not first-publication or historical-ACL proof.",
        calendar_policy="same weekday, 7..56 days past, nearest 4/minimum 2; no unversioned holiday calendar features",
        exclusion_policy="remove dates listed in existing activity context; historical occurrence registry, not an attendance feature",
        excluded_events=[r['event_date'] for r in plan if not r['eligible']],
        source_records=ledgers,panel_sha256=digest(PRIVATE/"official_hourly.csv")),ensure_ascii=False,indent=2),"utf-8")
    print("saved",output,flush=True)


if __name__=="__main__": main()
