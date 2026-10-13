"""Build baselines exclusively from timestamp-screened official OD versions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from analysis_v8.features import attach_end_features

KEYS = ["event_date", "station", "hour"]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def require_features(frame):
    for c in ['available_baseline','baseline_version_at','cutoff_at']:
        if c not in frame or frame[c].isna().any(): raise ValueError('Missing version-screened baseline')
    if (not np.isfinite(frame.available_baseline).all() or (frame.available_baseline<0).any()
        or (pd.to_datetime(frame.baseline_version_at,utc=True)>pd.to_datetime(frame.cutoff_at,utc=True)).any()):
        raise ValueError('Future or invalid baseline input')


def eligible_training(frame, year):
    fit_at=pd.Timestamp(f'{year}-01-01',tz='Asia/Taipei')
    return frame.loc[frame.year.lt(year) & (pd.to_datetime(frame.label_version_at,utc=True)<=fit_at)].copy()


def load(root):
    private=root/'.local/analysis_v13'
    ledger=json.loads((root/'docs/d1_od_version_sources.json').read_text('utf-8'))
    if sha(private/'official_hourly.csv') != ledger['panel_sha256']:
        raise ValueError('Official aggregate changed')
    versions={r['month']:pd.Timestamp(r['version_at']) for r in ledger['source_records']}
    official=pd.read_csv(private/'official_hourly.csv')
    official['date']=pd.to_datetime(official.date)
    snapshot=pd.read_csv(root/'.local/analysis_v8/both/analysis_input.csv',parse_dates=['event_date'])
    # Do not carry over any observed weather/attendance or old baseline input.
    frame=snapshot[KEYS+['year','y','is_weekend','day_of_year','year_index']].copy()
    info={'audit':{'d1_deployable':False,'model_feature_whitelist':[]},'manifest':[],'feature_eligibility':{}}
    frame,_,_=attach_end_features(frame,info,root,'docs/d1_show_end_sources.json',18)
    plan=json.loads((private/'baseline_day_plan.json').read_text('utf-8'))
    rows,audit=[],[]
    for entry in plan:
        date=pd.Timestamp(entry['event_date']);cutoff=date.tz_localize('Asia/Taipei')-pd.Timedelta(hours=6)
        if not entry['eligible']:
            audit.append(dict(event_date=str(date.date()),eligible=False,reason='no_pre_cutoff_current_version',days='[]'))
            continue
        days=pd.to_datetime(entry['days']);v=max(versions[d.strftime('%Y%m')] for d in days)
        lags=(date-days).days
        if not 2<=len(days)<=4 or len(set(days))!=len(days) or not ((lags>=7)&(lags<=56)&(lags%7==0)).all() or v>cutoff:
            raise ValueError('Invalid baseline source selection')
        past=official.loc[official.date.isin(days)]
        if len(past)!=24*len(days):raise ValueError('Incomplete baseline source panel')
        means=past.groupby(['station','hour'],as_index=False).y.mean().rename(columns={'y':'available_baseline'})
        means['event_date']=date;means['baseline_version_at']=v.isoformat();means['cutoff_at']=cutoff.isoformat()
        means['label_version_at']=versions[date.strftime('%Y%m')].isoformat()
        rows.append(means)
        audit.append(dict(event_date=str(date.date()),eligible=True,days=json.dumps(entry['days']),
            latest_observation=str(days.max().date()),max_version_at=v.isoformat(),cutoff_at=cutoff.isoformat(),
            min_lag_days=int(lags.min()),max_lag_days=int(lags.max()),reason='official_current_version_screened'))
    frame=frame.merge(pd.concat(rows),on=KEYS,validate='one_to_one')
    check=frame.merge(official.rename(columns={'date':'event_date','y':'official_y'})[KEYS+['official_y']],on=KEYS,validate='one_to_one')
    if len(check)!=len(frame) or not check.y.eq(check.official_y).all():raise ValueError('Official targets differ from historical reference')
    require_features(frame)
    if frame[KEYS].duplicated().any():raise ValueError('Duplicate model input')
    return frame.sort_values(KEYS).reset_index(drop=True),pd.DataFrame(audit)
