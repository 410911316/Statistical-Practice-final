"""Select training stations before fitting any transformations."""
import numpy as np
import pandas as pd
from analysis_v15.experiment import fit as base_fit, KEYS

ALL = {'北門','大橋頭站','雙連','民權西路'}


def check_panel(frame, stations):
    expected = set(stations)
    if not expected <= ALL or len(expected) != len(stations) or '北門' not in expected:
        raise ValueError('Invalid station cohort')
    if frame.empty or frame.duplicated(KEYS).any() or set(frame.station) != expected:
        raise ValueError('Empty, duplicate or wrong station panel')
    if not pd.to_datetime(frame.event_date).dt.year.eq(frame.year).all():
        raise ValueError('Year/date mismatch')
    if not np.isfinite(frame.y.to_numpy(float)).all() or (frame.y < 0).any():
        raise ValueError('Invalid target counts')
    for _, g in frame.groupby('event_date'):
        if len(g)!=6*len(expected) or set(g.station)!=expected:
            raise ValueError('Incomplete event station coverage')
        for _, h in g.groupby('station'):
            if set(h.hour)!=set(range(18,24)):
                raise ValueError('Incomplete target hours')


def select_train(frame, year, stations):
    if len(set(stations)) != len(stations) or not set(stations)<=ALL or '北門' not in stations:
        raise ValueError('Invalid station cohort')
    # Preserve the exact input ordering, so the four-station reference reproduces V15.
    train = frame.loc[frame.year.lt(year) & frame.station.isin(stations)].copy()
    check_panel(train, stations)
    if not pd.to_datetime(train.event_date).dt.year.lt(year).all():
        raise ValueError('Future data in training')
    return train


def fit_cohort(frame, year, stations, model, variant, groups, params, seed):
    train = select_train(frame, year, stations)
    fitted = base_fit(train, year, model, variant, groups, params, seed)
    if set(fitted.encoder.base.categories['station'])!=set(stations):
        raise ValueError('Excluded station entered encoding')
    return fitted, train


def common_metrics(frame, prediction):
    p=np.asarray(prediction,float); y=frame.y.to_numpy(float)
    if len(p)!=len(y) or not np.isfinite(p).all() or (p<0).any():
        raise ValueError('Invalid predictions')
    e=p-y
    return dict(mae=float(np.mean(abs(e))),rmse=float(np.sqrt(np.mean(e*e))),bias=float(np.mean(e)),
                events=int(frame.event_date.nunique()),rows=len(frame))
