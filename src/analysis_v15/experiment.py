"""Frozen 18:00 snapshot: prior source bins only, future labels never features."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from analysis_v12.models import Encoder as BaseEncoder

KEYS = ['event_date', 'station', 'hour']
DIRECTIONS = {'origin': ('origin_count', 'origin_complete'),
              'destination': ('destination_count_same_source_bin', 'destination_complete')}
VARIANTS = ['none'] + [f'{d}_w{w}' for d in ['origin', 'both'] for w in [1, 3, 5]]


def columns(variant):
    if variant == 'none':
        return []
    if variant not in VARIANTS:
        raise ValueError('Unknown snapshot variant')
    direction, w = variant.split('_w'); w = int(w)
    kinds = ['origin', 'destination'] if direction == 'both' else ['origin']
    stats = ['last', 'excess'] if w == 1 else ['last', 'sum', 'slope', 'excess']
    return [f'lag_{d}_{s}_w{w}' for d in kinds for s in stats]


def build_features(events, hourly, reference, cutoff=18):
    if cutoff != 18:
        raise ValueError('This protocol freezes source bins 13..17 at nominal 18:00')
    pairs = events[['event_date', 'station']].drop_duplicates().copy()
    pairs['event_date'] = pd.to_datetime(pairs.event_date)
    h = hourly.copy(); h['service_date'] = pd.to_datetime(h.service_date)
    h = h.loc[h.source_hour.isin(range(cutoff-5, cutoff))]
    if h.duplicated(['service_date', 'station', 'source_hour']).any():
        raise ValueError('Duplicate source hour')
    indexed = h.set_index(['service_date', 'station', 'source_hour']).sort_index()
    ref = reference.copy()
    ref['event_date'] = pd.to_datetime(ref.event_date)
    ref['candidate_date'] = pd.to_datetime(ref.candidate_date)
    ref = ref.loc[ref.selected.eq(True)]
    if ref.duplicated(['event_date','station','candidate_date']).any():
        raise ValueError('Duplicate reference day')

    def counts(day, station, bins, direction):
        value, complete = DIRECTIONS[direction]
        idx = pd.MultiIndex.from_product([[day], [station], bins], names=indexed.index.names)
        rows = indexed.reindex(idx)
        if rows[complete].isna().any() or not rows[complete].eq(True).all():
            raise ValueError(f'Missing/incomplete {direction} source bins: {day} {station}')
        x = rows[value].to_numpy(float)
        if not np.isfinite(x).all() or (x < 0).any():
            raise ValueError('Invalid source counts')
        return x

    features, audit = [], []
    for pair in pairs.itertuples(index=False):
        days = sorted(ref.loc[ref.event_date.eq(pair.event_date) & ref.station.eq(pair.station),
                              'candidate_date'].tolist())
        if not days or any(d >= pair.event_date for d in days):
            raise ValueError('Reference days must exist and precede the event')
        record = dict(event_date=pair.event_date, station=pair.station)
        for w in [1, 3, 5]:
            bins = list(range(cutoff-w, cutoff))
            for direction in DIRECTIONS:
                x = counts(pair.event_date, pair.station, bins, direction)
                b = np.array([counts(d, pair.station, bins, direction).sum() for d in days])
                stats = dict(last=x[-1], excess=x.sum()-b.mean())
                if w > 1:
                    stats.update(sum=x.sum(), slope=(x[-1]-x[0])/(w-1))
                for name, value in stats.items():
                    record[f'lag_{direction}_{name}_w{w}'] = float(value)
                audit.append(dict(event_date=str(pair.event_date.date()), station=pair.station,
                                  direction=direction, window=w, bins=bins, reference_days=[str(d.date()) for d in days],
                                  observed_sum=float(x.sum()), reference_sum_mean=float(b.mean())))
        features.append(record)
    return pd.DataFrame(features), audit


class Encoder:
    """Keep the archived design unchanged; standardize added columns on training only."""
    def fit(self, frame, groups, variant):
        self.variant = variant; self.groups = tuple(groups)
        self.base = BaseEncoder().fit(frame, tuple(groups))
        self.lag_columns = columns(variant)
        x = frame[self.lag_columns].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError('Missing snapshot inputs are not zero')
        self.means = x.mean(axis=0); self.scales = x.std(axis=0)
        self.keep = self.scales > 1e-12
        self.scales = np.where(self.keep, self.scales, 1.)
        self.names = self.base.names + [c for c,k in zip(self.lag_columns,self.keep) if k]
        return self

    def transform(self, frame):
        x = frame[self.lag_columns].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError('Invalid snapshot at prediction time')
        return np.column_stack([self.base.transform(frame),
                                ((x-self.means)/self.scales)[:,self.keep]])


@dataclass
class Fitted:
    encoder: Encoder
    estimator: object
    train_end: str
    train_rows: int

    def predict(self, frame):
        return np.maximum(0., self.estimator.predict(self.encoder.transform(frame))).astype(float)


def fit(frame, year, model, variant, groups, params, seed):
    if frame.empty or not pd.to_datetime(frame.event_date).dt.year.lt(year).all():
        raise ValueError('Training must use earlier years only')
    y = frame.y.to_numpy(float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid training labels')
    encoder = Encoder().fit(frame, groups, variant); x = encoder.transform(frame)
    if model == 'ridge':
        if params['lambda'] <= 0:
            raise ValueError('Ridge needs a positive penalty')
        estimator = Ridge(alpha=len(frame)*params['lambda'], fit_intercept=True, solver='svd')
    elif model == 'xgboost':
        estimator = XGBRegressor(objective='reg:squarederror', tree_method='hist', n_jobs=1,
                                 random_state=seed, **params)
    else:
        raise ValueError('Only prespecified models')
    estimator.fit(x, y)
    return Fitted(encoder, estimator, str(frame.event_date.max().date()), len(frame))
