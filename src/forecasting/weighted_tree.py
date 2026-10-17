"""Frozen weighted XGBoost; weights are planned-window overlaps."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from .features import Encoder

KEYS=['event_date','station','hour']


WINDOW=['planned_end_0_60_beimen','planned_end_60_120_beimen']


STATIONS={'北門','大橋頭站','雙連','民權西路'}


def panel_check(frame):
    if frame.empty or frame.duplicated(KEYS).any():raise ValueError('Empty/duplicate activity panel')
    if not np.isfinite(frame.y).all() or (frame.y<0).any():raise ValueError('Invalid count labels')
    if not frame.year.eq(frame.event_date.dt.year).all():raise ValueError('Year/date mismatch')
    for _,event in frame.groupby('event_date'):
        if len(event)!=24 or set(event.station)!=STATIONS or any(set(g.hour)!=set(range(18,24)) for _,g in event.groupby('station')):
            raise ValueError('Require complete four-station six-hour events')


def training_weights(frame,multiplier):
    panel_check(frame)
    if not np.isfinite(multiplier) or multiplier<1:raise ValueError('Invalid training multiplier')
    f=frame.reset_index(drop=True);raw=np.ones(len(f));audit=[]
    for date,g in f.groupby('event_date'):
        north=g.loc[g.station.eq('北門')];values=north[WINDOW].to_numpy(float)
        unknown=np.isnan(values).all()
        if unknown:q=np.zeros(len(north))
        else:
            if (not np.isfinite(values).all() or (values<0).any() or (values>1).any() or (values.sum(axis=1)>1+1e-8).any()):
                raise ValueError('Partial/invalid D1 planned window')
            q=values.sum(axis=1)
        raw[north.index]=1+(multiplier-1)*q
        raw[g.index]*=len(g)/raw[g.index].sum()
        audit.append(dict(event_date=str(date.date()),unknown_end=bool(unknown),
            boosted_hours=north.loc[q>0,'hour'].astype(int).tolist(),
            beimen_weight=float(raw[north.index].sum()),event_weight=float(raw[g.index].sum())))
    return raw,audit


def mae_nonnegative(y_true,y_pred):
    return float(np.mean(abs(y_true-np.maximum(y_pred,0))))


def estimator(params,trees,cfg,stopping=False):
    extra=dict(early_stopping_rounds=cfg['patience'],eval_metric=mae_nonnegative) if stopping else {}
    return XGBRegressor(objective='reg:squarederror',tree_method='hist',n_jobs=1,random_state=cfg['seed'],
                        n_estimators=int(trees),**params,**extra)


@dataclass
class Fitted:
    encoder: object
    estimator: object
    params: dict
    multiplier: int

    def predict(self,frame):
        return np.maximum(self.estimator.predict(self.encoder.transform(frame)),0)


def fit_final(train,year,params,multiplier,cfg,groups):
    if not train.year.lt(year).all():raise ValueError('Current/future labels in final fit')
    w,_=training_weights(train,multiplier);enc=Encoder().fit(train,tuple(groups));params=dict(params)
    trees=params.pop('n_estimators')
    model=estimator(params,trees,cfg).fit(enc.transform(train),train.y,sample_weight=None if multiplier==1 else w)
    return Fitted(enc,model,{**params,'n_estimators':trees},multiplier)
