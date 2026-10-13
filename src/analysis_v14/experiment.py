"""Keep stopping labels, hyperparameter selection labels and outer labels separate."""
from dataclasses import dataclass
from itertools import product
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from analysis_v12.models import Encoder

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


def shapes(cfg):
    return [dict(max_depth=d,learning_rate=e,min_child_weight=k,**cfg['fixed'])
            for d,e,k in product(cfg['max_depth'],cfg['learning_rate'],cfg['min_child_weight'])]


def split(frame,year,cfg):
    panel_check(frame)
    train=frame.loc[frame.year.lt(year)].sort_values(KEYS).reset_index(drop=True)
    valid=frame.loc[frame.year.eq(year)&frame.station.eq('北門')].sort_values(KEYS).reset_index(drop=True)
    if train.event_date.nunique()<cfg['min_base_events'] or valid.empty:raise ValueError('Insufficient prior events')
    stop_year=int(train.year.max())
    inner=train.loc[train.year.lt(stop_year)].reset_index(drop=True)
    stop=train.loc[train.year.eq(stop_year)&train.station.eq('北門')].reset_index(drop=True)
    active=inner.event_date.nunique()>=cfg['min_inner_training_events']
    if active and not inner.year.max()<stop.year.min()<year:raise ValueError('Stopping chronology violated')
    return train,valid,inner,stop,active


def mae_nonnegative(y_true,y_pred):
    return float(np.mean(abs(y_true-np.maximum(y_pred,0))))


def estimator(params,trees,cfg,stopping=False):
    extra=dict(early_stopping_rounds=cfg['patience'],eval_metric=mae_nonnegative) if stopping else {}
    return XGBRegressor(objective='reg:squarederror',tree_method='hist',n_jobs=1,random_state=cfg['seed'],
                        n_estimators=int(trees),**params,**extra)


def candidates(frame,year,multiplier,cfg,groups):
    train,valid,inner,stop,active=split(frame,year,cfg)
    enc=Encoder().fit(train,tuple(groups));x=enc.transform(train);z=enc.transform(valid)
    w,weight_audit=training_weights(train,multiplier)
    if active:
        enc_inner=Encoder().fit(inner,tuple(groups));xi=enc_inner.transform(inner);xs=enc_inner.transform(stop)
        wi,_=training_weights(inner,multiplier)
    fixed=[];es=[];records=[];fixed_grid=[]
    for j,params in enumerate(shapes(cfg)):
        trees=cfg['fallback_trees'];curve=[];stop_score=None
        if active:
            stopping=estimator(params,cfg['max_trees'],cfg,True).fit(xi,inner.y,
                sample_weight=None if multiplier==1 else wi,eval_set=[(xs,stop.y)],verbose=False)
            trees=int(stopping.best_iteration)+1
            curve=list(stopping.evals_result()['validation_0']['mae_nonnegative'])
            if trees!=int(np.argmin(curve))+1:raise ValueError('Wrong early-stopping minimum')
            stop_score=mae_nonnegative(stop.y.to_numpy(),stopping.predict(xs))
            if abs(stop_score-curve[trees-1])>1e-3:raise ValueError('Stopped prediction/score differ')
        full=estimator(params,cfg['max_trees'],cfg).fit(x,train.y,sample_weight=None if multiplier==1 else w)
        for n in cfg['fixed_trees']:
            fixed.append(np.maximum(full.predict(z,iteration_range=(0,n)),0));fixed_grid.append(dict(shape_id=j,trees=n,params=params))
        es.append(np.maximum(full.predict(z,iteration_range=(0,trees)),0))
        records.append(dict(shape_id=j,params=params,trees=trees,stopping_active=active,
            inner_train_events=int(inner.event_date.nunique()) if active else 0,
            inner_train_end=str(inner.event_date.max().date()) if active else None,
            stop_start=str(stop.event_date.min().date()) if active else None,
            stop_end=str(stop.event_date.max().date()) if active else None,
            stop_events=int(stop.event_date.nunique()) if active else 0,
            outer_year=int(year),curve=curve,stop_score=stop_score,
            cap_reached=bool(len(curve)==cfg['max_trees']),best_at_cap=trees==cfg['max_trees'],
            fallback=None if active else f"too_few_earlier_training_events_use_prespecified_{cfg['fallback_trees']}"))
    return dict(fixed=np.column_stack(fixed),es=np.column_stack(es)),dict(
        year=int(year),multiplier=multiplier,train_events=int(train.event_date.nunique()),
        train_start=str(train.event_date.min().date()),train_end=str(train.event_date.max().date()),
        validation_dates=[str(d.date()) for d in valid.event_date.unique()],
        stopping=records,fixed_grid=fixed_grid,weight_audit=weight_audit,encoder_columns=enc.names,
        inner_encoder_columns=enc_inner.names if active else [])


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


def select_past(timeline,matrix,year):
    if timeline.empty or not timeline.year.lt(year).all():raise ValueError('Current/future labels used to select')
    if timeline.duplicated(KEYS).any() or not timeline.station.eq('北門').all():raise ValueError('Invalid OOF keys')
    if any(len(g)!=6 or set(g.hour)!=set(range(18,24)) for _,g in timeline.groupby('event_date')):
        raise ValueError('Incomplete selection events')
    if matrix.shape[0]!=len(timeline) or not np.isfinite(matrix).all():raise ValueError('Invalid candidate predictions')
    losses=np.mean(abs(matrix-timeline.y.to_numpy()[:,None]),axis=0)
    j=int(np.argmin(losses));return j,losses
