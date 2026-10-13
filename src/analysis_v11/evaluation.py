"""Scores emphasize known scheduled egress windows, not observed peak locations."""
import numpy as np
import pandas as pd

WINDOW = ["planned_end_0_60_beimen", "planned_end_60_120_beimen"]
KEYS = ["event_date","station","hour"]


def checked(frame):
    if frame.empty or frame.duplicated(KEYS).any() or not frame.station.eq("北門").all():
        raise ValueError("Need unique nonempty Beimen panel")
    if not np.isfinite(frame.y.to_numpy(float)).all() or (frame.y < 0).any():
        raise ValueError("Invalid truth")
    for _, event in frame.groupby("event_date"):
        if len(event)!=6 or set(event.hour)!=set(range(18,24)):
            raise ValueError("Incomplete event hours")
    return frame


def window_weights(frame):
    """Equal events, within-event hourly overlap weights. Never disaggregate counts."""
    checked(frame)
    frame = frame.reset_index(drop=True)
    weights = np.zeros(len(frame))
    audit = []
    for date, event in frame.groupby("event_date"):
        values = event[WINDOW].to_numpy(float)
        fallback = None
        if np.isnan(values).all():
            q = np.ones(len(event))
            fallback = "unknown_D1_end_full_six_hours"
        else:
            if not np.isfinite(values).all() or (values<0).any() or (values>1).any() or (values.sum(axis=1)>1+1e-8).any():
                raise ValueError("Partial or invalid planned overlap")
            q = values.sum(axis=1)
            if q.sum()<=0:
                q = np.ones(len(event))
                fallback = "known_end_outside_observation_full_six_hours"
        weights[event.index] = q/q.sum()/frame.event_date.nunique()
        audit.append(dict(event_date=str(pd.Timestamp(date).date()),fallback=fallback,
                          known_overlap_hours=None if np.isnan(values).all() else float(values.sum()),
                          emphasized_source_hours=event.loc[q>0,"hour"].astype(int).tolist()))
    return weights,audit


def costs(frame, predictions, ratios=(1,2,3)):
    p = np.asarray(predictions,float)
    if p.ndim==1:p=p[:,None]
    if p.shape[0]!=len(frame) or not np.isfinite(p).all() or (p<0).any():
        raise ValueError("Invalid candidate predictions")
    if any(not np.isfinite(r) or r<=0 for r in ratios):
        raise ValueError("Invalid cost ratio")
    a,_ = window_weights(frame)
    e = p-frame.y.to_numpy(float)[:,None]
    under,over = a@np.maximum(-e,0),a@np.maximum(e,0)
    return {r:r*under+over for r in ratios}


def select_candidate(past, predictions, cutoff_year, ratio, count):
    checked(past)
    if not pd.to_datetime(past.event_date).dt.year.lt(cutoff_year).all():
        raise ValueError("Current/future labels entered tuning")
    if predictions.ndim!=2 or count<1 or count>predictions.shape[1]:
        raise ValueError("Invalid grid size")
    losses=costs(past,predictions[:,:count],(ratio,))[ratio]
    best=int(np.argmin(losses))
    return best,float(losses[best]),losses


def metrics(frame, prediction):
    p=np.asarray(prediction,float)
    score=costs(frame,p)
    a,audit=window_weights(frame)
    e=p-frame.y.to_numpy(float)
    full_weights=np.array([1/(frame.event_date.nunique()*n) for n in frame.event_date.map(frame.event_date.value_counts())])
    under,peak_abs,peak_hour=[],[],[]
    f=frame.assign(pred=p)
    for _,event in f.groupby("event_date"):
        event=event.sort_values("hour")
        actual=event.loc[event.y.idxmax()]
        under.append(max(0,float(actual.y-actual.pred)))
        peak_abs.append(abs(float(actual.y-actual.pred)))
        peak_hour.append(abs(float(actual.hour-event.loc[event.pred.idxmax(),"hour"])))
    return dict(mae=float(full_weights@abs(e)),rmse=float(np.sqrt(full_weights@(e**2))),
                bias=float(full_weights@e),window_under=float(a@np.maximum(-e,0)),window_over=float(a@np.maximum(e,0)),
                **{f"cost_{r}":float(score[r][0]) for r in (1,2,3)},
                peak_under=float(np.mean(under)),peak_abs=float(np.mean(peak_abs)),peak_hour=float(np.mean(peak_hour)),
                events=frame.event_date.nunique(),fallback_events=sum(x["fallback"] is not None for x in audit))

