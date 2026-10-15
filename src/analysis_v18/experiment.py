"""Two-coefficient correction learned only from forward-held-out residuals."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from analysis_v11.evaluation import WINDOW, KEYS, checked


def scheduled_gate(frame):
    """Only station and announcement-derived overlap enter this routing rule."""
    q = np.zeros((len(frame), 2))
    f = frame.reset_index(drop=True)
    for _, event in f.loc[f.station.eq("北門")].groupby("event_date"):
        values = event[WINDOW].to_numpy(float)
        if np.isnan(values).all():
            continue
        if (not np.isfinite(values).all() or (values < 0).any()
                or (values > 1).any() or (values.sum(axis=1) > 1+1e-9).any()):
            raise ValueError("Partial or invalid scheduled overlap")
        q[event.index] = values
    return q


def calibration_data(past, cutoff_year):
    """Whole North events; reject contemporaneous labels instead of silently dropping."""
    checked(past)
    f = past.reset_index(drop=True)
    dates = pd.to_datetime(f.event_date)
    if not dates.dt.year.lt(cutoff_year).all():
        raise ValueError("Current/future labels in correction training")
    train_end = pd.to_datetime(f.base_train_end)
    if not train_end.lt(pd.to_datetime(dates.dt.year.astype(str)+"-01-01")).all():
        raise ValueError("Residuals are not from forward-held-out predictions")
    if not (f.parameter_validation_end_year < dates.dt.year).all():
        raise ValueError("Hyperparameters used held-out year labels")
    b = f.base_prediction.to_numpy(float)
    if not np.isfinite(b).all() or (b < 0).any():
        raise ValueError("Invalid base prediction")
    q = scheduled_gate(f)
    a = np.zeros(len(f))
    eligible = []
    for date, event in f.groupby("event_date"):
        exposure = q[event.index].sum(axis=1)
        if exposure.sum() > 0:
            a[event.index] = exposure/exposure.sum()
            eligible.append(str(pd.Timestamp(date).date()))
    if eligible:
        a /= len(eligible)
    return f, b, q, a, eligible


@dataclass
class Correction:
    cutoff_year: int
    under_ratio: float
    bound: float
    penalty: float
    theta: list
    eligible_dates: list
    scale: float
    objective: float | None
    status: str

    def predict(self, frame, base):
        b = np.asarray(base, float)
        if b.shape != (len(frame),) or not np.isfinite(b).all() or (b < 0).any():
            raise ValueError("Invalid prediction vector")
        return b * (1 + scheduled_gate(frame) @ np.asarray(self.theta))

    def record(self):
        return asdict(self)


def fit_correction(past, cutoff_year, ratio=2, bound=.25, penalty=.1, min_events=6):
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Invalid underestimation ratio")
    if not np.isfinite(bound) or not 0 < bound < 1:
        raise ValueError("Correction bound must lie in (0,1)")
    if not np.isfinite(penalty) or penalty < 0 or min_events < 1:
        raise ValueError("Invalid regularization or support")
    f, b, q, a, dates = calibration_data(past, cutoff_year)
    record = dict(cutoff_year=int(cutoff_year), under_ratio=float(ratio),
                  bound=float(bound), penalty=float(penalty), eligible_dates=dates)
    if len(dates) < min_events:
        return Correction(**record, theta=[0., 0.], scale=1., objective=None,
                          status="insufficient_known_egress_events")
    keep = a > 0
    scale = max(1., float(a @ b))
    X = b[keep, None]*q[keep]/scale
    r = (f.y.to_numpy(float)[keep]-b[keep])/scale
    w = a[keep]; n = len(r)
    # Variables [theta(2), under(n), over(n), abs_theta(2)].
    c = np.r_[np.zeros(2), ratio*w, w, np.full(2, penalty)]
    Aeq = np.c_[X, np.eye(n), -np.eye(n), np.zeros((n, 2))]
    Aub = np.zeros((4, 2+2*n+2))
    Aub[:2, :2] = np.eye(2); Aub[:2, -2:] = -np.eye(2)
    Aub[2:, :2] = -np.eye(2); Aub[2:, -2:] = -np.eye(2)
    solved = linprog(c, A_ub=Aub, b_ub=np.zeros(4), A_eq=Aeq, b_eq=r,
                     bounds=[(-bound, bound)]*2+[(0, None)]*(2*n+2), method="highs")
    if not solved.success:
        raise RuntimeError(solved.message)
    return Correction(**record, theta=solved.x[:2].tolist(), scale=scale,
                      objective=float(solved.fun), status="fitted")
