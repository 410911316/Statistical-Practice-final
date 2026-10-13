"""Chronological convex stacking using symmetric or asymmetric absolute loss."""
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,eye,hstack
from analysis_v11.evaluation import window_weights,checked

MODELS=('ridge','gam','xgboost')


def solve(frame,cutoff_year,ratio=2,window=True):
    checked(frame)
    if not frame.year.lt(cutoff_year).all():raise ValueError('Future labels in fusion')
    if (pd.to_datetime(frame.label_version_at,utc=True)>pd.Timestamp(f'{cutoff_year}-01-01',tz='Asia/Taipei')).any():
        raise ValueError('Unreleased OOF label in fusion')
    if not frame.base_train_year.lt(frame.year).all():raise ValueError('In-sample base prediction')
    p=frame[list(MODELS)].to_numpy(float);y=frame.y.to_numpy(float)
    if not np.isfinite(p).all() or (p<0).any() or ratio<=0:raise ValueError('Invalid fusion inputs')
    if window: a,audit=window_weights(frame)
    else:
        a=1/(frame.event_date.nunique()*frame.event_date.map(frame.event_date.value_counts()).to_numpy(float));audit=[]
    n,k=p.shape
    # y-Pw=u-v, u is underestimation, v is overestimation. Only positive
    # residual parts remain at the optimum, except zero-weight rows are free.
    equality=hstack([csr_matrix(p),eye(n),-eye(n)],format='csr')
    from scipy.sparse import vstack
    equality=vstack([equality,csr_matrix(np.r_[np.ones(k),np.zeros(2*n)][None,:])],format='csr')
    solution=linprog(np.r_[np.zeros(k),ratio*a,a],A_eq=equality,b_eq=np.r_[y,1],
                     bounds=(0,None),method='highs')
    if not solution.success:raise RuntimeError(solution.message)
    w=solution.x[:k]
    if w.min()<-1e-8 or abs(w.sum()-1)>1e-8:raise ValueError('Invalid simplex weights')
    w=np.maximum(w,0);w/=w.sum()
    residual=y-p@w
    loss=float(a@(ratio*np.maximum(residual,0)+np.maximum(-residual,0)))
    vertices=a@(ratio*np.maximum(y[:,None]-p,0)+np.maximum(p-y[:,None],0))
    if loss>min(vertices)+1e-6 or abs(loss-solution.fun)>1e-6:raise ValueError('LP optimum check failed')
    return w,dict(ratio=ratio,window=window,past_events=int(frame.event_date.nunique()),
        fit_end=str(frame.event_date.max().date()),objective=loss,vertex_objectives=vertices.tolist(),
        solver='HiGHS residual positive/negative LP',window_audit=audit)
