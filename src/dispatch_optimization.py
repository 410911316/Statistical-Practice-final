"""Choose extra train insertion minutes against many sampled demand scenarios.

The queue is the same fluid bulk-service model as ``queueing.simulate_bulk_queue``
but vectorized over scenarios, and restricted to integer-minute train times
(official Beimen departures are whole minutes). The optimizer is a heuristic
(greedy insertion, then one-train-at-a-time relocation); it does not prove
global optimality and knows nothing about fleet, crew, turnback or signalling
limits beyond a configured minimum separation between departures.
"""
from __future__ import annotations

import numpy as np


class DirectionQueue:
    """Arrival scenarios for one direction: arrivals[s, i] arrive uniformly in minute i."""

    def __init__(self, arrivals, capacity, horizon, extra_capacity=None):
        arrivals=np.asarray(arrivals,dtype=float)
        if arrivals.ndim!=2 or (arrivals<0).any() or not np.isfinite(arrivals).all():
            raise ValueError('arrivals must be a finite nonnegative scenarios × minutes array')
        self.capacity=np.broadcast_to(np.asarray(capacity,dtype=float),arrivals.shape[:1]).copy()
        # Extra trains may arrive emptier than regular ones (e.g. short-turn starts).
        self.extra_capacity=(self.capacity.copy() if extra_capacity is None else
                             np.broadcast_to(np.asarray(extra_capacity,dtype=float),arrivals.shape[:1]).copy())
        if (self.capacity<0).any() or (self.extra_capacity<0).any(): raise ValueError('capacity must be nonnegative')
        width=max(arrivals.shape[1],int(horizon))
        padded=np.zeros((arrivals.shape[0],width))
        padded[:,:arrivals.shape[1]]=arrivals
        zero=np.zeros((arrivals.shape[0],1))
        # cumulative[:, m] = arrivals before minute m; area[:, m] = ∫_0^m cumulative(t) dt.
        self.cumulative=np.hstack([zero,padded.cumsum(axis=1)])
        self.area=np.hstack([zero,(self.cumulative[:,:-1]+padded/2).cumsum(axis=1)])
        self.demand=self.cumulative[:,-1]
        self.horizon=width

    def evaluate(self,train_times,extra_times=()):
        """Return (waiting person-minutes, residual queue, left-behind departures) per scenario."""
        extra=set(int(t) for t in extra_times)
        times=np.asarray(sorted(list(train_times)+list(extra_times)),dtype=int)
        if len(times) and (times[0]<0 or times[-1]>self.horizon or np.any(np.diff(times)<=0)):
            raise ValueError('train times must be distinct integer minutes within the horizon')
        queue=np.zeros(len(self.demand));wait=np.zeros(len(self.demand))
        left=np.zeros(len(self.demand),dtype=int);previous=0
        for t in times:
            span=t-previous
            wait+=queue*span+self.area[:,t]-self.area[:,previous]-self.cumulative[:,previous]*span
            queue=queue+self.cumulative[:,t]-self.cumulative[:,previous]
            queue=queue-np.minimum(queue,self.extra_capacity if t in extra else self.capacity)
            left+=queue>1e-7;previous=t
        # Arrivals after the last train never board: count them as residual, not as waits.
        queue=queue+self.demand-self.cumulative[:,previous]
        return wait,queue,left


def feasible_minutes(existing,separation,start,end):
    """Integer minutes in [start, end] at least ``separation`` from every departure."""
    existing=np.asarray(sorted(existing),dtype=float)
    candidates=np.arange(start,end+1)
    if not len(existing): return candidates
    gaps=np.abs(candidates[:,None]-existing[None,:]).min(axis=1)
    return candidates[gaps>=separation]


def plan_cost(queues,trains,extras,residual_penalty):
    """Mean over scenarios of total waiting person-minutes; unserved riders are penalized."""
    total=0.
    for direction,queue in queues.items():
        wait,residual,_=queue.evaluate(trains[direction],extras[direction])
        total+=np.mean(wait+residual_penalty*residual)
    return total


def optimize_insertions(queues,trains,n_extra,separation,window,residual_penalty=60.):
    """Greedy insertion of n_extra trains (either direction), then relocation passes.

    Returns {direction: sorted inserted minutes} for each n = 0..n_extra.
    Plans are nested: the n+1 plan starts from the improved n plan.
    """
    extras={d:[] for d in queues}
    plans=[{d:[] for d in queues}]
    best=plan_cost(queues,trains,extras,residual_penalty)
    for _ in range(n_extra):
        choice=None
        for d in queues:
            for m in feasible_minutes(list(trains[d])+extras[d],separation,*window):
                trial={**extras,d:extras[d]+[int(m)]}
                cost=plan_cost(queues,trains,trial,residual_penalty)
                if choice is None or cost<choice[0]-1e-9: choice=(cost,d,int(m))
        if choice is None: break
        best,d,m=choice;extras[d]=sorted(extras[d]+[m])
        improved=True
        while improved:
            improved=False
            for d in queues:
                for m in list(extras[d]):
                    others=[x for x in extras[d] if x!=m]
                    for alt in feasible_minutes(list(trains[d])+others,separation,*window):
                        trial={**extras,d:sorted(others+[int(alt)])}
                        cost=plan_cost(queues,trains,trial,residual_penalty)
                        if cost<best-1e-9:
                            best,extras,improved=cost,trial,True;break
                    if improved: break
                if improved: break
        plans.append({d:list(v) for d,v in extras.items()})
    return plans
