"""Trains needed per time slot for one direction, allowing both more and fewer trains.

A schedule is a headway per slot: departures step from ``start`` by the headway
of the slot the current departure falls in, so the headway itself guarantees
the minimum separation. Slots are filled in time order, choosing the LONGEST
headway (fewest trains) whose departures in that slot leave somebody behind in
at most ``epsilon`` of the scenarios. Boarding at a train depends only on
earlier trains, so the check for slot s does not depend on later slots. Without
left-behind riders nobody waits longer than one headway, so the maximum
headway also caps the 95th-percentile wait. The forward search is a heuristic:
a sparse slot can let riders pile up for the next one, which is handled by a
bounded backtrack rather than a global optimization.
"""
from __future__ import annotations

import numpy as np


def schedule_from_headways(headways,slot_minutes,end,start=0):
    """Integer departure minutes in [start, end) stepping by each slot's headway."""
    if any(int(h)!=h or h<1 for h in headways): raise ValueError('headways must be positive integers')
    times,t=[],int(start)
    while t<end:
        times.append(t)
        t+=int(headways[min(t//slot_minutes,len(headways)-1)])
    return times


def trains_per_slot(times,slot_minutes,slots):
    counts=np.zeros(slots,dtype=int)
    for t in times:
        if 0<=t//slot_minutes<slots: counts[int(t//slot_minutes)]+=1
    return counts


def _slot_share(queue,chosen,h,s,slots,slot_minutes,end):
    # Later slots reuse h; they cannot affect boarding at this slot's trains.
    times,left=queue.left_behind_by_train(schedule_from_headways(chosen+[h]*(slots-s),slot_minutes,end))
    in_slot=(times//slot_minutes)==s
    return float(left[:,in_slot].any(axis=1).mean()) if in_slot.any() else 0.


def required_headways(queue,slots,slot_minutes,end,options,epsilon,max_backtrack=2):
    """Return (headway per slot, left-behind share per slot, feasible flag per slot).

    ``queue`` is a dispatch_optimization.DirectionQueue over sampled scenarios.
    If a slot fails even at the shortest headway, riders piled up during a
    sparse earlier slot: the latest of the previous ``max_backtrack`` slots that
    can still shorten is capped one option shorter and the search resumes from
    there. A slot still failing is kept at the shortest headway and flagged
    infeasible instead of being silently passed.
    """
    options=sorted(set(int(h) for h in options),reverse=True)
    chosen,shares,feasible,cap=[],[],[],{}
    s=0
    while s<slots:
        allowed=[h for h in options if h<=cap.get(s,options[0])]
        for h in allowed:
            share=_slot_share(queue,chosen,h,s,slots,slot_minutes,end)
            if share<=epsilon: break
        if share<=epsilon:
            chosen.append(h);shares.append(share);feasible.append(True);s+=1
            continue
        j=s-1
        while j>=0 and chosen[j]==options[-1]: j-=1
        if j>=0 and s-j<=max_backtrack:
            # Caps only tighten, so the search terminates.
            cap[j]=options[options.index(chosen[j])+1]
            del chosen[j:],shares[j:],feasible[j:]
            s=j
            continue
        chosen.append(options[-1]);shares.append(share);feasible.append(False);s+=1
    return chosen,shares,feasible
