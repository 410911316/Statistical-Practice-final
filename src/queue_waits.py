"""Per-passenger FIFO waiting-time distribution for the fluid bulk queue.

Built on ``queueing.simulate_bulk_queue`` without changing it, so the frozen
V1 scenario audit hash of queueing.py stays valid. The method reads waits as
horizontal distances between the cumulative arrival curve A(t) and the
cumulative boarding curve D(t) (Newell's cumulative curves). Within one
arrival bin the arrival density is constant, so each (arrival bin, train)
overlap is a uniform distribution of waits; the whole distribution is their
mixture. Same assumptions as the simulator: uniform arrivals within each bin,
instantaneous boarding, first-come-first-served, no abandonment.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from queueing import simulate_bulk_queue

DEFAULT_QUANTILES = (0.5, 0.9, 0.95)


def _wait_pieces(counts: list[float], dt: float,
                 train_records: list[dict[str, float]]) -> list[tuple[float, float, float]]:
    """Return (mass, shortest wait, longest wait) for every bin-train overlap."""
    pieces: list[tuple[float, float, float]] = []
    boarded_before, k = 0.0, 0
    position = 0.0  # cumulative arrivals before the current bin
    for index, count in enumerate(counts):
        if count <= 0:
            continue
        rate, start, end = count / dt, index * dt, position + count
        low = position
        while low < end and k < len(train_records):
            boarded_after = boarded_before + train_records[k]["boarded"]
            high = min(end, boarded_after)
            if high > low:
                # FIFO: arrival time is linear in cumulative position inside a bin.
                first_arrival = start + (low - position) / rate
                last_arrival = start + (high - position) / rate
                train_time = train_records[k]["time_minutes"]
                pieces.append((high - low, train_time - last_arrival, train_time - first_arrival))
                low = high
            if boarded_after <= low:
                boarded_before, k = boarded_after, k + 1
            else:
                break
        position = end
    return pieces


def _share_at_most(pieces: list[tuple[float, float, float]], wait: float) -> float:
    total = 0.0
    for mass, shortest, longest in pieces:
        if wait >= longest:
            total += mass
        elif wait > shortest:
            total += mass * (wait - shortest) / (longest - shortest)
    return total


def fifo_wait_distribution(
    arrivals: Sequence[float],
    train_times: Sequence[float],
    capacities: Sequence[float],
    interval_minutes: float = 1.0,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    thresholds_minutes: Sequence[float] = (),
) -> dict[str, Any]:
    """Return mean, quantiles, maximum and exceedance shares of FIFO waits.

    Inputs mean the same as in ``simulate_bulk_queue`` (initial queue is 0:
    an earlier queue has unknown arrival times). Quantiles and shares are only
    reported when every arrival boarded; censored runs return None instead of
    silently dropping the passengers still waiting at the horizon.
    """
    if any(not 0 < q < 1 for q in quantiles):
        raise ValueError("quantiles must be strictly between 0 and 1")
    queue = simulate_bulk_queue(arrivals, train_times, capacities, interval_minutes)
    dt, total = queue["interval_minutes"], queue["total_arrivals"]
    counts = [float(v) for v in arrivals]
    output: dict[str, Any] = {
        "model": "fifo_cumulative_curve_waits",
        "total_arrivals": total, "censored_passengers": queue["residual_queue"],
        "cleared": queue["cleared"], "queue_mean_wait_minutes": queue["mean_wait_minutes"],
        "mean_wait_minutes": None, "max_wait_minutes": None,
        "quantiles": {q: None for q in quantiles},
        "share_waiting_longer_than": {t: None for t in thresholds_minutes},
    }
    if not queue["cleared"] or total <= 0:
        return output
    pieces = _wait_pieces(counts, dt, queue["train_records"])
    longest = max(p[2] for p in pieces)
    output["mean_wait_minutes"] = sum(m * (a + b) / 2 for m, a, b in pieces) / total
    output["max_wait_minutes"] = longest
    for q in quantiles:
        low, high = 0.0, longest
        for _ in range(100):
            middle = (low + high) / 2
            if _share_at_most(pieces, middle) >= q * total:
                high = middle
            else:
                low = middle
        output["quantiles"][q] = high
    for threshold in thresholds_minutes:
        output["share_waiting_longer_than"][threshold] = 1 - _share_at_most(pieces, threshold) / total
    return output
