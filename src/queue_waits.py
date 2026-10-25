"""Per-passenger FIFO waiting-time distribution for the fluid bulk queue.

Built on ``queueing.simulate_bulk_queue`` without changing it. The method
reads waits as horizontal distances between the cumulative arrival curve A(t)
and the cumulative boarding curve D(t) (Newell's cumulative curves). Within
one arrival bin the arrival density is constant, so each (arrival bin, train)
overlap is a uniform distribution of waits; the whole distribution is their
mixture. Same assumptions as the simulator: uniform arrivals within each bin,
instantaneous boarding, first-come-first-served, no abandonment.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

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


def fifo_wait_pieces(
    arrivals: Sequence[float],
    train_times: Sequence[float],
    capacities: Sequence[float],
    interval_minutes: float = 1.0,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run the bulk queue and return (queue result, pieces array).

    Pieces are rows of (passengers, shortest wait, longest wait). Pieces from
    several queues (e.g. both directions) can be stacked and summarized
    together with ``summarize_wait_pieces``.
    """
    queue = simulate_bulk_queue(arrivals, train_times, capacities, interval_minutes)
    pieces = _wait_pieces([float(v) for v in arrivals], queue["interval_minutes"],
                          queue["train_records"])
    return queue, np.array(pieces, dtype=float).reshape(-1, 3)


def _share_at_most(pieces: np.ndarray, waits: np.ndarray) -> np.ndarray:
    mass, shortest, longest = pieces[:, 0], pieces[:, 1], pieces[:, 2]
    width = longest - shortest
    # Rounding can leave zero-width slivers: treat them as a point mass, not 0/0.
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(width > 0, np.clip((waits[:, None] - shortest) / width, 0, 1),
                            (waits[:, None] >= longest).astype(float))
    return fraction @ mass


def summarize_wait_pieces(
    pieces: np.ndarray,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    thresholds_minutes: Sequence[float] = (),
) -> dict[str, Any]:
    """Mean, maximum, quantiles and exceedance shares of a piece mixture."""
    if any(not 0 < q < 1 for q in quantiles):
        raise ValueError("quantiles must be strictly between 0 and 1")
    total = float(pieces[:, 0].sum()) if len(pieces) else 0.0
    if total <= 0:
        return {"mean_wait_minutes": None, "max_wait_minutes": None,
                "quantiles": {q: None for q in quantiles},
                "share_waiting_longer_than": {t: None for t in thresholds_minutes}}
    longest = float(pieces[:, 2].max())
    targets = np.asarray(quantiles, dtype=float) * total
    low, high = np.zeros(len(targets)), np.full(len(targets), longest)
    for _ in range(60):
        middle = (low + high) / 2
        enough = _share_at_most(pieces, middle) >= targets
        high, low = np.where(enough, middle, high), np.where(enough, low, middle)
    exceed = (1 - _share_at_most(pieces, np.asarray(thresholds_minutes, dtype=float)) / total
              if thresholds_minutes else [])
    return {
        "mean_wait_minutes": float(pieces[:, 0] @ (pieces[:, 1] + pieces[:, 2]) / 2 / total),
        "max_wait_minutes": longest,
        "quantiles": {q: float(v) for q, v in zip(quantiles, high)},
        "share_waiting_longer_than": {t: float(v) for t, v in zip(thresholds_minutes, exceed)},
    }


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
    queue, pieces = fifo_wait_pieces(arrivals, train_times, capacities, interval_minutes)
    output: dict[str, Any] = {
        "model": "fifo_cumulative_curve_waits",
        "total_arrivals": queue["total_arrivals"], "censored_passengers": queue["residual_queue"],
        "cleared": queue["cleared"], "queue_mean_wait_minutes": queue["mean_wait_minutes"],
        "mean_wait_minutes": None, "max_wait_minutes": None,
        "quantiles": {q: None for q in quantiles},
        "share_waiting_longer_than": {t: None for t in thresholds_minutes},
    }
    if queue["cleared"]:
        output.update(summarize_wait_pieces(pieces, quantiles, thresholds_minutes))
    return output
