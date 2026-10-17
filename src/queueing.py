"""Pure, deterministic queueing helpers for explicitly assumed rail scenarios.

Counts may be fractional: this is a fluid scenario calculation, not a fitted
stochastic queue. Time is measured in minutes from the start of the first bin.
Uniform arrivals within each bin and instantaneous boarding are assumptions.
Remaining train capacity and usable waiting space must be supplied by callers;
the module contains no Taipei Metro capacity or operational timetable defaults.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any


def _nonnegative(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _positive(value: Any, name: str) -> float:
    result = _nonnegative(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def mmc_metrics(arrival_rate: float, service_rate: float, servers: int) -> dict[str, Any]:
    """Return stationary M/M/c metrics for parallel service counters.

    Both rates use persons/minute; service_rate is per server. The assumptions
    are Poisson arrivals, independent exponential service, FCFS, infinite
    waiting room and stationary rates. ``servers`` must not mean train count.
    At rho >= 1 stationary probabilities and mean waits are undefined (None).
    """
    arrival = _nonnegative(arrival_rate, "arrival_rate")
    service = _positive(service_rate, "service_rate")
    if isinstance(servers, bool) or not isinstance(servers, Integral) or servers < 1:
        raise ValueError("servers must be a positive integer")
    servers = int(servers)
    rho = arrival / (servers * service)
    output: dict[str, Any] = {
        "arrival_rate": arrival, "service_rate": service, "servers": servers,
        "utilization": rho, "stable": rho < 1,
        "model": "stationary_M_M_c_parallel_service_counters",
    }
    if rho >= 1:
        output.update({
            "empty_probability": None, "wait_probability": None,
            "mean_wait_minutes": None, "mean_queue": None,
            "mean_system_minutes": None, "mean_system_count": None,
            "status": "no_finite_stationary_mean",
        })
        return output
    if arrival == 0:
        p0, p_wait = 1.0, 0.0
    else:
        offered = arrival / service
        # Erlang B recurrence avoids the very large factorials in Erlang C.
        erlang_b = 1.0
        for count in range(1, servers + 1):
            numerator = offered * erlang_b
            erlang_b = numerator / (count + numerator)
        p_wait = erlang_b / (1.0 - rho + rho * erlang_b)
        log_terms = [n * math.log(offered) - math.lgamma(n + 1)
                     for n in range(servers)]
        log_terms.append(servers * math.log(offered) - math.lgamma(servers + 1)
                         - math.log1p(-rho))
        peak = max(log_terms)
        log_denominator = peak + math.log(sum(math.exp(t - peak) for t in log_terms))
        p0 = math.exp(-log_denominator)
    wait = p_wait / (servers * service - arrival)
    system_time = wait + 1.0 / service
    output.update({
        "empty_probability": p0, "wait_probability": p_wait,
        "mean_wait_minutes": wait, "mean_queue": arrival * wait,
        "mean_system_minutes": system_time,
        "mean_system_count": arrival * system_time, "status": "stationary",
    })
    return output


def disaggregate_hourly(
    hourly_counts: Sequence[float],
    direction_shares: Mapping[str, float],
    minute_profile: Sequence[float] | None = None,
    walk_delay_minutes: int = 0,
) -> dict[str, list[float]]:
    """Allocate hourly counts to directions and minute bins without loss.

    Shares must sum to one. A supplied profile is 60 nonnegative proportions
    summing to one and is applied to every direction/hour. If omitted, the
    explicit assumption is uniform allocation (1/60 per minute). A fixed
    integer walking delay prepends zero bins, preserving the tail after the
    last hour. This does not infer true minute-level or walking trajectories.
    """
    counts = [_nonnegative(v, "hourly_counts") for v in hourly_counts]
    if not direction_shares:
        raise ValueError("direction_shares must not be empty")
    shares = {direction: _nonnegative(share, "direction_shares")
              for direction, share in direction_shares.items()}
    if any(not isinstance(direction, str) or not direction for direction in shares):
        raise ValueError("direction names must be nonempty strings")
    if not math.isclose(math.fsum(shares.values()), 1.0, abs_tol=1e-9, rel_tol=0):
        raise ValueError("direction_shares must sum to one")
    if (isinstance(walk_delay_minutes, bool)
            or not isinstance(walk_delay_minutes, Integral) or walk_delay_minutes < 0):
        raise ValueError("walk_delay_minutes must be a nonnegative integer")
    profile = ([1.0 / 60] * 60 if minute_profile is None
               else [_nonnegative(v, "minute_profile") for v in minute_profile])
    if len(profile) != 60 or not math.isclose(math.fsum(profile), 1.0, abs_tol=1e-9, rel_tol=0):
        raise ValueError("minute_profile must contain 60 proportions summing to one")
    return {
        direction: [0.0] * int(walk_delay_minutes)
        + [hour * share * fraction for hour in counts for fraction in profile]
        for direction, share in shares.items()
    }


def simulate_bulk_queue(
    arrivals: Sequence[float],
    train_times: Sequence[float],
    capacities: Sequence[float],
    interval_minutes: float = 1.0,
    initial_queue: float = 0.0,
    waiting_space_limit: float | None = None,
) -> dict[str, Any]:
    """Integrate a uniform-within-bin fluid queue with instantaneous boarding.

    ``arrivals[i]`` is the number arriving uniformly in [i*dt, (i+1)*dt).
    ``capacities[k]`` is the *remaining* capacity of the kth train at this
    station/direction. Trains must have strictly increasing nonnegative times.
    The horizon ends at the later of the last arrival bin and the last train;
    no additional service is silently invented to clear a queue.

    ``max_queue`` and the waiting-space check include the instant immediately
    before boarding. The limit is space available for waiting passengers after
    reserving space for alighters/circulation, not total platform certification.
    ``mean_wait_minutes`` is only a complete cohort mean when initial_queue=0
    and everyone has boarded. Censored runs still report their integrated area.
    """
    counts = [_nonnegative(v, "arrivals") for v in arrivals]
    times = [_nonnegative(v, "train_times") for v in train_times]
    capacity = [_nonnegative(v, "capacities") for v in capacities]
    dt = _positive(interval_minutes, "interval_minutes")
    initial = _nonnegative(initial_queue, "initial_queue")
    limit = None if waiting_space_limit is None else _nonnegative(waiting_space_limit, "waiting_space_limit")
    if len(times) != len(capacity):
        raise ValueError("train_times and capacities must have the same length")
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError("train_times must be strictly increasing")
    arrival_end = len(counts) * dt
    horizon = max(arrival_end, times[-1] if times else 0.0)
    boundaries = {i * dt: i for i in range(len(counts) + 1)}
    service_events = dict(zip(times, capacity))
    event_times = sorted(set(boundaries) | set(service_events))
    queue, area, peak_queue, boarded_total = initial, 0.0, initial, 0.0
    previous, rate = 0.0, 0.0
    trace: list[dict[str, Any]] = []
    train_records: list[dict[str, float]] = []
    for time in event_times:
        elapsed = time - previous
        new_arrivals = rate * elapsed
        area += queue * elapsed + 0.5 * rate * elapsed * elapsed
        queue += new_arrivals
        pre_service = queue
        peak_queue = max(peak_queue, pre_service)
        available = service_events.get(time, 0.0)
        boarded = min(queue, available)
        queue -= boarded
        boarded_total += boarded
        row = {
            "time_minutes": time, "arrivals_since_previous": new_arrivals,
            "queue_before_service": pre_service, "available_capacity": available,
            "boarded": boarded, "queue_after_service": queue,
            "cumulative_waiting_person_minutes": area,
            "is_train_event": time in service_events,
        }
        trace.append(row)
        if time in service_events:
            train_records.append({
                "time_minutes": time, "available_capacity": available,
                "waiting_before": pre_service, "boarded": boarded, "left_behind": queue,
            })
        if time in boundaries:
            index = boundaries[time]
            rate = counts[index] / dt if index < len(counts) else 0.0
        previous = time
    total_arrivals = math.fsum(counts)
    total_demand = initial + total_arrivals
    tolerance = 1e-9 * max(1.0, total_demand)
    cleared = queue <= tolerance
    conservation_error = total_demand - boarded_total - queue
    return {
        "model": "deterministic_fluid_bulk_service_scenario",
        "assumptions": [
            "Arrivals are uniform within every input interval.",
            "Boarding occurs instantaneously at each supplied train time.",
            "Capacities are available spaces at this station, not whole-train capacities.",
            "No abandonment, rerouting or additional transfer demand is inferred.",
            "Waiting-space feasibility is not full platform or timetable safety validation.",
        ],
        "interval_minutes": dt, "arrival_horizon_minutes": arrival_end,
        "horizon_minutes": horizon, "initial_queue": initial,
        "total_arrivals": total_arrivals, "total_boarded": boarded_total,
        "residual_queue": queue, "cleared": cleared,
        "conservation_error": conservation_error,
        "waiting_person_minutes": area,
        "mean_wait_minutes": area / total_arrivals if cleared and initial == 0 and total_arrivals > 0 else None,
        "mean_observed_window_wait_minutes": area / total_demand if cleared and total_demand > 0 else None,
        "max_queue": peak_queue, "waiting_space_limit": limit,
        "waiting_space_exceeded": None if limit is None else peak_queue > limit + 1e-9,
        "train_count": len(times), "train_records": train_records, "trace": trace,
    }


def compare_headway_scenarios(
    arrivals: Sequence[float],
    scenarios: Sequence[Mapping[str, Any]],
    interval_minutes: float = 1.0,
) -> list[dict[str, Any]]:
    """Compare explicitly configured headways for one directional arrival path.

    Each scenario requires name, headway_minutes, available_capacity,
    first_train_minutes, clearance_minutes and a nonempty assumptions list.
    Optional waiting_space_limit has the same limited meaning as in the queue
    simulator. Trains are generated up to arrival_end + clearance_minutes;
    clearance_minutes is a service extension, not a guarantee of clearance.
    Results preserve input order: this function neither declares an optimal
    schedule nor invents operating costs, fleet limits or track constraints.
    """
    dt = _positive(interval_minutes, "interval_minutes")
    counts = [_nonnegative(v, "arrivals") for v in arrivals]
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for scenario in scenarios:
        required = {"name", "headway_minutes", "available_capacity", "first_train_minutes",
                    "clearance_minutes", "assumptions"}
        missing = required - set(scenario)
        if missing:
            raise ValueError(f"scenario missing required fields: {', '.join(sorted(missing))}")
        name = scenario["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("scenario names must be unique nonempty strings")
        names.add(name)
        assumptions = scenario["assumptions"]
        if (isinstance(assumptions, (str, bytes)) or not isinstance(assumptions, Sequence)
                or not assumptions or any(not isinstance(a, str) or not a.strip() for a in assumptions)):
            raise ValueError("scenario assumptions must be a nonempty list of strings")
        headway = _positive(scenario["headway_minutes"], "headway_minutes")
        capacity = _nonnegative(scenario["available_capacity"], "available_capacity")
        first = _nonnegative(scenario["first_train_minutes"], "first_train_minutes")
        clearance = _nonnegative(scenario["clearance_minutes"], "clearance_minutes")
        end = len(counts) * dt + clearance
        train_count = max(0, int(math.floor((end - first) / headway + 1e-12)) + 1)
        if train_count > 100_000:
            raise ValueError("scenario creates too many train events; check time units")
        times = [first + k * headway for k in range(train_count)]
        metrics = simulate_bulk_queue(
            counts, times, [capacity] * train_count, dt,
            waiting_space_limit=scenario.get("waiting_space_limit"),
        )
        # Keep the comparison horizon fixed even if the last service is earlier.
        if metrics["horizon_minutes"] < end:
            tail = end - metrics["horizon_minutes"]
            metrics["waiting_person_minutes"] += metrics["residual_queue"] * tail
            metrics["horizon_minutes"] = end
            metrics["trace"].append({
                "time_minutes": end, "arrivals_since_previous": 0.0,
                "queue_before_service": metrics["residual_queue"],
                "available_capacity": 0.0, "boarded": 0.0,
                "queue_after_service": metrics["residual_queue"],
                "cumulative_waiting_person_minutes": metrics["waiting_person_minutes"],
                "is_train_event": False,
            })
        result.append({
            "scenario_name": name,
            "config": {
                "headway_minutes": headway, "available_capacity": capacity,
                "first_train_minutes": first, "clearance_minutes": clearance,
                "waiting_space_limit": scenario.get("waiting_space_limit"),
            },
            "assumptions": list(assumptions), "metrics": metrics,
        })
    return result
