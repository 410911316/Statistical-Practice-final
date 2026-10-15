"""Date-only nested training subsets; no future labels used in selection."""
import numpy as np
import pandas as pd
from analysis_v16.experiment import select_train, check_panel


def subset_plans(frame, year, stations, counts, repeats, seed):
    if not counts or counts != sorted(set(counts)) or any(type(n) is not int or n < 2 for n in counts):
        raise ValueError("Need unique increasing whole-event counts >= 2")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Need positive repeats")
    full = select_train(frame, year, stations)
    days = sorted(pd.to_datetime(full.event_date.unique()))
    if len(days) < min(counts):
        raise ValueError("Not enough prior events")
    sizes = [n for n in counts if n < len(days)]
    plans = []

    def add(protocol, replicate, chosen):
        dates = [str(pd.Timestamp(d).date()) for d in sorted(chosen)]
        plans.append(dict(year=year, protocol=protocol, replicate=replicate,
                          n_events=len(dates), train_dates=dates,
                          subset_id=f"{year}_{protocol}_{replicate}_{len(dates)}"))

    # Full data are fitted only once, then used as the common endpoint of all curves.
    add("full", -1, days)
    for rep in range(repeats):
        rng = np.random.default_rng(np.random.SeedSequence([seed, year, rep]))
        permutation = rng.permutation(len(days))
        for n in sizes:
            add("event_subsample", rep, [days[i] for i in permutation[:n]])
    for n in sizes:
        add("oldest_first", -1, days[:n])
        add("recent_first", -1, days[-n:])
    return plans


def materialize(frame, plan, stations):
    dates = pd.to_datetime(plan["train_dates"])
    if len(dates) != plan["n_events"] or dates.duplicated().any() or not (dates.year < plan["year"]).all():
        raise ValueError("Invalid, duplicate or future training dates")
    full = select_train(frame, plan["year"], stations)
    train = full.loc[full.event_date.isin(dates)].copy()
    if set(train.event_date.unique()) != set(dates.to_numpy()):
        raise ValueError("Missing requested event")
    check_panel(train, stations)
    return train


def aggregate(scores):
    """Descriptive subset-composition bands, not confidence intervals."""
    rows = []
    for record in scores.to_dict("records"):
        protocols = ["event_subsample", "oldest_first", "recent_first"] if record["protocol"] == "full" else [record["protocol"]]
        for protocol in protocols:
            rows.append({**record, "protocol": protocol})
    expanded = pd.DataFrame(rows)
    metrics = ["mae", "rmse", "peak_abs", "peak_under", "cost_2"]
    result = []
    keys = ["year", "model", "protocol", "n_events", "scope"]
    for key, group in expanded.groupby(keys, sort=True):
        record = dict(zip(keys, key))
        record["n_fits"] = len(group)
        for metric in metrics:
            values = group[metric].to_numpy(float)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_p10"] = float(np.quantile(values, .1))
            record[f"{metric}_p90"] = float(np.quantile(values, .9))
        result.append(record)
    return pd.DataFrame(result)
