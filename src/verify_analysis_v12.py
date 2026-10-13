"""Independently recompute chronology, grid selections and every public metric."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["event_date", "station", "hour"]


def independent(f, p):
    p = np.asarray(p)
    if p.ndim == 1:
        p = p[:, None]
    absolute, squared, biases, under, over, peak_under, peak_abs, peak_hour = [], [], [], [], [], [], [], []
    fallback = 0
    for date in sorted(f.event_date.unique()):
        mask = f.event_date.eq(date).to_numpy()
        e = p[mask] - f.loc[mask, "y"].to_numpy()[:, None]
        absolute.append(abs(e).mean(axis=0))
        squared.append((e * e).mean(axis=0))
        biases.append(e.mean(axis=0))
        event = f.loc[mask]
        windows = event[["planned_end_0_60_beimen", "planned_end_60_120_beimen"]].to_numpy()
        if np.isnan(windows).all() or np.nansum(windows) == 0:
            w = np.ones(6) / 6
            fallback += 1
        else:
            assert np.isfinite(windows).all()
            w = windows.sum(axis=1) / windows.sum()
        under.append(w @ np.maximum(-e, 0))
        over.append(w @ np.maximum(e, 0))
        k = np.argmax(event.y.to_numpy())
        peak_under.append(np.maximum(-e[k], 0))
        peak_abs.append(abs(e[k]))
        h = event.hour.to_numpy()
        peak_hour.append(abs(h[k] - h[np.argmax(p[mask], axis=0)]))
    u, o = np.mean(under, axis=0), np.mean(over, axis=0)
    return dict(mae=np.mean(absolute, axis=0), rmse=np.sqrt(np.mean(squared, axis=0)),
                bias=np.mean(biases, axis=0), window_under=u, window_over=o,
                cost_1=u+o, cost_2=2*u+o, cost_3=3*u+o,
                peak_under=np.mean(peak_under, axis=0), peak_abs=np.mean(peak_abs, axis=0),
                peak_hour=np.mean(peak_hour, axis=0), events=len(absolute), fallback_events=fallback)


def verify():
    out = ROOT / "result/analysis_v12"
    private = ROOT / ".local/analysis_v12"
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    for path, expected in manifest["source_hashes"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    assert hashlib.sha256((private / "predictions.csv").read_bytes()).hexdigest() == manifest["predictions_sha256"]
    frame = pd.read_csv(private / "analysis_input.csv", parse_dates=["event_date"])
    pred = pd.read_csv(private / "predictions.csv", parse_dates=["event_date"])
    assert not pred.duplicated(["variant", "model", "control", *KEYS]).any()
    # All event preceding-month intervals really end before D-1 18:00.
    prior_end = (frame.event_date.dt.to_period("M") - 1).dt.end_time
    assert (prior_end < frame.event_date - pd.Timedelta(hours=6)).all()
    choices = json.loads((out / "selections.json").read_text("utf-8"))
    folds = json.loads((out / "folds.json").read_text("utf-8"))
    for f in folds:
        assert f["train_end"] < f["valid_start"]
        assert f["max_gam_normal_residual"] < 1e-6
    worst_selection, worst_metric = 0., 0.
    for c in choices:
        rows, matrices = [], []
        for f in folds:
            i = f["identity"]
            if i["variant"] != c["variant"] or i["model"] != c["model"] or i["year"] >= c["year"]:
                continue
            path = private / c["variant"] / c["model"] / f"{i['year']}.npz"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == f["sha256"]
            rows.append(frame.loc[frame.year.eq(i["year"]) & frame.station.eq("北門")].sort_values(KEYS))
            matrices.append(np.load(path)["prediction"])
        past = pd.concat(rows, ignore_index=True)
        costs = independent(past, np.vstack(matrices))["cost_2"]
        worst_selection = max(worst_selection, float(np.max(abs(costs - c["all_costs"]))))
        assert abs(costs[c["candidate_id"]] - costs.min()) < 1e-8
        assert past.event_date.nunique() == c["inner_events"]
        current = frame.loc[frame.year.eq(c["year"]) & frame.station.eq("北門")].sort_values(KEYS)
        matrix = np.load(private / c["variant"] / c["model"] / f"{c['year']}.npz")["prediction"]
        for control, idx in (("retuned", c["candidate_id"]), ("fixed_core", c["fixed_candidate_id"])):
            observed = pred.loc[pred.variant.eq(c["variant"]) & pred.model.eq(c["model"]) &
                                pred.control.eq(control) & pred.year.eq(c["year"])].sort_values(KEYS)
            assert observed[KEYS].reset_index(drop=True).equals(current[KEYS].reset_index(drop=True))
            np.testing.assert_allclose(observed.prediction, matrix[:, idx], atol=1e-8, rtol=0)
    count = 0
    for table in ("summary", "annual", "events"):
        records = pd.read_csv(out / "tables" / (table + ".csv"))
        for r in records.to_dict("records"):
            f = pred.loc[pred.variant.eq(r["variant"]) & pred.model.eq(r["model"]) & pred.control.eq(r["control"])]
            if table == "annual":
                f = f.loc[f.year.eq(r["year"])]
            elif table == "events":
                f = f.loc[f.event_date.eq(pd.Timestamp(r["event_date"]))]
            for name, value in independent(f.reset_index(drop=True), f.prediction).items():
                error = float(np.max(abs(value - r[name])))
                worst_metric = max(worst_metric, error)
                assert error < 1e-7, (table, name, error)
            count += 1
    worst_legacy = 0.
    for variant, old_variant in (("core", "core"), ("fire_weather", "both")):
        legacy = pd.read_csv(ROOT / f".local/analysis_v11/{old_variant}_predictions.csv", parse_dates=["event_date"])
        for model in ("ridge", "gam", "xgboost"):
            c = pred.loc[pred.variant.eq(variant) & pred.model.eq(model) & pred.control.eq("retuned")].sort_values(KEYS)
            p = legacy.sort_values(KEYS)["expanded_r2_" + model].to_numpy()
            error = float(np.max(abs(c.prediction.to_numpy() - p)))
            worst_legacy = max(worst_legacy, error)
            assert error < 1e-7
    result = dict(status="passed", verified_choices=len(choices), verified_folds=len(folds),
                  verified_metric_rows=count, max_selection_difference=worst_selection,
                  max_metric_difference=worst_metric, core_and_both_v11_max_prediction_difference=worst_legacy,
                  max_gam_normal_residual=max(f["max_gam_normal_residual"] for f in folds),
                  independent_of_production_scoring_functions=True,
                  status_is_numerical_and_temporal_integrity_not_proof_of_D1_publication=True)
    (out / "independent_validation.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
