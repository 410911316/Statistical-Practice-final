"""Independently reconstruct v10 metrics and LP decisions from saved predictions."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["ridge", "gam", "xgboost"]
KEYS = ["event_date", "station", "hour"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def independent_lp(frame, scope):
    data = frame if scope == "all" else frame.loc[frame.station.eq("北門")]
    p, y = data[MODELS].to_numpy(float), data.y.to_numpy(float)
    n = len(data)
    a = 1/(data.event_date.nunique()*data.event_date.map(data.event_date.value_counts()).to_numpy())
    scale = max(1., float(np.max(y)), float(np.max(p)))
    # Equality formulation: P w - r_plus + r_minus = y.
    # Distinct from production's absolute-error inequality formulation.
    constraints = vstack([
        hstack([csr_matrix(p/scale), -eye(n), eye(n)]),
        csr_matrix(np.r_[np.ones(3), np.zeros(2*n)][None, :]),
    ]).tocsr()
    result = linprog(np.r_[np.zeros(3), a, a], A_eq=constraints,
                     b_eq=np.r_[y/scale, 1.], bounds=(0, None), method="highs-ipm")
    if not result.success:
        raise AssertionError(result.message)
    w = result.x[:3]
    return w, float(a @ np.abs(p @ w-y))


def verify():
    output = ROOT / "result/analysis_v10"
    config = read(ROOT / "configs/analysis_v10.json")
    manifest = read(output / "run_manifest.json")
    training = read(output / "beimen_only/training_manifest.json")
    checked_hashes = 0
    for item in manifest["source_files"] + manifest["code"] + training["output_files"]:
        assert sha(ROOT / item["path"]) == item["sha256"], item["path"]
        checked_hashes += 1
    for path, expected in training["input_code_hashes"].items():
        assert sha(ROOT / path) == expected, path
        checked_hashes += 1
    base_config = json.loads((ROOT / "configs/analysis_v8_both.json").read_text("utf-8-sig"))
    assert training["config"] == base_config
    assert len(base_config["xgb_grid"]) == 90
    local = pd.read_csv(ROOT / ".local/analysis_v10/beimen_only/temporal_predictions.csv", parse_dates=["event_date"])
    source = pd.read_csv(ROOT / ".local/analysis_v8/both/analysis_input.csv", parse_dates=["event_date"])
    source = source.loc[source.station.eq("北門")].sort_values(KEYS).reset_index(drop=True)
    assert len(source) == 126 and source.event_date.nunique() == 21
    folds = read(output / "beimen_only/folds.json")
    models = joblib.load(ROOT / ".local/analysis_v10/beimen_only/models.joblib")
    model_difference = 0.
    for fold in folds:
        year = fold["validation_year"]
        assert fold["train_stations"] == ["北門"]
        past = source.loc[source.year.lt(year)]
        assert len(past) == fold["train_rows"]
        assert past.event_date.nunique() == fold["train_events"]
        assert str(past.event_date.max().date()) == fold["train_end"]
        future = source.loc[source.year.eq(year)].sort_values(KEYS)
        cached = local.loc[local.year.eq(year)].sort_values(KEYS)
        np.testing.assert_array_equal(future[KEYS].to_numpy(), cached[KEYS].to_numpy())
        np.testing.assert_array_equal(future.y.to_numpy(), cached.y.to_numpy())
        for m in MODELS:
            assert models[year][m].info() == fold["base_models"][m]
            difference = np.max(np.abs(models[year][m].predict(future)-cached[m].to_numpy()))
            model_difference = max(model_difference, float(difference))
    assert model_difference < 1e-7
    inner = pd.read_csv(output / "beimen_only/temporal_folds.csv")
    assert (pd.to_datetime(inner.train_end) < pd.to_datetime(inner.validation_start)).all()
    assert (pd.to_datetime(inner.validation_start) <= pd.to_datetime(inner.parent_train_end)).all()
    decisions = read(output / "gate_decisions.json")
    lp_differences, gate_differences = [], []
    lp_count = 0
    for d in decisions:
        if d["regime"] == "local":
            meta = local.loc[local.year.lt(d["outer_year"])].copy()
        else:
            meta = pd.read_csv(ROOT / f".local/analysis_v8/{d['variant']}/meta_predictions.csv",
                               parse_dates=["event_date"])
            meta = meta.loc[meta.outer_year.eq(d["outer_year"])]
        assert meta.year.max() < d["outer_year"]
        assert (pd.to_datetime(meta.base_train_end).dt.year < meta.year).all()
        _, optimum = independent_lp(meta, d["fit_scope"])
        lp_count += 1
        fitting = meta if d["fit_scope"] == "all" else meta.loc[meta.station.eq("北門")]
        saved = np.array([d["final_weights"][m] for m in MODELS])
        assert saved.min() >= -1e-8 and abs(saved.sum()-1) < 1e-8
        losses = fitting.assign(error=np.abs(fitting[MODELS].to_numpy() @ saved-fitting.y.to_numpy()))
        objective = losses.groupby("event_date").error.mean().mean()
        lp_differences.append(abs(objective-optimum))
        assert abs(objective-d["final_oof_mae"]) < 1e-7
        validation = []
        for year in sorted(meta.year.unique()):
            past = meta.loc[meta.year.lt(year)]
            if past.event_date.nunique() < config["gate_min_weight_training_events"]:
                continue
            w, _ = independent_lp(past, d["fit_scope"])
            lp_count += 1
            later = meta.loc[meta.year.eq(year) & meta.station.eq("北門")].copy()
            later["fusion_error"] = np.abs(later[MODELS].to_numpy() @ w-later.y.to_numpy())
            later["xgb_error"] = np.abs(later.xgboost-later.y)
            validation.append(later.groupby("event_date")[["fusion_error","xgb_error"]].mean().assign(year=year))
        chosen = "xgboost"
        if validation:
            valid = pd.concat(validation)
            assert len(valid) == d["validation_events"]
            for key, col in [("fusion_validation_mae","fusion_error"),("xgb_validation_mae","xgb_error")]:
                gate_differences.append(abs(float(valid[col].mean())-d[key]))
            sufficient = len(valid) >= config["gate_min_validation_events"] and valid.year.nunique() >= config["gate_min_validation_years"]
            if sufficient and valid.fusion_error.mean() < valid.xgb_error.mean()-config["gate_numerical_tolerance"]:
                chosen = "fusion"
        else:
            assert d["validation_events"] == 0
        assert chosen == d["chosen"], (d["variant"], d["outer_year"], d["regime"])
    assert max(lp_differences) < 1e-5
    assert max(gate_differences) < 1e-5

    summary = pd.read_csv(output / "tables/summary.csv")
    metric_difference, applied_difference = 0., 0.
    frames = {v:pd.read_csv(ROOT / f".local/analysis_v10/{v}_paired_predictions.csv", parse_dates=["event_date"])
              for v in config["cached_variants"]}
    for d in decisions:
        f = frames[d["variant"]].loc[lambda x: x.year.eq(d["outer_year"])]
        base = f[[f"{d['regime']}_{m}" for m in MODELS]].to_numpy()
        fusion = base @ np.array([d["final_weights"][m] for m in MODELS])
        gate = fusion if d["chosen"] == "fusion" else base[:,2]
        for kind, prediction in (("fusion",fusion), ("gate",gate)):
            applied_difference = max(applied_difference, float(np.max(np.abs(prediction-f[f"{d['regime']}_{kind}_{d['fit_scope']}"]))))
    for row in summary.itertuples():
        frame = frames[row.variant]
        assert len(frame) == 54 and frame.event_date.nunique() == 9
        per_event = []
        for _, event in frame.groupby("event_date"):
            event = event.sort_values("hour")
            error = event[row.model].to_numpy()-event.y.to_numpy()
            peak = np.argmax(event.y.to_numpy())
            signed = -error[peak]
            hour_diff = abs(event.hour.iloc[peak]-event.hour.iloc[np.argmax(event[row.model].to_numpy())])
            per_event.append([np.mean(abs(error)), np.mean(error**2), np.mean(error),
                              max(0,signed), abs(signed), hour_diff])
        means = np.mean(per_event, axis=0)
        for key, value in zip(["mae","mse","bias","peak_underprediction","peak_absolute_error","peak_hour_mae","rmse"],
                              [*means, np.sqrt(means[1])]):
            metric_difference = max(metric_difference, abs(float(value)-getattr(row,key)))
    assert metric_difference < 1e-7
    assert applied_difference < 1e-7
    checked = [*output.glob("tables/*.csv"), output/"gate_decisions.json", output/"run_manifest.json",
               output/"beimen_only/training_manifest.json"]
    checked += list((ROOT/".local/analysis_v10").glob("*_paired_predictions.csv"))
    audit = dict(status="passed", scope="implementation_and_saved_predictions_not_new_generalization_evidence",
                 source_code_and_output_hashes_checked=checked_hashes, local_model_bundles_repredicted=len(models),
                 unchanged_xgb_grid_candidates=90, inner_base_folds_checked=len(inner),
                 independently_rebuilt_lp_problems=lp_count, gate_decisions_checked=len(decisions),
                 max_lp_objective_difference=max(lp_differences), max_gate_validation_difference=max(gate_differences),
                 max_base_prediction_difference=model_difference, max_applied_prediction_difference=applied_difference,
                 summary_rows_checked=len(summary), max_summary_metric_difference=metric_difference,
                 verifier_sha256=sha(Path(__file__)),
                 output_files=[{"path":p.relative_to(ROOT).as_posix(),"sha256":sha(p)} for p in checked])
    (output/"independent_validation.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in audit.items() if k!="output_files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    verify()

