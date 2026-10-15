"""Learning curves with whole-event subsets and fixed future-year validation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from analysis_v17.experiment import subset_plans, materialize, aggregate
from analysis_v15.experiment import fit, KEYS
from analysis_v16.experiment import check_panel
from analysis_v11.evaluation import metrics, WINDOW
from analysis_v13.data import sha
from run_analysis_v8 import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"result/analysis_v17"
PRIVATE = ROOT/".local/analysis_v17"


def run():
    cfg = json.loads((ROOT/"configs/analysis_v17.json").read_text("utf8"))
    parent = json.loads((ROOT/"result/analysis_v16/manifest.json").read_text("utf8"))
    assert cfg["stations"] == ["北門", "大橋頭站", "雙連"]
    assert cfg["outer_years"] == [2024, 2025, 2026] and cfg["models"] == ["ridge", "xgboost"]
    assert cfg["variant"] == "none" and cfg["repeats"] == 10
    assert cfg["event_counts"] == [6, 9, 12, 15, 18] and cfg["protocols"] == ["event_subsample", "oldest_first", "recent_first"]
    for path, digest in parent["source_hashes"].items():
        if sha(ROOT/path) != digest:
            raise ValueError(f"Changed archived source: {path}")
    ref_path = ROOT/".local/analysis_v16/predictions.csv"
    assert sha(ref_path) == parent["prediction_sha256"]
    f = pd.read_csv(ROOT/cfg["input"], parse_dates=["event_date"])
    oldfits = json.loads((ROOT/"result/analysis_v16/fits.json").read_text("utf8"))
    ref = pd.read_csv(ref_path, parse_dates=["event_date"])
    groups = parent["groups"]
    fits, predictions, scores, controls = [], [], [], []
    OUT.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    (OUT/"tables").mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        for year in cfg["outer_years"]:
            test_all = f.loc[f.year.eq(year) & f.station.isin(cfg["stations"])].sort_values(KEYS).copy()
            check_panel(test_all, cfg["stations"])
            test = test_all.loc[test_all.station.eq("北門")]
            assert test.event_date.nunique() == 3 and len(test) == 18
            plans = subset_plans(f, year, cfg["stations"], cfg["event_counts"], cfg["repeats"], cfg["subset_seed"])
            for plan in plans:
                train = materialize(f, plan, cfg["stations"])
                north_train = train.loc[train.station.eq("北門")].sort_values(KEYS)
                for model in cfg["models"]:
                    old = next(r for r in oldfits if r["year"] == year and r["cohort"] == "three"
                               and r["model"] == model and r["variant"] == "none")
                    fitted = fit(train, year, model, "none", groups, old["params"], cfg["model_seed"])
                    run_id = f"{plan['subset_id']}_{model}"
                    meta = {k: plan[k] for k in ["year", "protocol", "replicate", "n_events"]}
                    record = dict(**plan, run_id=run_id, model=model, params=old["params"],
                                  train_rows=len(train), train_end=fitted.train_end,
                                  encoded_columns=len(fitted.encoder.names), feature_names=fitted.encoder.names,
                                  train_mean=fitted.encoder.base.mean.to_dict(),
                                  train_scale=fitted.encoder.base.scale.to_dict(),
                                  imputation=fitted.encoder.base.imputation_values,
                                  sklearn_alpha=float(fitted.estimator.alpha) if model == "ridge" else None)
                    fits.append(record)
                    for scope, block in [("train", north_train), ("validation", test)]:
                        p = fitted.predict(block)
                        saved = block[KEYS+["y", *WINDOW]].copy()
                        saved["prediction"] = p
                        for k, v in dict(**meta, run_id=run_id, model=model, scope=scope).items():
                            saved[k] = v
                        predictions.append(saved)
                        scores.append(dict(**meta, run_id=run_id, model=model, scope=scope, **metrics(block, p)))
                        if plan["protocol"] == "full" and scope == "validation":
                            reference = ref.loc[ref.year.eq(year) & ref.station.eq("北門")].sort_values(KEYS)
                            assert reference[KEYS].reset_index(drop=True).equals(block[KEYS].reset_index(drop=True))
                            error = float(np.max(abs(p-reference[f"three_{model}_none"].to_numpy())))
                            assert error < 1e-6
                            controls.append(dict(year=year, model=model, max_error=error))
            print(f"Completed fixed validation year {year}: {len(plans)*2} fits", flush=True)
    p = pd.concat(predictions, ignore_index=True)
    s = pd.DataFrame(scores)
    p.to_csv(PRIVATE/"predictions.csv", index=False, encoding="utf-8-sig")
    s.to_csv(OUT/"tables/scores.csv", index=False, encoding="utf-8-sig")
    a = aggregate(s)
    a.to_csv(OUT/"tables/curves.csv", index=False, encoding="utf-8-sig")
    write_json(OUT/"fits.json", fits)
    own = [Path(__file__), ROOT/"configs/analysis_v17.json", *sorted((ROOT/"src/analysis_v17").glob("*.py")),
           ROOT/"result/analysis_v16/manifest.json", ROOT/"result/analysis_v16/fits.json", ref_path]
    source_hashes = {**parent["source_hashes"], **{path.relative_to(ROOT).as_posix(): sha(path) for path in own}}
    write_json(OUT/"manifest.json", dict(config=cfg, groups=groups, source_hashes=source_hashes,
        prediction_sha256=sha(PRIVATE/"predictions.csv"), fits_sha256=sha(OUT/"fits.json"),
        scores_sha256=sha(OUT/"tables/scores.csv"), curves_sha256=sha(OUT/"tables/curves.csv"),
        versions=parent["versions"], models_fitted=len(fits), controls=controls, deployed=False,
        independent_unseen_test=False, parameter_search=False,
        limitations=parent["limitations"]+[
            "Activity subsampling changes composition; no causal estimate of sample-size benefit",
            "Bands describe ten overlapping past-event subsets; not confidence intervals or new future events",
            "Frozen hyperparameters and background inputs retain information from wider past data",
            "No new same-day lag inputs; none variant only; no optimum size selected on validation"]))
    print(a.loc[a.protocol.eq("event_subsample") & a.scope.eq("validation"),
                ["year", "model", "n_events", "mae_mean", "mae_p10", "mae_p90", "peak_abs_mean"]].to_string(index=False))


if __name__ == "__main__":
    run()
