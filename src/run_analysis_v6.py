"""Low-complexity GAM and Ridge/GAM/XGBoost chronological fusion experiments."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd

from analysis_v1.data import load_analysis_data
from analysis_v1.d1_features import attach_d1_features
from analysis_v1.d1_reporting import build_tuning_figures
from analysis_v6.models import BASE_MODELS, Trainer, fit_fusion, metrics_table, vif_table

ROOT = Path(__file__).resolve().parents[1]


def json_value(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=json_value, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def validate_config(config, root=ROOT):
    if config.get("include_historical_baseline") is not False:
        raise ValueError("V6 excludes the unverified historical OD baseline")
    if config.get("station_hour_interaction") is not True:
        raise ValueError("V6 requires a shared station x hour design")
    if config.get("d1_cutoff_hour") != 18 or config.get("fusion_top_k") != 2:
        raise ValueError("V6 prespecifies D-1 18:00 and top-two historical OOF selection")
    if config.get("linear_design") != "full_shared_with_training_rank_filter":
        raise ValueError("V6 uses full OLS, not forward selection")
    expected = {"gam_n_splines": 5, "gam_spline_order": 3, "gam_min_known_events": 6,
                "gam_min_unique": 8, "gam_fusion_models": ["ridge", "gam", "xgboost"]}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("V6 prespecifies low-complexity GAM support and three fusion members")
    if config.get("gam_lambdas") != [.1, 1, 10, 100, 1000]:
        raise ValueError("V6 prespecifies the chronological smoothness grid")
    root = Path(root).resolve()
    output, private = (root / config["output_dir"]).resolve(), (root / config["private_output_dir"]).resolve()
    if not output.is_relative_to(root / "result/analysis_v6") or not private.is_relative_to(root / ".local/analysis_v6"):
        raise ValueError("V6 outputs must not overwrite previous versions or expose private predictions")
    return output, private


def run(args):
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    output, private = validate_config(config)
    frame, data_info = load_analysis_data(ROOT)
    audit = None
    if config["d1_features"]:
        frame, data_info, audit = attach_d1_features(frame, data_info, ROOT, config["d1_source_files"],
                                                   config["d1_features"], cutoff_hour=18)
    data_info["feature_eligibility"]["baseline_hourly_56d"] = {
        "use": "excluded_from_all_v6_models_and_fusion",
        "reason": "Historical OD publication timestamps are unverified."}
    data_info["audit"]["model_feature_whitelist"].remove("baseline_hourly_56d")
    data_info["audit"]["baseline_excluded_from_fitted_models"] = True
    if args.require_d1 and not data_info["audit"]["d1_deployable"]:
        raise ValueError("歷史訓練標籤／日曆發布版本仍未全核實；新欄位合格不代表完整 D−1 上線驗證。")
    if args.validate_only:
        print("共用設計與來源閘門驗證通過；仍屬回顧式向前驗證。", flush=True)
        return
    for path in (output / "tables", output / "figures", private):
        path.mkdir(parents=True, exist_ok=True)
    if audit is not None:
        write_csv(audit, output / "tables/d1_feature_audit.csv")
    print(json.dumps(data_info["activity_counts"], ensure_ascii=False), flush=True)
    trainer = Trainer(config)
    predictions, folds, meta, fitted = trainer.backtest(frame, progress=lambda msg: print(msg, flush=True))
    metrics = metrics_table(predictions)
    write_csv(metrics, output / "tables/model_metrics.csv")
    for name, field in (("yearly", "year"), ("station", "station"), ("event", "event_date")):
        grouped = pd.concat([metrics_table(part).assign(**{field: key})
                             for key, part in predictions.groupby(field)], ignore_index=True)
        write_csv(grouped, output / f"tables/{name}_metrics.csv")
    vifs, coefficients, curves, extrapolation = [], [], [], []
    for year, bundle in fitted.items():
        train, test = frame.loc[frame.year < year], frame.loc[frame.year == year]
        curve = bundle["gam"].effect_curves(train).rename(columns={
            "column": "feature", "x_raw": "value", "centered_partial_effect": "effect"})
        term_audit = {term["column"]: term for term in bundle["gam"].terms_audit}
        for field, source in (("training_min", "training_min_raw"),
                              ("training_max", "training_max_raw"),
                              ("training_events", "known_training_events")):
            curve[field] = curve["feature"].map(lambda name: term_audit[name][source])
        curves.append(curve.assign(outer_year=year))
        extrapolation.extend({"outer_year": year, **row} for row in bundle["gam"].extrapolation_audit(test))
        for model in ("ols", "ridge", "lasso"):
            fit = bundle[model]
            vifs.append(vif_table(fit, frame.loc[frame.year < year], year))
            beta = fit.estimator if model == "ols" else np.r_[fit.estimator.intercept_, fit.estimator.coef_]
            coefficients.extend({"outer_year": year, "model": model, "column": name,
                                 "coefficient": float(value), "nonzero": bool(abs(value) > 1e-10),
                                 "scale": "training_standardized_continuous_and_0_1_dummy_no_causal_inference"}
                                for name, value in zip(["intercept", *fit.encoder.names], beta))
    write_csv(pd.concat(vifs, ignore_index=True), output / "tables/training_vif.csv")
    write_csv(pd.DataFrame(coefficients), output / "tables/training_coefficients.csv")
    write_csv(pd.DataFrame([{"feature": key, **value} for key, value in data_info["feature_eligibility"].items()]),
              output / "tables/feature_eligibility.csv")
    write_csv(pd.concat(curves, ignore_index=True), output / "tables/gam_effect_curves.csv")
    write_csv(pd.DataFrame(extrapolation), output / "tables/gam_extrapolation.csv")
    write_json(output / "folds.json", folds)
    write_csv(frame, private / "analysis_input.csv")
    write_csv(predictions, private / "outer_predictions.csv")
    write_csv(meta, private / "meta_predictions.csv")
    joblib.dump(fitted, private / "outer_models.joblib")

    # A final research fit is saved separately and never used to score outer tests.
    final_bundle = trainer.fit_bundle(frame)
    final_meta = trainer.build_meta(frame)
    final_weights, final_fusion = fit_fusion(final_meta, top_k=config["fusion_top_k"])
    joblib.dump({"models": final_bundle, "fusion_weights": final_weights, "model_order": BASE_MODELS,
                 "status": "research_only_not_verified_d1", "train_end": str(frame.event_date.max().date())},
                private / "research_models.joblib")
    choices = pd.DataFrame(trainer.selection_log)
    for name in ("groups", "params"):
        choices[name] = choices[name].map(lambda v: json.dumps(v, ensure_ascii=False))
    write_csv(choices, output / "tables/inner_selection.csv")
    write_csv(pd.DataFrame(trainer.fold_log).drop_duplicates(), output / "tables/temporal_folds.csv")
    build_tuning_figures(choices, folds, output, audit)

    sources = [Path(__file__), config_path, *sorted((ROOT / "src/analysis_v1").glob("*.py")),
               *sorted((ROOT / "src/analysis_v5").glob("*.py")), *sorted((ROOT / "src/analysis_v6").glob("*.py")), ROOT / "uv.lock", ROOT / "pyproject.toml"]
    manifest = {
        "analysis_version": config["version"], "evaluation_kind": config["evaluation_kind"],
        "config": config, "data": data_info, "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "matplotlib", "duckdb", "pygam")},
        "code": [{"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sources],
        "outer_test": {"years": config["outer_years"], "events": int(predictions.event_date.nunique()), "rows": len(predictions)},
        "full_data_research_fit": {"fusion_weights": final_weights, "fusion_audit": final_fusion,
                                   "deployable": False, "warning": "Not used to score outer tests."},
        "design": "V5 inputs and estimators unchanged; GAM replaces supported continuous terms by training-only cubic P-splines.",
    }
    write_json(output / "run_manifest.json", manifest)
    print(metrics.to_string(index=False), flush=True)
    print(f"第六版結果：{output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/analysis_v6_both.json"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--require-d1", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
