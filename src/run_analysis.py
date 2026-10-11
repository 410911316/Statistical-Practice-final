"""Run versioned retrospective, temporally validated analysis; never writes SQL."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd

from analysis_v1.data import load_analysis_data
from analysis_v1.models import BASE_MODELS, Trainer, metrics_table, solve_stack, vif_table


ROOT = Path(__file__).resolve().parents[1]


def json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_value, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def resolve_run_paths(args, config):
    output = Path(args.output or ROOT / config.get("output_dir", "result/analysis_v1")).resolve()
    private = Path(args.private_output or ROOT / config.get("private_output_dir", ".local/analysis_v1")).resolve()
    reference = ROOT / config["comparison_reference"] if config.get("comparison_reference") else None
    reference_manifest = ROOT / config["comparison_manifest"] if config.get("comparison_manifest") else None
    if reference is not None:
        reference = reference.resolve()
        if reference_manifest is None:
            raise ValueError("A version comparison requires its reference run manifest")
        reference_manifest = reference_manifest.resolve()
        if private == reference.parent or output == reference_manifest.parent:
            raise ValueError("Comparison output must not overwrite the reference run")
    return output, private, reference, reference_manifest


def verify_comparison_inputs(reference, reference_manifest, data_info):
    """Require identical source versions before attributing differences to models."""
    if not reference.is_file():
        raise FileNotFoundError(f"Missing original private predictions: {reference}")
    previous = json.loads(reference_manifest.read_text(encoding="utf-8"))
    old_hashes = {row["path"]: row["sha256"] for row in previous["data"]["manifest"]}
    new_hashes = {row["path"]: row["sha256"] for row in data_info["manifest"]}
    if old_hashes != new_hashes:
        raise ValueError("Comparison source hashes differ from v1; do not confound a data revision with model changes")
    return {"reference_predictions_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "reference_manifest_sha256": hashlib.sha256(reference_manifest.read_bytes()).hexdigest(),
            "same_input_hashes": True, "reference_version": previous["analysis_version"]}


def run(args):
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    frame, data_info = load_analysis_data(ROOT)
    print(json.dumps(data_info["activity_counts"], ensure_ascii=False), flush=True)
    output, private, reference, reference_manifest = resolve_run_paths(args, config)
    comparison_info = verify_comparison_inputs(reference, reference_manifest, data_info) if reference else None
    d1_audit = None
    if config.get("d1_features"):
        from analysis_v1.d1_features import attach_d1_features
        frame, data_info, d1_audit = attach_d1_features(
            frame, data_info, ROOT, config["d1_source_files"], config["d1_features"],
            cutoff_hour=config.get("d1_cutoff_hour", 18))
        if comparison_info is not None:
            comparison_info["same_input_hashes"] = False
            comparison_info["same_existing_input_hashes"] = True
            comparison_info["added_source_files"] = data_info["audit"]["new_d1_source_files"]
    if not config.get("include_historical_baseline", True):
        data_info["feature_eligibility"]["baseline_hourly_56d"] = {
            "use": "retrospective_comparator_only_excluded_from_models",
            "reason": "Historical OD publication timestamps are unverified; no baseline enters any fitted model."}
        data_info["audit"]["model_feature_whitelist"].remove("baseline_hourly_56d")
        data_info["audit"]["baseline_excluded_from_fitted_models"] = True
    if args.require_d1 and not data_info["audit"]["d1_deployable"]:
        raise ValueError("尚未核實歷史 OD／日曆發布版本，不能宣稱已驗證 D−1 預測。請補齊 available_at 與預報版本後再建立正式部署資料。")
    if args.validate_only:
        print("資料驗證通過；目前僅支援回顧式向前驗證。", flush=True)
        return
    for path in (output / "figures", output / "tables", private):
        path.mkdir(parents=True, exist_ok=True)
    if d1_audit is not None:
        write_csv(d1_audit, output / "tables/d1_feature_audit.csv")
    trainer = Trainer(config)
    predictions, folds, meta, fitted = trainer.backtest(frame, progress=lambda message: print(message, flush=True))
    metrics = metrics_table(predictions)
    write_csv(metrics, output / "tables/model_metrics.csv")
    eligibility = [{"feature": name, **record} for name, record in data_info["feature_eligibility"].items()]
    write_csv(pd.DataFrame(eligibility), output / "tables/feature_eligibility.csv")
    yearly = pd.concat([metrics_table(block).assign(year=int(year)) for year, block in predictions.groupby("year")], ignore_index=True)
    write_csv(yearly, output / "tables/yearly_metrics.csv")
    station_metrics = pd.concat([metrics_table(block).assign(station=station) for station, block in predictions.groupby("station")], ignore_index=True)
    write_csv(station_metrics, output / "tables/station_metrics.csv")
    event_records = []
    for day, block in predictions.groupby("event_date"):
        event_records.append(metrics_table(block).assign(event_date=day.strftime("%Y-%m-%d")))
    write_csv(pd.concat(event_records, ignore_index=True), output / "tables/event_metrics.csv")
    vifs, coefficients = [], []
    for year, bundle in fitted.items():
        train = frame.loc[frame.year < year]
        for model in ("ols", "ridge"):
            fit = bundle[model]
            vifs.append(vif_table(fit, train, year))
            if not fit.fallback:
                beta = fit.estimator if model == "ols" else np.r_[fit.estimator.intercept_, fit.estimator.coef_]
                coefficients.extend({"outer_year": year, "model": model, "column": name, "coefficient": float(value),
                                     "scale": "training_standardized_numeric_and_0_1_dummy_no_inference"}
                                    for name, value in zip(["intercept", *fit.encoder.names], beta))
    vif = pd.concat(vifs, ignore_index=True)
    write_csv(vif, output / "tables/training_vif.csv")
    write_csv(pd.DataFrame(coefficients), output / "tables/training_coefficients.csv")
    write_csv(pd.DataFrame(trainer.fold_log).drop_duplicates(), output / "tables/temporal_folds.csv")
    choices = pd.DataFrame(trainer.selection_log)
    for name in ("groups", "params"):
        choices[name] = choices[name].map(lambda value: json.dumps(value, ensure_ascii=False))
    write_csv(choices, output / "tables/inner_selection.csv")
    write_json(output / "folds.json", folds)
    # Observation-level predictions and serialized estimators stay ignored/private.
    write_csv(frame, private / "analysis_input.csv")
    write_csv(predictions, private / "outer_predictions.csv")
    write_csv(meta, private / "meta_predictions.csv")
    joblib.dump(fitted, private / "outer_models.joblib")
    final_bundle = trainer.fit_bundle(frame)
    final_meta = trainer.build_meta(frame)
    final_weights, final_lp = solve_stack(final_meta[list(BASE_MODELS)], final_meta.y, final_meta.event_date)
    joblib.dump({"models": final_bundle, "weights": final_weights, "model_order": BASE_MODELS,
                 "status": "research_only_not_verified_d1", "train_end": str(frame.event_date.max().date())}, private / "research_models.joblib")
    from analysis_v1.eda import build_eda
    from analysis_v1.queue_report import build_queue_report
    from analysis_v1.reporting import build_model_figures, write_report
    eda = build_eda(frame, data_info, output)
    queue = build_queue_report(predictions, config, output)
    build_model_figures(predictions, metrics, folds, vif, output)
    sources = [Path(__file__), config_path, *sorted((ROOT / "src/analysis_v1").glob("*.py")), ROOT / "uv.lock"]
    manifest = {"analysis_version": config["version"], "evaluation_kind": config["evaluation_kind"],
                "config": config, "data": data_info, "python": platform.python_version(),
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "matplotlib", "duckdb")},
                "code": [{"path": p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name,
                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
                "outer_test": {"years": config["outer_years"], "events": int(predictions.event_date.nunique()), "rows": len(predictions)},
                "full_data_research_fit": {"weights": dict(zip(BASE_MODELS, final_weights)), "lp": final_lp,
                                           "deployable": False, "warning": "These weights are not used to score the outer tests."},
                "eda": eda, "queue": queue}
    if reference is not None:
        from analysis_v1.comparison import write_comparison
        original = pd.read_csv(reference, parse_dates=["event_date"])
        # New features may change OLS; baseline remains a paired control.
        controls = config.get("comparison_unchanged_models", ["baseline", "ols"])
        if "baseline" not in controls or not set(controls) <= {"baseline", "ols", "ridge", "xgboost"}:
            raise ValueError("Comparison must retain the baseline control")
        keys = ["event_date", "station", "hour"]
        before = original.sort_values(keys).reset_index(drop=True)
        after = predictions.sort_values(keys).reset_index(drop=True)
        if not before[keys].equals(after[keys]) or not np.allclose(
                before[controls], after[controls], rtol=0, atol=1e-8):
            raise ValueError("Unchanged model controls differ; comparison requires investigation")
        write_comparison(original, predictions, output / "comparison",
                         title=config.get("comparison_title"))
        manifest["comparison"] = {**comparison_info, "matched_unchanged_models": controls,
                                  "ols_and_baseline_controls_match": "ols" in controls,
                                  "report": "comparison/report.md", "confirmatory_test": False}
    if config.get("tuning_diagnostics"):
        from analysis_v1.d1_reporting import build_tuning_figures
        build_tuning_figures(choices, folds, output, d1_audit)
    write_json(output / "run_manifest.json", manifest)
    write_report(output, metrics, folds, data_info, queue, config=config, private_output=private)
    print(metrics.to_string(index=False), flush=True)
    print(f"分析結果：{output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/analysis_v1.json"))
    parser.add_argument("--output", default=None, help="Defaults to the versioned directory in config")
    parser.add_argument("--private-output", default=None, help="Defaults to the versioned private directory in config")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--require-d1", action="store_true", help="Stop unless historical information availability has been verified.")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"分析停止：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
