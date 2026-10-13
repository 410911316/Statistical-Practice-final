"""Inspect error complementarity and compare Beimen weighting/refitting/gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v10.experiment import (MODELS, KEYS, checked_predictions, gate_from_past,
    apply_decision, complementary_errors, evaluate_beimen, summarize)
from analysis_v10.training import refit_beimen, fingerprint
from compare_fusion_losses import load_verified
from run_analysis_v8 import write_json

ROOT = Path(__file__).resolve().parents[1]
SHARED_METHODS = ["shared_ridge", "shared_gam", "shared_xgboost", "shared_equal",
                  "shared_fusion_all", "shared_fusion_beimen", "shared_gate_all", "shared_gate_beimen"]
LOCAL_METHODS = ["local_ridge", "local_gam", "local_xgboost", "local_equal", "local_fusion_beimen", "local_gate_beimen"]


def assemble(shared_frames, root, local_predictions, config):
    outputs, decisions, validation_events, pairs, help_rows = {}, [], [], [], []
    local = checked_predictions(local_predictions)
    for variant, outer in shared_frames.items():
        meta_source = pd.read_csv(root / f".local/analysis_v8/{variant}/meta_predictions.csv")
        frame = outer.loc[outer.station.eq("北門"), KEYS + ["year", "y", "train_end"]].copy().reset_index(drop=True)
        for m in MODELS:
            frame[f"shared_{m}"] = outer.loc[outer.station.eq("北門"), m].to_numpy()
        frame["shared_equal"] = frame[[f"shared_{m}" for m in MODELS]].mean(axis=1)
        if variant == config["primary_variant"]:
            lp = local.loc[local.year.isin(config["outer_years"])].sort_values(KEYS).reset_index(drop=True)
            if not frame[KEYS].equals(lp[KEYS]) or not np.array_equal(frame.y.to_numpy(), lp.y.to_numpy()):
                raise ValueError("Local/shared Beimen outer panels differ")
            if not pd.to_datetime(frame.train_end).equals(lp.base_train_end):
                raise ValueError("Local/shared training cutoffs differ")
            for m in MODELS:
                frame[f"local_{m}"] = lp[m].to_numpy()
            frame["local_equal"] = lp[list(MODELS)].mean(axis=1)
        for year in config["outer_years"]:
            shared_meta = checked_predictions(meta_source.loc[meta_source.outer_year.eq(year)], cutoff_year=year)
            plans = [("shared", shared_meta, "all"), ("shared", shared_meta, "beimen")]
            if variant == config["primary_variant"]:
                local_meta = local.loc[local.year.lt(year)]
                sm = shared_meta.loc[shared_meta.station.eq("北門")].reset_index(drop=True)
                lm = local_meta.reset_index(drop=True)
                if not sm[KEYS].equals(lm[KEYS]) or not np.array_equal(sm.y.to_numpy(), lm.y.to_numpy()):
                    raise ValueError("Local/shared past OOF labels differ")
                plans.append(("local", local_meta, "beimen"))
            # Diagnostics are recorded separately from gate inputs and do not select models.
            for regime in sorted({p[0] for p in plans}):
                data = next(p[1] for p in plans if p[0] == regime)
                pair, help_table = complementary_errors(data)
                pairs.append(pair.assign(variant=variant, regime=regime, outer_year=year, layer="past_oof"))
                help_rows.append(help_table.assign(variant=variant, regime=regime, outer_year=year, layer="past_oof"))
            for regime, meta, fit_scope in plans:
                decision, events = gate_from_past(meta, fit_scope, year, config)
                decisions.append({"variant": variant, "regime": regime, **decision})
                validation_events.append(events.assign(variant=variant, regime=regime, fit_scope=fit_scope, outer_year=year))
                mask = frame.year.eq(year)
                base = frame.loc[mask, [f"{regime}_{m}" for m in MODELS]].rename(columns={f"{regime}_{m}": m for m in MODELS})
                weights = np.array([decision["final_weights"][m] for m in MODELS])
                frame.loc[mask, f"{regime}_fusion_{fit_scope}"] = base.to_numpy() @ weights
                frame.loc[mask, f"{regime}_gate_{fit_scope}"] = apply_decision(base, decision)
        # Reproduce the already audited four-station MAE fusion before publishing.
        expected = outer.loc[outer.station.eq("北門"), "mae_fusion"].to_numpy()
        if not np.allclose(frame.shared_fusion_all, expected, rtol=0, atol=1e-6):
            raise ValueError("Shared all-station fusion no longer reproduces v9")
        outputs[variant] = frame
        for regime in (["shared", "local"] if variant == config["primary_variant"] else ["shared"]):
            data = frame[KEYS + ["year", "y"]].copy()
            data["base_train_end"] = frame.train_end
            for m in MODELS:
                data[m] = frame[f"{regime}_{m}"]
            pair, help_table = complementary_errors(data)
            pairs.append(pair.assign(variant=variant, regime=regime, outer_year=0, layer="outer_descriptive_only"))
            help_rows.append(help_table.assign(variant=variant, regime=regime, outer_year=0, layer="outer_descriptive_only"))
    nonempty = [frame for frame in validation_events if not frame.empty]
    validation = pd.concat(nonempty, ignore_index=True) if nonempty else validation_events[0].iloc[:0]
    return outputs, decisions, validation, pd.concat(pairs, ignore_index=True), pd.concat(help_rows, ignore_index=True)


def run(args):
    root = ROOT
    config_path = root / "configs/analysis_v10.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    shared, _, _, old_manifests, sources = load_verified(root)
    local, training_manifest = refit_beimen(root, reuse=args.reuse_local)
    frames, decisions, validation, pairs, help_table = assemble(shared, root, local, config)
    metrics, event_metrics = [], []
    for variant, frame in frames.items():
        methods = SHARED_METHODS + (LOCAL_METHODS if variant == config["primary_variant"] else [])
        scores, events = evaluate_beimen(frame, methods)
        metrics.append(scores.assign(variant=variant))
        event_metrics.append(events.assign(variant=variant))
    scores, events = pd.concat(metrics, ignore_index=True), pd.concat(event_metrics, ignore_index=True)
    annual = summarize(events, ("variant", "year", "model"))
    output, private = root / "result/analysis_v10", root / ".local/analysis_v10"
    (output / "tables").mkdir(parents=True, exist_ok=True)
    for name, frame in [("summary", scores), ("event_metrics", events), ("annual_metrics", annual),
                        ("gate_validation_events", validation), ("error_pairs", pairs), ("conditional_help", help_table)]:
        frame.to_csv(output / "tables" / f"{name}.csv", index=False, encoding="utf-8-sig")
    for variant, frame in frames.items():
        frame.to_csv(private / f"{variant}_paired_predictions.csv", index=False, encoding="utf-8-sig")
    write_json(output / "gate_decisions.json", decisions)
    sources += [{"path": str(p.relative_to(root)).replace("\\", "/"), "sha256": fingerprint(p)} for p in
                [config_path, root / "result/analysis_v10/beimen_only/training_manifest.json"]]
    code = [Path(__file__), root / "src/analysis_v10/experiment.py", root / "src/analysis_v10/training.py",
            root / "src/analysis_v10/reporting.py", root / "src/compare_fusion_losses.py"]
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "version": "v10", "config": config,
                "source_files": sources, "d1_deployable": False, "d1_limitations": old_manifests["both"]["data"]["audit"],
                "changes": ["same_base_predictions_beimen_weights", "nested_temporal_fusion_adoption", "beimen_only_base_refit"],
                "local_training_events": training_manifest["events"], "outer_events": 9,
                "code": [{"path": str(p.relative_to(root)).replace("\\", "/"), "sha256": fingerprint(p)} for p in code if p.exists()]}
    write_json(output / "run_manifest.json", manifest)
    from analysis_v10.reporting import build_report
    build_report(frames, scores, events, pairs, help_table, decisions, output)
    print(scores.loc[scores.variant.eq("both"), ["model", "mae", "rmse", "peak_underprediction"]].to_string(index=False))
    print("Output:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-local", action="store_true", help="Reuse only hash-matching completed Beimen refit")
    run(parser.parse_args())
