"""Beimen refit using the existing v7/v8 estimators and unchanged tuning grids."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from analysis_v7.models import Trainer
from run_analysis_v8 import write_json

MODELS = ("ridge", "gam", "xgboost")
KEYS = ["event_date", "station", "hour"]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refit_beimen(root, reuse=False, progress=print):
    root = Path(root)
    paths = [root / ".local/analysis_v8/both/analysis_input.csv", root / "configs/analysis_v8_both.json",
             root / "src/analysis_v1/models.py", root / "src/analysis_v5/models.py",
             root / "src/analysis_v7/models.py", Path(__file__)]
    hashes = {str(p.relative_to(root)).replace("\\", "/"): fingerprint(p) for p in paths}
    private, output = root / ".local/analysis_v10/beimen_only", root / "result/analysis_v10/beimen_only"
    manifest_path = output / "training_manifest.json"
    if reuse:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["input_code_hashes"] != hashes:
            raise ValueError("Beimen-only cache source/code changed; refit required")
        for item in manifest["output_files"]:
            if fingerprint(root / item["path"]) != item["sha256"]:
                raise ValueError("Beimen-only cached output changed")
        return pd.read_csv(private / "temporal_predictions.csv"), manifest
    frame = pd.read_csv(paths[0], parse_dates=["event_date"])
    frame = frame.loc[frame.station.eq("北門")].sort_values(KEYS).reset_index(drop=True)
    if len(frame) != 126 or frame.event_date.nunique() != 21:
        raise ValueError("Expected same 21 events and six hours per Beimen event")
    if frame.duplicated(KEYS).any() or not frame.groupby("event_date").hour.apply(lambda v: set(v) == set(range(18,24))).all():
        raise ValueError("Incomplete Beimen input panel")
    config = json.loads(paths[1].read_text(encoding="utf-8-sig"))
    trainer, pieces, audits, models = Trainer(config), [], [], {}
    # Each calendar-year prediction is trained and tuned only on earlier years.
    # Reusing these genuinely OOF predictions in later outer folds is safe.
    for year in sorted(frame.year.unique()):
        train, valid = frame.loc[frame.year.lt(year)], frame.loc[frame.year.eq(year)]
        if train.event_date.nunique() < config["min_base_train_events"]:
            continue
        progress(f"Beimen-only {year}: {train.event_date.nunique()} past events, {valid.event_date.nunique()} validation events", flush=True)
        bundle = trainer.fit_bundle(train)
        piece = valid[KEYS + ["year", "y"]].copy()
        for model in MODELS:
            piece[model] = bundle[model].predict(valid)
        if not np.isfinite(piece[list(MODELS)].to_numpy()).all() or (piece[list(MODELS)] < 0).any().any():
            raise RuntimeError("Invalid local base predictions")
        piece["base_train_end"] = str(train.event_date.max().date())
        pieces.append(piece)
        audits.append({"validation_year": int(year), "train_events": int(train.event_date.nunique()),
                       "train_rows": len(train), "train_stations": sorted(train.station.unique()),
                       "train_end": str(train.event_date.max().date()), "validation_start": str(valid.event_date.min().date()),
                       "base_models": {m: bundle[m].info() for m in MODELS}})
        models[int(year)] = {m: bundle[m] for m in MODELS}
    predictions = pd.concat(pieces, ignore_index=True)
    for p in (private, output):
        p.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(private / "temporal_predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump(models, private / "models.joblib")
    write_json(output / "folds.json", audits)
    selections = [r for r in trainer.legacy.selection_log if r["model"] in MODELS] + trainer.gam_selection_log
    write_json(output / "inner_selection.json", selections)
    pd.DataFrame(trainer.legacy.fold_log + trainer.gam_fold_log).to_csv(output / "temporal_folds.csv", index=False, encoding="utf-8-sig")
    write_json(output / "gam_numerical_diagnostics.json", trainer.diagnostic_log)
    generated = [private / "temporal_predictions.csv", private / "models.joblib", output / "folds.json",
                 output / "inner_selection.json", output / "temporal_folds.csv", output / "gam_numerical_diagnostics.json"]
    manifest = {"training_scope": "Beimen_only", "events": 21, "input_rows": 126,
                "used_models": list(MODELS), "config": config, "input_code_hashes": hashes,
                "base_algorithm": "unchanged_v8_Trainer_including_train_only_encoders_and_nested_tuning",
                "note": "Legacy trainer also fits OLS/Lasso internally; only Ridge/GAM/XGBoost are used. Single-station constants and aliases are dropped by the original train-only encoder.",
                "output_files": [{"path": str(p.relative_to(root)).replace("\\", "/"), "sha256": fingerprint(p)} for p in generated]}
    write_json(manifest_path, manifest)
    return predictions, manifest


if __name__ == "__main__":
    refit_beimen(Path(__file__).resolve().parents[2])
