"""Prespecified feature ablations; selected and frozen tuning controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis_v12.data import load, check_variant
from analysis_v12.models import Encoder, GAMEncoder, fit_one
from analysis_v11.evaluation import KEYS, WINDOW, select_candidate, metrics
from compare_fusion_losses import load_verified
from run_analysis_v8 import write_json

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("ridge", "gam", "xgboost")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidates(frame, base, cfg, variant, model, signature, reuse):
    private = ROOT / ".local/analysis_v12" / variant / model
    private.mkdir(parents=True, exist_ok=True)
    grid, groups = cfg["grids"][model], tuple(base["candidate_groups"])
    parts, predictions, audits = [], [], []
    for year in sorted(frame.year.unique()):
        train = frame.loc[frame.year < year]
        if train.event_date.nunique() < 2:
            continue
        valid = frame.loc[frame.year.eq(year) & frame.station.eq("北門")].sort_values(KEYS).reset_index(drop=True)
        cache, recordpath = private / f"{year}.npz", private / f"{year}.json"
        identity = dict(signature=signature, model=model, variant=variant, year=int(year))
        if reuse and cache.exists() and recordpath.exists():
            audit = json.loads(recordpath.read_text("utf-8"))
            if audit["identity"] != identity or audit["sha256"] != digest(cache):
                raise ValueError("Changed candidate cache: rerun without --reuse")
            matrix = np.load(cache)["prediction"]
        else:
            encoder = GAMEncoder().fit(train, groups, base) if model == "gam" else Encoder().fit(train, groups)
            matrix = np.empty((len(valid), len(grid)))
            certificate = 0.
            for index, parameters in enumerate(grid):
                fitted = fit_one(train, model, groups, parameters, base, encoder)
                matrix[:, index] = fitted.predict(valid)
                if model == "gam":
                    certificate = max(certificate, fitted.certificate["relative_normal_equation_residual"])
            np.savez_compressed(cache, prediction=matrix)
            audit = dict(identity=identity, sha256=digest(cache), train_events=int(train.event_date.nunique()),
                         train_end=str(train.event_date.max().date()), valid_start=str(valid.event_date.min().date()),
                         valid_events=int(valid.event_date.nunique()), candidates=len(grid),
                         max_gam_normal_residual=certificate, encoded_columns=encoder.names,
                         dropped_constants=encoder.dropped_constants)
            write_json(recordpath, audit)
        if matrix.shape != (len(valid), len(grid)) or not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("Invalid candidate predictions")
        parts.append(valid[KEYS + ["year", "y", *WINDOW]])
        predictions.append(matrix)
        audits.append(audit)
        print(f"{variant} {model} {year}: completed {len(grid)} candidates", flush=True)
    return pd.concat(parts, ignore_index=True), np.vstack(predictions), audits


def run(reuse=False):
    cfg = json.loads((ROOT / "configs/analysis_v12.json").read_text("utf-8"))
    gridcfg = json.loads((ROOT / cfg["grid_source"]).read_text("utf-8"))
    basecore = json.loads((ROOT / "configs/analysis_v8_core.json").read_text("utf-8"))
    baseboth = json.loads((ROOT / "configs/analysis_v8_both.json").read_text("utf-8"))
    shared, _, _, manifests, sources = load_verified(ROOT)
    source, google, baseline, baseline_meta = load(ROOT)
    private = ROOT / ".local/analysis_v12"
    output = ROOT / "result/analysis_v12"
    (output / "tables").mkdir(parents=True, exist_ok=True)
    source.to_csv(private / "analysis_input.csv", index=False, encoding="utf-8-sig")
    google.to_csv(output / "tables/google_monthly_audit.csv", index=False, encoding="utf-8-sig")
    baseline.to_csv(output / "tables/baseline_availability.csv", index=False, encoding="utf-8-sig")
    source.groupby("event_date")[["planned_finale", "planned_enhanced_nonfinale"]].first().to_csv(
        output / "tables/event_types.csv", encoding="utf-8-sig")
    paths = [Path(__file__), *sorted((ROOT / "src/analysis_v12").glob("*.py")),
             ROOT / "src/analysis_v1/models.py", ROOT / "src/analysis_v1/data.py",
             ROOT / "src/analysis_v5/models.py", ROOT / "src/analysis_v7/models.py",
             ROOT / "src/analysis_v11/models.py", ROOT / "src/analysis_v11/evaluation.py",
             ROOT / "configs/analysis_v12.json", ROOT / "configs/analysis_v11.json",
             ROOT / "configs/analysis_v8_core.json", ROOT / "configs/analysis_v8_both.json",
             ROOT / "docs/d1_event_type_sources.json", ROOT / "docs/d1_show_end_sources.json",
             ROOT / "docs/d1_weather_sources.json", private / "analysis_input.csv"]
    signature = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    fixed = json.loads((ROOT / "result/analysis_v11/selections.json").read_text("utf-8"))
    fixed = {(x["model"], x["outer_year"]): x["candidate_id"] for x in fixed
             if x["variant"] == "core" and x["grid"] == "expanded" and x["ratio"] == 2}
    all_pred, choices, folds = [], [], []
    # BLAS pools are constrained before concurrent independent model families.
    with threadpool_limits(limits=1):
        for variant, extra in cfg["variants"].items():
            base = dict(basecore)
            requested = set(basecore["candidate_groups"]) | set(extra)
            # Preserve the archived canonical order. Exact-rank filtering and
            # tree split ties must not introduce a second change in an ablation.
            canonical = baseboth["candidate_groups"]
            base["candidate_groups"] = [g for g in canonical if g in requested]
            base["candidate_groups"] += [g for g in extra if g not in canonical]
            check_variant(variant, base["candidate_groups"], cfg)
            with ThreadPoolExecutor(max_workers=3) as pool:
                jobs = {m: pool.submit(candidates, source, base, gridcfg, variant, m, signature, reuse) for m in MODELS}
                for model, job in jobs.items():
                    timeline, matrix, audits = job.result()
                    folds.extend(audits)
                    for year in cfg["outer_years"]:
                        before, now = timeline.year.lt(year).to_numpy(), timeline.year.eq(year).to_numpy()
                        past = timeline.loc[before].reset_index(drop=True)
                        best, loss, losses = select_candidate(past, matrix[before], year, 2, len(gridcfg["grids"][model]))
                        choices.append(dict(variant=variant, model=model, year=year, candidate_id=best,
                                            params=gridcfg["grids"][model][best], inner_cost=loss,
                                            inner_events=int(past.event_date.nunique()),
                                            all_costs=losses.tolist(), fixed_candidate_id=fixed[model, year]))
                        for control, index in (("retuned", best), ("fixed_core", fixed[model, year])):
                            part = timeline.loc[now].copy()
                            part["prediction"] = matrix[now, index]
                            part["variant"], part["model"], part["control"] = variant, model, control
                            part["sensitivity_only"] = variant in cfg["sensitivity_only"]
                            all_pred.append(part)
    pred = pd.concat(all_pred, ignore_index=True)
    # Public summaries exclude the full raw hourly panel.
    pred.to_csv(private / "predictions.csv", index=False, encoding="utf-8-sig")
    summary, annual, events = [], [], []
    for (variant, model, control), f in pred.groupby(["variant", "model", "control"], sort=False):
        common = dict(variant=variant, model=model, control=control,
                      sensitivity_only=variant in cfg["sensitivity_only"])
        summary.append({**common, **metrics(f.reset_index(drop=True), f.prediction)})
        for year, y in f.groupby("year"):
            annual.append({**common, "year": int(year), **metrics(y.reset_index(drop=True), y.prediction)})
        for date, e in f.groupby("event_date"):
            peak = e.loc[e.y.idxmax()]
            events.append({**common, "event_date": str(date.date()), **metrics(e.reset_index(drop=True), e.prediction),
                           "true_peak_hour": int(peak.hour), "true_peak": float(peak.y),
                           "prediction_at_true_peak": float(peak.prediction)})
    for name, rows in (("summary", summary), ("annual", annual), ("events", events)):
        pd.DataFrame(rows).to_csv(output / "tables" / (name + ".csv"), index=False, encoding="utf-8-sig")
    write_json(output / "selections.json", choices)
    write_json(output / "folds.json", folds)
    write_json(output / "manifest.json", dict(config=cfg, d1_deployable=False,
        limitations=["Previously inspected nine outer events; exploratory",
                     "Historical OD label release and calendar vintages unverified",
                     "Google query/geography/retrieval window unknown; lagging does not repair vintage",
                     "Baseline source observation dates are past, publication dates unverified"],
        baseline_metadata=baseline_meta, old_audit=manifests["both"]["data"]["audit"],
        source_hashes=signature, legacy_sources=sources,
        raw_google_sha256=digest(ROOT / "data/original_data/google_網路聲量.csv"),
        candidate_fits=sum(f["candidates"] for f in folds), predictions_sha256=digest(private / "predictions.csv"),
        events=21, outer_events=9, outer_rows_per_model=54))
    print(pd.DataFrame(summary).query("model == 'xgboost' and control == 'retuned'")[
        ["variant", "mae", "rmse", "cost_2", "peak_under"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true")
    run(parser.parse_args().reuse)
