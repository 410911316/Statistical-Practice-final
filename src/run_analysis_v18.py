"""Reproduce three-station bases, then fit bounded scheduled-egress corrections."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from analysis_v18.experiment import fit_correction
from analysis_v16.experiment import fit_cohort, check_panel
from analysis_v11.evaluation import select_candidate, metrics, KEYS, WINDOW
from analysis_v13.data import sha
from run_analysis_v8 import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"result/analysis_v18"
PRIVATE = ROOT/".local/analysis_v18"


def archived_parameters(root, year, model, cfg, used_paths):
    """Re-select from old candidate forecasts only on years strictly before year."""
    config_path = root/"configs/analysis_v11.json"
    source_path = root/".local/analysis_v8/core/analysis_input.csv"
    old = json.loads(config_path.read_text("utf8"))
    data = pd.read_csv(source_path, parse_dates=["event_date"])
    grid = old["grids"][model]
    frames, matrices = [], []
    used_paths.update([config_path, source_path])
    for valid_year in sorted(data.year.unique()):
        valid_year = int(valid_year)
        if valid_year >= year:
            continue
        prefix = root/f".local/analysis_v11/core/{model}/{valid_year}"
        if data.loc[data.year.lt(valid_year)].event_date.nunique() < old["candidate_prediction_start_min_train_events"]:
            continue
        record = json.loads(prefix.with_suffix(".json").read_text("utf8"))
        matrix_path = prefix.with_suffix(".npz")
        if sha(matrix_path) != record["sha256"]:
            raise ValueError("Changed candidate cache")
        for path, digest in record["source"]["signature"].items():
            actual = root/Path(path.replace("\\", "/"))
            if sha(actual) != digest:
                raise ValueError(f"Changed archived selection source: {path}")
            used_paths.add(actual)
        valid = data.loc[data.year.eq(valid_year) & data.station.eq("北門")].sort_values(KEYS).reset_index(drop=True)
        train = data.loc[data.year.lt(valid_year)]
        assert record["source"]["validation_year"] == valid_year
        assert record["source"]["train_end"] == str(train.event_date.max().date())
        assert pd.Timestamp(record["source"]["train_end"]).year < valid_year
        assert record["candidate_count"] == len(grid)
        assert record["validation_events"] == valid.event_date.nunique()
        matrix = np.load(matrix_path)["predictions"]
        assert matrix.shape == (len(valid), len(grid))
        frames.append(valid); matrices.append(matrix)
        used_paths.update([matrix_path, prefix.with_suffix(".json")])
    past = pd.concat(frames, ignore_index=True)
    choice, loss, _ = select_candidate(past, np.concatenate(matrices), year, cfg["tuning_ratio"], len(grid))
    return grid[choice], dict(candidate_index=choice, validation_cost=float(loss),
        parameter_validation_end_year=int(past.year.max()), selection_events=int(past.event_date.nunique()),
        selection_scope="archived_four_station_core_grid",
        candidate_count=len(grid))


def run():
    cfg = json.loads((ROOT/"configs/analysis_v18.json").read_text("utf8"))
    parent = json.loads((ROOT/"result/analysis_v16/manifest.json").read_text("utf8"))
    for path, digest in parent["source_hashes"].items():
        if sha(ROOT/path) != digest:
            raise ValueError(f"Changed archived input: {path}")
    ref_path = ROOT/".local/analysis_v16/predictions.csv"
    assert sha(ref_path) == parent["prediction_sha256"]
    frame = pd.read_csv(ROOT/cfg["input"], parse_dates=["event_date"])
    reference = pd.read_csv(ref_path, parse_dates=["event_date"])
    oldfits = json.loads((ROOT/"result/analysis_v16/fits.json").read_text("utf8"))
    groups = parent["groups"]
    sources = {Path(__file__), ROOT/"configs/analysis_v18.json", ROOT/cfg["input"],
               ROOT/"result/analysis_v16/manifest.json", ROOT/"result/analysis_v16/fits.json", ref_path,
               *sorted((ROOT/"src/analysis_v18").glob("*.py"))}
    fits, blocks, controls = [], [], []
    OUT.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    (OUT/"tables").mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        for year in cfg["oof_years"]:
            target = frame.loc[frame.year.eq(year) & frame.station.isin(cfg["stations"])].sort_values(KEYS).copy()
            check_panel(target, cfg["stations"])
            for model in cfg["models"]:
                params, selection = archived_parameters(ROOT, year, model, cfg, sources)
                fitted, train = fit_cohort(frame, year, cfg["stations"], model, "none", groups, params, cfg["model_seed"])
                assert train.event_date.nunique() >= cfg["min_base_events"]
                prediction = fitted.predict(target)
                if year in cfg["outer_years"]:
                    previous = next(r for r in oldfits if r["year"]==year and r["cohort"]=="three"
                                    and r["variant"]=="none" and r["model"]==model)
                    assert params == previous["params"], "Historical selection no longer reproduces control"
                    ref = reference.loc[reference.year.eq(year)].sort_values(KEYS)
                    assert target[KEYS].reset_index(drop=True).equals(ref[KEYS].reset_index(drop=True))
                    delta = float(np.max(abs(prediction-ref[f"three_{model}_none"].to_numpy())))
                    assert delta < 1e-6, delta
                    controls.append(dict(year=year, model=model, max_error=delta, rows=len(target)))
                saved = target[KEYS+["year", "y", *WINDOW]].copy()
                saved["base_prediction"] = prediction
                saved["model"] = model
                saved["base_train_end"] = fitted.train_end
                saved["parameter_validation_end_year"] = selection["parameter_validation_end_year"]
                blocks.append(saved.loc[saved.station.eq("北門")])
                fits.append(dict(year=year, model=model, params=params, **selection,
                    train_end=fitted.train_end, train_rows=len(train), train_events=int(train.event_date.nunique()),
                    feature_names=fitted.encoder.names))
            print(f"Historical forward prediction {year} completed", flush=True)
    oof = pd.concat(blocks, ignore_index=True)
    oof.to_csv(PRIVATE/"temporal_oof.csv", index=False, encoding="utf-8-sig")
    predictions, corrections = [], []
    for year in cfg["outer_years"]:
        for model in cfg["models"]:
            history = oof.loc[oof.year.lt(year) & oof.model.eq(model)].copy()
            target = oof.loc[oof.year.eq(year) & oof.model.eq(model)].copy()
            target["base"] = target.base_prediction
            for ratio in cfg["under_ratios"]:
                corrector = fit_correction(history, year, ratio, cfg["coefficient_bound"],
                                           cfg["l1_penalty"], cfg["min_correction_events"])
                target[f"correction_{ratio}"] = corrector.predict(target, target.base_prediction)
                corrections.append(dict(model=model, **corrector.record()))
            predictions.append(target)
    p = pd.concat(predictions, ignore_index=True)
    p.to_csv(PRIVATE/"predictions.csv", index=False, encoding="utf-8-sig")
    summary, annual, events = [], [], []
    for model, model_data in p.groupby("model"):
        for method in ["base", *[f"correction_{r}" for r in cfg["under_ratios"]]]:
            summary.append(dict(model=model, method=method, **metrics(model_data, model_data[method])))
            for year, block in model_data.groupby("year"):
                annual.append(dict(model=model, method=method, year=int(year), **metrics(block, block[method])))
                for date, event in block.groupby("event_date"):
                    peak = event.sort_values("hour").loc[event.y.idxmax()]
                    events.append(dict(model=model, method=method, event_date=str(date.date()),
                        actual_peak_hour=int(peak.hour), actual_peak=float(peak.y), predicted_at_actual_peak=float(peak[method]),
                        **metrics(event, event[method])))
    for name, records in [("summary", summary), ("annual", annual), ("events", events)]:
        pd.DataFrame(records).to_csv(OUT/f"tables/{name}.csv", index=False, encoding="utf-8-sig")
    write_json(OUT/"fits.json", fits)
    write_json(OUT/"corrections.json", corrections)
    generated = [PRIVATE/"temporal_oof.csv", PRIVATE/"predictions.csv", OUT/"fits.json", OUT/"corrections.json",
                 *sorted((OUT/"tables").glob("*.csv"))]
    write_json(OUT/"manifest.json", dict(config=cfg, groups=groups,
        source_hashes={**parent["source_hashes"], **{p.relative_to(ROOT).as_posix(): sha(p) for p in sources}},
        output_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in generated},
        controls=controls, base_fits=len(fits), corrections=len(corrections),
        d1_deployable=False, deployed=False, independent_unseen_test=False,
        limitations=parent["limitations"]+[
            "No new same-day features; old publication and OD-clock limitations remain",
            "Historical base selection uses archived four-station core tuning before each predicted year",
            "Correction uses two bounded coefficients; 25% cap, penalty .1, six-event support are study assumptions",
            "Scheduled end is neither actual dispersal nor measured platform arrival",
            "Nine repeated development events cannot certify generalization; no automatic replacement"]))
    print(pd.DataFrame(summary)[["model","method","mae","rmse","peak_abs","peak_under","window_under","window_over","cost_2"]].to_string(index=False))


if __name__ == "__main__":
    run()
