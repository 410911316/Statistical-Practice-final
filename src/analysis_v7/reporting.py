"""Paired audits and reports for the GAM numerical/support repair.

This module only reads completed fits. Repeated outer scores describe a repair
on inspected activities; they never select a hyperparameter or fusion member.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_v1.reporting import markdown_table, save, style
from analysis_v6 import reporting as previous

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = previous.VARIANTS
ALL_MODELS = previous.ALL_MODELS
OLD_MODELS = previous.OLD_MODELS
OLD_BASE_MODELS = previous.OLD_BASE_MODELS
FUSION_MODELS = previous.FUSION_MODELS
KEYS = previous.KEYS
HOURS = previous.HOURS
STATIONS = previous.STATIONS
LABELS = {**previous.LABELS, "gam_equal": "三模型等權", "gam_stack": "三模型學權重"}
VERSIONS = ("v6", "v7")
VERSION_LABELS = {"v6": "修復前", "v7": "修復後"}
COMPARE_MODELS = ("ridge", "gam", "xgboost", "gam_equal", "gam_stack", "stack")
FIT_CODE = {*previous.PRESERVED_CODE, "src/analysis_v6/models.py", "src/run_analysis_v7.py", "src/analysis_v7/models.py"}
OOF_MODELS = (*OLD_BASE_MODELS, "gam")


def validate_manifests(manifests):
    """Require the four ablations to use one source, fitting code and design."""
    reference, shared_sources = None, {}
    for variant in VARIANTS:
        manifest = manifests[variant]
        config = manifest["config"]
        if config.get("include_historical_baseline") is not False or config.get("d1_cutoff_hour") != 18:
            raise ValueError("D-1 18:00 and exclusion of unverified baseline are required")
        if config.get("station_hour_interaction") is not True or config.get("linear_design") != "full_shared_with_training_rank_filter":
            raise ValueError("Preserved common station/hour inputs required")
        expected = previous.FEATURES[variant]
        if config.get("d1_features", []) != expected or config.get("candidate_groups") != previous.CALENDAR + expected:
            raise ValueError("Prespecified candidate groups changed")
        if config.get("gam_fusion_models") != list(FUSION_MODELS) or config.get("fusion_top_k") != 2:
            raise ValueError("Fusion members or historical selection changed")
        settings = {key: value for key, value in config.items() if key not in previous.VARY_KEYS}
        inputs = previous._hashes(manifest["data"]["manifest"])
        code = previous._hashes(manifest["code"])
        fit = {path: code[path] for path in FIT_CODE if path in code}
        if set(fit) != FIT_CODE:
            raise ValueError("Missing critical v7 fitting code hashes")
        base = {path: value for path, value in inputs.items() if path not in previous.D1_PATHS}
        current = (settings, base, fit, manifest.get("packages", {}))
        if reference is not None and current != reference:
            raise ValueError("Variants change settings, fitting packages, source data or code")
        reference = current
        required = ({"docs/d1_program_sources.json"} if variant in ("fireworks", "both") else set())
        required |= ({"docs/d1_weather_sources.json"} if variant in ("weather", "both") else set())
        if not required.issubset(inputs):
            raise ValueError("D1 source hashes missing")
        for path in previous.D1_PATHS.intersection(inputs):
            if path in shared_sources and shared_sources[path] != inputs[path]:
                raise ValueError("D1 source hashes differ across variants")
            shared_sources[path] = inputs[path]
    return {"base_input_hashes": reference[1], "d1_source_hashes": shared_sources,
            "model_code_hashes": reference[2], "same_settings_except_feature_groups": True}


def validate_repair_pair(current, baseline, current_manifest, baseline_manifest):
    """Allow only GAM controls/code to change from v6 and reproduce old eight."""
    current, baseline = previous.align_panels(current, baseline)
    new, old = current_manifest["config"], baseline_manifest["config"]
    non_gam = lambda data: {key: value for key, value in data.items()
                            if key not in previous.VERSION_KEYS and not key.startswith("gam_")}
    if non_gam(new) != non_gam(old):
        raise ValueError("Non-GAM model settings changed during repair")
    if previous._hashes(current_manifest["data"]["manifest"]) != previous._hashes(baseline_manifest["data"]["manifest"]):
        raise ValueError("Repair comparison source data changed")
    new_code, old_code = previous._hashes(current_manifest["code"]), previous._hashes(baseline_manifest["code"])
    for path in {*previous.PRESERVED_CODE, "src/analysis_v6/models.py"}:
        if path not in old_code or path not in new_code or new_code[path] != old_code[path]:
            raise ValueError("Legacy fitting or preprocessing code changed")
    if current_manifest.get("packages", {}) != baseline_manifest.get("packages", {}):
        raise ValueError("Fitting package versions changed during repair")
    differences = {}
    for model in OLD_MODELS:
        differences[model] = float(np.max(abs(current[model].to_numpy() - baseline[model].to_numpy())))
        if not np.allclose(current[model], baseline[model], rtol=1e-12, atol=1e-8):
            raise ValueError(f"Legacy {model} predictions changed during repair")
    changed_controls = {key: {"before": old.get(key), "after": new.get(key)} for key in sorted(set(new) | set(old))
                        if key.startswith("gam_") and new.get(key) != old.get(key)}
    return {"same_panel_truth_and_cutoff": True, "same_non_gam_settings_sources_packages": True,
            "prediction_atol": 1e-8, "prediction_rtol": 1e-12, "legacy_max_abs_difference": differences,
            "changed_gam_controls": changed_controls}


def deduplicate_oof(source):
    """Count each past forecast once, never once per outer fit reusing it."""
    required = set(KEYS + ["year", "outer_year", "base_train_end", "y", *OOF_MODELS])
    if required - set(source):
        raise ValueError(f"Missing OOF fields: {sorted(required - set(source))}")
    frame = source.copy()
    frame["event_date"] = pd.to_datetime(frame.event_date, errors="raise")
    frame["base_train_end"] = pd.to_datetime(frame.base_train_end, errors="raise")
    if frame.empty or frame[KEYS + ["base_train_end", "outer_year"]].isna().any().any():
        raise ValueError("Empty OOF data or missing keys/cutoffs")
    if not frame.year.eq(frame.event_date.dt.year).all() or not (frame.outer_year > frame.year).all():
        raise ValueError("OOF dates or outer chronology invalid")
    if not (frame.base_train_end < frame.event_date).all():
        raise ValueError("OOF base training must precede each activity")
    if frame.duplicated(["outer_year", *KEYS]).any():
        raise ValueError("Duplicate OOF activity within one outer fit")
    if not np.isfinite(frame[["y", *OOF_MODELS]].to_numpy(float)).all() or (frame[["y", *OOF_MODELS]] < 0).any().any():
        raise ValueError("OOF truth/predictions must be finite and nonnegative")
    for _, part in frame.groupby(KEYS):
        if part.base_train_end.nunique() != 1 or part.y.nunique() != 1:
            raise ValueError("Repeated OOF forecast changes truth or base cutoff")
        if not np.allclose(part[list(OOF_MODELS)], part[list(OOF_MODELS)].iloc[0].to_numpy(), rtol=1e-12, atol=1e-8):
            raise ValueError("Repeated OOF forecast differs between outer fits")
    unique = frame.sort_values([*KEYS, "outer_year"]).drop_duplicates(KEYS).drop(columns="outer_year").reset_index(drop=True)
    expected = set(itertools.product(STATIONS, HOURS))
    for _, part in unique.groupby("event_date"):
        if len(part) != len(expected) or set(zip(part.station, part.hour)) != expected:
            raise ValueError("Incomplete OOF station/hour panel")
        if part.base_train_end.nunique() != 1:
            raise ValueError("One OOF event must have one common base cutoff")
    return unique


def validate_oof_pair(current, baseline):
    current, baseline = deduplicate_oof(current), deduplicate_oof(baseline)
    if not current[KEYS + ["base_train_end", "y"]].equals(baseline[KEYS + ["base_train_end", "y"]]):
        raise ValueError("OOF repair samples, truths or training cutoffs differ")
    for model in OLD_BASE_MODELS:
        if not np.allclose(current[model], baseline[model], rtol=1e-12, atol=1e-8):
            raise ValueError(f"OOF legacy {model} predictions changed")
    return {"events": int(current.event_date.nunique()), "unique_rows": len(current),
            "same_old_four_predictions": True, "deduplicated_outer_reuse": True}


def oof_health_table(source):
    frame, rows = deduplicate_oof(source), []
    for year, yearly in frame.groupby("year"):
        for scope, block in (("all", yearly), ("beimen", yearly.loc[yearly.station.eq("北門")])):
            for model in OOF_MODELS:
                error = block[model] - block.y
                event = pd.DataFrame({"event_date": block.event_date, "ae": abs(error), "bias": error})
                event = event.groupby("event_date")[["ae", "bias"]].mean()
                rows.append({"oof_year": int(year), "scope": scope, "model": model, "events": len(event), "rows": len(block),
                             "event_mae": float(event.ae.mean()), "event_bias": float(event.bias.mean()),
                             "max_prediction": float(block[model].max()), "max_actual": float(block.y.max()),
                             "max_absolute_error": float(abs(error).max()),
                             "scope_note": "past_oof_training_diagnostic_deduplicated_not_outer_test"})
    return pd.DataFrame(rows)


def validate_diagnostics(diagnostic, extrapolation, config, outer, meta):
    """Audit all grid trials and omitted/active inputs, not only final fits."""
    keys = ["layer", "parent_train_end", "train_end", "validation_start", "validation_end", "validation_year", "lambda", "ridge_alpha"]
    required = {*keys, "train_events", "validation_events", "event_mae", "max_raw_prediction", "min_raw_prediction",
                "max_clipped_prediction", "max_truth", "n_prediction_rows", "normal_equation_relative_residual", "numerical_verified"}
    if required - set(diagnostic) or diagnostic.empty:
        raise ValueError("Missing all-layer GAM numerical diagnostics")
    if set(diagnostic.layer) != {"inner_grid", "middle_oof", "outer"} or diagnostic.duplicated(keys).any():
        raise ValueError("Unexpected or duplicated GAM diagnostic evaluations")
    numeric = ["event_mae", "max_raw_prediction", "min_raw_prediction", "max_clipped_prediction", "max_truth", "normal_equation_relative_residual"]
    if not np.isfinite(diagnostic[numeric].to_numpy(float)).all():
        raise ValueError("Nonfinite GAM diagnostic")
    if not diagnostic.numerical_verified.eq(True).all() or not diagnostic.normal_equation_relative_residual.between(0, 1e-6).all():
        raise ValueError("Unverified GAM numerical fit")
    if not (pd.to_datetime(diagnostic.train_end) < pd.to_datetime(diagnostic.validation_start)).all():
        raise ValueError("GAM diagnostic training reaches validation data")
    if not (pd.to_datetime(diagnostic.train_end) <= pd.to_datetime(diagnostic.parent_train_end)).all():
        raise ValueError("GAM diagnostic base fit exceeds parent training cutoff")
    past = diagnostic.loc[~diagnostic.layer.eq("outer")]
    if not (pd.to_datetime(past.validation_end) <= pd.to_datetime(past.parent_train_end)).all():
        raise ValueError("Inner or OOF validation exceeds parent training cutoff")
    if not diagnostic.n_prediction_rows.eq(diagnostic.validation_events * len(STATIONS) * len(HOURS)).all():
        raise ValueError("GAM diagnostic panel row count changed")
    expected = set(itertools.product(config["gam_lambdas"], config["gam_ridge_alphas"]))
    for _, block in diagnostic.loc[diagnostic.layer.eq("inner_grid")].groupby(["parent_train_end", "validation_year"]):
        if len(block) != len(expected) or set(zip(block["lambda"], block.ridge_alpha)) != expected:
            raise ValueError("GAM grid audit omitted a candidate")
    unique = deduplicate_oof(meta)
    # Reported selected middle/outer evaluations must agree with the prediction
    # files; the inner-grid table is a search diagnostic, never final accuracy.
    for row in diagnostic.loc[~diagnostic.layer.eq("inner_grid")].itertuples(index=False):
        block = (outer if row.layer == "outer" else unique)
        block = block.loc[block.year.eq(row.validation_year)]
        if block.empty and row.layer == "middle_oof":
            # Final research-fit OOF may extend one year past the OOF used for
            # outer weights. That same chronological fit is the saved last
            # outer prediction, so verify it instead of silently skipping it.
            block = outer.loc[outer.year.eq(row.validation_year)]
        if block.empty:
            raise ValueError("GAM diagnostic forecast absent from predictions")
        cutoff = "train_end" if "train_end" in block else "base_train_end"
        if not pd.to_datetime(block[cutoff]).eq(pd.Timestamp(row.train_end)).all():
            raise ValueError("Selected GAM diagnostic training cutoff differs from predictions")
        error = abs(block.gam - block.y).groupby(block.event_date).mean().mean()
        expected_values = np.array([error, block.gam.max(), block.y.max(), len(block)], float)
        actual_values = np.array([row.event_mae, row.max_clipped_prediction, row.max_truth, row.n_prediction_rows], float)
        if not np.allclose(actual_values, expected_values, rtol=1e-10, atol=1e-7):
            raise ValueError("Selected GAM diagnostics disagree with saved predictions")
    audit_required = {*keys, "column", "known_training_events", "active", "missing_indicator_active", "support_gate_passed", "gate_uses_y"}
    if audit_required - set(extrapolation) or extrapolation.empty or extrapolation.duplicated(keys + ["column"]).any():
        raise ValueError("Missing/duplicated complete GAM candidate support audit")
    expected_columns = set(config["candidate_groups"]) - set(config["gam_binary_groups"])
    diag_keys = diagnostic[keys].copy()
    matched = extrapolation.merge(diag_keys.assign(_recorded=True), on=keys, how="left", validate="many_to_one")
    if not matched._recorded.eq(True).all() or len(extrapolation) != len(diagnostic) * len(expected_columns):
        raise ValueError("GAM extrapolation audit does not cover every evaluation")
    for _, block in extrapolation.groupby(keys):
        if set(block.column) != expected_columns:
            raise ValueError("GAM candidate omitted from extrapolation audit")
    low = extrapolation.known_training_events < config["gam_min_known_events"]
    if extrapolation.loc[low, ["active", "missing_indicator_active", "support_gate_passed"]].any().any():
        raise ValueError("Unsupported continuous GAM feature or missing indicator remained active")
    if not extrapolation.gate_uses_y.eq(False).all():
        raise ValueError("GAM support gate used response data")
    return {"all_layers": sorted(diagnostic.layer.unique()), "evaluations": len(diagnostic),
            "candidate_feature_audits": len(extrapolation), "all_grid_candidates_audited": True,
            "all_fits_numerically_verified": True, "unsupported_inputs_and_indicators_omitted": True,
            "max_normal_equation_relative_residual": float(diagnostic.normal_equation_relative_residual.max()),
            "max_prediction_any_grid_or_selected_evaluation": float(diagnostic.max_clipped_prediction.max()),
            "warning": "all_grid_diagnostics_are_not_selected_model_generalization_scores"}


def peak_table(frame):
    frame, rows = previous.validate_panel(frame), []
    for date, block in frame.loc[frame.station.eq("北門")].groupby("event_date"):
        block = block.sort_values("hour")
        actual = block.loc[block.y.idxmax()]
        for model in COMPARE_MODELS:
            predicted = block.loc[block[model].idxmax()]
            rows.append({"event_date": date.strftime("%Y-%m-%d"), "year": int(actual.year), "model": model,
                         "actual_peak_hour": int(actual.hour), "actual_peak_trips": float(actual.y),
                         "prediction_at_actual_peak": float(actual[model]), "predicted_peak_hour": int(predicted.hour),
                         "predicted_max_trips": float(predicted[model]),
                         "signed_underprediction": float(actual.y - actual[model]),
                         "positive_underprediction": float(max(0., actual.y - actual[model])),
                         "peak_hour_absolute_error": int(abs(actual.hour - predicted.hour))})
    return pd.DataFrame(rows)


def paired_changes(summary):
    keys = ["variant", "scope", "model"]
    values = ["mae", "rmse", "peak_underprediction_mean", "peak_absolute_error_mean", "peak_hour_mae"]
    if summary.duplicated(["version", *keys]).any():
        raise ValueError("Duplicate paired summary cell")
    old, new = (summary.loc[summary.version.eq(version), keys + values] for version in VERSIONS)
    result = old.merge(new, on=keys, suffixes=("_v6", "_v7"), how="outer", validate="one_to_one", indicator=True)
    if not result._merge.eq("both").all():
        raise ValueError("Incomplete before/after metric comparison")
    result = result.drop(columns="_merge")
    for value in values:
        result[f"{value}_change_v7_minus_v6"] = result[f"{value}_v7"] - result[f"{value}_v6"]
    return result


def read_variants(root):
    root, frames, manifests, folds, meta = Path(root), {}, {}, {}, {}
    checks = {"repair_pairs": {}, "v5_reproduction": {}, "oof_pairs": {}, "prediction_hashes": {},
              "meta_prediction_hashes": {}, "all_layer_diagnostics": {}}
    reference = None
    for version in VERSIONS:
        frames[version], manifests[version], folds[version], meta[version] = {}, {}, {}, {}
        for variant in VARIANTS:
            source = root / f".local/analysis_{version}" / variant / "outer_predictions.csv"
            frame = previous.validate_panel(pd.read_csv(source))
            if reference is not None:
                previous._same_panel(reference, frame)
            else:
                reference = frame
            public = root / f"result/analysis_{version}" / variant
            manifest = json.loads((public / "run_manifest.json").read_text(encoding="utf-8"))
            outer = manifest["outer_test"]
            if (outer["rows"], outer["events"], outer["years"]) != (len(frame), frame.event_date.nunique(), sorted(frame.year.unique())):
                raise ValueError("Manifest/prediction sample mismatch")
            fold = json.loads((public / "folds.json").read_text(encoding="utf-8"))
            previous.validate_fusion_predictions(frame, fold)
            frames[version][variant], manifests[version][variant], folds[version][variant] = frame, manifest, fold
            meta_path = root / f".local/analysis_{version}" / variant / "meta_predictions.csv"
            meta[version][variant] = pd.read_csv(meta_path)
            checks["prediction_hashes"][f"{version}/{variant}"] = hashlib.sha256(source.read_bytes()).hexdigest()
            checks["meta_prediction_hashes"][f"{version}/{variant}"] = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    checks["v6_design"] = previous.validate_manifests(manifests["v6"])
    checks["v7_design"] = validate_manifests(manifests["v7"])
    for variant in VARIANTS:
        checks["repair_pairs"][variant] = validate_repair_pair(frames["v7"][variant], frames["v6"][variant], manifests["v7"][variant], manifests["v6"][variant])
        checks["oof_pairs"][variant] = validate_oof_pair(meta["v7"][variant], meta["v6"][variant])
        public = root / "result/analysis_v7" / variant / "tables"
        checks["all_layer_diagnostics"][variant] = validate_diagnostics(
            pd.read_csv(public / "gam_validation_diagnostics.csv"), pd.read_csv(public / "gam_all_extrapolation.csv"),
            manifests["v7"][variant]["config"], frames["v7"][variant], meta["v7"][variant])
        v5_path = root / ".local/analysis_v5" / variant / "outer_predictions.csv"
        v5_frame = pd.read_csv(v5_path)
        v5_manifest = json.loads((root / "result/analysis_v5" / variant / "run_manifest.json").read_text(encoding="utf-8"))
        checks["v5_reproduction"][variant] = {
            version: previous.validate_old_reproduction(frames[version][variant], v5_frame, manifests[version][variant], v5_manifest)
            for version in VERSIONS}
        checks["v5_reproduction"][variant]["v5_prediction_sha256"] = hashlib.sha256(v5_path.read_bytes()).hexdigest()
    return frames, manifests, folds, meta, checks


def comparison_figure(summary, output):
    style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), squeeze=False)
    for ax, (variant, title) in zip(axes.flat, VARIANTS.items()):
        for index, version in enumerate(VERSIONS):
            block = summary.loc[summary.version.eq(version) & summary.variant.eq(variant) & summary.scope.eq("beimen")].set_index("model").loc[list(COMPARE_MODELS)]
            bars = ax.barh(np.arange(len(block)) + (index - .5) * .35, block.mae, height=.33,
                           color=("#AAAAAA", "#46796F")[index], label=VERSION_LABELS[version])
            ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
        ax.set(yticks=np.arange(len(COMPARE_MODELS)), yticklabels=[LABELS[m] for m in COMPARE_MODELS],
               title=title, xlabel="北門活動等權 MAE（人次／小時；越低越好）")
        ax.invert_yaxis()
        maximum = summary.loc[summary.variant.eq(variant) & summary.scope.eq("beimen") & summary.model.isin(COMPARE_MODELS), "mae"].max()
        ax.set_xlim(0, maximum * 1.19)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("GAM 修復前後與固定比較模型｜相同九場外層活動\nRidge、XGBoost 與舊四模型融合保持逐值不變", fontsize=14)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .045, 1, .93))
    save(fig, output / "figures/repair_comparison.png")


def profiles_figure(frames, output):
    style()
    before, after = frames["v6"]["both"], frames["v7"]["both"]
    dates = sorted(after.event_date.unique())
    fig, axes = plt.subplots(int(np.ceil(len(dates) / 3)), 3, figsize=(15, 3.3 * np.ceil(len(dates) / 3)), squeeze=False)
    for ax, date in zip(axes.flat, dates):
        select = lambda f: f.loc[f.event_date.eq(date) & f.station.eq("北門")].sort_values("hour")
        old, new = select(before), select(after)
        for model, label, color, linestyle in (("y", "實際", "#151515", "-"), ("xgboost", "XGBoost", "#A98543", ":"),
                                               ("gam", "修復後 GAM", "#6B87AB", "-."), ("gam_stack", "修復後學權重", "#377C6C", "--")):
            ax.plot(new.hour, new[model], linestyle, color=color, linewidth=1.8, label=label)
        ax.plot(old.hour, old.gam_stack, "--", color="#AC7777", linewidth=1.5, label="修復前學權重")
        ax.set(title=f"{pd.Timestamp(date):%Y-%m-%d}", xticks=HOURS, xlabel="來源小時", ylabel="北門起站人次", ylim=(0, None))
        ax.grid(axis="y", alpha=.2)
    for ax in list(axes.flat)[len(dates):]:
        ax.axis("off")
    fig.suptitle("北門逐場曲線｜固定煙火＋天氣組｜GAM 與融合修復前後\n修復數值失穩，不代表所有尖峰都能預測", fontsize=14)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, .055, 1, .93))
    save(fig, output / "figures/beimen_repair_profiles.png")


def weights_figure(weights, output):
    style()
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    colors = {"ridge": "#7D96AB", "gam": "#4A8978", "xgboost": "#B7944F"}
    for ax, (variant, label) in zip(axes, VARIANTS.items()):
        years = sorted(weights.loc[weights.variant.eq(variant), "outer_year"].unique())
        ticks, texts = [], []
        for offset, year in enumerate(years):
            for i, version in enumerate(VERSIONS):
                x = 2.5 * offset + i
                part = weights.loc[weights.version.eq(version) & weights.variant.eq(variant) & weights.outer_year.eq(year) & weights.fusion.eq("gam_stack")].set_index("base_model")
                bottom = 0.
                for model in FUSION_MODELS:
                    value = float(part.loc[model, "weight"])
                    ax.bar(x, value, bottom=bottom, width=.8, color=colors[model], label=LABELS[model] if offset == 0 and i == 0 else None)
                    if value >= .13:
                        ax.text(x, bottom + value / 2, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
                    bottom += value
                ticks.append(x)
                texts.append(f"{year}\n{'前' if version == 'v6' else '後'}")
        ax.set(title=label, xticks=ticks, xticklabels=texts, ylim=(0, 1.02))
    axes[0].set_ylabel("非負且合計為 1 的融合權重")
    fig.suptitle("修復前後：只以過去 OOF 預測學權重\n權重不是變數重要性，也不保證外層優於最佳單一模型", fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .1, 1, .85))
    save(fig, output / "figures/fusion_weight_repair.png")


def oof_figure(health, output):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, metric, title in zip(axes, ("event_mae", "max_prediction"), ("過去 OOF 活動等權 MAE", "過去 OOF 最大單列預測")):
        for version, color in zip(VERSIONS, ("#AC7777", "#377C6C")):
            part = health.loc[health.version.eq(version) & health.variant.eq("both") & health.scope.eq("all") & health.model.eq("gam")].sort_values("oof_year")
            ax.plot(part.oof_year, part[metric], "o-", color=color, label=VERSION_LABELS[version])
        if metric == "max_prediction":
            ax.plot(part.oof_year, part.max_actual, "--", color="#333333", label="同年度最大實際值")
        ax.set(title=title, xlabel="OOF 驗證年度", ylabel="人次（對數刻度）", yscale="log", xticks=part.oof_year)
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle("固定煙火＋天氣組：GAM 早期驗證是否仍失穩\n重複外層所用的相同 OOF 已去重；這是融合訓練診斷，不是外層測試成績", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .86))
    save(fig, output / "figures/gam_oof_health.png")


def write_report(summary, changes, weights, health, output):
    selected = changes.loc[changes.variant.eq("both") & changes.scope.eq("beimen") & changes.model.isin(COMPARE_MODELS)].set_index("model").loc[list(COMPARE_MODELS)].reset_index()
    table = selected[["model", "mae_v6", "mae_v7", "mae_change_v7_minus_v6", "peak_underprediction_mean_v6", "peak_underprediction_mean_v7"]].copy()
    table.model = table.model.map(LABELS)
    table.columns = ["模型", "修前 MAE", "修後 MAE", "MAE 差（後−前）", "修前尖峰低估", "修後尖峰低估"]
    parts = ["# 第七版：GAM 修復後重跑融合", "", "本版固定原資料、D−1 18:00 來源閘門、候選變數與時間切分，只修復 GAM 的支援範圍／數值穩定性設計。"
             "四組特徵、九場共同外層活動與原八種方法都保留。所有比較先驗證來源雜湊、套件版本、切分，以及舊方法逐值重現。", "",
             "## 固定煙火＋天氣組：北門結果", "", markdown_table(table.round(2)), "",
             "此表 MAE 是北門六個時段先每場平均、再跨場平均的逐時絕對誤差。尖峰低估在每場北門實際最高時段計算正向低估量，高估記為零；不能用模型自己選的最高值代替。", "",
             "![修復前後 MAE](figures/repair_comparison.png)", "", "## 融合權重與早期 OOF 診斷", "",
             "等權融合固定 Ridge、GAM、XGBoost 各 1/3；學權重仍僅依過去向前 OOF 預測，以活動等權 MAE 估計非負且合計為 1 的權重。"
             "本版不因外層表現更換成員、刪除年份或事後選誤差最小的一組特徵。", "",
             "![融合權重](figures/fusion_weight_repair.png)", "", "![GAM OOF 健康](figures/gam_oof_health.png)", ""]
    learned = selected.set_index("model").loc["gam_stack"]
    direction = "增加" if learned.mae_change_v7_minus_v6 > 0 else "降低"
    overall = changes.loc[changes.variant.eq("both") & changes.scope.eq("all")].set_index("model")
    bm = selected.set_index("model")
    parts += [f"固定煙火＋天氣組，學權重融合的全站 MAE {overall.loc['gam_stack', 'mae_v6']:.2f} → {overall.loc['gam_stack', 'mae_v7']:.2f}；"
              f"北門 {learned.mae_v6:.2f} → {learned.mae_v7:.2f}（{direction} {abs(learned.mae_change_v7_minus_v6):.2f}）。"
              f"同組 XGBoost 全站／北門 MAE 為 {overall.loc['xgboost', 'mae_v7']:.2f}／{bm.loc['xgboost', 'mae_v7']:.2f}。"
              "修復讓過去 OOF 預測更穩定，但融合仍未勝過 XGBoost；GAM 得到非零權重不等於外層勝出。"
              "也不能依此事後調權重或刪掉年份。", ""]
    diagnostic = health.loc[health.variant.eq("both") & health.scope.eq("all") & health.model.eq("gam"), ["version", "oof_year", "events", "event_mae", "max_prediction", "max_actual"]].copy()
    diagnostic.version = diagnostic.version.map(VERSION_LABELS)
    diagnostic.columns = ["版本", "OOF 年度", "活動數", "MAE", "最大單列預測", "最大實際值"]
    parts += [markdown_table(diagnostic.round(2)), "",
              "OOF 健康表將不同外層年度重複使用的同一筆過去預測去重，並先檢查重複值一致。"
              "它描述訓練權重時可見的預測穩定性，不是未見年度的泛化成績。", "", "## 北門尖峰與限制", "",
              "![北門九場曲線](figures/beimen_repair_profiles.png)", "",
              "- 修復數值失穩與改善尖峰預測是兩件事。若三個模型同時低估，非負且總和為 1 的融合無法超過當筆最高的基模型預測。",
              "- 站時列的應變數是起站人次，不是獨立參加人數，也不是已驗證的分鐘月台到達率。",
              "- 新增節目與天氣欄位維持 D−1 來源閘門；早年歷史標籤發布時間／預定活動日期版本未全核實，仍不可宣稱完整即時部署驗證。",
              "- 九場外層活動已反覆查看，本次屬探索性修復比較；改善需要未見活動再驗證。",
              "- 精確修復規格、懲罰公式與實作細節見根目錄 [process.md](../../process.md)。", "",
              "完整資料：`tables/metrics_before_after.csv`、`tables/paired_changes.csv`、`tables/yearly_before_after.csv`、"
              "`tables/beimen_actual_peaks.csv`、`tables/oof_health.csv`、`tables/fusion_weights_before_after.csv`。"
              "來源與重現檢查見 `comparison_audit.json`。", ""]
    (output / "report.md").write_text("\n".join(parts), encoding="utf-8")


def build_comparison(root=ROOT, output=None):
    root = Path(root)
    output = Path(output) if output is not None else root / "result/analysis_v7"
    frames, manifests, folds, meta, checks = read_variants(root)
    summary, annual, peaks, health, weights = [], [], [], [], []
    for version in VERSIONS:
        weights.append(previous.fusion_table(folds[version]).assign(version=version))
        for variant, frame in frames[version].items():
            labels = {"version": version, "variant": variant, "variant_label": VARIANTS[variant]}
            summary.append(previous.metrics_table(frame).assign(**labels))
            peaks.append(peak_table(frame).assign(**labels))
            health.append(oof_health_table(meta[version][variant]).assign(**labels))
            for year, block in frame.groupby("year"):
                annual.append(previous.metrics_table(block).assign(**labels, year=int(year)))
    summary, annual, peaks, health, weights = (pd.concat(parts, ignore_index=True) for parts in (summary, annual, peaks, health, weights))
    changes = paired_changes(summary)
    for directory in ("tables", "figures"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    for name, frame in (("metrics_before_after", summary), ("paired_changes", changes), ("yearly_before_after", annual),
                        ("beimen_actual_peaks", peaks), ("oof_health", health), ("fusion_weights_before_after", weights)):
        frame.to_csv(output / "tables" / f"{name}.csv", index=False, encoding="utf-8-sig")
    comparison_figure(summary, output)
    profiles_figure(frames, output)
    weights_figure(weights, output)
    oof_figure(health, output)
    write_report(summary, changes, weights, health, output)
    checks["comparison_scope"] = "paired_exploratory_repair_on_inspected_outer_activities_not_model_selection"
    checks["oof_scope"] = "past_training_diagnostics_with_outer_reuse_deduplicated"
    checks["reporting_code"] = {"path": "src/analysis_v7/reporting.py", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "comparison_audit.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(output), "events": int(summary.events.max()), "versions": list(VERSIONS), "variants": list(VARIANTS)}


if __name__ == "__main__":
    print(json.dumps(build_comparison(), ensure_ascii=False, indent=2))
