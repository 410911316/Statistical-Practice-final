"""Paired, fixed-split evaluation of preannounced show-end predictors.

This module reports completed fits; it cannot select features, models or weights
from outer-test outcomes. All eleven methods may change when inputs change.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_v1.reporting import markdown_table, save, style
from analysis_v6 import reporting as common
from analysis_v7 import reporting as repair
from analysis_v8.features import END_FEATURES, STATION_CODES

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = common.VARIANTS
ALL_MODELS = common.ALL_MODELS
FUSION_MODELS = common.FUSION_MODELS
KEYS, HOURS, STATIONS = common.KEYS, common.HOURS, common.STATIONS
LABELS = repair.LABELS
VERSIONS = ("v7", "v8")
VERSION_LABELS = {"v7": "原模型", "v8": "加入展演結束訊號"}
COMPARE_MODELS = ("ols", "ridge", "lasso", "gam", "xgboost", "gam_equal", "gam_stack")
END_SOURCE = "docs/d1_show_end_sources.json"
END_CONFIG = {"baseline_variant", "end_source_file", "end_signal_policy"}
END_POLICY = "station_specific_overlap_0_60_and_60_120_minutes"
ENCODER_CODE = "src/analysis_v1/models.py"
PRESERVED_CODE = (*common.PRESERVED_CODE, "src/analysis_v6/models.py", "src/analysis_v7/models.py")
PRESERVED_CODE = set(PRESERVED_CODE) - {ENCODER_CODE}
FIT_CODE = {*PRESERVED_CODE, ENCODER_CODE, "src/run_analysis_v8.py", "src/analysis_v8/features.py"}


def validate_pair(current, baseline, current_manifest, baseline_manifest):
    """Require identical samples/settings, allowing only eight extra inputs."""
    current, baseline = common.align_panels(current, baseline)
    new, old = current_manifest["config"], baseline_manifest["config"]
    if new.get("candidate_groups") != old.get("candidate_groups", []) + list(END_FEATURES):
        raise ValueError("Only eight prespecified show-end candidates may be appended")
    ignored = common.VERSION_KEYS | END_CONFIG | {"candidate_groups"}
    if {k: v for k, v in new.items() if k not in ignored} != {k: v for k, v in old.items() if k not in ignored}:
        raise ValueError("Model grids or D-1 input settings changed")
    if new.get("end_source_file") != END_SOURCE or new.get("end_signal_policy") != END_POLICY:
        raise ValueError("Unexpected end-source or feature policy")
    before = common._hashes(baseline_manifest["data"]["manifest"])
    after = common._hashes(current_manifest["data"]["manifest"])
    allowed_extra = {END_SOURCE, "docs/d1_program_sources.json"}
    if (END_SOURCE not in after or any(after.get(k) != v for k, v in before.items())
            or set(after) - set(before) - allowed_extra):
        raise ValueError("Existing data or program/weather source changed")
    before_code, after_code = (common._hashes(m["code"]) for m in (baseline_manifest, current_manifest))
    if any(k not in before_code or k not in after_code or before_code[k] != after_code[k] for k in PRESERVED_CODE):
        raise ValueError("Existing fitting/preprocessing code changed")
    if ENCODER_CODE not in before_code or ENCODER_CODE not in after_code:
        raise ValueError("Missing shared encoder hashes")
    if current_manifest.get("packages", {}) != baseline_manifest.get("packages", {}):
        raise ValueError("Fitting package versions changed")
    return {"same_panel_truth_and_cutoffs": True, "same_model_settings_packages_and_existing_sources": True,
            "added_candidates": list(END_FEATURES), "all_models_refit_with_new_inputs": True,
            "prediction_changes_allowed": list(ALL_MODELS),
            "shared_encoder_extension": {"path": ENCODER_CODE, "before_sha256": before_code[ENCODER_CODE],
                "after_sha256": after_code[ENCODER_CODE],
                "reviewed_scope": "Exact eight planned-end names added to optional whitelist and [0,1] range validation; manually reviewed code delta, not an automatic algorithm-equivalence proof"}}


def validate_manifests(manifests):
    reference, shared = None, {}
    for variant, manifest in manifests.items():
        config = manifest["config"]
        if config.get("candidate_groups") != common.CALENDAR + common.FEATURES[variant] + list(END_FEATURES):
            raise ValueError("Prespecified feature groups changed")
        if config.get("baseline_variant") != variant or config.get("end_source_file") != END_SOURCE or config.get("end_signal_policy") != END_POLICY:
            raise ValueError("Incorrect baseline variant or show-end configuration")
        if config.get("include_historical_baseline") is not False or config.get("d1_cutoff_hour") != 18:
            raise ValueError("D-1 cutoff and historical-baseline exclusion changed")
        if config.get("d1_features") != common.FEATURES[variant]:
            raise ValueError("Existing D-1 features changed")
        code, inputs = common._hashes(manifest["code"]), common._hashes(manifest["data"]["manifest"])
        if not FIT_CODE.issubset(code) or END_SOURCE not in inputs:
            raise ValueError("Missing fitting-code or show-end-source hashes")
        varying = common.VARY_KEYS | {"baseline_variant"}
        current = ({k: v for k, v in config.items() if k not in varying},
                   {k: v for k, v in inputs.items() if k not in common.D1_PATHS},
                   {k: code[k] for k in FIT_CODE}, manifest.get("packages", {}))
        if reference is not None and current != reference:
            raise ValueError("Variants changed tuning, fitting code, packages or shared data")
        reference = current
        required = ({"docs/d1_program_sources.json"} if variant in ("fireworks", "both") else set())
        required |= ({"docs/d1_weather_sources.json"} if variant in ("weather", "both") else set())
        if not required.issubset(inputs):
            raise ValueError("Missing existing D-1 source manifest")
        for path in (common.D1_PATHS | {END_SOURCE}).intersection(inputs):
            if path in shared and shared[path] != inputs[path]:
                raise ValueError("Variants disagree on source hashes")
            shared[path] = inputs[path]
    if set(manifests) != set(VARIANTS):
        raise ValueError("All four variants required")
    return {"same_settings_except_candidate_groups": True, "source_hashes": shared,
            "model_code_hashes": reference[2], "main_variant_prespecified": "both"}


def validate_input_pair(current, baseline):
    """Check complete training data, not merely the nine test-event keys."""
    if set(current) != set(baseline) | set(END_FEATURES):
        raise ValueError("Input columns changed beyond the eight show-end predictors")
    for name, data in (("current", current), ("baseline", baseline)):
        if data.duplicated(KEYS).any() or data[KEYS].isna().any().any():
            raise ValueError(f"Duplicate/missing {name} input keys")
    new = current.sort_values(KEYS).reset_index(drop=True)
    old = baseline.sort_values(KEYS).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(new[list(old)], old, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError("Existing input values, truths or activity rows changed") from exc
    expected = {(s, h) for s in STATIONS for h in HOURS}
    if any(len(block) != len(expected) or set(zip(block.station, block.hour)) != expected
           for _, block in new.groupby("event_date")):
        raise ValueError("Incomplete training activity panel")
    return {"events": int(new.event_date.nunique()), "rows": len(new),
            "old_input_columns_bit_identical": True, "new_feature_columns": list(END_FEATURES)}


def validate_end_audit(inputs, audit, outer_dates=None):
    required = {"event_date", "eligible", "scope", "planned_end_hour", "cutoff_at", "effective_available_at"}
    if required - set(audit) or audit.event_date.duplicated().any() or audit.empty:
        raise ValueError("Missing or duplicated event-level end audit")
    if set(audit.event_date) != set(inputs.event_date):
        raise ValueError("End audit must retain every training event")
    eligible = audit.eligible.eq(True)
    if not audit.eligible.isin([True, False]).all():
        raise ValueError("End eligibility must be explicit boolean")
    known = audit.loc[eligible]
    if not known.scope.isin(["fireworks_only", "combined_sky_program"]).all() or not audit.loc[~eligible, "scope"].eq("unknown").all():
        raise ValueError("End-time scope is not a fireworks/combined/unknown distinction")
    cutoff = pd.to_datetime(known.cutoff_at, utc=True, errors="raise")
    available = pd.to_datetime(known.effective_available_at, utc=True, errors="raise")
    if cutoff.isna().any() or available.isna().any() or not (available <= cutoff).all():
        raise ValueError("Known end time is unavailable at D-1 cutoff")
    expected_cutoff = (pd.to_datetime(known.event_date) - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).dt.tz_localize("Asia/Taipei").dt.tz_convert("UTC")
    if not cutoff.reset_index(drop=True).equals(expected_cutoff.reset_index(drop=True)):
        raise ValueError("End audit cutoff must be D-1 18:00 Taipei")
    for row in audit.itertuples(index=False):
        block = inputs.loc[inputs.event_date.eq(row.event_date)]
        values = block[list(END_FEATURES)].to_numpy(float)
        if not row.eligible:
            if not np.isnan(values).all():
                raise ValueError("Unknown show end must remain missing in all eight predictors")
            continue
        if not np.isfinite(row.planned_end_hour) or not 0 <= row.planned_end_hour <= 24:
            raise ValueError("Invalid eligible show end")
        for code, station in STATION_CODES.items():
            for start, end in ((0, 60), (60, 120)):
                left, right = row.planned_end_hour * 60 + start, row.planned_end_hour * 60 + end
                expected = np.maximum(0, np.minimum((block.hour + 1) * 60, right) - np.maximum(block.hour * 60, left))
                expected = expected * block.station.eq(station) / 60.0
                column = f"planned_end_{start}_{end}_{code}"
                if not np.allclose(block[column], expected, rtol=0, atol=1e-9, equal_nan=False):
                    raise ValueError("Show-end overlap predictor does not match event timing and station")
    result = {"events": len(audit), "known_events": int(eligible.sum()), "unknown_events": int((~eligible).sum()),
              "scope_counts": {str(k): int(v) for k, v in audit.scope.value_counts().items()},
              "all_rows_retained": True, "all_eight_overlap_features_recomputed_independently": True,
              "not_measured_egress_time_or_stage_end": True}
    if outer_dates is not None:
        dates = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in outer_dates}
        test = audit.loc[audit.event_date.isin(dates)]
        result.update(outer_events=len(test), outer_known_events=int(test.eligible.sum()))
    return result


def validate_oof_pair(current, baseline):
    new, old = repair.deduplicate_oof(current), repair.deduplicate_oof(baseline)
    if not new[KEYS + ["base_train_end", "y"]].equals(old[KEYS + ["base_train_end", "y"]]):
        raise ValueError("Past OOF activity keys, truth or training cutoffs changed")
    return {"unique_events": int(new.event_date.nunique()), "rows": len(new), "same_past_oof_panel": True}


def peak_table(frame):
    frame, rows = common.validate_panel(frame), []
    for date, block in frame.loc[frame.station.eq("北門")].groupby("event_date"):
        block = block.sort_values("hour")
        actual = block.loc[block.y.idxmax()]
        for model in ALL_MODELS:
            maximum = block.loc[block[model].idxmax()]
            signed = float(actual.y - actual[model])
            rows.append({"event_date": date.strftime("%Y-%m-%d"), "year": int(actual.year), "model": model,
                         "actual_peak_hour": int(actual.hour), "actual_peak_trips": float(actual.y),
                         "prediction_at_actual_peak": float(actual[model]), "predicted_peak_hour": int(maximum.hour),
                         "predicted_max_trips": float(maximum[model]), "signed_underprediction": signed,
                         "positive_underprediction": max(0., signed), "peak_absolute_error": abs(signed),
                         "peak_hour_absolute_error": int(abs(actual.hour - maximum.hour))})
    return pd.DataFrame(rows)


def paired_changes(summary):
    keys = ["variant", "scope", "model"]
    metrics = ["mae", "rmse", "peak_underprediction_mean", "peak_signed_underprediction_mean", "peak_absolute_error_mean", "peak_hour_mae"]
    if summary.duplicated(["version", *keys]).any():
        raise ValueError("Duplicate paired summary cell")
    old, new = (summary.loc[summary.version.eq(v), keys + metrics] for v in VERSIONS)
    paired = old.merge(new, on=keys, suffixes=("_v7", "_v8"), how="outer", validate="one_to_one", indicator=True)
    if not paired._merge.eq("both").all():
        raise ValueError("Incomplete paired model comparison")
    paired = paired.drop(columns="_merge")
    for metric in metrics:
        paired[f"{metric}_change_v8_minus_v7"] = paired[f"{metric}_v8"] - paired[f"{metric}_v7"]
    return paired


def read_variants(root):
    root, frames, manifests, folds, meta, input_frames, end_audits = Path(root), {}, {}, {}, {}, {}, {}
    checks = {"pairs": {}, "input_pairs": {}, "oof_pairs": {}, "end_audits": {}, "all_layer_gam_diagnostics": {}, "file_hashes": {}}
    reference = None
    for version in VERSIONS:
        frames[version], manifests[version], folds[version], meta[version], input_frames[version] = {}, {}, {}, {}, {}
        for variant in VARIANTS:
            public, private = root / f"result/analysis_{version}/{variant}", root / f".local/analysis_{version}/{variant}"
            frame = common.validate_panel(pd.read_csv(private / "outer_predictions.csv"))
            if reference is not None:
                common._same_panel(reference, frame)
            reference = frame
            manifest = json.loads((public / "run_manifest.json").read_text(encoding="utf-8"))
            summary = manifest["outer_test"]
            if (summary["rows"], summary["events"], summary["years"]) != (len(frame), frame.event_date.nunique(), sorted(frame.year.unique())):
                raise ValueError("Manifest and prediction sample differ")
            fold = json.loads((public / "folds.json").read_text(encoding="utf-8"))
            common.validate_fusion_predictions(frame, fold)
            frames[version][variant], manifests[version][variant], folds[version][variant] = frame, manifest, fold
            meta[version][variant] = pd.read_csv(private / "meta_predictions.csv")
            input_frames[version][variant] = pd.read_csv(private / "analysis_input.csv")
            for filename in ("outer_predictions.csv", "meta_predictions.csv", "analysis_input.csv"):
                checks["file_hashes"][f"{version}/{variant}/{filename}"] = hashlib.sha256((private / filename).read_bytes()).hexdigest()
    checks["v7_design"] = repair.validate_manifests(manifests["v7"])
    checks["v8_design"] = validate_manifests(manifests["v8"])
    parent = "docs/d1_program_sources.json"
    if checks["v8_design"]["source_hashes"].get(parent) != checks["v7_design"]["d1_source_hashes"].get(parent):
        raise ValueError("New end-time provenance changed the existing parent program source")
    for variant in VARIANTS:
        checks["pairs"][variant] = validate_pair(frames["v8"][variant], frames["v7"][variant], manifests["v8"][variant], manifests["v7"][variant])
        checks["input_pairs"][variant] = validate_input_pair(input_frames["v8"][variant], input_frames["v7"][variant])
        checks["oof_pairs"][variant] = validate_oof_pair(meta["v8"][variant], meta["v7"][variant])
        cutoff_keys = ("outer_year", "train_events", "test_events", "train_end", "test_start")
        if [{k: f[k] for k in cutoff_keys} for f in folds["v7"][variant]] != [{k: f[k] for k in cutoff_keys} for f in folds["v8"][variant]]:
            raise ValueError("Outer folds changed")
        tables = root / "result/analysis_v8" / variant / "tables"
        end_audits[variant] = pd.read_csv(tables / "end_feature_audit.csv")
        checks["end_audits"][variant] = validate_end_audit(input_frames["v8"][variant], end_audits[variant], frames["v8"][variant].event_date.unique())
        if checks["input_pairs"][variant]["events"] != 21 or checks["input_pairs"][variant]["rows"] != 504:
            raise ValueError("Current project comparison requires all 21 activities / 504 rows")
        if len(frames["v8"][variant]) != 216 or frames["v8"][variant].event_date.nunique() != 9:
            raise ValueError("Current project comparison requires the same nine outer activities")
        if checks["end_audits"][variant]["scope_counts"] != {"fireworks_only": 16, "combined_sky_program": 2, "unknown": 3}:
            raise ValueError("Prespecified 16 fireworks + 2 combined + 3 unknown coverage changed")
        checks["all_layer_gam_diagnostics"][variant] = repair.validate_diagnostics(
            pd.read_csv(tables / "gam_validation_diagnostics.csv"), pd.read_csv(tables / "gam_all_extrapolation.csv"),
            manifests["v8"][variant]["config"], frames["v8"][variant], meta["v8"][variant])
    for variant in VARIANTS:
        if not end_audits[variant].equals(end_audits["both"]):
            raise ValueError("End-time evidence differs between variants")
    return frames, manifests, folds, end_audits, checks


def comparison_figure(summary, output):
    style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), squeeze=False)
    for ax, (variant, title) in zip(axes.flat, VARIANTS.items()):
        block = summary.loc[summary.variant.eq(variant) & summary.scope.eq("beimen")]
        for index, version in enumerate(VERSIONS):
            values = block.loc[block.version.eq(version)].set_index("model").loc[list(COMPARE_MODELS), "mae"]
            bars = ax.barh(np.arange(len(values)) + (index - .5) * .35, values, height=.33, color=("#AAAAAA", "#46796F")[index], label=VERSION_LABELS[version])
            ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
        ax.set(yticks=np.arange(len(COMPARE_MODELS)), yticklabels=[LABELS[m] for m in COMPARE_MODELS], title=title,
               xlabel="北門活動等權 MAE（人次／小時；越低越好）", xlim=(0, block.loc[block.model.isin(COMPARE_MODELS), "mae"].max() * 1.2))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("展演結束訊號 × 車站：加入前後北門誤差\n相同九場活動、原調參網格與時間切分；所有模型重新估計", fontsize=14)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .045, 1, .93))
    save(fig, output / "figures/end_signal_comparison.png")


def profiles_figure(frames, output):
    style()
    before, after = frames["v7"]["both"], frames["v8"]["both"]
    dates = sorted(after.event_date.unique())
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), squeeze=False)
    for ax, date in zip(axes.flat, dates):
        select = lambda f: f.loc[f.event_date.eq(date) & f.station.eq("北門")].sort_values("hour")
        old, new = select(before), select(after)
        ax.plot(new.hour, new.y, "-", color="#111111", lw=2, label="實際")
        for model, color, label in (("xgboost", "#A98543", "XGBoost"), ("gam_stack", "#377C6C", "三模型學權重")):
            ax.plot(old.hour, old[model], ":", color=color, alpha=.8, label=f"原 {label}")
            ax.plot(new.hour, new[model], "-o", markersize=3, color=color, label=f"新增後 {label}")
        ax.set(title=f"{pd.Timestamp(date):%Y-%m-%d}", xticks=HOURS, xlabel="來源小時", ylabel="北門起站人次", ylim=(0, None))
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("北門九場活動逐時曲線｜固定煙火＋天氣組\n評估尖峰高低與出現時段；展演結束訊號不是實際散場人流", fontsize=14)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .075, 1, .93))
    save(fig, output / "figures/beimen_end_signal_profiles.png")


def weights_figure(weights, output):
    style()
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    colors = {"ridge": "#7D96AB", "gam": "#4A8978", "xgboost": "#B7944F"}
    for ax, (variant, label) in zip(axes, VARIANTS.items()):
        ticks, labels = [], []
        for position, year in enumerate(sorted(weights.outer_year.unique())):
            for i, version in enumerate(VERSIONS):
                x = position * 2.5 + i
                block = weights.loc[weights.variant.eq(variant) & weights.version.eq(version) & weights.outer_year.eq(year) & weights.fusion.eq("gam_stack")].set_index("base_model")
                bottom = 0.
                for model in FUSION_MODELS:
                    value = float(block.loc[model, "weight"])
                    ax.bar(x, value, bottom=bottom, width=.8, color=colors[model], label=LABELS[model] if position == i == 0 else None)
                    if value >= .13:
                        ax.text(x, bottom + value / 2, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
                    bottom += value
                ticks.append(x)
                labels.append(f"{year}\n{'原' if version == 'v7' else '新增'}")
        ax.set(title=label, xticks=ticks, xticklabels=labels, ylim=(0, 1.02))
    axes[0].set_ylabel("非負且合計為 1 的權重")
    fig.suptitle("加入展演結束訊號前後：仍僅從過去 OOF 學習融合權重\n權重不是因果效果，也不保證勝過最佳單一模型", fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .1, 1, .85))
    save(fig, output / "figures/end_signal_fusion_weights.png")


def coverage_figure(audit, output):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.5]})
    names = ["fireworks_only", "combined_sky_program", "unknown"]
    values = audit.scope.value_counts().reindex(names, fill_value=0)
    bars = axes[0].bar(["煙火結束", "整套天空展演", "未知，保留缺值"], values, color=["#527F76", "#9A835D", "#B9B9B9"])
    axes[0].bar_label(bars, padding=3)
    axes[0].set(ylabel="活動場次", title="D−1 可用的預定結束時間", ylim=(0, max(values) * 1.2))
    # Teaching illustration independent of observations or model fitting.
    end = 20 + 20 / 60
    for i, (lo, hi, label) in enumerate(((0, 60, "結束後 0–60 分鐘"), (60, 120, "結束後 60–120 分鐘"))):
        hours = np.array(HOURS)
        overlap = np.maximum(0., np.minimum((hours + 1) * 60, end * 60 + hi) - np.maximum(hours * 60, end * 60 + lo))
        axes[1].bar(hours + (i - .5) * .35, overlap, width=.33, color=("#527F76", "#9A835D")[i], label=label)
    axes[1].set(xticks=HOURS, xlabel="來源小時（20 代表 20:00–21:00）", ylabel="每小時重疊分鐘數", title="示例：20:20 結束；輸入 = 重疊分鐘／60", ylim=(0, 60))
    axes[1].legend(frameon=False)
    fig.suptitle("八個預先指定的特徵 = 兩個結束後時間窗 × 四個車站\n右圖只說明編碼；不假設這些分鐘均勻有人流，也不是步行延遲模型", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .85))
    save(fig, output / "figures/end_signal_coverage.png")


def write_report(summary, changes, coverage, output):
    main = changes.loc[changes.variant.eq("both") & changes.scope.eq("beimen")].set_index("model").loc[list(COMPARE_MODELS)]
    table = main.reset_index()[["model", "mae_v7", "mae_v8", "mae_change_v8_minus_v7", "peak_absolute_error_mean_v7", "peak_absolute_error_mean_v8"]]
    table.model = table.model.map(LABELS)
    table.columns = ["模型", "原 MAE", "新增後 MAE", "MAE 差（新−原）", "原尖峰絕對誤差", "新增後尖峰絕對誤差"]
    total = changes.loc[changes.variant.eq("both") & changes.scope.eq("all")].set_index("model")
    parts = ["# 第八版：加入事前公告的展演結束訊號", "", "本版新增『預定展演結束後兩個一小時時間窗 × 四個車站』共八個候選變數，重跑十一種方法。"
             "基準為修復後第七版；既有輸入、GAM 修復、調參網格、D−1 18:00 閘門與向前時間切分維持不變。"
             "主要比較事先固定為煙火＋天氣組，其他三組用於拆解特徵效果。", "", "## 資料與特徵", "",
             f"- 共 {coverage['events']} 場活動、504 筆站時資料；{coverage['known_events']} 場可用預定結束時間，{coverage['unknown_events']} 場維持未知。",
             "- 16 場是煙火結束、2 場是整套天空展演結束；它們不等於舞台結束或民眾實際散場。",
             "- 對來源小時與結束後 0–60／60–120 分鐘求重疊分鐘數，除以 60 後乘上各車站指標（0–1 小時）。未知活動的八欄全部留缺值，沒有刪除該活動或填入事後實際結束時間。",
             "- 相同九場外層活動、216 筆站時資料全部保留。所有模型重新估計，因此既有模型預測可以改變。",
             "- 共用 Encoder 僅擴充這八欄的允許名單與 0–1 範圍檢查；第七版模型演算法本體不變，差異另經程式審查與測試。", "",
             "![結束時間涵蓋與特徵示例](figures/end_signal_coverage.png)", "", "## 固定煙火＋天氣組：北門結果", "", markdown_table(table.round(2)), "",
             "MAE 先在每場的六個小時計算，再跨場平均；差值小於零表示改善。尖峰絕對誤差在每場實際最大人次的小時計算，沒有用模型自己選出的最高值取代。", ""]
    for model in ("xgboost", "gam_equal", "gam_stack"):
        m, a = main.loc[model], total.loc[model]
        direction = "降低" if m.mae_change_v8_minus_v7 < 0 else "增加"
        parts.append(f"- {LABELS[model]}：全站 MAE {a.mae_v7:.2f} → {a.mae_v8:.2f}；北門 MAE {m.mae_v7:.2f} → {m.mae_v8:.2f}（{direction} {abs(m.mae_change_v8_minus_v7):.2f}）。")
    xgb, fusion = main.loc["xgboost"], main.loc["gam_stack"]
    relation = "仍高於" if fusion.mae_v8 > xgb.mae_v8 else "低於" if fusion.mae_v8 < xgb.mae_v8 else "等於"
    time_direction = "增加" if xgb.peak_hour_mae_v8 > xgb.peak_hour_mae_v7 else "降低" if xgb.peak_hour_mae_v8 < xgb.peak_hour_mae_v7 else "不變"
    parts += ["", f"**在同九場北門比較中，學權重融合的 MAE {fusion.mae_v8:.2f} {relation} XGBoost 的 {xgb.mae_v8:.2f}。**"
              "特徵改善不代表融合一定優於單一模型。",
              f"XGBoost 的尖峰人次絕對誤差由 {xgb.peak_absolute_error_mean_v7:.2f} → {xgb.peak_absolute_error_mean_v8:.2f}，"
              f"但尖峰出現小時的平均絕對誤差為 {xgb.peak_hour_mae_v7:.4f} → {xgb.peak_hour_mae_v8:.4f} 小時（{time_direction}）。"
              "因此不能把人次誤差降低解讀成尖峰時點也全面改善。", "",
              "![北門 MAE 比較](figures/end_signal_comparison.png)", "", "### 四組特徵的配對結果", "",
              "以下完整展示四組結果；主要結論維持事先固定的煙火＋天氣組，不依外層最低分改選組合。", ""]
    ablations = changes.loc[changes.scope.eq("beimen") & changes.model.isin(["xgboost", "gam_stack"]),
                           ["variant", "model", "mae_v7", "mae_v8", "mae_change_v8_minus_v7"]].copy()
    ablations.variant = ablations.variant.map(VARIANTS)
    ablations.model = ablations.model.map(LABELS)
    ablations.columns = ["原有特徵組", "模型", "原北門 MAE", "新增後北門 MAE", "MAE 差（新−原）"]
    parts += [markdown_table(ablations.round(2)), "", "## 北門尖峰與融合權重", "",
              "![北門逐場曲線](figures/beimen_end_signal_profiles.png)", "", "![融合權重](figures/end_signal_fusion_weights.png)", "",
              "等權融合固定 Ridge、GAM、XGBoost 各 1/3；學權重仍僅利用過去向前 OOF 預測，最小化活動等權 MAE，權重非負且合計為 1。"
              "新增時間特徵可能改變單一模型、過去 OOF 誤差與融合權重；外層結果沒有用來調權重、刪活動或更換模型。", "", "## 解讀與後續", "",
              "- 這次測試的是『事前公告的時間是否有助辨認逐時人流』。它不直接觀測離場者、步行時間、進閘或月台到達，亦不預設必定改善。",
              "- 起站人次不是獨立參加人數；若要接排隊模型，仍需分鐘內到達形狀、方向分流、步行延遲、列車剩餘容量及班距。",
              "- 21 場活動不是 504 場獨立樣本。結束時間變化有限，與原站時設計可能高度相關；秩檢查及正則化在各訓練折內進行。",
              "- 三模型若同時低估，非負且和為 1 的融合無法超過該筆最高基模型預測；不要因融合形式較複雜就宣稱較準。",
              "- 同九場外層活動已被多次查看，本版屬探索性比較，不能宣稱新的獨立驗證；須以未見活動確認。",
              "- 新增來源通過現有 D−1 閘門，原歷史標籤發布時間與早期活動日期版本仍未全部核實，尚未具備完整事前部署認證。",
              "- 實作、公式與原始公告整理見 [process.md](../../process.md) 與 [來源紀錄](../../docs/d1_show_end_sources.json)。", "",
              "完整表格：`tables/metrics_before_after.csv`、`tables/paired_changes.csv`、`tables/yearly_before_after.csv`、"
              "`tables/beimen_actual_peaks.csv`、`tables/fusion_weights_before_after.csv`、`tables/end_time_coverage.csv`。"
              "來源、同筆比較與 GAM 全層診斷見 `comparison_audit.json`。", ""]
    (output / "report.md").write_text("\n".join(parts), encoding="utf-8")


def build_comparison(root=ROOT, output=None):
    root = Path(root)
    output = Path(output) if output is not None else root / "result/analysis_v8"
    frames, manifests, folds, end_audits, checks = read_variants(root)
    summaries, annual, peaks, weights = [], [], [], []
    for version in VERSIONS:
        weights.append(common.fusion_table(folds[version]).assign(version=version))
        for variant, frame in frames[version].items():
            labels = {"version": version, "variant": variant, "variant_label": VARIANTS[variant]}
            summaries.append(common.metrics_table(frame).assign(**labels))
            peaks.append(peak_table(frame).assign(**labels))
            for year, block in frame.groupby("year"):
                annual.append(common.metrics_table(block).assign(**labels, year=int(year)))
    summary, annual, peaks, weights = (pd.concat(parts, ignore_index=True) for parts in (summaries, annual, peaks, weights))
    changes = paired_changes(summary)
    for name in ("tables", "figures"):
        (output / name).mkdir(parents=True, exist_ok=True)
    for name, table in (("metrics_before_after", summary), ("paired_changes", changes), ("yearly_before_after", annual),
                        ("beimen_actual_peaks", peaks), ("fusion_weights_before_after", weights), ("end_time_coverage", end_audits["both"])):
        table.to_csv(output / "tables" / f"{name}.csv", index=False, encoding="utf-8-sig")
    comparison_figure(summary, output)
    profiles_figure(frames, output)
    weights_figure(weights, output)
    coverage_figure(end_audits["both"], output)
    write_report(summary, changes, checks["end_audits"]["both"], output)
    checks["comparison_scope"] = "exploratory_paired_feature_addition_on_previously_inspected_outer_events"
    checks["reporting_code"] = {"path": "src/analysis_v8/reporting.py", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "comparison_audit.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(output), "outer_events": int(summary.events.max()), "versions": list(VERSIONS), "variants": list(VARIANTS)}
