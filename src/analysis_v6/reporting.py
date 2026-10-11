"""Paired temporal comparisons of GAM and prespecified three-model fusion.

This reporting module never fits or selects a model. Historical v5 comparators
must reproduce before a new comparison is published; outer scores only describe
the already inspected test activities, not a new independent validation set.
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
from analysis_v5.reporting import selection_table as old_selection_table
from analysis_v5.reporting import validate_panel as validate_old_panel

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {"core": "基本欄位", "fireworks": "只加煙火時間", "weather": "只加天氣預報", "both": "煙火＋天氣"}
OLD_BASE_MODELS = ("ols", "ridge", "lasso", "xgboost")
OLD_MODELS = (*OLD_BASE_MODELS, "equal", "stack", "select_best", "select_stack")
FUSION_MODELS = ("ridge", "gam", "xgboost")
ALL_MODELS = (*OLD_MODELS, "gam", "gam_equal", "gam_stack")
DISPLAY_MODELS = ("ols", "ridge", "lasso", "gam", "xgboost", "equal", "stack", "select_best", "select_stack", "gam_equal", "gam_stack")
LABELS = {"y": "實際", "ols": "完整 OLS", "ridge": "Ridge", "lasso": "Lasso", "gam": "GAM", "xgboost": "XGBoost",
          "equal": "舊四模型等權", "stack": "舊四模型學權重", "select_best": "舊 OOF 選一", "select_stack": "舊 OOF 前二融合",
          "gam_equal": "新三模型等權", "gam_stack": "新三模型學權重"}
COLORS = {"y": "#161616", "ols": "#84939B", "ridge": "#537A95", "lasso": "#7D8C5D", "gam": "#438879", "xgboost": "#B38D4D",
          "equal": "#A6A6A6", "stack": "#8C6C93", "select_best": "#6C7676", "select_stack": "#777777",
          "gam_equal": "#749C9B", "gam_stack": "#9A6559"}
KEYS = ["event_date", "station", "hour"]
STATIONS = ("北門", "大橋頭站", "民權西路", "雙連")
HOURS = tuple(range(18, 24))
CALENDAR = ["is_weekend", "day_of_year", "year_index"]
FIREWORKS = ["planned_fireworks_minutes", "planned_fireworks_start_hour"]
WEATHER = ["forecast_temperature_c", "forecast_wind_speed_m_s", "forecast_cloud_cover_pct"]
FEATURES = {"core": [], "fireworks": FIREWORKS, "weather": WEATHER, "both": FIREWORKS + WEATHER}
MEAN_MODELS = ("y", *FUSION_MODELS, "gam_equal", "gam_stack", "stack")
VARY_KEYS = {"version", "report_title", "output_dir", "private_output_dir", "experiment_notes",
             "candidate_groups", "d1_features", "d1_source_files"}
VERSION_KEYS = {"version", "report_title", "output_dir", "private_output_dir", "experiment_notes"}
D1_PATHS = {"docs/d1_program_sources.json", "docs/d1_weather_sources.json"}
FIT_CODE = {"src/run_analysis_v6.py", "src/analysis_v6/models.py", "src/analysis_v5/models.py",
            "src/analysis_v1/models.py", "src/analysis_v1/data.py", "src/analysis_v1/d1_features.py"}
PRESERVED_CODE = FIT_CODE - {"src/run_analysis_v6.py", "src/analysis_v6/models.py"}
EFFECT_LABELS = {"day_of_year": "年內日序", "year_index": "年度索引", "planned_fireworks_minutes": "預定煙火長度（分鐘）",
                 "planned_fireworks_start_hour": "預定煙火開始（小時）", "forecast_temperature_c": "D−1 氣溫預報（°C）",
                 "forecast_wind_speed_m_s": "D−1 風速預報（m/s）", "forecast_cloud_cover_pct": "D−1 雲量預報（%）"}


def validate_panel(source: pd.DataFrame) -> pd.DataFrame:
    """Require complete event panels and chronological cutoffs for all methods."""
    frame = validate_old_panel(source)
    missing = set(ALL_MODELS) - set(frame)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if not np.isfinite(frame[list(ALL_MODELS)].to_numpy(float)).all():
        raise ValueError("Nonfinite predictions")
    if (frame[list(ALL_MODELS)].to_numpy(float) < 0).any():
        raise ValueError("Person counts must be nonnegative")
    return frame


def align_panels(left, right):
    left, right = validate_panel(left), validate_panel(right)
    _same_panel(left, right)
    return left, right


def _same_panel(left, right):
    if not left[KEYS].equals(right[KEYS]):
        raise ValueError("Comparators have different holdout panels")
    if not np.array_equal(left.y.to_numpy(float), right.y.to_numpy(float)):
        raise ValueError("Comparators have different truths")
    if not left.train_end.equals(right.train_end):
        raise ValueError("Comparators have different training cutoffs")


def metrics_table(predictions):
    """Event-balanced losses; peak error is measured at each actual peak hour."""
    frame, rows = validate_panel(predictions), []
    for scope, block in (("all", frame), ("beimen", frame.loc[frame.station.eq("北門")])):
        for model in ALL_MODELS:
            error = block[model] - block.y
            event = pd.DataFrame({"event_date": block.event_date, "ae": abs(error), "se": error**2, "bias": error})
            event = event.groupby("event_date")[["ae", "se", "bias"]].mean()
            peaks = []
            for (date, _), part in block.groupby(["event_date", "station"]):
                part = part.sort_values("hour")
                actual, predicted = part.loc[part.y.idxmax()], part.loc[part[model].idxmax()]
                signed = float(actual.y - actual[model])
                peaks.append({"event_date": date, "signed": signed, "absolute": abs(signed),
                              "positive": max(0., signed), "hour_error": abs(float(actual.hour - predicted.hour))})
            peak = pd.DataFrame(peaks).groupby("event_date").mean().mean()
            rows.append({"scope": scope, "model": model, "events": len(event), "rows": len(block),
                         "mae": float(event.ae.mean()), "rmse": float(np.sqrt(event.se.mean())),
                         "bias_pred_minus_actual": float(event.bias.mean()), "peak_absolute_error_mean": float(peak.absolute),
                         "peak_underprediction_mean": float(peak.positive), "peak_signed_underprediction_mean": float(peak.signed),
                         "peak_hour_mae": float(peak.hour_error)})
    return pd.DataFrame(rows)


def mean_event_counts(frames):
    rows, reference = [], None
    for variant, label in VARIANTS.items():
        frame = validate_panel(frames[variant])
        if reference is not None:
            _same_panel(reference, frame)
        else:
            reference = frame
        for scope, part in (("all", frame), ("beimen", frame.loc[frame.station.eq("北門")])):
            totals = part.groupby("event_date")[["y", *ALL_MODELS]].sum()
            actual = float(totals.y.mean())
            for model in ("y", *ALL_MODELS):
                mean = float(totals[model].mean())
                rows.append({"variant": variant, "variant_label": label, "scope": scope, "model": model,
                             "events": len(totals), "stations": part.station.nunique(), "hours_per_station": len(HOURS),
                             "mean_event_origin_trips": mean, "actual_mean_event_origin_trips": actual,
                             "mean_bias_trips": mean - actual, "mean_bias_pct": 100 * (mean / actual - 1) if actual else np.nan,
                             "unit": "origin_person_trips_per_event_18_to_23_not_unique_attendance",
                             "aggregation": "mean_over_events_of_sum_over_stations_and_hours"})
    return pd.DataFrame(rows)


def _hashes(items):
    result = {item["path"].replace("\\", "/"): item["sha256"] for item in items}
    if len(result) != len(items):
        raise ValueError("Duplicate manifest paths")
    return result


def validate_manifests(manifests):
    """Feature ablation must not change source, cutoff, tuning or model code."""
    reference, sources = None, {}
    for variant in VARIANTS:
        m, expected = manifests[variant], FEATURES[variant]
        config = m["config"]
        if config.get("include_historical_baseline") is not False or config.get("d1_cutoff_hour") != 18:
            raise ValueError("All variants must exclude unverified baseline and use D-1 18:00")
        if config.get("station_hour_interaction") is not True or config.get("linear_design") != "full_shared_with_training_rank_filter":
            raise ValueError("All variants require the old shared station/hour design")
        if config.get("d1_features", []) != expected or config.get("candidate_groups") != CALENDAR + expected:
            raise ValueError(f"{variant}: incorrect prespecified feature group")
        if config.get("fusion_top_k") != 2:
            raise ValueError("Historical top-two selection rule must be preserved")
        if config.get("gam_fusion_models") != list(FUSION_MODELS):
            raise ValueError("New fusion members must be prespecified Ridge/GAM/XGBoost")
        settings = {k: v for k, v in config.items() if k not in VARY_KEYS}
        inputs, code = _hashes(m["data"]["manifest"]), _hashes(m["code"])
        base = {k: v for k, v in inputs.items() if k not in D1_PATHS}
        fit = {k: v for k, v in code.items() if k in FIT_CODE}
        if set(fit) != FIT_CODE:
            raise ValueError(f"{variant}: missing critical model code hashes")
        current = (settings, base, fit, m.get("packages", {}))
        if reference is not None and current != reference:
            raise ValueError(f"{variant}: comparison changes settings, source data or model code")
        reference = current
        required = ({"docs/d1_program_sources.json"} if variant in ("fireworks", "both") else set())
        required |= ({"docs/d1_weather_sources.json"} if variant in ("weather", "both") else set())
        if not required.issubset(inputs):
            raise ValueError(f"{variant}: D1 source hashes missing")
        for path in D1_PATHS.intersection(inputs):
            if path in sources and sources[path] != inputs[path]:
                raise ValueError(f"{variant}: shared D1 source hashes differ")
            sources[path] = inputs[path]
    return {"base_input_hashes": reference[1], "d1_source_hashes": sources, "model_code_hashes": reference[2],
            "same_settings_except_feature_groups": True, "all_variants_exclude_historical_baseline": True}


def validate_old_reproduction(current, old, current_manifest, old_manifest):
    """Only the new GAM specification may differ from the archived v5 setup."""
    current, old = validate_panel(current), validate_old_panel(old)
    _same_panel(current, old)
    new_config, old_config = current_manifest["config"], old_manifest["config"]
    old_keys = set(old_config) - VERSION_KEYS
    changed = [key for key in old_keys if key not in new_config or new_config[key] != old_config[key]]
    if changed:
        raise ValueError(f"Historical v5 settings changed: {sorted(changed)}")
    # New-only controls must be explicitly about the GAM addition, not hidden
    # alterations of a preserved model or evaluation rule.
    extra = set(new_config) - set(old_config)
    if any(not key.startswith("gam_") for key in extra):
        raise ValueError(f"Unexpected new settings alter historical comparison: {sorted(extra)}")
    if _hashes(current_manifest["data"]["manifest"]) != _hashes(old_manifest["data"]["manifest"]):
        raise ValueError("Historical v5 source data changed")
    new_code, old_code = _hashes(current_manifest["code"]), _hashes(old_manifest["code"])
    if any(path not in new_code or path not in old_code or new_code[path] != old_code[path] for path in PRESERVED_CODE):
        raise ValueError("Historical v5 fitting/preprocessing code changed")
    old_packages = old_manifest.get("packages", {})
    if any(current_manifest.get("packages", {}).get(name) != version for name, version in old_packages.items()):
        raise ValueError("Historical v5 fitting package versions changed")
    differences = {}
    for model in OLD_MODELS:
        differences[model] = float(np.max(abs(current[model].to_numpy() - old[model].to_numpy())))
        if not np.allclose(current[model], old[model], rtol=1e-12, atol=1e-8):
            raise ValueError(f"Historical v5 {model} predictions failed reproduction")
    return {"same_panel_truth_and_cutoff": True, "same_old_settings_and_sources": True,
            "prediction_atol": 1e-8, "prediction_rtol": 1e-12, "max_abs_difference": differences}


def fusion_table(folds_by_variant):
    # Validate the historical ranking/design constraints unchanged.
    old = old_selection_table(folds_by_variant)
    rows = []
    for variant, folds in folds_by_variant.items():
        for fold in folds:
            weights = fold["fusion_weights"]
            for fusion in ("gam_equal", "gam_stack"):
                current = weights[fusion]
                if set(current) != set(FUSION_MODELS) or not np.isfinite(list(current.values())).all():
                    raise ValueError("GAM fusion needs exactly Ridge/GAM/XGBoost finite weights")
                if min(current.values()) < -1e-9 or not np.isclose(sum(current.values()), 1.):
                    raise ValueError("GAM fusion weights must be nonnegative and sum to one")
                if fusion == "gam_equal" and not np.allclose(list(current.values()), 1 / 3):
                    raise ValueError("GAM equal weights must each be one third")
                for model, weight in current.items():
                    rows.append({"variant": variant, "variant_label": VARIANTS[variant], "outer_year": fold["outer_year"],
                                 "train_end": fold["train_end"], "fusion": fusion, "base_model": model, "weight": float(weight),
                                 "scope": "past_temporal_oof_training_diagnostic_not_outer_score"})
    return pd.concat([old, pd.DataFrame(rows)], ignore_index=True)


def validate_fusion_predictions(frame, folds):
    if sorted(f["outer_year"] for f in folds) != sorted(frame.year.unique()):
        raise ValueError("Fold years mismatch")
    required_fusions = {"equal", "stack", "select_best", "select_stack", "gam_equal", "gam_stack"}
    for fold in folds:
        part = frame.loc[frame.outer_year.eq(fold["outer_year"])]
        if pd.Timestamp(fold["train_end"]) != part.train_end.iloc[0] or pd.Timestamp(fold["test_start"]) != part.event_date.min():
            raise ValueError("Fold/prediction chronology mismatch")
        if set(fold["fusion_weights"]) != required_fusions:
            raise ValueError("Missing or unexpected fusion method")
        for fusion, weights in fold["fusion_weights"].items():
            models = FUSION_MODELS if fusion.startswith("gam_") else OLD_BASE_MODELS
            if set(weights) != set(models):
                raise ValueError("Fusion model membership mismatch")
            calculated = sum(part[m].to_numpy() * weights[m] for m in models)
            if not np.allclose(calculated, part[fusion].to_numpy(), rtol=1e-9, atol=1e-7):
                raise ValueError("Fusion predictions do not match recorded weights")


def read_variants(root):
    root, frames, manifests, folds, hashes, reproductions = Path(root), {}, {}, {}, {}, {}
    reference = None
    for variant in VARIANTS:
        source = root / ".local/analysis_v6" / variant / "outer_predictions.csv"
        frame = validate_panel(pd.read_csv(source))
        if reference is not None:
            _same_panel(reference, frame)
        else:
            reference = frame
        public = root / "result/analysis_v6" / variant
        manifest = json.loads((public / "run_manifest.json").read_text(encoding="utf-8"))
        outer = manifest["outer_test"]
        if outer["rows"] != len(frame) or outer["events"] != frame.event_date.nunique() or outer["years"] != sorted(frame.year.unique()):
            raise ValueError(f"{variant}: manifest/prediction sample mismatch")
        variant_folds = json.loads((public / "folds.json").read_text(encoding="utf-8"))
        validate_fusion_predictions(frame, variant_folds)
        old_path = root / ".local/analysis_v5" / variant / "outer_predictions.csv"
        if not old_path.exists():
            raise ValueError(f"{variant}: missing private v5 predictions; reproduce v5 before publishing v6 comparisons")
        old_manifest = json.loads((root / "result/analysis_v5" / variant / "run_manifest.json").read_text(encoding="utf-8"))
        reproductions[variant] = validate_old_reproduction(frame, pd.read_csv(old_path), manifest, old_manifest)
        reproductions[variant]["v5_prediction_sha256"] = hashlib.sha256(old_path.read_bytes()).hexdigest()
        frames[variant], manifests[variant], folds[variant] = frame, manifest, variant_folds
        hashes[variant] = hashlib.sha256(source.read_bytes()).hexdigest()
    checks = validate_manifests(manifests)
    checks.update(prediction_hashes=hashes, v5_reproduction=reproductions)
    return frames, manifests, folds, checks


def comparison_figure(summary, output):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 12))
    colors, ys = ("#A4A4A4", "#678895", "#A89D72", "#424242"), np.arange(len(DISPLAY_MODELS))
    for ax, scope, title in zip(axes, ("all", "beimen"), ("全部四站", "北門")):
        for i, ((variant, label), color) in enumerate(zip(VARIANTS.items(), colors)):
            part = summary.loc[summary.variant.eq(variant) & summary.scope.eq(scope)].set_index("model").loc[list(DISPLAY_MODELS)]
            bars = ax.barh(ys + (i - 1.5) * .19, part.mae, height=.18, label=label, color=color)
            ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
        ax.set_yticks(ys, [LABELS[m] for m in DISPLAY_MODELS])
        ax.invert_yaxis()
        ax.set(title=title, xlabel="活動等權 MAE（人次／站／小時；越低越好）")
        ax.set_xlim(0, summary.loc[summary.scope.eq(scope), "mae"].max() * 1.15)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("第六版：加入 GAM，固定 Ridge＋GAM＋XGBoost 等權／學權重融合\n"
                 "與原八種方法比較；新增變數僅接受 D−1 18:00 前資訊", fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .04, 1, .95))
    save(fig, output / "figures/model_comparison.png")


def mean_figure(means, output):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    for ax, scope, title in zip(axes, ("all", "beimen"), ("四站合計", "北門")):
        part = means.loc[means.variant.eq("both") & means.scope.eq(scope)].set_index("model").loc[list(MEAN_MODELS)]
        bars = ax.bar(range(len(part)), part.mean_event_origin_trips, color=[COLORS[m] for m in MEAN_MODELS], width=.7)
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in part.mean_event_origin_trips], padding=5, fontsize=10)
        ax.set_xticks(range(len(part)), [LABELS[m] for m in MEAN_MODELS], rotation=25, ha="right")
        ax.set(title=title, ylabel="每場活動平均起站人次（18–23 時）", ylim=(0, part.mean_event_origin_trips.max() * 1.2))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"實際與模型預測平均人次｜固定煙火＋天氣組、相同 {int(part.events.iloc[0])} 場外層活動\n"
                 "每場先加總各站六小時，再跨場平均；不是獨立參加人數", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .9))
    save(fig, output / "figures/mean_event_counts.png")


def profiles_figure(frame, output):
    style()
    dates = sorted(frame.event_date.unique())
    sets = ((FUSION_MODELS, "beimen_profiles.png", "Ridge、GAM、XGBoost"),
            (("gam_equal", "gam_stack", "stack"), "beimen_fusion_profiles.png", "新三模型融合與舊四模型融合"))
    for models, filename, subtitle in sets:
        fig, axes = plt.subplots(int(np.ceil(len(dates) / 3)), 3, figsize=(15, 3.2 * np.ceil(len(dates) / 3)), squeeze=False)
        for ax, date in zip(axes.flat, dates):
            part = frame.loc[frame.event_date.eq(date) & frame.station.eq("北門")].sort_values("hour")
            ax.plot(part.hour, part.y, "o-", color=COLORS["y"], linewidth=2, label=LABELS["y"])
            for i, model in enumerate(models):
                ax.plot(part.hour, part[model], ("--", "-.", ":")[i], color=COLORS[model], linewidth=1.8, label=LABELS[model])
            ax.set(title=f"{pd.Timestamp(date):%Y-%m-%d}", xticks=HOURS, xlabel="來源小時", ylabel="起站人次", ylim=(0, None))
            ax.grid(axis="y", alpha=.2)
        for ax in list(axes.flat)[len(dates):]:
            ax.axis("off")
        overlap = ("；新舊學權重曲線重合" if "gam_stack" in models and
                   np.allclose(frame.gam_stack, frame["stack"], rtol=1e-9, atol=1e-7) else "")
        fig.suptitle(f"北門全部外層活動｜固定煙火＋天氣組｜{subtitle}\n平均誤差改善不代表每場尖峰都改善{overlap}", fontsize=13)
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=len(models) + 1, frameon=False)
        fig.tight_layout(rect=(0, .06, 1, .92))
        save(fig, output / "figures" / filename)


def fusion_figure(weights, output):
    style()
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    for ax, (variant, label) in zip(axes, VARIANTS.items()):
        part = weights.loc[weights.variant.eq(variant) & weights.fusion.eq("gam_stack")]
        matrix = part.pivot(index="outer_year", columns="base_model", values="weight").reindex(columns=FUSION_MODELS)
        bottom = np.zeros(len(matrix))
        for model in FUSION_MODELS:
            ax.bar(range(len(matrix)), matrix[model], bottom=bottom, color=COLORS[model], label=LABELS[model])
            for i, (value, base) in enumerate(zip(matrix[model], bottom)):
                if value >= .08:
                    ax.text(i, base + value / 2, f"{value:.2f}", ha="center", va="center", fontsize=10, color="white")
            bottom += matrix[model].to_numpy()
        ax.set(xticks=range(len(matrix)), xticklabels=matrix.index, ylim=(0, 1.03), title=label)
    axes[0].set_ylabel("非負、總和為 1 的權重")
    fig.suptitle("固定 Ridge＋GAM＋XGBoost，僅用過去 OOF 預測學融合權重\n權重可為零；它不是變數的重要性或因果貢獻", fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .09, 1, .86))
    save(fig, output / "figures/fusion_weights.png")


def effect_figure(curves, output, year=2026):
    required = {"outer_year", "feature", "value", "effect", "training_min", "training_max", "training_events"}
    if required - set(curves):
        raise ValueError("Missing GAM effect curve provenance")
    part = curves.loc[curves.outer_year.eq(year)].copy()
    if part.empty or not np.isfinite(part[["value", "effect", "training_min", "training_max"]].to_numpy(float)).all():
        raise ValueError("Missing/nonfinite prespecified GAM effect curves")
    if (part.value < part.training_min - 1e-9).any() or (part.value > part.training_max + 1e-9).any():
        raise ValueError("GAM effect display extrapolates outside training support")
    features = list(part.feature.unique())
    style()
    columns = 2 if len(features) <= 4 else 3
    rows = int(np.ceil(len(features) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.4 * columns, 3.6 * rows), squeeze=False)
    for ax, feature in zip(axes.flat, features):
        block = part.loc[part.feature.eq(feature)].sort_values("value")
        ax.plot(block.value, block.effect, color=COLORS["gam"], linewidth=2)
        ax.axhline(0, color="#999999", linestyle="--", linewidth=.8)
        ax.set(xlabel=EFFECT_LABELS.get(feature, feature), ylabel="中心化項效果（人次）")
        ax.set_title(f"{int(block.training_events.iloc[0])} 場已知訓練資料｜範圍 {block.training_min.iloc[0]:.1f}–{block.training_max.iloc[0]:.1f}", fontsize=10)
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(features):]:
        ax.axis("off")
    fig.suptitle(f"GAM 已平滑連續變數的預測項｜固定煙火＋天氣組 {year} 外層模型\n"
                 f"{year - 1} 年以前訓練；非負截斷前的中心化加法分量，無信賴區間、非因果", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .9 if rows == 1 else .94))
    save(fig, output / "figures/gam_effect_curves.png")


def write_report(summary, annual, means, weights, output):
    parts = ["# 第六版：GAM 與 Ridge／GAM／XGBoost 融合", "",
             "固定比較四組特徵；所有新增節目與天氣變數，只接受活動前一天 18:00 前已發布的資訊。"
             "原第五版八種方法的設定與來源保持一致，並逐值驗證預測重現。GAM 與兩種新融合均使用相同外層活動評分。", "",
             "## 選模型的理由", "",
             "- **Ridge**：提供收縮後的線性與既定車站 × 時段交互預測，保留相關訊號。",
             "- **GAM**：用受懲罰的低複雜度平滑函數描述連續變數的彎曲關係，仍能逐項畫圖；類別與車站 × 時段結構另行保留。",
             "- **XGBoost**：透過條件切分描述門檻與多變數交互規律。",
             "- 三者的函數結構不同，是事前選擇融合成員的理由；是否互補仍須由外層誤差檢查，不能因模型名稱不同便假定有效。", "",
             "## 實際結果", ""]
    for scope, label in (("all", "全部四站"), ("beimen", "北門")):
        table = summary.loc[summary.scope.eq(scope)].pivot(index="model", columns="variant", values="mae").reindex(index=DISPLAY_MODELS, columns=VARIANTS)
        table.index, table.columns = [LABELS[m] for m in table.index], list(VARIANTS.values())
        parts += [f"### {label} MAE（人次／站／小時；越低越好）", "", markdown_table(table.round(2).reset_index(names="模型")), ""]
    both = summary.loc[summary.variant.eq("both") & summary.scope.eq("beimen")].set_index("model")
    old, equal, learned = (float(both.loc[m, "mae"]) for m in ("stack", "gam_equal", "gam_stack"))
    if abs(learned - old) < 1e-8:
        change = "新學權重與舊融合的 MAE 相同。"
    else:
        direction = "降低" if learned < old else "增加"
        change = f"新學權重相對舊融合的 MAE {direction} {abs(learned - old):,.2f}。"
    gam_weights = weights.loc[weights.variant.eq("both") & weights.fusion.eq("gam_stack") & weights.base_model.eq("gam"), "weight"]
    zero_note = ("此組三個外層年度的 GAM 權重均為零，因此新融合只剩 Ridge 與 XGBoost；"
                 "舊融合也只有這兩者有非零權重，兩組融合曲線重合。" if np.allclose(gam_weights, 0.) else "")
    parts += [f"固定煙火＋天氣組：北門舊四模型學權重 MAE {old:,.2f}；新三模型等權 {equal:,.2f}、學權重 {learned:,.2f}。"
              f"{change}{zero_note}融合可能優於某些單一模型，也可能輸給另一個單一模型。", "",
              "![共同外層比較](figures/model_comparison.png)", "", "## 訓練與融合流程", "",
              "1. 先固定 Ridge、GAM、XGBoost；本版不依外層成績決定成員。",
              "2. 每個外層年度只用較早活動；各內層以向前切分選 Ridge 懲罰、GAM 平滑強度與 XGBoost 設定。",
              "3. 對過去活動產生 OOF 預測。新等權融合各占 1/3；新學權重融合以過去 OOF 的活動等權 MAE，估計非負且總和為 1 的三個權重。",
              "4. 到外層年度才計分。OOF 用來訓練權重，OOF 誤差不是最終測試成績；權重也不是因果貢獻。",
              "5. 舊 OLS、Ridge、Lasso、XGBoost、四模型等權、四模型學權重、OOF 選一及前二融合都保留作同樣本對照。", "",
              "![融合權重](figures/fusion_weights.png)", "", "## 平均人次及北門尖峰", "",
              "固定展示煙火＋天氣組，沒有依外層最小誤差挑圖。先加總每場 18–23 時的起站人次，再跨活動平均；"
              "它不是獨立參加人數。平均總量接近也不代表尖峰時段正確。", "",
              "![平均人次](figures/mean_event_counts.png)", "", "![三模型北門逐場](figures/beimen_profiles.png)", "",
              "![融合北門逐場](figures/beimen_fusion_profiles.png)", "", "## GAM 的可解釋性", "",
              "下圖固定使用 2026 外層訓練模型（訓練資料止於 2025 年），只畫訓練觀測範圍。"
              "每個函數以訓練平均項效果中心化，表示未做非負截斷前的加法分量；不是完整人流預測、不是因果效果，也沒有推論用信賴區間。", "",
              "![GAM 預測項](figures/gam_effect_curves.png)", "", "## 限制與文獻", "",
              "- 只有 21 場活動，外層共同評分 9 場；504 個站時列不能當成 504 個獨立活動。小樣本下平滑曲線與融合權重都可能不穩定。",
              "- 新欄位有 D−1 來源閘門，但早年訓練標籤／日曆版本時間及回顧挑選已舉行活動的限制仍在；不能宣稱完整 D−1 即時部署驗證。",
              "- 這批外層年度已反覆檢查，屬探索性回顧比較。更換模型後的改善仍需未見活動確認。",
              "- [Hastie 與 Tibshirani（1986）：Generalized Additive Models](https://doi.org/10.1214/ss/1177013604)：GAM 的加法平滑結構。",
              "- [Eilers 與 Marx（1996）：Flexible Smoothing with B-splines and Penalties](https://doi.org/10.1214/ss/1038425655)：基底與差分懲罰的平滑原理。",
              "- [Hoerl 與 Kennard（1970）：Ridge Regression](https://doi.org/10.1080/00401706.1970.10488634)：係數收縮。",
              "- [Chen 與 Guestrin（2016）：XGBoost](https://arxiv.org/abs/1603.02754)：梯度提升樹。",
              "- [Breiman（1996）：Stacked Regressions](https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf)：以留出預測學融合；本研究的 MAE、總和為 1 與時間切分是指定實作，不是原文最佳性保證。", "",
              "完整數值：`tables/metrics_by_variant.csv`、`tables/yearly.csv`、`tables/mean_event_counts.csv`、`tables/fusion_weights.csv`。"
              "`comparison_audit.json` 留下來源雜湊、共同切分與第五版重現檢查。", ""]
    (output / "report.md").write_text("\n".join(parts), encoding="utf-8")


def build_comparison(root=ROOT, output=None):
    root = Path(root)
    output = Path(output) if output is not None else root / "result/analysis_v6"
    frames, manifests, folds, checks = read_variants(root)
    # Validate all inputs before writing a report that could be mistaken for a
    # successful comparison.
    weights = fusion_table(folds)
    summary, annual = [], []
    for variant, frame in frames.items():
        summary.append(metrics_table(frame).assign(variant=variant, variant_label=VARIANTS[variant]))
        for year, block in frame.groupby("year"):
            annual.append(metrics_table(block).assign(variant=variant, variant_label=VARIANTS[variant], year=int(year)))
    summary, annual = pd.concat(summary, ignore_index=True), pd.concat(annual, ignore_index=True)
    means = mean_event_counts(frames)
    curves = pd.read_csv(root / "result/analysis_v6/both/tables/gam_effect_curves.csv")
    for directory in ("tables", "figures"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    for name, frame in (("metrics_by_variant", summary), ("yearly", annual), ("mean_event_counts", means), ("fusion_weights", weights)):
        frame.to_csv(output / "tables" / f"{name}.csv", index=False, encoding="utf-8-sig")
    checks["comparison_scope"] = "paired_exploratory_temporal_outer_holdouts_not_model_selection"
    checks["reporting_code"] = {"path": "src/analysis_v6/reporting.py", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    comparison_figure(summary, output)
    mean_figure(means, output)
    profiles_figure(frames["both"], output)
    fusion_figure(weights, output)
    effect_figure(curves, output)
    write_report(summary, annual, means, weights, output)
    (output / "comparison_audit.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(output), "report": str(output / "report.md"), "events": int(summary.events.max()),
            "variants": list(VARIANTS), "models": list(ALL_MODELS)}


if __name__ == "__main__":
    print(json.dumps(build_comparison(), ensure_ascii=False, indent=2))
