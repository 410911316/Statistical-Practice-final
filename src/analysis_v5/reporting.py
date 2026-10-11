"""Audited comparisons of common-design models on paired temporal holdouts.

This module never fits or chooses a model. The ``both`` mean/profile display is
prespecified and is not selected by the repeatedly inspected outer test scores.
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

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {"core": "基本欄位", "fireworks": "只加煙火時間", "weather": "只加天氣預報", "both": "煙火＋天氣"}
BASE_MODELS = ("ols", "ridge", "lasso", "xgboost")
ALL_MODELS = (*BASE_MODELS, "equal", "stack", "select_best", "select_stack")
LABELS = {"y": "實際", "ols": "完整設計 OLS", "ridge": "Ridge", "lasso": "Lasso", "xgboost": "XGBoost",
          "equal": "四模型等權", "stack": "四模型 Stacking", "select_best": "過去 OOF 選一", "select_stack": "過去 OOF 前二融合"}
COLORS = {"y": "#161616", "ols": "#84939B", "ridge": "#537A95", "lasso": "#7D8C5D", "xgboost": "#B38D4D",
          "equal": "#A6A6A6", "stack": "#8C6C93", "select_best": "#6C7676", "select_stack": "#37786C"}
KEYS = ["event_date", "station", "hour"]
STATIONS = ("北門", "大橋頭站", "民權西路", "雙連")
HOURS = tuple(range(18, 24))
CALENDAR = ["is_weekend", "day_of_year", "year_index"]
FIREWORKS = ["planned_fireworks_minutes", "planned_fireworks_start_hour"]
WEATHER = ["forecast_temperature_c", "forecast_wind_speed_m_s", "forecast_cloud_cover_pct"]
FEATURES = {"core": [], "fireworks": FIREWORKS, "weather": WEATHER, "both": FIREWORKS + WEATHER}
MEAN_MODELS = ("y", *BASE_MODELS, "stack", "select_stack")
VARY_KEYS = {"version", "report_title", "output_dir", "private_output_dir", "experiment_notes",
             "candidate_groups", "d1_features", "d1_source_files"}
D1_PATHS = {"docs/d1_program_sources.json", "docs/d1_weather_sources.json"}
FIT_CODE = {"src/run_analysis_v5.py", "src/analysis_v5/models.py", "src/analysis_v1/models.py",
            "src/analysis_v1/data.py", "src/analysis_v1/d1_features.py"}


def validate_panel(source: pd.DataFrame) -> pd.DataFrame:
    """Require complete common activity panels and genuine chronological cutoffs."""
    required = set(KEYS + ["y", "year", "outer_year", "train_end", *ALL_MODELS])
    if required - set(source):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(source))}")
    frame = source.copy()
    frame["event_date"] = pd.to_datetime(frame.event_date, errors="raise")
    frame["train_end"] = pd.to_datetime(frame.train_end, errors="raise")
    if frame.empty or frame[KEYS + ["train_end"]].isna().any().any():
        raise ValueError("Empty data or missing holdout keys/cutoffs")
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate event/station/hour keys")
    if not np.isfinite(frame[["hour", "y", *ALL_MODELS]].to_numpy(float)).all():
        raise ValueError("Nonfinite truth or predictions")
    if (frame[["y", *ALL_MODELS]].to_numpy(float) < 0).any():
        raise ValueError("Person counts must be nonnegative")
    if not (frame.train_end < frame.event_date).all():
        raise ValueError("Training must end before each heldout event")
    for column in ("year", "outer_year"):
        if not frame[column].eq(frame.event_date.dt.year).all():
            raise ValueError(f"{column} disagrees with event date")
    expected = set(itertools.product(STATIONS, HOURS))
    for date, part in frame.groupby("event_date"):
        if set(zip(part.station, part.hour)) != expected or len(part) != len(expected):
            raise ValueError(f"{date:%Y-%m-%d}: incomplete station/hour panel")
    if (frame.groupby("outer_year").train_end.nunique() != 1).any():
        raise ValueError("One outer year must use one common training cutoff")
    return frame.sort_values(KEYS).reset_index(drop=True)


def align_panels(left, right):
    left, right = validate_panel(left), validate_panel(right)
    if not left[KEYS].equals(right[KEYS]):
        raise ValueError("Variants have different holdout panels")
    if not np.array_equal(left.y.to_numpy(float), right.y.to_numpy(float)):
        raise ValueError("Variants have different truths")
    if not left.train_end.equals(right.train_end):
        raise ValueError("Variants have different training cutoffs")
    return left, right


def metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Activity-balanced hourly loss; peak errors use each observed peak hour."""
    frame = validate_panel(predictions)
    records = []
    for scope, block in (("all", frame), ("beimen", frame.loc[frame.station.eq("北門")])):
        for model in ALL_MODELS:
            error = block[model] - block.y
            event = pd.DataFrame({"event_date": block.event_date, "ae": abs(error), "se": error**2, "bias": error})
            event = event.groupby("event_date")[["ae", "se", "bias"]].mean()
            peaks = []
            for (date, _), part in block.groupby(["event_date", "station"]):
                part = part.sort_values("hour")
                actual, pred = part.loc[part.y.idxmax()], part.loc[part[model].idxmax()]
                signed = float(actual.y - actual[model])
                peaks.append({"event_date": date, "signed": signed, "absolute": abs(signed),
                              "positive": max(0., signed), "hour_error": abs(float(actual.hour - pred.hour))})
            peak = pd.DataFrame(peaks).groupby("event_date").mean().mean()
            records.append({"scope": scope, "model": model, "events": len(event), "rows": len(block),
                            "mae": float(event.ae.mean()), "rmse": float(np.sqrt(event.se.mean())),
                            "bias_pred_minus_actual": float(event.bias.mean()),
                            "peak_absolute_error_mean": float(peak.absolute),
                            "peak_underprediction_mean": float(peak.positive),
                            "peak_signed_underprediction_mean": float(peak.signed),
                            "peak_hour_mae": float(peak.hour_error)})
    return pd.DataFrame(records)


def mean_event_counts(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average activity totals, not fitted training values or unique attendance."""
    records, reference = [], None
    for variant, label in VARIANTS.items():
        frame = validate_panel(frames[variant])
        if reference is not None:
            align_panels(reference, frame)
        else:
            reference = frame
        for scope, part in (("all", frame), ("beimen", frame.loc[frame.station.eq("北門")])):
            totals = part.groupby("event_date")[["y", *ALL_MODELS]].sum()
            actual = float(totals.y.mean())
            for model in ("y", *ALL_MODELS):
                mean = float(totals[model].mean())
                records.append({"variant": variant, "variant_label": label, "scope": scope, "model": model,
                                "events": len(totals), "stations": part.station.nunique(), "hours_per_station": len(HOURS),
                                "mean_event_origin_trips": mean, "actual_mean_event_origin_trips": actual,
                                "mean_bias_trips": mean - actual,
                                "mean_bias_pct": 100 * (mean / actual - 1) if actual else np.nan,
                                "unit": "origin_person_trips_per_event_18_to_23_not_unique_attendance",
                                "aggregation": "mean_over_events_of_sum_over_stations_and_hours"})
    return pd.DataFrame(records)


def _hashes(items):
    result = {item["path"].replace("\\", "/"): item["sha256"] for item in items}
    if len(result) != len(items):
        raise ValueError("Duplicate manifest paths")
    return result


def validate_manifests(manifests):
    """Ablations may change features, not fitting code, data, grids or cutoffs."""
    reference_design = reference_base = reference_code = None
    shared_sources = {}
    for variant in VARIANTS:
        manifest, expected = manifests[variant], FEATURES[variant]
        config = manifest["config"]
        if config.get("include_historical_baseline") is not False or config.get("d1_cutoff_hour") != 18:
            raise ValueError("All variants must exclude unverified baseline and use D-1 18:00")
        if config.get("station_hour_interaction") is not True or config.get("linear_design") != "full_shared_with_training_rank_filter":
            raise ValueError("All variants require the shared full design and station/hour interactions")
        if config.get("d1_features", []) != expected or config.get("candidate_groups") != CALENDAR + expected:
            raise ValueError(f"{variant}: incorrect prespecified feature group")
        if config.get("fusion_top_k") != 2:
            raise ValueError("Prespecified selection must retain two models")
        design = {k: v for k, v in config.items() if k not in VARY_KEYS}
        inputs, code = _hashes(manifest["data"]["manifest"]), _hashes(manifest["code"])
        base = {k: v for k, v in inputs.items() if k not in D1_PATHS}
        fit = {k: v for k, v in code.items() if k in FIT_CODE}
        if set(fit) != FIT_CODE:
            raise ValueError(f"{variant}: missing critical model code hashes")
        if reference_design is not None:
            if design != reference_design or base != reference_base or fit != reference_code:
                raise ValueError(f"{variant}: comparison changes settings, source data or model code")
        else:
            reference_design, reference_base, reference_code = design, base, fit
        required = ({"docs/d1_program_sources.json"} if variant in ("fireworks", "both") else set())
        required |= ({"docs/d1_weather_sources.json"} if variant in ("weather", "both") else set())
        if not required.issubset(inputs):
            raise ValueError(f"{variant}: D1 source hashes missing")
        for path in D1_PATHS.intersection(inputs):
            if path in shared_sources and shared_sources[path] != inputs[path]:
                raise ValueError(f"{variant}: shared D1 source hashes differ")
            shared_sources[path] = inputs[path]
    return {"base_input_hashes": reference_base, "d1_source_hashes": shared_sources,
            "model_code_hashes": reference_code, "same_settings_except_feature_groups": True,
            "all_variants_exclude_historical_baseline": True}


def selection_table(folds_by_variant):
    """Inspect recorded past-OOF choices; never reselect using outer outcomes."""
    records = []
    for variant, folds in folds_by_variant.items():
        for fold in folds:
            if pd.Timestamp(fold["train_end"]) >= pd.Timestamp(fold["test_start"]):
                raise ValueError("Fusion training cutoff must precede outer test")
            audit, weights = fold["fusion_audit"], fold["fusion_weights"]
            scores = audit["oof_event_mae"]
            if set(scores) != set(BASE_MODELS) or not np.isfinite(list(scores.values())).all():
                raise ValueError("Missing or nonfinite OOF scores")
            ranked = audit["ranked_models"]
            if set(ranked) != set(BASE_MODELS) or len(ranked) != len(BASE_MODELS):
                raise ValueError("OOF ranking must contain each base model once")
            if any(scores[a] > scores[b] + 1e-8 for a, b in zip(ranked, ranked[1:])):
                raise ValueError("OOF ranking disagrees with past OOF MAE")
            if audit["selected_models"] != ranked[:2] or audit["selected_best_model"] != ranked[0]:
                raise ValueError("Selected models disagree with prespecified OOF ranking")
            infos = fold["base_models"]
            shared_columns = infos[BASE_MODELS[0]]["columns"]
            shared_groups = infos[BASE_MODELS[0]]["groups"]
            if any(infos[m]["columns"] != shared_columns or infos[m]["groups"] != shared_groups for m in BASE_MODELS):
                raise ValueError("Base models do not share the same encoded design")
            for fusion in ("equal", "stack", "select_best", "select_stack"):
                current = weights[fusion]
                if set(current) != set(BASE_MODELS) or not np.isfinite(list(current.values())).all():
                    raise ValueError("Fusion weights missing or nonfinite")
                if min(current.values()) < -1e-9 or not np.isclose(sum(current.values()), 1.):
                    raise ValueError("Fusion weights must be nonnegative and sum to one")
                for model in BASE_MODELS:
                    weight = current[model]
                    if fusion == "select_stack" and model not in ranked[:2] and abs(weight) > 1e-9:
                        raise ValueError("Unselected model has nonzero selected-fusion weight")
                    if fusion == "select_best" and not np.isclose(weight, float(model == ranked[0])):
                        raise ValueError("Best-single weights disagree with OOF choice")
                    if fusion == "equal" and not np.isclose(weight, .25):
                        raise ValueError("Equal-weight comparator must retain all four equally")
                    records.append({"variant": variant, "variant_label": VARIANTS[variant],
                                    "outer_year": fold["outer_year"], "train_end": fold["train_end"],
                                    "fusion": fusion, "base_model": model, "weight": float(weight),
                                    "oof_event_mae": scores[model], "oof_rank": ranked.index(model) + 1,
                                    "selected_top2": model in ranked[:2],
                                    "selected_best": model == ranked[0],
                                    "scope": "past_temporal_oof_training_diagnostic_not_outer_score"})
    return pd.DataFrame(records)


def read_variants(root):
    root = Path(root)
    frames, manifests, folds, hashes = {}, {}, {}, {}
    reference = None
    for variant in VARIANTS:
        source = root / ".local/analysis_v5" / variant / "outer_predictions.csv"
        frame = validate_panel(pd.read_csv(source))
        if reference is not None:
            align_panels(reference, frame)
        else:
            reference = frame
        public = root / "result/analysis_v5" / variant
        manifest = json.loads((public / "run_manifest.json").read_text(encoding="utf-8"))
        outer = manifest["outer_test"]
        if outer["rows"] != len(frame) or outer["events"] != frame.event_date.nunique() or outer["years"] != sorted(frame.year.unique()):
            raise ValueError(f"{variant}: manifest and predictions have different samples")
        variant_folds = json.loads((public / "folds.json").read_text(encoding="utf-8"))
        if sorted(f["outer_year"] for f in variant_folds) != sorted(frame.year.unique()):
            raise ValueError(f"{variant}: fold years mismatch")
        for fold in variant_folds:
            part = frame.loc[frame.outer_year.eq(fold["outer_year"])]
            if pd.Timestamp(fold["train_end"]) != part.train_end.iloc[0] or pd.Timestamp(fold["test_start"]) != part.event_date.min():
                raise ValueError(f"{variant}: fold/prediction chronology mismatch")
            for fusion, weight in fold["fusion_weights"].items():
                values = sum(part[m].to_numpy() * weight[m] for m in BASE_MODELS)
                if not np.allclose(values, part[fusion].to_numpy(), rtol=1e-9, atol=1e-7):
                    raise ValueError(f"{variant}: fusion predictions do not match recorded weights")
        frames[variant], manifests[variant], folds[variant] = frame, manifest, variant_folds
        hashes[variant] = hashlib.sha256(source.read_bytes()).hexdigest()
    checks = validate_manifests(manifests)
    checks["prediction_hashes"] = hashes
    return frames, manifests, folds, checks


def comparison_figure(summary, output):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.8))
    colors, ys = ("#A4A4A4", "#678895", "#A89D72", "#424242"), np.arange(len(ALL_MODELS))
    for ax, scope, title in zip(axes, ("all", "beimen"), ("全部四站", "北門")):
        for i, ((variant, label), color) in enumerate(zip(VARIANTS.items(), colors)):
            part = summary.loc[summary.variant.eq(variant) & summary.scope.eq(scope)].set_index("model").loc[list(ALL_MODELS)]
            bars = ax.barh(ys + (i - 1.5) * .19, part.mae, height=.18, label=label, color=color)
            ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
        ax.set_yticks(ys, [LABELS[m] for m in ALL_MODELS])
        ax.invert_yaxis()
        ax.set(title=title, xlabel="活動等權 MAE（人次／站／小時；越低越好）")
        ax.set_xlim(0, summary.loc[summary.scope.eq(scope), "mae"].max() * 1.16)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("同一設計矩陣：一般回歸、Ridge、Lasso、XGBoost 與四種融合／選擇規則\n"
                 "新增變數只使用 D−1 18:00 前資訊；共同外層活動之探索性比較", fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .06, 1, .93))
    save(fig, output / "figures/model_comparison.png")


def mean_figure(means, output):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax, scope, title in zip(axes, ("all", "beimen"), ("四站合計", "北門")):
        part = means.loc[means.variant.eq("both") & means.scope.eq(scope)].set_index("model").loc[list(MEAN_MODELS)]
        bars = ax.bar(range(len(part)), part.mean_event_origin_trips, color=[COLORS[m] for m in MEAN_MODELS], width=.7)
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in part.mean_event_origin_trips], padding=5, fontsize=10)
        ax.set_xticks(range(len(part)), [LABELS[m].replace(" ", "\n") for m in MEAN_MODELS], rotation=20, ha="right")
        ax.set(title=title, ylabel="每場活動平均起站人次（18–23 時）", ylim=(0, part.mean_event_origin_trips.max() * 1.2))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"實際與模型預測平均人次｜固定展示煙火＋天氣版、相同 {int(part.events.iloc[0])} 場外層活動\n"
                 "先加總每場各站六個小時，再跨場平均；不是獨立參加人數", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .9))
    save(fig, output / "figures/mean_event_counts.png")


def profiles_figure(frame, output):
    style()
    dates = sorted(frame.event_date.unique())
    for models, filename, subtitle in ((BASE_MODELS, "beimen_profiles.png", "四個基本模型"),
                                      (("stack", "select_stack", "select_best"), "beimen_fusion_profiles.png", "融合與選擇規則")):
        fig, axes = plt.subplots(int(np.ceil(len(dates) / 3)), 3, figsize=(15, 3.2 * np.ceil(len(dates) / 3)), squeeze=False)
        for ax, date in zip(axes.flat, dates):
            part = frame.loc[frame.event_date.eq(date) & frame.station.eq("北門")].sort_values("hour")
            ax.plot(part.hour, part.y, "o-", color=COLORS["y"], linewidth=2, label=LABELS["y"])
            for model in models:
                line_style = {"stack": "-", "select_stack": "--", "select_best": "-."}.get(model, "--")
                ax.plot(part.hour, part[model], line_style, color=COLORS[model], linewidth=1.6, label=LABELS[model])
            ax.set(title=f"{pd.Timestamp(date):%Y-%m-%d}", xticks=HOURS, xlabel="來源小時", ylabel="起站人次", ylim=(0, None))
            ax.grid(axis="y", alpha=.2)
        for ax in list(axes.flat)[len(dates):]:
            ax.axis("off")
        overlap = ("；兩種 Stacking 曲線重合" if "stack" in models and
                   np.allclose(frame["stack"], frame["select_stack"], rtol=1e-9, atol=1e-7) else "")
        fig.suptitle(f"北門全部外層活動｜煙火＋天氣版｜{subtitle}\n逐場檢查平均誤差是否掩蓋尖峰低估{overlap}", fontsize=13)
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=len(models) + 1, frameon=False)
        fig.tight_layout(rect=(0, .06, 1, .92))
        save(fig, output / "figures" / filename)


def fusion_figure(selections, output):
    style()
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.0), sharey=True)
    for column, (variant, label) in enumerate(VARIANTS.items()):
        for row, (fusion, title) in enumerate((("stack", "四模型 Stacking"), ("select_stack", "過去 OOF 前二融合"))):
            ax = axes[row, column]
            part = selections.loc[selections.variant.eq(variant) & selections.fusion.eq(fusion)]
            matrix = part.pivot(index="outer_year", columns="base_model", values="weight").reindex(columns=BASE_MODELS)
            bottom = np.zeros(len(matrix))
            for model in BASE_MODELS:
                ax.bar(range(len(matrix)), matrix[model], bottom=bottom, color=COLORS[model], label=LABELS[model])
                for i, (value, base) in enumerate(zip(matrix[model], bottom)):
                    if value >= .08:
                        ax.text(i, base + value / 2, f"{value:.2f}", ha="center", va="center", fontsize=9, color="white")
                bottom += matrix[model].to_numpy()
            ax.set(xticks=range(len(matrix)), xticklabels=matrix.index, ylim=(0, 1.04), title=f"{label}\n{title}")
            if column == 0:
                ax.set_ylabel("非負、總和為 1 的權重")
    fig.suptitle("模型入選與融合權重均只由較早年度 OOF 預測決定\n前二模型的權重仍可為零；入選不保證兩者都有貢獻", fontsize=13)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .06, 1, .9))
    save(fig, output / "figures/fusion_selection.png")


def write_report(summary, annual, means, selections, output):
    parts = ["# 第五版：共同設計矩陣、Lasso 與先選模型再融合", "",
             "本報告為已完成的共同外層回顧比較。所有新增特徵只接受活動前一天 18:00 前的公告／天氣預報；"
             "未核實當時發布時間的一般日基準已排除。早年訓練標籤的發布時點及已舉行樣本的回顧選取仍有限制，不能宣稱完整即時部署驗證。", "",
             "四個模型在每個訓練折使用相同候選欄位、車站 × 時段交互作用、訓練內缺值處理與完全相依欄位移除。"
             "一般回歸為完整 OLS，這版沒有 Forward Selection；係數方法比較因此不再混入不同設計矩陣。", "",
             "## 實際結果", ""]
    for scope, label in (("all", "全部四站 MAE"), ("beimen", "北門 MAE")):
        table = summary.loc[summary.scope.eq(scope)].pivot(index="model", columns="variant", values="mae").reindex(index=ALL_MODELS, columns=VARIANTS)
        table.index = [LABELS[m] for m in table.index]
        table.columns = [VARIANTS[v] for v in table.columns]
        parts += [f"### {label}（人次／站／小時；越低越好）", "", markdown_table(table.round(2).reset_index(names="模型")), ""]
    both = summary.loc[summary.variant.eq("both") & summary.scope.eq("beimen")].set_index("model")
    old, selected = float(both.loc["stack", "mae"]), float(both.loc["select_stack", "mae"])
    change = selected - old
    change_text = ("兩者 MAE 相同" if abs(change) < .005
                   else f"誤差{'降低' if change < 0 else '增加'} {abs(change):,.2f}")
    both_weights = selections.loc[selections.variant.eq("both") & selections.fusion.isin(("stack", "select_stack"))]
    paired_weights = both_weights.pivot(index=["outer_year", "base_model"], columns="fusion", values="weight")
    same_weights = np.allclose(paired_weights["stack"], paired_weights["select_stack"], rtol=1e-9, atol=1e-9)
    weight_note = (" 本版兩種規則在每個年度得到相同權重：全四模型融合已將未入選模型設成零，"
                   "先排除它們沒有改變預測。" if same_weights else "")
    parts += [f"固定煙火＋天氣版：北門四模型 Stacking MAE 為 {old:,.2f}，先選前二再融合為 {selected:,.2f}，"
              f"{change_text}。{weight_note}這是共同外層結果，不能預先假定選模型再融合必然改善。", "",
              "![全部模型比較](figures/model_comparison.png)", "", "## 選模型與融合的規則", "",
              "1. 每一外層年度只用較早活動；更內層向前驗證調 Ridge、Lasso、XGBoost。",
              "2. 中層產生過去活動的 OOF 預測，用活動等權 MAE 排名四個模型，固定保留前二。",
              "3. 在相同過去 OOF 上估計非負、總和為 1 的 MAE 融合權重；同時保留全四模型融合、等權及只選一個的對照。",
              "4. 到外層年度才計分。排名與權重擬合共用 OOF，因此 OOF 誤差只是訓練診斷；成效只能看外層。",
              "5. 本批外層資料已被多次檢查，結果屬探索性回顧比較；下一個尚未觀察的新場次才適合獨立確認。", "",
              "![各年融合權重](figures/fusion_selection.png)", "", "## 平均人次", "",
              "圖固定展示煙火＋天氣版，不依外層最小誤差挑版本。每一活動先加總 18–23 時的站點起站人次，再對相同外層活動取平均。"
              "這是捷運起站人次，不是獨立參加人數；不同逐時曲線也可能有相同平均，所以須同時檢查逐時與尖峰誤差。", "",
              "![平均人次](figures/mean_event_counts.png)", "", "## 北門逐場曲線", "",
              "![北門基本模型](figures/beimen_profiles.png)", "", "![北門融合模型](figures/beimen_fusion_profiles.png)", "",
              "完整數值：`tables/metrics_by_variant.csv`、`tables/yearly.csv`、`tables/mean_event_counts.csv`、"
              "`tables/selected_models_weights.csv`。各版本另保留選參、係數與資料可得性稽核。", ""]
    (output / "report.md").write_text("\n".join(parts), encoding="utf-8")


def build_comparison(root=ROOT, output=None):
    root = Path(root)
    output = Path(output) if output is not None else root / "result/analysis_v5"
    for name in ("tables", "figures"):
        (output / name).mkdir(parents=True, exist_ok=True)
    frames, manifests, folds, checks = read_variants(root)
    summary, annual = [], []
    for variant, frame in frames.items():
        summary.append(metrics_table(frame).assign(variant=variant, variant_label=VARIANTS[variant]))
        for year, block in frame.groupby("year"):
            annual.append(metrics_table(block).assign(variant=variant, variant_label=VARIANTS[variant], year=int(year)))
    summary, annual = pd.concat(summary, ignore_index=True), pd.concat(annual, ignore_index=True)
    means, selections = mean_event_counts(frames), selection_table(folds)
    for name, frame in (("metrics_by_variant", summary), ("yearly", annual), ("mean_event_counts", means), ("selected_models_weights", selections)):
        frame.to_csv(output / "tables" / f"{name}.csv", index=False, encoding="utf-8-sig")
    checks["comparison_scope"] = "paired_exploratory_temporal_outer_holdouts_not_model_selection"
    checks["reporting_code"] = {"path": "src/analysis_v5/reporting.py",
                                 "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "comparison_audit.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison_figure(summary, output)
    mean_figure(means, output)
    profiles_figure(frames["both"], output)
    fusion_figure(selections, output)
    write_report(summary, annual, means, selections, output)
    return {"output": str(output), "report": str(output / "report.md"), "events": int(summary.events.max()),
            "variants": list(VARIANTS), "models": list(ALL_MODELS)}


if __name__ == "__main__":
    print(json.dumps(build_comparison(), ensure_ascii=False, indent=2))
