"""Compare prespecified D1 feature ablations using paired outer predictions only.

No model is fitted or selected here. Every average uses complete activity panels,
not training fitted values or rows that happen to survive a missing-value filter.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_v1.comparison import align_predictions
from analysis_v1.models import ALL_MODELS, metrics_table
from analysis_v1.reporting import LABELS, markdown_table, save, style

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"core": "基本欄位", "fireworks": "只加煙火時間", "weather": "只加天氣預報", "both": "煙火＋天氣"}
CALENDAR = ["is_weekend", "day_of_year", "year_index"]
FIREWORKS = ["planned_fireworks_minutes", "planned_fireworks_start_hour"]
WEATHER = ["forecast_temperature_c", "forecast_wind_speed_m_s", "forecast_cloud_cover_pct"]
FEATURES = {"core": [], "fireworks": FIREWORKS, "weather": WEATHER, "both": FIREWORKS + WEATHER}
STATIONS = ("北門", "大橋頭站", "民權西路", "雙連")
HOURS = tuple(range(18, 24))
MODELS = ("ols", "ridge", "xgboost", "stack")
MEAN_MODELS = ("y", *MODELS)
MEAN_LABELS = {"y": "實際", "ols": "一般回歸", "ridge": "Ridge", "xgboost": "XGBoost", "stack": "混合模型\nStacking"}
DESIGN_VARY_KEYS = {
    "version", "report_title", "output_dir", "private_output_dir", "experiment_notes",
    "candidate_groups", "d1_features", "d1_source_files", "comparison_reference",
    "comparison_manifest", "comparison_title", "comparison_unchanged_models",
}
D1_PATHS = {"docs/d1_program_sources.json", "docs/d1_weather_sources.json"}
FIT_CODE = {"src/run_analysis.py", "src/analysis_v1/models.py", "src/analysis_v1/data.py", "src/analysis_v1/d1_features.py"}


def validate_panel(source):
    """A total is meaningful only with every prespecified station/hour present."""
    if "train_end" not in source:
        raise ValueError("train_end is required to check chronological holdout")
    frame, _ = align_predictions(source, source)
    expected = set(itertools.product(STATIONS, HOURS))
    for date, part in frame.groupby("event_date"):
        if set(zip(part.station, part.hour)) != expected:
            raise ValueError(f"{date:%Y-%m-%d}: incomplete station/hour panel")
    return frame


def _hashes(items):
    result = {item["path"].replace("\\", "/"): item["sha256"] for item in items}
    if len(result) != len(items):
        raise ValueError("Duplicate source paths in manifest")
    return result


def validate_manifests(manifests):
    """Allow feature-group changes; reject changed data, tuning or model code."""
    reference_design, reference_inputs, reference_code = None, None, None
    source_hashes = {}
    grid_expected = set(itertools.product((1, 2, 3, 4, 5), (50, 100, 250), (1, 6, 24), (1., 10.)))
    for variant in VARIANTS:
        manifest = manifests[variant]
        config = manifest["config"]
        if config.get("include_historical_baseline") is not False:
            raise ValueError(f"{variant}: unverified historical baseline must be excluded")
        if config.get("d1_cutoff_hour") != 18:
            raise ValueError(f"{variant}: D-1 cutoff must be 18:00")
        if config.get("d1_features", []) != FEATURES[variant]:
            raise ValueError(f"{variant}: incorrect ablation features")
        if config.get("candidate_groups") != CALENDAR + FEATURES[variant]:
            raise ValueError(f"{variant}: incorrect candidate groups")
        grid = config.get("xgb_grid", [])
        grid_values = {tuple(row[k] for k in ("max_depth", "n_estimators", "min_child_weight", "reg_lambda")) for row in grid}
        if len(grid) != 90 or grid_values != grid_expected:
            raise ValueError(f"{variant}: expected the same complete 90-candidate grid")
        design = {key: value for key, value in config.items() if key not in DESIGN_VARY_KEYS}
        inputs = _hashes(manifest["data"]["manifest"])
        base_inputs = {key: value for key, value in inputs.items() if key not in D1_PATHS}
        code = {key: value for key, value in _hashes(manifest["code"]).items() if key in FIT_CODE}
        if set(code) != FIT_CODE:
            raise ValueError(f"{variant}: modeling code hashes are incomplete")
        if reference_design is not None:
            if design != reference_design:
                raise ValueError(f"{variant}: model/evaluation settings differ beyond feature groups")
            if base_inputs != reference_inputs:
                raise ValueError(f"{variant}: base source input hashes differ")
            if code != reference_code:
                raise ValueError(f"{variant}: modeling code differs")
        else:
            reference_design, reference_inputs, reference_code = design, base_inputs, code
        required_sources = ({"docs/d1_program_sources.json"} if variant in ("fireworks", "both") else set())
        required_sources |= ({"docs/d1_weather_sources.json"} if variant in ("weather", "both") else set())
        if not required_sources.issubset(inputs):
            raise ValueError(f"{variant}: D1 source hashes missing")
        for path in D1_PATHS.intersection(inputs):
            if path in source_hashes and source_hashes[path] != inputs[path]:
                raise ValueError(f"{variant}: shared D1 source hashes differ")
            source_hashes[path] = inputs[path]
    return {"base_input_hashes": reference_inputs, "d1_source_hashes": source_hashes,
            "model_code_hashes": reference_code, "all_variants_exclude_historical_baseline": True,
            "candidate_count": 90, "same_settings_except_feature_groups": True}


def read_variants(root):
    root = Path(root)
    frames, manifests, prediction_hashes = {}, {}, {}
    reference = None
    for variant in VARIANTS:
        path = root / ".local/analysis_v4" / variant / "outer_predictions.csv"
        frame = validate_panel(pd.read_csv(path, parse_dates=["event_date"]))
        if reference is not None:
            _, frame = align_predictions(reference, frame)
            if not pd.to_datetime(reference.train_end).equals(pd.to_datetime(frame.train_end)):
                raise ValueError(f"{variant}: training cutoff differs")
        else:
            reference = frame
        frames[variant] = frame
        manifest_path = root / "result/analysis_v4" / variant / "run_manifest.json"
        manifests[variant] = json.loads(manifest_path.read_text(encoding="utf-8"))
        outer = manifests[variant]["outer_test"]
        if outer["rows"] != len(frame) or outer["events"] != frame.event_date.nunique():
            raise ValueError(f"{variant}: manifest/prediction sample mismatch")
        if sorted(frame.event_date.dt.year.unique()) != outer["years"]:
            raise ValueError(f"{variant}: manifest/prediction years mismatch")
        prediction_hashes[variant] = hashlib.sha256(path.read_bytes()).hexdigest()
    checks = validate_manifests(manifests)
    checks["prediction_hashes"] = prediction_hashes
    return frames, manifests, checks


def ablation_metrics(frames):
    """Event-balanced hourly errors and errors at fixed observed peak hours."""
    rows, annual = [], []
    reference = validate_panel(frames["core"])
    for variant, label in VARIANTS.items():
        frame = validate_panel(frames[variant])
        align_predictions(reference, frame)
        for scope, block in (("all", frame), ("beimen", frame.loc[frame.station.eq("北門")])):
            metrics = metrics_table(block).set_index("model")
            peaks = []
            for (date, station), part in block.groupby(["event_date", "station"]):
                part = part.sort_values("hour")
                actual = part.loc[part.y.idxmax()]
                for model in MODELS:
                    signed = float(actual.y - actual[model])
                    peaks.append({"event_date": date, "model": model, "signed": signed,
                                  "absolute": abs(signed), "positive": max(0., signed)})
            peaks = pd.DataFrame(peaks)
            for model in MODELS:
                peak = peaks.loc[peaks.model.eq(model)].groupby("event_date")[["signed", "absolute", "positive"]].mean().mean()
                value = metrics.loc[model]
                rows.append({"variant": variant, "variant_label": label, "scope": scope, "model": model,
                             "events": int(value.events), "rows": int(value.rows), "mae": float(value.mae),
                             "rmse": float(value.rmse), "bias_pred_minus_actual": float(value.bias_pred_minus_actual),
                             "peak_absolute_error_mean": float(peak.absolute),
                             "peak_underprediction_mean": float(peak.positive),
                             "peak_signed_underprediction_mean": float(peak.signed),
                             "peak_hour_mae": float(value.peak_hour_mae)})
            for year, part in block.groupby(block.event_date.dt.year):
                by_year = metrics_table(part)
                for value in by_year.loc[by_year.model.isin(MODELS)].to_dict("records"):
                    annual.append({"variant": variant, "scope": scope, "year": int(year), **value})
    summary = pd.DataFrame(rows)
    core = summary.loc[summary.variant.eq("core"), ["scope", "model", "mae", "peak_absolute_error_mean"]]
    core = core.rename(columns={"mae": "core_mae", "peak_absolute_error_mean": "core_peak_absolute_error_mean"})
    summary = summary.merge(core, on=["scope", "model"], validate="many_to_one")
    summary["mae_change_from_core"] = summary.mae - summary.core_mae
    summary["peak_absolute_error_change_from_core"] = summary.peak_absolute_error_mean - summary.core_peak_absolute_error_mean
    return summary, pd.DataFrame(annual)


def mean_event_counts(frames):
    """First sum 18–23 origin trips within each activity, then average activities."""
    rows = []
    reference = None
    for variant, label in VARIANTS.items():
        frame = validate_panel(frames[variant])
        if reference is not None:
            align_predictions(reference, frame)
        else:
            reference = frame
        for scope, block in (("all", frame), ("beimen", frame.loc[frame.station.eq("北門")])):
            totals = block.groupby("event_date")[list(MEAN_MODELS)].sum()
            actual = float(totals.y.mean())
            for model in MEAN_MODELS:
                mean = float(totals[model].mean())
                rows.append({"variant": variant, "variant_label": label, "scope": scope, "model": model,
                             "events": len(totals), "stations": block.station.nunique(), "hours_per_station": len(HOURS),
                             "mean_event_origin_trips": mean, "actual_mean_event_origin_trips": actual,
                             "mean_bias_trips": mean - actual,
                             "mean_bias_pct": 100 * (mean / actual - 1) if actual else np.nan,
                             "unit": "origin_person_trips_per_event_18_to_23_not_unique_attendance",
                             "aggregation": "mean_over_events_of_sum_over_stations_and_hours"})
    return pd.DataFrame(rows)


def ablation_figure(summary, output):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    colors = ("#9B9B9B", "#648696", "#8E8E64", "#454545")
    y = np.arange(len(MODELS))
    for ax, scope, title in zip(axes, ("all", "beimen"), ("全部四站", "北門")):
        for i, ((variant, label), color) in enumerate(zip(VARIANTS.items(), colors)):
            part = summary.loc[summary.variant.eq(variant) & summary.scope.eq(scope)].set_index("model").loc[list(MODELS)]
            bars = ax.barh(y + (i - 1.5) * .18, part.mae, height=.18, color=color, label=label)
            ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
        ax.set_yticks(y, [LABELS[m] for m in MODELS])
        ax.invert_yaxis()
        ax.set(title=title, xlabel="活動等權 MAE（人次／站／小時；越低越好）")
        ax.set_xlim(0, summary.loc[summary.scope.eq(scope), "mae"].max() * 1.19)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"煙火時間與天氣分開加入｜相同 {int(summary.iloc[0].events)} 場外層活動\n"
                 "四版均排除未核實發布時間的一般日基準，使用相同內層選參規則", fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, .88))
    save(fig, output / "figures/feature_ablation.png")


def mean_figure(means, output):
    """Both-feature model is fixed in advance; do not pick a variant by test MAE."""
    style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    colors = ("#222222", "#B1B1B1", "#7A8C91", "#A09174", "#5C7668")
    for ax, scope, title in zip(axes, ("all", "beimen"), ("四站合計", "北門")):
        part = means.loc[means.variant.eq("both") & means.scope.eq(scope)].set_index("model").loc[list(MEAN_MODELS)]
        bars = ax.bar(range(len(part)), part.mean_event_origin_trips, color=colors, width=.68)
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in part.mean_event_origin_trips], padding=5, fontsize=11)
        ax.set(xticks=range(len(part)), xticklabels=[MEAN_LABELS[m] for m in MEAN_MODELS],
               title=title, ylabel="每場活動平均起站人次（18–23 時）",
               ylim=(0, part.mean_event_origin_trips.max() * 1.2))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    events = int(means.iloc[0].events)
    fig.suptitle(f"實際與三模型、混合模型的平均人次｜煙火＋天氣版\n"
                 f"先加總每場六小時，再平均相同 {events} 場外層活動；不是獨立參加人數", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .88))
    save(fig, output / "figures/mean_event_counts.png")


def selected_settings(root, frames):
    """Summarize actual outer-trained settings, not the final full-data refit."""
    rows = []
    for variant in VARIANTS:
        folds = json.loads((Path(root) / "result/analysis_v4" / variant / "folds.json").read_text(encoding="utf-8"))
        expected_years = set(frames[variant].event_date.dt.year)
        if {fold["outer_year"] for fold in folds} != expected_years or len(folds) != len(expected_years):
            raise ValueError(f"{variant}: selected settings do not cover the prediction years")
        for fold in folds:
            block = frames[variant].loc[frames[variant].event_date.dt.year.eq(fold["outer_year"])]
            if not pd.to_datetime(block.train_end).eq(pd.Timestamp(fold["train_end"])).all():
                raise ValueError(f"{variant}: selected setting/prediction training cutoff mismatch")
            models = fold["base_models"]
            if any(models[model]["includes_historical_baseline"] for model in ("ols", "ridge", "xgboost")):
                raise ValueError(f"{variant}: fitted model includes unverified baseline")
            rows.append({"variant": variant, "outer_year": fold["outer_year"], "train_end": fold["train_end"],
                         "ols_selected_groups": json.dumps(models["ols"]["groups"], ensure_ascii=False),
                         "ridge_lambda": models["ridge"]["params"]["lambda"],
                         **{key: models["xgboost"]["params"][key] for key in ("max_depth", "n_estimators", "min_child_weight", "reg_lambda")},
                         **{f"stack_weight_{model}": weight for model, weight in fold["weights"].items()}})
    return pd.DataFrame(rows)


def write_report(summary, means, frames, annual, settings, output):
    rows = []
    for variant, label in VARIANTS.items():
        for model in MODELS:
            part = summary.loc[summary.variant.eq(variant) & summary.model.eq(model)].set_index("scope")
            rows.append({"輸入": label, "方法": LABELS[model], "四站 MAE": part.loc["all", "mae"],
                         "北門 MAE": part.loc["beimen", "mae"],
                         "北門 MAE 變化": part.loc["beimen", "mae_change_from_core"],
                         "北門尖峰絕對誤差": part.loc["beimen", "peak_absolute_error_mean"]})
    average_rows = []
    for model in MEAN_MODELS:
        part = means.loc[means.variant.eq("both") & means.model.eq(model)].set_index("scope")
        average_rows.append({"方法": MEAN_LABELS[model].replace("\n", " "),
                             "四站平均起站人次": part.loc["all", "mean_event_origin_trips"],
                             "北門平均起站人次": part.loc["beimen", "mean_event_origin_trips"]})
    n = frames["core"].event_date.nunique()
    xgb = summary.loc[summary.scope.eq("beimen") & summary.model.eq("xgboost")].set_index("variant")
    def change(variant):
        pct = 100 * (xgb.loc[variant, "mae"] / xgb.loc["core", "mae"] - 1)
        return f"{'增加' if pct >= 0 else '減少'} {abs(pct):.1f}%"
    latest_year = int(annual.year.max())
    last_xgb = annual.loc[annual.scope.eq("beimen") & annual.model.eq("xgboost") & annual.year.eq(latest_year)].set_index("variant")
    ols_unchanged = all(np.allclose(frames["core"].ols, frames[variant].ols, rtol=0, atol=1e-8) for variant in VARIANTS)
    no_new_ols = all(not set(json.loads(groups)).intersection(FIREWORKS + WEATHER) for groups in settings.ols_selected_groups)
    ols_note = "一般回歸的 Forward Selection 沒有選入煙火或天氣欄位，因此四組預測完全相同。" if ols_unchanged and no_new_ols else "一般回歸選入欄位見設定表。"
    text = f"""# 煙火時間與天氣：分組加入的實際比較

共同使用 **{n} 場外層活動、每場四站 × 18–23 時**。四組均排除尚未核實發布時間的 56 日一般日基準；只有候選變數組合不同，不能與保留舊基準的第二版混為同一比較。

## 比較設計

| 組別 | 候選輸入 |
|---|---|
| 基本欄位 | 車站、時段、日期三欄；Ridge 保留車站 × 時段交互作用 |
| 只加煙火時間 | 基本欄位＋預定施放分鐘數、預定開始時間 |
| 只加天氣預報 | 基本欄位＋預報溫度、風速、雲量 |
| 煙火＋天氣 | 基本欄位＋以上五欄 |

新增資料仍須有證據顯示活動前一天 **18:00（臺北時間）** 已發布，天氣使用當時預報，不使用事後觀測；未知留缺值，補值、標準化、選變數及選參只在較早訓練資料內執行。無人機與整體節目長度在各外層訓練期均沒有可用觀測，不能辨識效果，本次不列入。

每組各自依同一規則重選參數。因此比較的是「加入一組候選變數後，整套建模流程的表現」，同時包含選變數、正則化與融合權重的變化，**不是煙火或天氣對人流的因果效應**。歷史訓練標籤的發布時間與實際舉辦樣本篩選仍有回顧限制；不能把新欄位的 D1 查核等同整條流程已通過事前部署驗證。

## 結果

**XGBoost 北門 MAE**：基本欄位為 {xgb.loc['core', 'mae']:.2f}；只加煙火為 {xgb.loc['fireworks', 'mae']:.2f}（{change('fireworks')}），只加天氣為 {xgb.loc['weather', 'mae']:.2f}（{change('weather')}），兩者都加為 {xgb.loc['both', 'mae']:.2f}（{change('both')}）。這支持本批樣本下煙火時間較有幫助、天氣組合拖累 XGBoost；不能推論真實天氣沒有影響，也不能把合併誤差拆成兩個可相加的因果效果。

只加煙火也不是每年都改善：{latest_year} 年北門 XGBoost MAE 為 {last_xgb.loc['fireworks', 'mae']:.2f}，基本欄位為 {last_xgb.loc['core', 'mae']:.2f}。{ols_note} 各模型的變數、Ridge 懲罰與融合權重見[實際選出設定](tables/selected_settings.csv)；加入新欄位會改變整體擬合及選參，不能假設一定更準。

MAE 越小越好；「北門 MAE 變化」以基本欄位為基準，正值表示變差。尖峰均鎖定每場每站實際最高的小時，同值取最早小時；絕對誤差同時懲罰高估與低估。

{markdown_table(pd.DataFrame(rows))}

![分組加入的四站與北門 MAE](figures/feature_ablation.png)

完整結果見 [分組指標](tables/group_ablation.csv)及[逐年度指標](tables/yearly_ablation.csv)。樣本只有 {n} 場外層活動；曾查看這些年度結果後修訂模型，故屬探索性回顧比較，不以此次成績挑出「已驗證最優」模型。平均總人次也不能取代逐時與尖峰誤差：不同時段的高低估可能互相抵銷。

## 平均人次長條圖

使用預先指定的「煙火＋天氣」版，不根據外層分數選版。**先將一場活動四站（或北門）18–23 時人次相加，再對共同 {n} 場活動取平均**；三個基模型和 Stacking 都是留出預測。

`平均人次 = (1 / 活動數) × Σ活動 Σ車站 Σ18–23時 預測起站人次`

實際值把公式中的預測換成觀測值。這是來源時段的**起站人次**，不是活動現場獨立人數、進出合計或月台到達量；不能直接當成排隊理論的分鐘到達率。圖為類別平均值的長條圖，並非連續分布直方圖。

{markdown_table(pd.DataFrame(average_rows))}

![實際、三模型與混合模型平均人次](figures/mean_event_counts.png)

所有四組的平均值與偏差見 [mean_event_counts.csv](tables/mean_event_counts.csv)。

## 90 組如何選

`5 個樹深 × 3 個樹數 × 3 個 min_child_weight × 2 個 reg_lambda = 90 組`。各外層年度只用較早年度的時間驗證折，比較逐場等權 MAE，選出網格中最低的一組，再用較早資料重訓及預測當年；不使用當年外層真值選參。這是**有限網格搜尋**，不是所有可能參數中的全域最佳解，也不是把 90 個模型混合。Stacking 則另用三個基模型的過去留出預測估計非負且合計為 1 的權重。

## 重建

完成四個 `analysis_v4_*.json` 設定的分析後執行：

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 src/compare_analysis_v4.py
```

程式會拒絕缺少車站／小時、改變真值／訓練截止時間、不同資料雜湊或不同選參規則的比較。[comparison_checks.json](comparison_checks.json) 保存來源及逐列預測檔的雜湊；逐列預測仍留在忽略提交的 `.local/analysis_v4/`。
"""
    (output / "report.md").write_text(text, encoding="utf-8")


def build_comparison(root=ROOT, output=None):
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / "result/analysis_v4"
    frames, manifests, checks = read_variants(root)
    summary, annual = ablation_metrics(frames)
    means = mean_event_counts(frames)
    settings = selected_settings(root, frames)
    for directory in (output / "tables", output / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    for name, table in (("group_ablation", summary), ("yearly_ablation", annual), ("mean_event_counts", means), ("selected_settings", settings)):
        table.to_csv(output / f"tables/{name}.csv", index=False, encoding="utf-8-sig")
    (output / "comparison_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ablation_figure(summary, output)
    mean_figure(means, output)
    write_report(summary, means, frames, annual, settings, output)
    return summary, means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    summary, means = build_comparison(args.root, args.output)
    print(summary.loc[summary.scope.eq("beimen"), ["variant", "model", "mae", "mae_change_from_core", "peak_absolute_error_mean"]].to_string(index=False))
    print(means.loc[means.variant.eq("both"), ["scope", "model", "mean_event_origin_trips"]].to_string(index=False))


if __name__ == "__main__":
    main()
