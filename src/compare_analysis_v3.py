"""Reproduce a paired four-version report without fitting or selecting a model.

Run after all four versioned analyses have finished. Prediction rows stay in the
ignored .local directory; only grouped metrics, one illustrative peak, and figures
are written to result/analysis_v3.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_v1.comparison import align_predictions
from analysis_v1.models import ALL_MODELS, BASE_MODELS, metrics_table
from analysis_v1.reporting import LABELS, markdown_table, save, style

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = (
    ("analysis_v2", "第二版：原 36 組"),
    ("analysis_v3_tuning", "只擴大調參：90 組"),
    ("analysis_v3_with_baseline", "加 D1 新欄：保留舊基準"),
    ("analysis_v3", "主版：排除未核實基準"),
)
PEAK_DATE = pd.Timestamp("2026-08-15")


def read_versions(root):
    """All versions must use the same rows, targets, and historical control."""
    root = Path(root)
    frames, reference = {}, None
    for version, _ in VERSIONS:
        path = root / ".local" / version / "outer_predictions.csv"
        frame = pd.read_csv(path, parse_dates=["event_date"])
        if reference is None:
            reference, frame = align_predictions(frame, frame)
        else:
            _, frame = align_predictions(reference, frame)
            if not np.allclose(reference.baseline, frame.baseline, rtol=0, atol=1e-8):
                raise ValueError(f"{version}: the unchanged baseline control differs")
        frames[version] = frame
    return frames


def version_metrics(frames):
    """Use each observed event/station peak's same hour for every model/version."""
    summaries = []
    for version, label in VERSIONS:
        frame = frames[version]
        for scope, block in (("all", frame), ("beimen", frame.loc[frame.station.eq("北門")])):
            metrics = metrics_table(block).set_index("model")
            at_peaks = []
            for (day, station), part in block.groupby(["event_date", "station"]):
                part = part.sort_values("hour")
                actual = part.loc[part.y.idxmax()]  # Earliest observed maximum on ties.
                for model in ALL_MODELS:
                    signed = float(actual.y - actual[model])
                    at_peaks.append({"event_date": day, "station": station, "model": model,
                                     "positive": max(0., signed), "signed": signed,
                                     "absolute": abs(signed)})
            peaks = pd.DataFrame(at_peaks)
            for model in ALL_MODELS:
                # First average stations within an activity, then average activities.
                p = peaks.loc[peaks.model.eq(model)].groupby("event_date")[["positive", "signed", "absolute"]].mean().mean()
                row = metrics.loc[model]
                summaries.append({"version": version, "version_label": label, "scope": scope,
                                  "model": model, "events": int(row.events), "rows": int(row.rows),
                                  "mae": float(row.mae), "rmse": float(row.rmse),
                                  "bias_pred_minus_actual": float(row.bias_pred_minus_actual),
                                  "peak_underprediction_mean": float(p.positive),
                                  "peak_signed_underprediction_mean": float(p.signed),
                                  "peak_absolute_error_mean": float(p.absolute),
                                  "peak_hour_mae": float(row.peak_hour_mae),
                                  "peak_alignment": "same_actual_event_station_peak_hour_earliest_tie",
                                  "evaluation_scope": "paired_retrospective_comparison_not_parameter_selection"})
    return pd.DataFrame(summaries)


def illustrative_peak(frames):
    reference = frames[VERSIONS[0][0]]
    part = reference.loc[reference.event_date.eq(PEAK_DATE) & reference.station.eq("北門")].sort_values("hour")
    if part.empty:
        raise ValueError("The prespecified 2026-08-15 Beimen example is missing")
    actual = part.loc[part.y.idxmax()]
    rows = []
    for version, label in VERSIONS:
        frame = frames[version]
        row = frame.loc[frame.event_date.eq(PEAK_DATE) & frame.station.eq("北門")
                        & frame.hour.eq(actual.hour)].iloc[0]
        for model in BASE_MODELS:
            prediction = float(row[model])
            signed = float(actual.y - prediction)
            rows.append({"version": version, "version_label": label, "event_date": PEAK_DATE.strftime("%Y-%m-%d"),
                         "station": "北門", "actual_peak_hour": int(actual.hour),
                         "actual_peak": float(actual.y), "model": model,
                         "prediction_at_actual_peak": prediction,
                         "signed_underprediction": signed,
                         "signed_underprediction_pct": 100*signed/actual.y if actual.y else np.nan})
    return pd.DataFrame(rows)


def comparison_figure(summary, output):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.1))
    colors = ["#B0B0B0", "#557992", "#538873", "#AF784C"]
    y = np.arange(len(ALL_MODELS))
    for ax, scope, title in zip(axes, ("all", "beimen"), ("全部四站", "北門")):
        for index, ((version, label), color) in enumerate(zip(VERSIONS, colors)):
            part = summary.loc[summary.version.eq(version) & summary.scope.eq(scope)].set_index("model").loc[list(ALL_MODELS)]
            bars = ax.barh(y + (index-1.5)*.18, part.mae, height=.18, color=color, label=label)
            ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
        ax.set_yticks(y, [LABELS[model] for model in ALL_MODELS])
        ax.invert_yaxis()
        ax.set(title=title, xlabel="活動等權 MAE（人次／站／小時，越低越好）")
        ax.set_xlim(0, summary.loc[summary.scope.eq(scope), "mae"].max() * 1.2)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    reference = summary.loc[summary.version.eq(VERSIONS[0][0]) & summary.scope.eq("all")].iloc[0]
    fig.suptitle(f"相同 {int(reference.events)} 場外層活動：調參、加入 D1 新欄與排除未核實基準分開比較\n"
                 "一般日基準方法是回顧對照；主版三模型與融合的輸入均排除此基準", fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(.5, .015), frameon=False)
    fig.tight_layout(rect=(0, .11, 1, .89))
    save(fig, output / "figures/version_comparison.png")


def write_report(summary, peak, frames, output):
    rows = []
    for version, label in VERSIONS:
        for model in ALL_MODELS:
            group = summary.loc[summary.version.eq(version) & summary.model.eq(model)].set_index("scope")
            rows.append({"版本": label, "方法": LABELS[model], "全站 MAE": group.loc["all", "mae"],
                         "北門 MAE": group.loc["beimen", "mae"],
                         "北門尖峰正向低估": group.loc["beimen", "peak_underprediction_mean"],
                         "北門尖峰絕對誤差": group.loc["beimen", "peak_absolute_error_mean"]})
    peak_rows = []
    for version, label in VERSIONS:
        part = peak.loc[peak.version.eq(version)].set_index("model")
        peak_rows.append({"版本": label, "實際人次": float(part.iloc[0].actual_peak),
                          **{LABELS[model]: float(part.loc[model, "prediction_at_actual_peak"]) for model in BASE_MODELS}})
    source = frames[VERSIONS[0][0]]
    years = "、".join(map(str, sorted(source.event_date.dt.year.unique())))
    text = f"""# D1 輸入與 XGBoost 調參：四版本配對比較

使用完全相同的 **{years} 年 {source.event_date.nunique()} 場活動、{len(source)} 列**；逐一檢查日期、車站、小時、真值及一般日基準對照不變。本檔由保存的預測重建，不重新擬合，也不以外層成績挑參數。

## 四個版本比較什麼

1. **第二版**：原輸入、Ridge 車站 × 時段、XGBoost 原 36 組網格。
2. **只擴大調參**：原輸入不變；XGBoost 擴為 90 組。與第二版相比，用來看調參範圍的改變。
3. **加 D1 新欄、保留舊基準**：同一 90 組網格，加入已核實截止前的節目等新欄，保留 56 日一般日基準。與版本 2 相比，用來看增加候選輸入的影響；仍不能聲稱原基準已核實 D1 可用。
4. **主版、排除未核實基準**：排除尚未核實發布時間的 56 日基準。此版與前版的差異包括**拿掉舊輸入**，不能把所有升降都歸因於無人機、煙火或天氣。

新欄只接受活動前一天 18:00（臺北時間）前的已核實來源版本；預定節目不以實際演出回填，天氣必須有預報發布時間及目標時段，不拿事後觀測替代。未知保持缺值，補值及缺值指標只在訓練折學習；整欄未知不產生可學係數。逐筆來源、截止時間與採用原因見 [D1 審計](tables/d1_feature_audit.csv)，資料覆蓋見 [覆蓋率表](tables/d1_feature_coverage.csv)。

各模型都只用較早年度訓練及內層選參；三模型融合權重也由較早年度的時間留出預測估計。研究者已看過這些外層年度並據以修改模型，所以結果仍是**探索性回顧比較，尚非新的確認性測試或完整 D1 部署驗證**。歷史標籤發布時間、日曆版本與事後篩選「實際舉辦」樣本的界線仍須保留。

## 全部方法的結果

MAE 按每場活動等權計算。先以每場每站的**實際最高小時**固定比較位置，再看各版預測；同值取最早小時。正向低估為 `max(實際−預測, 0)`，高估不抵銷低估；絕對尖峰誤差同時懲罰高估與低估。表中的一般日基準只作回顧對照，並未重新成為主版三模型的輸入。

{markdown_table(pd.DataFrame(rows))}

![四版本全站及北門 MAE](figures/version_comparison.png)

完整 RMSE、有符號尖峰誤差、尖峰時刻誤差見 [version_comparison.csv](tables/version_comparison.csv)。有符號尖峰低估採 `實際−預測`，負值代表高估。圖表列出全部版本與方法，不挑最好的一個活動或外層年度當成整體改善證據。

## 固定示例：2026/8/15 北門 {int(peak.iloc[0].actual_peak_hour)} 點

這是先前已指定的示例，三模型均比較同一個實際尖峰小時；並未改選各模型預測最高的小時。

{markdown_table(pd.DataFrame(peak_rows))}

有符號低估比例另見 [示例完整表](tables/version_20260815_peak.csv)。單場數值不能取代上面的 {source.event_date.nunique()} 場配對指標。

## 如何解讀調參與後續用途

樹深決定每棵樹可分割的複雜度，樹數決定逐步修正的次數；增加兩者不保證更準。新版預先比較 `max_depth=[1,2,3,4,5]`、`n_estimators=[50,100,250]`、`min_child_weight=[1,6,24]`、`reg_lambda=[1,10]`，共 90 組，學習率維持 0.05。實際選中設定及其**內層** MAE 見 [參數表](tables/xgb_selected_params.csv)及[調參熱圖](figures/xgb_tuning_heatmap.png)。熱圖每格對其餘候選取最低內層 MAE，沒有使用當年外層測試真值選參。

即使尖峰預測改善，目前的目標仍是來源時段起站人次。分鐘內到達形狀、方向比例、步行延遲、列車剩餘容量及營運限制尚待核實；不能直接把這些逐時點預測當成排隊到達率或已可執行的調度保證。

## 重建本報告

先完成四個版本，再於專案根目錄執行：

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 src/compare_analysis_v3.py
```

需要本機 `.local/analysis_v2`、`.local/analysis_v3_tuning`、`.local/analysis_v3_with_baseline`、`.local/analysis_v3` 內的 `outer_predictions.csv`。完整逐列預測維持不提交；本報告只輸出彙整指標、圖表及上述固定示例。
"""
    (output / "version_report.md").write_text(text, encoding="utf-8")


def build_comparison(root=ROOT, output=None):
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / "result/analysis_v3"
    frames = read_versions(root)
    summary, peak = version_metrics(frames), illustrative_peak(frames)
    for directory in (output / "tables", output / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "tables/version_comparison.csv", index=False, encoding="utf-8-sig")
    peak.to_csv(output / "tables/version_20260815_peak.csv", index=False, encoding="utf-8-sig")
    comparison_figure(summary, output)
    write_report(summary, peak, frames, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    summary = build_comparison(args.root, args.output)
    print(summary.loc[summary.scope.eq("beimen"), ["version", "model", "mae", "peak_absolute_error_mean"]].to_string(index=False))
    print(f"四版本彙整：{args.output or args.root / 'result/analysis_v3'}")


if __name__ == "__main__":
    main()
