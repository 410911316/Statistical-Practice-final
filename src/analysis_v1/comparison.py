"""Paired retrospective comparison of unchanged outer-event prediction keys.

Only event/station summaries and figures are public; the full prediction frames
remain in their private run directories. This module never fits or selects a model.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import ALL_MODELS, metrics_table
from .reporting import LABELS, markdown_table, save, style

KEYS = ["event_date", "station", "hour"]
METRICS = ["mae", "rmse", "bias_pred_minus_actual",
           "peak_underprediction_mean", "peak_hour_mae"]


def align_predictions(old: pd.DataFrame, new: pd.DataFrame):
    """Reject changed test cases, duplicate rows, changed truth or invalid values."""
    frames = []
    for label, source in (("old", old), ("new", new)):
        missing = set(KEYS + ["y", *ALL_MODELS]) - set(source.columns)
        if missing:
            raise ValueError(f"{label}: missing columns {sorted(missing)}")
        frame = source.copy()
        frame["event_date"] = pd.to_datetime(frame.event_date, errors="raise")
        if frame.empty or frame[KEYS].isna().any().any():
            raise ValueError(f"{label}: empty data or missing comparison key")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{label}: duplicate event/station/hour keys")
        if not np.isfinite(frame[["hour", "y", *ALL_MODELS]].to_numpy(float)).all():
            raise ValueError(f"{label}: nonfinite truth or predictions")
        if (frame[["y", *ALL_MODELS]].to_numpy(float) < 0).any():
            raise ValueError(f"{label}: person counts must be nonnegative")
        if not ((frame.hour % 1 == 0) & frame.hour.between(0, 23)).all():
            raise ValueError(f"{label}: hour must be an integer between 0 and 23")
        if "train_end" in frame:
            ends = pd.to_datetime(frame.train_end, errors="raise")
            if ends.isna().any() or not (ends < frame.event_date).all():
                raise ValueError(f"{label}: training must end before each test event")
        for column in ("year", "outer_year"):
            if column in frame and not frame[column].eq(frame.event_date.dt.year).all():
                raise ValueError(f"{label}: {column} disagrees with event date")
        frames.append(frame.sort_values(KEYS).reset_index(drop=True))
    left, right = frames
    if not left[KEYS].equals(right[KEYS]):
        raise ValueError("Old/new comparison keys differ; do not compare different test cases")
    if not np.array_equal(left.y.to_numpy(float), right.y.to_numpy(float)):
        raise ValueError("Old/new target values differ")
    if not left.station.eq("北門").any():
        raise ValueError("Comparison requires Beimen (北門) observations")
    return left, right


def comparison_tables(old: pd.DataFrame, new: pd.DataFrame):
    """Build event-balanced metrics and predictions at identical actual peaks."""
    old, new = align_predictions(old, new)
    metrics = []
    for version, frame in (("before", old), ("after", new)):
        for scope, block in (("all_stations", frame),
                             ("beimen", frame.loc[frame.station.eq("北門")])):
            metrics.append(metrics_table(block).assign(version=version, scope=scope))
    metrics = pd.concat(metrics, ignore_index=True)
    before = metrics.loc[metrics.version.eq("before")].drop(columns="version")
    after = metrics.loc[metrics.version.eq("after")].drop(columns="version")
    change = before.merge(after, on=["scope", "model", "events", "rows"],
                          suffixes=("_before", "_after"), validate="one_to_one")
    for metric in METRICS:
        change[f"{metric}_change"] = change[f"{metric}_after"] - change[f"{metric}_before"]
    for metric in ("mae", "rmse", "peak_underprediction_mean"):
        denominator = change[f"{metric}_before"].replace(0., np.nan)
        change[f"{metric}_reduction_pct"] = 100 * (1 - change[f"{metric}_after"] / denominator)

    peaks = []
    for day, left in old.loc[old.station.eq("北門")].groupby("event_date", sort=True):
        left = left.sort_values("hour")
        right = new.loc[new.event_date.eq(day) & new.station.eq("北門")].sort_values("hour")
        actual = left.loc[left.y.idxmax()]  # Sorted hours implement earliest tie.
        right_at_peak = right.loc[right.hour.eq(actual.hour)].iloc[0]
        for model in ALL_MODELS:
            row = {"event_date": day, "station": "北門", "model": model,
                   "actual_peak_hour": int(actual.hour), "actual_peak": float(actual.y)}
            for version, block, at_peak in (("before", left, actual),
                                             ("after", right, right_at_peak)):
                prediction = float(at_peak[model])
                peak = block.loc[block[model].idxmax()]
                signed_under = float(actual.y) - prediction
                row.update({
                    f"prediction_at_actual_peak_{version}": prediction,
                    f"signed_underprediction_{version}": signed_under,
                    f"positive_underprediction_{version}": max(0., signed_under),
                    f"signed_underprediction_pct_{version}": 100 * signed_under / actual.y if actual.y else np.nan,
                    f"positive_underprediction_pct_{version}": 100 * max(0., signed_under) / actual.y if actual.y else np.nan,
                    f"predicted_peak_hour_{version}": int(peak.hour),
                    f"predicted_peak_value_{version}": float(peak[model]),
                    f"peak_hour_error_{version}": abs(int(peak.hour) - int(actual.hour)),
                })
            row["positive_underprediction_reduction"] = row["positive_underprediction_before"] - row["positive_underprediction_after"]
            row["absolute_peak_error_before"] = abs(row["signed_underprediction_before"])
            row["absolute_peak_error_after"] = abs(row["signed_underprediction_after"])
            peaks.append(row)
    peaks = pd.DataFrame(peaks)
    peak_summary = []
    for model, block in peaks.groupby("model", sort=True):
        row = {"model": model, "events": len(block),
               "events_positive_under_reduced": int((block.positive_underprediction_reduction > 1e-8).sum()),
               "events_absolute_peak_error_reduced": int((block.absolute_peak_error_after < block.absolute_peak_error_before - 1e-8).sum())}
        for version in ("before", "after"):
            for metric in ("positive_underprediction", "positive_underprediction_pct",
                           "signed_underprediction", "absolute_peak_error", "peak_hour_error"):
                row[f"mean_{metric}_{version}"] = block[f"{metric}_{version}"].mean()
            row[f"events_underpredicted_{version}"] = int((block[f"signed_underprediction_{version}"] > 1e-8).sum())
        peak_summary.append(row)
    return {"metrics_before_after": metrics, "metrics_change": change,
            "beimen_event_peaks": peaks, "beimen_peak_summary": pd.DataFrame(peak_summary)}


def _figures(old, new, tables, output):
    style()
    figures = output / "figures"
    days = sorted(old.event_date.unique())
    columns = min(3, len(days))
    rows = int(np.ceil(len(days) / columns))
    # Separate model figures avoid five nearly overlapping curves in each panel.
    for model in ("ridge", "xgboost"):
        fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.4 * rows),
                                 squeeze=False, constrained_layout=True)
        for ax, day in zip(axes.flat, days):
            left = old.loc[old.event_date.eq(day) & old.station.eq("北門")].sort_values("hour")
            right = new.loc[new.event_date.eq(day) & new.station.eq("北門")].sort_values("hour")
            ax.plot(left.hour, left.y, "o-", color="#111111", label="實際", lw=1.7)
            ax.plot(left.hour, left[model], "s--", color="#999999", label="改版前", lw=1.4, markersize=4)
            ax.plot(right.hour, right[model], "o-", color="#466D8C", label="改版後", lw=1.5, markersize=4)
            ax.set(title=f"{pd.Timestamp(day):%Y-%m-%d}", xticks=sorted(left.hour.unique()),
                   xlabel="來源小時", ylabel="北門來源時段起站人次", ylim=(0, None))
            ax.grid(axis="y", alpha=.2)
        for ax in list(axes.flat)[len(days):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=9)
        fig.suptitle(f"北門 {len(days)} 場完整曲線｜{LABELS[model]} 改版前後（回顧式比較）")
        save(fig, figures / f"beimen_profiles_{model}.png")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), constrained_layout=True)
    for ax, model in zip(axes, ("ridge", "xgboost")):
        peaks = tables["beimen_event_peaks"].loc[lambda x: x.model.eq(model)]
        x = np.arange(len(peaks))
        ax.bar(x - .18, peaks.signed_underprediction_pct_before, width=.36, color="#AAAAAA", label="改版前")
        ax.bar(x + .18, peaks.signed_underprediction_pct_after, width=.36, color="#466D8C", label="改版後")
        ax.axhline(0, color="#333333", lw=.8)
        ax.set(xticks=x, xticklabels=peaks.event_date.dt.strftime("%Y-%m-%d"),
               ylabel="（實際 − 預測）／實際（%）", title=LABELS[model])
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=.2)
        ax.legend()
    fig.suptitle("同一實際尖峰時段的低估比例｜越接近 0 越好；負值代表高估")
    save(fig, figures / "beimen_peak_underprediction.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, scope, title in zip(axes, ("all_stations", "beimen"), ("全部車站", "北門")):
        part = tables["metrics_change"].loc[lambda x: x.scope.eq(scope)].set_index("model").loc[list(ALL_MODELS)]
        x = np.arange(len(part))
        ax.bar(x - .18, part.mae_before, width=.36, color="#AAAAAA", label="改版前")
        ax.bar(x + .18, part.mae_after, width=.36, color="#466D8C", label="改版後")
        ax.set(xticks=x, xticklabels=[LABELS[m] for m in part.index], ylabel="活動等權 MAE（人次／站／小時）", title=title)
        ax.tick_params(axis="x", labelrotation=25)
        ax.legend()
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("相同測試活動、相同真值：整體準確度與北門準確度分開比較")
    save(fig, figures / "metrics_before_after.png")


def _report(tables, old, output):
    changes = tables["metrics_change"]
    peaks = tables["beimen_event_peaks"]
    summary = tables["beimen_peak_summary"].set_index("model")
    rows = []
    for model in ("ols", "ridge", "xgboost", "equal", "stack", "baseline"):
        total = changes.loc[changes.model.eq(model) & changes.scope.eq("all_stations")].iloc[0]
        beimen = changes.loc[changes.model.eq(model) & changes.scope.eq("beimen")].iloc[0]
        rows.append({"模型": LABELS[model], "全站 MAE 前": total.mae_before,
                     "全站 MAE 後": total.mae_after, "北門 MAE 前": beimen.mae_before,
                     "北門 MAE 後": beimen.mae_after,
                     "北門尖峰低估 前": beimen.peak_underprediction_mean_before,
                     "北門尖峰低估 後": beimen.peak_underprediction_mean_after})
    lines = []
    for model in ("ridge", "xgboost"):
        row = changes.loc[changes.model.eq(model) & changes.scope.eq("beimen")].iloc[0]
        item = summary.loc[model]
        status = "下降" if row.mae_after < row.mae_before else "上升" if row.mae_after > row.mae_before else "不變"
        peak_status = "下降" if row.peak_underprediction_mean_after < row.peak_underprediction_mean_before else "上升" if row.peak_underprediction_mean_after > row.peak_underprediction_mean_before else "不變"
        lines.append(f"- **{LABELS[model]}**：北門 MAE {row.mae_before:.2f} → {row.mae_after:.2f}（{status}）；實際尖峰時正向低估 {row.peak_underprediction_mean_before:.2f} → {row.peak_underprediction_mean_after:.2f} 人次（{peak_status}）。{int(item.events_positive_under_reduced)}/{int(item.events)} 場正向低估減少，{int(item.events_absolute_peak_error_reduced)}/{int(item.events)} 場尖峰絕對誤差減少；改版後仍有 {int(item.events_underpredicted_after)} 場低估。")
    event_rows = []
    for day, block in peaks.groupby("event_date", sort=True):
        r, x = block.loc[block.model.eq("ridge")].iloc[0], block.loc[block.model.eq("xgboost")].iloc[0]
        event_rows.append({"活動日": f"{day:%Y-%m-%d}", "實際尖峰小時": r.actual_peak_hour,
                           "實際人次": r.actual_peak, "Ridge 前": r.prediction_at_actual_peak_before,
                           "Ridge 後": r.prediction_at_actual_peak_after,
                           "XGB 前": x.prediction_at_actual_peak_before,
                           "XGB 後": x.prediction_at_actual_peak_after})
    years = "、".join(map(str, sorted(old.event_date.dt.year.unique())))
    text = f"""# 車站 × 時段 Ridge 與 XGBoost 調參：改版前後比較

## 比較範圍

- 使用完全相同的 {years} 年、{old.event_date.nunique()} 場、{len(old)} 列外層預測；逐一核對活動日、車站、小時及真值。圖表與指標由保存的預測產生。
- **這是看過既有結果後的回顧式模型修訂比較，並非全新、未接觸的確認性測試。**不能再把這些年度當成未見過的最終考試；未來活動或另保留資料才適合確認泛化能力。
- 模型仍須以較早年度做內層調參與融合，這份比較表本身不選參數。不因北門改善就宣稱全站均改善，也不以逐時列數當獨立活動數做顯著性檢定。
- MAE 與 RMSE 按活動等權計算。尖峰先取每場北門真值最高的小時，再比較兩版在**同一小時**的預測；同值取最早小時。正向低估是 `max(實際 − 預測, 0)`，高估不抵銷低估；另保留有符號誤差與絕對誤差，以免高估被誤說成改善。

## 指標與結果

{markdown_table(pd.DataFrame(rows))}

{chr(10).join(lines)}

![全站與北門誤差](figures/metrics_before_after.png)

完整 RMSE、偏誤、峰值時刻誤差與改變量見 [metrics_change.csv](tables/metrics_change.csv)；正向低估下降比例以原值為分母，原值為零時不定義。

## 每場北門實際尖峰

{markdown_table(pd.DataFrame(event_rows))}

![北門尖峰低估比例](figures/beimen_peak_underprediction.png)

完整六種方法及預測尖峰時間見 [beimen_event_peaks.csv](tables/beimen_event_peaks.csv)。圖中負值代表高估，並非負的風險。逐時曲線如下，避免只比較單一峰值而漏掉時間錯位或其他時段偏誤。

![Ridge 全部活動](figures/beimen_profiles_ridge.png)
![XGBoost 全部活動](figures/beimen_profiles_xgboost.png)

## 銜接排隊的界線

這些數字仍是來源時段起站人次。較低的平均誤差或尖峰低估，不代表已識別月台的分鐘到達、方向別人數及列車剩餘容量，也不代表已有可行調度。須核實原始時段意義、補分鐘內到達形狀與步行延遲，再以批次登車、滯留守恆及營運限制做情境比較；本報告不把改善的回顧式預測稱為調度保證。
"""
    (output / "report.md").write_text(text, encoding="utf-8")


def write_comparison(old: pd.DataFrame, new: pd.DataFrame, output: Path):
    """Write four aggregate tables, four figures, and a self-contained report."""
    old, new = align_predictions(old, new)
    tables = comparison_tables(old, new)
    output = Path(output)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / "tables" / f"{name}.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    _figures(old, new, tables, output)
    _report(tables, old, output)
    return tables
