"""Readable figures and an evidence-based Chinese report for completed runs."""
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from .models import ALL_MODELS

LABELS = {"baseline": "一般日基準", "ols": "Forward OLS", "ridge": "Ridge", "xgboost": "XGBoost", "equal": "等權平均", "stack": "MAE Stacking"}


def style():
    available = {font.name for font in font_manager.fontManager.ttflist}
    chinese_font = next((name for name in ("Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei") if name in available), None)
    if chinese_font is None:
        raise RuntimeError("Install Microsoft JhengHei or Noto Sans CJK TC to render Chinese figures")
    plt.rcParams.update({"font.family": [chinese_font, "DejaVu Sans"], "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def build_model_figures(predictions, metrics, folds, vif, output: Path):
    style()
    figures = output / "figures"
    values = metrics.set_index("model").loc[list(ALL_MODELS)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for ax, metric, title in zip(axes, ("mae", "rmse"), ("活動等權 MAE", "活動等權 RMSE")):
        bars = ax.bar(range(6), values[metric], color=["#999999", "#466D8C", "#59806E", "#AE8653", "#887BA2", "#333333"])
        ax.set_xticks(range(6), [LABELS[m] for m in ALL_MODELS], rotation=25, ha="right")
        ax.bar_label(bars, fmt="%.0f", padding=3)
        ax.set_ylim(0, values[metric].max() * 1.22)
        ax.set(ylabel="人次／站／小時", title=title)
    years_text = "、".join(str(y) for y in sorted(predictions.year.unique()))
    fig.suptitle(f"共同外層測試：{years_text} 年 {predictions.event_date.nunique()} 場活動（回顧式驗證）")
    save(fig, figures / "model_comparison.png")

    errors = pd.DataFrame({m: abs(predictions[m] - predictions.y) for m in ALL_MODELS})
    errors["event_date"] = predictions.event_date.to_numpy()
    errors = errors.groupby("event_date").mean()
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    im = ax.imshow(errors, cmap="YlOrBr", aspect="auto")
    ax.set_xticks(range(6), [LABELS[m] for m in ALL_MODELS])
    ax.set_yticks(range(len(errors)), errors.index.strftime("%Y-%m-%d"))
    for i in range(len(errors)):
        for j in range(6):
            ax.text(j, i, f"{errors.iloc[i,j]:.0f}", ha="center", va="center", color="white" if errors.iloc[i,j] > errors.to_numpy().max() * .62 else "black")
    fig.colorbar(im, ax=ax, label="每場 MAE（人次／站／小時）")
    ax.set_title("逐場誤差：平均表現不可取代跨活動穩定性")
    save(fig, figures / "event_errors.png")

    block = predictions.loc[predictions.year == max(predictions.year)]
    days, stations = sorted(block.event_date.unique()), ["北門", "大橋頭站", "雙連", "民權西路"]
    fig, axes = plt.subplots(len(days), 4, figsize=(15, 3.3 * len(days)), squeeze=False, constrained_layout=True)
    for i, day in enumerate(days):
        for j, station in enumerate(stations):
            part = block.loc[block.event_date.eq(day) & block.station.eq(station)].sort_values("hour")
            ax = axes[i, j]
            ax.plot(part.hour, part.y, "o-", color="#111111", label="實際")
            ax.plot(part.hour, part.baseline, "--", color="#999999", label="一般日基準")
            ax.plot(part.hour, part["stack"], "s-", color="#547D98", label="Stacking")
            ax.set(title=f"{pd.Timestamp(day):%Y-%m-%d}｜{station}", xticks=list(range(18,24)), xlabel="來源小時", ylabel="起站人次")
            ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{max(predictions.year)} 年全部 {len(days)} 場外層預測｜未將當年真值交給訓練程序")
    save(fig, figures / "heldout_profiles_latest.png")

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    x, bottom = np.arange(len(folds)), np.zeros(len(folds))
    for model, color in zip(("ols", "ridge", "xgboost"), ("#466D8C", "#59806E", "#AE8653")):
        vals = [fold["weights"][model] for fold in folds]
        ax.bar(x, vals, bottom=bottom, color=color, label=LABELS[model])
        for i, (val, base) in enumerate(zip(vals, bottom)):
            if val > .07:
                ax.text(i, base + val/2, f"{val:.2f}", ha="center", va="center", color="white")
        bottom += vals
    ax.set(xticks=x, xticklabels=[str(f["outer_year"]) for f in folds], ylabel="非負、總和為 1 的權重", title="每一外層年度：只由較早年度 OOF 學得權重", ylim=(0,1.18))
    ax.legend(loc="upper center", ncol=3)
    save(fig, figures / "stack_weights.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    residual = predictions.y - predictions.ols
    axes[0].scatter(predictions.ols, residual, color="#547D98", alpha=.75, s=18)
    axes[0].axhline(0, color="black", lw=.8)
    axes[0].set(xlabel="外層 OLS 預測", ylabel="實際 − 預測", title="外層殘差：偏誤與異質變異的描述診斷")
    for yr, g in predictions.assign(residual=residual).groupby("year"):
        curve = g.groupby("hour").residual.mean()
        axes[1].plot(curve.index, curve.values, "o-", label=str(yr))
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set(xlabel="來源小時", ylabel="平均殘差", title="同場時段相依；不套用逐列獨立推論")
    axes[1].legend()
    save(fig, figures / "ols_residuals.png")

    latest = vif.loc[(vif.outer_year == vif.outer_year.max()) & vif.model.eq("ridge")].sort_values("vif")
    fig, ax = plt.subplots(figsize=(11, max(5, len(latest) * .28)), constrained_layout=True)
    displayed = np.minimum(latest.vif.to_numpy(), 50.)
    bars = ax.barh(latest.column, displayed, color="#71818C")
    for bar, value in zip(bars, latest.vif):
        ax.text(bar.get_width() + .1, bar.get_y() + bar.get_height()/2, "∞" if not np.isfinite(value) else f"{value:.2f}", va="center", fontsize=8)
    ax.set(xlim=(0, max(displayed) * 1.2 + 1), xlabel="VIF（顯示上限 50；原值見表格）", title=f"{vif.outer_year.max()} 外層訓練資料的候選設計診斷｜不依門檻自動刪變數")
    save(fig, figures / "training_vif.png")


def markdown_table(frame):
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    for record in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{x:.2f}" if isinstance(x, (float, np.floating)) else str(x) for x in record) + " |")
    return "\n".join(rows)


def write_report(output, metrics, folds, data_info, queue, config=None, private_output=None):
    config = config or {}
    root = Path(__file__).resolve().parents[2]
    def repo_link(filename):
        try:
            return Path(os.path.relpath(root / filename, output)).as_posix()
        except ValueError:  # Different Windows drives cannot have relative links.
            return (root / filename).as_posix()
    baseline = float(metrics.set_index("model").loc["baseline", "mae"])
    best = metrics.iloc[0]
    best_rmse = metrics.loc[metrics.rmse.idxmin()]
    best_peak = metrics.loc[metrics.peak_underprediction_mean.idxmin()]
    counts = data_info["activity_counts"]
    source_years = sorted(counts["by_year"])
    test_years = [fold["outer_year"] for fold in folds]
    year_label = "、".join(map(str, test_years))
    queue_shares = "／".join(f"{name} {value:.0%}" for name, value in queue["directions"].items())
    presentation = metrics[["model", "mae", "rmse", "peak_underprediction_mean", "peak_hour_mae"]].copy()
    presentation["model"] = presentation.model.map(LABELS)
    presentation.columns = ["方法", "MAE", "RMSE", "實際尖峰時平均低估人次", "尖峰時刻 MAE（小時）"]
    rows = []
    for fold in folds:
        rows.append({"測試年": fold["outer_year"], "訓練活動": fold["train_events"], "測試活動": fold["test_events"],
                     "OLS額外選入": "、".join(fold["base_models"]["ols"]["groups"]) or "無",
                     "Ridge λ": f'{fold["base_models"]["ridge"]["params"]["lambda"]:.3g}',
                     "OLS權重": fold["weights"]["ols"], "Ridge權重": fold["weights"]["ridge"], "XGB權重": fold["weights"]["xgboost"]})
    title = config.get("report_title", "第一版人流預測分析結果")
    interaction_note = ("本版 Ridge 在原主效果外新增 15 個車站 × 小時交互項；OLS 保持原設計作對照。"
                        f"XGBoost 不另加這組乘積欄，改由樹分裂學習交互，使用 {len(config.get('xgb_grid', []))} 組預先指定的參數搜尋。"
                        if config.get("ridge_station_hour_interaction") else "本版線性模型只有站別與小時主效果，尚未加入交互作用。")
    if config.get("d1_features"):
        interaction_note = ("Ridge 保留 15 個站別 × 時段交互項；三模型均可使用經時間閘門核對的新候選，"
                            "OLS 另在訓練內選變數。XGBoost 透過樹分裂學交互作用，"
                            f"搜尋 {len(config['xgb_grid'])} 組預先固定設定。")
    core_note = ("固定核心為同站同時段一般日基準、站別及小時" if config.get("include_historical_baseline", True)
                 else "固定核心只有站別及小時；發布時間未核實的一般日基準只列為回顧式對照，不進入三模型或融合")
    candidate_note = "、".join(config.get("candidate_groups", ["is_weekend", "day_of_year", "year_index"]))
    d1_note = ""
    if config.get("d1_features"):
        d1_note = ("\n\n新節目／天氣輸入逐筆要求可核對的來源版本在活動前一天 18:00（臺北時間）以前發布；"
                   "預報須另核對初始化、有效時間及發布時間。未知保留 NaN，補值與缺值指標只在訓練折學習；"
                   "整欄未知會被丟棄，不能把未知無人機當成沒有。歷史 OD 訓練標籤發布時間、日曆版本與已辦活動篩選仍屬回顧條件，"
                   "新增輸入通過時間檢查不代表全流程已驗證部署。\n\n"
                   "[逐筆來源與截止時間](tables/d1_feature_audit.csv)\n\n"
                   "![新增變數可用率](figures/d1_feature_coverage.png)")
    tuning_note = ("\n\n![XGBoost 訓練內調參熱圖](figures/xgb_tuning_heatmap.png)\n\n"
                   "熱圖每格是在該樹深／樹數下，其他參數取最低內層 MAE；不使用外層真值。"
                   "[選定參數](tables/xgb_selected_params.csv)／[完整熱圖數值](tables/xgb_tuning_surface.csv)。"
                   if config.get("tuning_diagnostics") else "")
    comparison_note = ("原版與本版的同日期比較見 [改版比較報告](comparison/report.md)。本次改動由第一版結果啟發，"
                       "同一批外層年份已被檢視，因此是探索式回顧比較，不能當作全新未接觸的確認性測試。"
                       if config.get("comparison_reference") else "")
    private_label = Path(private_output).relative_to(root).as_posix() if private_output and Path(private_output).is_relative_to(root) else "設定指定的私有資料夾"
    text = f"""# {title}

本次已實際執行資料核對、變數描述、Forward OLS、Ridge、XGBoost、等權平均與 MAE Stacking，並完成列車批次容量的假設情境試算。

{comparison_note}

## 1. 結論與適用範圍

- 資料為 {source_years[0]}–{source_years[-1]} 年 **{counts['held_dates']} 場**已舉辦活動、四站、18–23 時，合計 **{counts['rows']} 列**。展開後的列數不是獨立活動數。
- 原始候選共 {counts['context_dates']} 個日期；取消／延期未納入。因此分析是條件於活動已舉辦的人流比較，未建立活動取消機率模型。
- 共同比較為 **{year_label} 年 {int(best.events)} 場、{int(best.rows)} 列**；每年僅使用更早年度訓練，選變數與混合權重都在訓練資料內完成。
- 本次平均 MAE 最低為 **{LABELS[best.model]}：{best.mae:.2f} 人次／站／小時**，一般日基準為 {baseline:.2f}，相較降低 {(1-best.mae/baseline)*100:.1f}%。這是本次歷史評估結果，不能保證未來活動也最佳。
- RMSE 最低為 **{LABELS[best_rmse.model]}：{best_rmse.rmse:.2f}**；實際尖峰時低估指標最低為 **{LABELS[best_peak.model]}：{best_peak.peak_underprediction_mean:.2f} 人次**。平均準確度與尖峰風險應分開評估。
- **目前屬回顧式向前驗證，尚未核實真實 D−1 可用性。**歷史 OD 發布時間與日曆版本仍缺證據；事後天氣、無人機實況及實際節目長度沒有放入預測。
- 測試僅 {len(folds)} 個年度，不以展開後的列數作獨立樣本進行模型優劣顯著性檢定，也不提供過度精確的信賴區間。
- 這些歷史年份先前已做探索，本次是目前流程的歷史模型比較，並非完全未接觸資料的確認性測試。

## 2. 模型比較

{markdown_table(presentation)}

MAE／RMSE 先按活動平均；目前每場同為 24 列。尖峰低估先找每場每站實際最高時段，再計正向低估；尖峰時刻誤差以預測最大值時段與實際最大值時段比較，同值取較早時段。低 MAE 不等於尖峰風險最低。

![模型比較](figures/model_comparison.png)
![逐場誤差](figures/event_errors.png)

完整逐年、逐場結果見 [年度指標](tables/yearly_metrics.csv) 與 [每場指標](tables/event_metrics.csv)。較早年份訓練場次更少，2023 活動型態及平假日改變也會影響跨年泛化。

## 3. 變數、選模與權重

{core_note}；候選為 {candidate_note}。OLS 用內層時間 MAE 做 Forward Selection，Ridge／XGBoost 使用全部合格候選。連續欄位標準化只使用當次訓練資料；VIF 是設計診斷，不作機械式刪欄門檻。年度與日序可能代表活動型態差異，不是因果效果。

{d1_note}

{interaction_note}

{markdown_table(pd.DataFrame(rows))}

{tuning_note}

![權重](figures/stack_weights.png)
![VIF](figures/training_vif.png)
![殘差](figures/ols_residuals.png)

三個基礎模型的較早年度 OOF 預測經非負且總和為 1 的 MAE 線性規劃求權重；測試真值不參與求解。MAE 最佳化僅保證不劣於可行單模型的**同一份中層訓練損失**，不保證外層或未來誤差較小。

## 4. {max(test_years)} 年全部外層預測

![最近測試年全部活動四站](figures/heldout_profiles_latest.png)

圖顯示該年全部活動與四站，未只挑表現較好的案例。請同時核對尖峰高度、時刻與鄰站誤差，不能只用整體 MAE 決定調度。各站數值見 [車站指標](tables/station_metrics.csv)。逐筆來源與模型物件留在本機 `{private_label}`，Git 提供彙整圖表。

## 5. 資料探索

![年度活動數](figures/eda_activity_counts.png)
![資料完整度](figures/eda_completeness.png)
![活動層次相關](figures/eda_spearman.png)

探索圖由 `eda.py` 產生；活動層次關聯係數只描述這些場次之間的共變，實測天氣僅供事後探索。缺測、微量雨與無人機未知狀態都不能直接補成零。相關係數配對樣本數與描述統計見 [tables](tables/)，輸入資格見 [變數使用表](tables/feature_eligibility.csv)。

## 6. 如何銜接排隊與列車調度

![假設排隊情境](figures/queue_scenarios.png)

- 使用設定指定的 {queue['event_date']} {queue['station']} 外層 {queue['forecast_model']} 預測；方向比例假設為 {queue_shares}、逐分鐘均勻到達、步行 {queue['walk_delay_minutes']} 分鐘。每班剩餘容量與班距詳見情境表，不是查得的北捷營運值。
- 計算累積等待人分鐘、最大隊列、殘留人數與完整清空時的平均等待。若未清空，不冒充完整平均等待。
- **這是條件式容量情境，不是已估計的真實候車或最佳列車時刻表。**來源小時計數的物理到站意義、方向、分鐘尖峰與列車可用容量仍待核實。
- M/M/c 函式另供平行服務櫃台使用；列車用批次登車服務模型。尚未取得車隊、折返、軌道最小班距、班表與站台容量限制，不宣稱完成調度最佳化。

情境數字見 [排隊比較表](tables/queue_scenarios.csv)。補齊營運條件後，再以等待人分鐘及營運成本為目標，加上班距、車隊、折返及安全容量限制求解。

{queue.get('interpretation', '')}

## 7. 重現與文獻

- [專案入口與主要文獻]({repo_link('README.md')})
- [完整實作方法、公式、資料條件]({repo_link('process.md')})
- [資料與程式 SHA-256、環境與參數](run_manifest.json)
- [年度模型設定](folds.json)、[內層選模紀錄](tables/inner_selection.csv)、[時間切分](tables/temporal_folds.csv)

圖表與指標由程式生成，沒有以預設數字代替分析；排隊情境中的容量與方向比例則明確屬人為假設。
"""
    (output / "report.md").write_text(text, encoding="utf-8")
