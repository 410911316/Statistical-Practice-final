"""Build Chinese station-ablation report and plots from verified model outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis_v1.reporting import style, save
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/analysis_v16"
LABELS = {
    "none": "原模型", "origin_w1": "進站 1 時段", "origin_w3": "進站 3 時段",
    "origin_w5": "進站 5 時段", "both_w1": "進站＋目的站 1 時段",
    "both_w3": "進站＋目的站 3 時段", "both_w5": "進站＋目的站 5 時段",
}
ORDER = list(LABELS)
COHORTS = {"four": "四站訓練", "three": "三站訓練"}
COLORS = {"four": "#7b858b", "three": "#3b7d69"}
STATIONS = ["北門", "大橋頭站", "雙連"]


def table(frame, columns, titles):
    rows = ["| " + " | ".join(titles) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    for record in frame.to_dict("records"):
        cells = [f"{record[c]:,.2f}" if isinstance(record[c], (float, np.floating)) else str(record[c])
                 for c in columns]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def run():
    check = json.loads((OUT/"independent_validation.json").read_text("utf8"))
    assert check["passed"], "Run independent validation before reporting"
    summary = pd.read_csv(OUT/"tables/summary.csv")
    annual = pd.read_csv(OUT/"tables/annual.csv")
    station = pd.read_csv(OUT/"tables/stations.csv")
    common = pd.read_csv(OUT/"tables/common_three.csv")
    later = pd.read_csv(OUT/"tables/conservative.csv")
    paired = pd.read_csv(OUT/"tables/paired_comparison.csv")
    pred = pd.read_csv(ROOT/".local/analysis_v16/predictions.csv", parse_dates=["event_date"])
    north = pred.loc[pred.station.eq("北門")]
    for frame in (summary, annual, station, common, later, paired):
        frame["input"] = frame.variant.map(LABELS)
        if "cohort" in frame:
            frame["cohort_label"] = frame.cohort.map(COHORTS)
        if "model" in frame:
            frame["model_label"] = frame.model.map({"ridge": "Ridge", "xgboost": "XGBoost"})
    base = summary.loc[summary.variant.eq("none")].copy()
    xb = base.loc[base.model.eq("xgboost")].set_index("cohort")
    style()
    figures = OUT/"figures"
    figures.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, metric, title in zip(
        axes.flat, ["mae", "rmse", "peak_abs", "cost_2"],
        ["全六時段 MAE", "全六時段 RMSE", "實際尖峰時段平均絕對誤差", "散場低估 2 倍／高估 1 倍誤差"],
    ):
        bars = ax.bar(range(2), xb.loc[list(COHORTS), metric],
                      color=list(COLORS.values()), width=.58)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=12)
        ax.set_xticks(range(2), list(COHORTS.values()))
        ax.set_ylim(0, xb[metric].max()*1.22)
        ax.set(title=title, ylabel="人次／來源時段（越低越好）")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("移除民權西路：北門整體 MAE 變差，尖峰誤差降低\n原 XGBoost、相同九場 54 筆、固定參數；回顧比較", fontsize=14)
    save(fig, figures/"comparison.png")

    short = ["原模型", "進站\n1 時段", "進站\n3 時段", "進站\n5 時段",
             "兩方向\n1 時段", "兩方向\n3 時段", "兩方向\n5 時段"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.3), constrained_layout=True)
    for ax, model in zip(axes, ["ridge", "xgboost"]):
        block = summary.loc[summary.model.eq(model)]
        for i, cohort in enumerate(COHORTS):
            values = block.loc[block.cohort.eq(cohort)].set_index("variant").loc[ORDER, "mae"]
            bars = ax.bar(np.arange(7)+(i-.5)*.35, values, width=.33,
                          color=COLORS[cohort], label=COHORTS[cohort])
            ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=10)
        ax.set_xticks(range(7), short)
        ax.set_ylim(0, block.mae.max()*1.19)
        ax.set_title({"ridge": "Ridge", "xgboost": "XGBoost"}[model])
        ax.set_ylabel("北門 MAE（人次／來源時段）")
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2)
    fig.suptitle("所有七種輸入完整比較｜相同九場、未重新選參數\n前段人流版本假設活動當天能取得資料，並非 D−1 輸入", fontsize=14)
    save(fig, figures/"lag_comparison.png")

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for ax, (date, group) in zip(axes.flat, north.groupby("event_date")):
        g = group.sort_values("hour")
        for name, label, color, ls in [
            ("y", "實際", "#222222", "-"),
            ("four_xgboost_none", "四站訓練", COLORS["four"], "--"),
            ("three_xgboost_none", "三站訓練", COLORS["three"], "-."),
        ]:
            ax.plot(g.hour, g[name], label=label, color=color, ls=ls, marker="o", ms=3)
        ax.set(title=str(date.date()), xticks=list(range(18, 24)),
               xlabel="來源時段（鐘點定義待核實）", ylabel="北門進站人次")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.15)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3)
    fig.suptitle("原 XGBoost：四站與三站訓練的九場北門預測\n各年度僅用更早年份訓練；三站並非每場皆改善", fontsize=14)
    save(fig, figures/"beimen_profiles.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for ax, metric in zip(axes, ["mae", "rmse"]):
        g = station.loc[station.model.eq("xgboost") & station.variant.eq("none")]
        for i, cohort in enumerate(COHORTS):
            values = g.loc[g.cohort.eq(cohort)].set_index("station").loc[STATIONS, metric]
            bars = ax.bar(np.arange(3)+(i-.5)*.34, values, width=.32,
                          color=COLORS[cohort], label=COHORTS[cohort])
            ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set_xticks(range(3), ["北門", "大橋頭", "雙連"])
        ax.set_ylim(0, g[metric].max()*1.22)
        ax.set(title=metric.upper(), ylabel="人次／來源時段")
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2)
    fig.suptitle("逐站比較｜固定原 XGBoost 參數、同九場各 54 筆\n比較共同三站，避免改變評估範圍造成假改善", fontsize=14)
    save(fig, figures/"stations.png")

    comparisons = []
    for model in ["xgboost", "ridge"]:
        z = paired.loc[paired.model.eq(model)].set_index("variant").loc[ORDER].reset_index()
        comparisons.append(table(z, ["input", "mae_four", "mae_three", "mae_improvement_pct"],
                                 ["輸入", "四站 MAE", "三站 MAE", "三站改善率 %"]))
    annual_x = annual.loc[annual.model.eq("xgboost") & annual.variant.eq("none")]
    station_x = station.loc[station.model.eq("xgboost") & station.variant.eq("none")]
    later_x = later.loc[later.model.eq("xgboost") & later.variant.isin(["none", "origin_w1"])]
    peaks = north.loc[north.groupby("event_date").y.idxmax(),
                      ["event_date", "hour", "y", "four_xgboost_none", "three_xgboost_none"]].copy()
    peaks["event_date"] = peaks.event_date.dt.strftime("%Y-%m-%d")
    c = common.loc[common.model.eq("xgboost") & common.variant.eq("none")].set_index("cohort")
    report = f"""# 第十六版：減少訓練車站能否改善北門預測

**已完成北門、大橋頭、雙連三站訓練，與原四站訓練比較。原 XGBoost 的北門 MAE 473.91 → 491.32，惡化 3.7%；實際尖峰時段絕對誤差 749.02 → 641.23，降低 14.4%。沒有證據可直接說「站太多就是不準的原因」。**

本輪屬固定參數的探索性回顧比較；九場已反覆用於開發，不是全新未見測試，也未自動更換網站模型。

## 1. 只改什麼

- 四站：北門、大橋頭、雙連、民權西路；三站：**北門、大橋頭、雙連**。原始資料車站名稱是「大橋頭站」。
- 在標準化、車站編碼、建模之前就移除民權西路的訓練列；三站版不會保留該站虛擬變數。
- 沿用 V15 的同一資料快照、候選輸入與各年度固定參數。模型是 **Ridge、XGBoost**；各跑原模型、進站前 1／3／5 時段、再加目的站前 1／3／5 時段，共七種輸入。
- 沒有重新調參、early stopping、散場訓練加權或新增融合。這是接續上一輪的車站比較，不是將歷次所有方法重新比賽。
- 全資料 21 場：四站 504 列、三站 378 列。2024／2025／2026 分別以更早的 12／15／18 場訓練。
- 主要評估同樣九場北門、來源時段 18–23，共 54 列；次要評估同樣九場共同三站、162 列。不拿四站平均與三站平均直接比較。
- 四站控制的 42 組年度／模型／输入預測全部重現 V15。兩組總共完成 **84 個擬合**。
- Ridge 固定的是平均平方損失的懲罰 λ；因三站樣本量減少，實際 sklearn alpha=n×λ 隨 n 改變。這讓每筆資料平均損失對懲罰的相對強度相同。

## 2. 北門原模型：整体與尖峰有取捨

{table(base, ["model_label", "cohort_label", "mae", "rmse", "peak_abs", "peak_under", "cost_2"], ["模型", "訓練範圍", "MAE", "RMSE", "尖峰絕對誤差", "尖峰正向低估", "散場 2:1 誤差"])}

- XGBoost：全六時段 MAE 惡化 **3.7%**，RMSE 略降 **0.9%**；尖峰絕對誤差降低 **14.4%**，尖峰正向低估降低 **15.7%**，散場 2:1 誤差降低 **3.8%**。
- Ridge：MAE 845.44 → 854.66，也略差；這次沒有因少一站而全面改善。
- **尖峰絕對誤差**：先在每場實際人流找最高的來源時段，再計算該時段 |實際−預測|，最後九場平均；不是兩條曲線各自最高值之差。
- **尖峰正向低估**：同一實際尖峰時段的 max(實際−預測,0) 再平均；高估以 0 計，不與低估互相抵銷。
- **散場 2:1**：沿用事前預定結束後兩小時與來源時段的重疊權重，各場等權；計算 2×低估＋高估。若事前結束未知，按原協定改用全六時段。它是加權預測誤差，不是候車人數、分鐘或金錢。

![北門四項指標](figures/comparison.png)

## 3. 按年與逐場確認

{table(annual_x, ["year", "cohort_label", "mae", "rmse", "peak_abs"], ["年度", "訓練範圍", "MAE", "RMSE", "尖峰絕對誤差"])}

三站版只在 2025 年的年度 MAE 較好；2024、2026 較差。尖峰誤差在 2024／2025 改善，2026 反而增加，不能用整體平均推論每场皆改善。

{table(peaks, ["event_date", "hour", "y", "four_xgboost_none", "three_xgboost_none"], ["活動日期", "實際尖峰來源時段", "實際", "四站預測", "三站預測"])}

![九場北門曲線](figures/beimen_profiles.png)

## 4. 前段人流版本也完整比較

以下改善率＝100×(1−三站 MAE／四站 MAE)；正值較好，負值較差。兩組每一列都使用同一輸入版本，沒有挑最好的窗口作結論。

### XGBoost

{comparisons[0]}

### Ridge

{comparisons[1]}

XGBoost 的「進站＋目的站」三種窗口在三站版都比各自四站版好；例如 1 時段 **643.45 → 577.47**，但仍比不加前段人流的四站 **473.91** 差。只加進站前段的三種窗口反而更差。新資訊與訓練範圍的組合有影響，但沒有穩定勝出的新版本。

![全部七種輸入](figures/lag_comparison.png)

## 5. 其他兩站

{table(station_x, ["station", "cohort_label", "mae", "rmse"], ["車站", "訓練範圍", "MAE", "RMSE"])}

大橋頭 MAE 降低，北門與雙連上升。共同三站同一批 162 列的總 MAE **{c.loc['four', 'mae']:.2f} → {c.loc['three', 'mae']:.2f}**，RMSE **{c.loc['four', 'rmse']:.2f} → {c.loc['three', 'rmse']:.2f}**。所以沒有「三站整體更準」的結果。

![逐站結果](figures/stations.png)

## 6. 來源時段 19–23 的敏感度檢查

{table(later_x, ["input", "cohort_label", "mae", "rmse"], ["XGBoost 輸入", "訓練範圍", "MAE", "RMSE"])}

此處只看九場 × 五個來源時段＝45 列，兩組都同樣排除來源時段 18。不與主要六時段 MAE 直接排名；全部七種輸入保存在 tables/conservative.csv。

## 7. 可以如何解讀

1. **尚不能認定是車站太多造成誤差。**移除一站同時減少訓練列、改變車站與時段的設計矩陣、改變樹分裂；本輪無法把變化歸因於某一站「污染」。
2. **全時段與散場目標不同。**三站原 XGBoost 的尖峰與散場平均誤差有改善，但一般 MAE 變差，2026 尖峰也沒有改善；不據此宣布新最佳模型。
3. **這次固定參數是為了比較車站範圍。**不能因此認定三站已調到最佳，也沒有藉外層九場挑站、挑窗口、挑模型部署。
4. 若後續把車站範圍、窗口、調參加入模型選擇，必須在每次預測的較早年份內決定，之後仍需要未見活動檢驗；這九場的反覆比較只能作探索。
5. 排隊與調度尚需方向、分鐘到達分布、班距與剩餘容量；本輪的每來源時段誤差不能直接轉成已驗證候車分鐘。

## 8. 可得性界線

- 不新增事後天氣、聲量或預測目標時段真值；預定結束訊號沿用原來源稽核。
- 原 474 版的一般日與歷史 OD 發布時點尚未完全核實，因此 none 版本也仍是回顧分析，不能當成已驗證的 D−1 部署表現。
- 含前段人流的版本假設名義 18:00 能取得來源時段 13–17；這是活動當天更新，並非前一天預測。公開月更新 OD 未證明可即時供應。
- 原 OD 時段按進站或出站鐘點，以及區間起／終點尚待核實。目的站數量不直接寫成實際刷卡出站時刻。
- 三站各用自己的前段人流，預測北門時不把另外兩站同場未來真值當輸入。
- 列數不是獨立活動數；不作逐列獨立顯著性檢定。

## 9. 重現與檢核

~~~powershell
.\\.venv\\Scripts\\python.exe -X utf8 src/run_analysis_v16.py
.\\.venv\\Scripts\\python.exe -X utf8 src/verify_analysis_v16.py
.\\.venv\\Scripts\\python.exe -X utf8 src/report_analysis_v16.py
.\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s tests -p 'test_analysis_v*.py'
~~~

需先具備 V15 及其前代本機快照、模型設定與封存來源；來源雜湊變更即停止，不下載或捏造替代資料。本機輸入、逐列預測與 84 個模型保留在 .local；公開結果在根目錄 result/analysis_v16。

- [獨立檢核](independent_validation.json)：43 個來源雜湊、84 模型重載、504 列指標、42 組四站控制重現全部通過。
- 最大重载預測差 {check['max_prediction_error']:.2e}；控制差 {check['max_control_error']:.2e}；指標差 {check['max_metric_error']:.2e}。
- 檢查移除車站沒有進入編碼、每年只用過去訓練、固定參數相同，以及 Ridge 懲罰與訓練標準化正確。
- 新增 12 項單元測試涵蓋面板完整性、訓練年隔離、被排除站／未來標籤擾動不影響模型、只用訓練標準化、Ridge nλ 與手算指標。
- [設定](../../configs/analysis_v16.json)／[全部參數](fits.json)／[北門指標](tables/summary.csv)／[逐場指標](tables/events.csv)／[同範圍三站指標](tables/common_three.csv)／[配對差異](tables/paired_comparison.csv)。
- 資料與方法沿用[第十五版](../analysis_v15/report.md)，實作公式見 [process.md 第 27 節](../../process.md#27-第十六版三站與四站訓練的配對比較)。
"""
    report = report.replace("输入", "輸入").replace("整体", "整體").replace("每场", "每場").replace("重载", "重載")
    (OUT/"report.md").write_text(report, encoding="utf8")
    print(json.dumps({"report": str(OUT/"report.md"), "figures": len(list(figures.glob("*.png")))}, ensure_ascii=False))


if __name__ == "__main__":
    run()
