"""Chinese comparison of bounded egress correction; no post-hoc winner selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis_v1.reporting import style, save
from report_analysis_v17 import table
from analysis_v13.data import sha

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"result/analysis_v18"
MODEL={"ridge":"Ridge","xgboost":"XGBoost"}
METHOD={"base":"三站共用模型","correction_1":"共用＋1:1 修正","correction_2":"共用＋2:1 修正"}


def run():
    check=json.loads((OUT/"independent_validation.json").read_text("utf8"))
    manifest=json.loads((OUT/"manifest.json").read_text("utf8"))
    assert check["passed"]
    for path,digest in manifest["output_hashes"].items():assert sha(ROOT/path)==digest,path
    s=pd.read_csv(OUT/"tables/summary.csv")
    annual=pd.read_csv(OUT/"tables/annual.csv")
    events=pd.read_csv(OUT/"tables/events.csv")
    p=pd.read_csv(ROOT/".local/analysis_v18/predictions.csv",parse_dates=["event_date"])
    corrections=json.loads((OUT/"corrections.json").read_text("utf8"))
    style();figdir=OUT/"figures";figdir.mkdir(exist_ok=True)
    colors=["#72869d","#beb6a6","#397e69"]
    xgb=s.loc[s.model.eq("xgboost")].set_index("method").loc[list(METHOD)]
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,key,title in zip(axes.flat,["mae","rmse","peak_abs","cost_2"],
                            ["全部六時段 MAE","全部六時段 RMSE","實際尖峰時段絕對誤差","事前散場窗口：低估 2 倍＋高估"]):
        b=ax.bar(range(3),xgb[key],color=colors,width=.6)
        ax.bar_label(b,fmt="%.1f",padding=4)
        ax.set(xticks=range(3),xticklabels=["原共用","＋1:1 修正","＋2:1 修正"],title=title,
               ylabel="人次／來源時段",ylim=(0,xgb[key].max()*1.2))
        ax.grid(axis="y",alpha=.16);ax.set_axisbelow(True)
    fig.suptitle("XGBoost 共用模型＋少量散場修正｜同樣九場北門，越低越好\n2:1 是主要實驗；1:1 是敏感度比較。散場損失下降，但尖峰誤差增加。",fontsize=13)
    save(fig,figdir/"comparison.png")

    fig,axes=plt.subplots(3,3,figsize=(14,10.4),constrained_layout=True,sharey=True)
    for ax,(date,g) in zip(axes.flat,p.loc[p.model.eq("xgboost")].groupby("event_date")):
        g=g.sort_values("hour")
        q=g[["planned_end_0_60_beimen","planned_end_60_120_beimen"]].sum(axis=1)
        for hour,weight in zip(g.hour,q):
            if weight>0:ax.axvspan(hour-.45,hour+.45,color="#d8c09b",alpha=.14+.18*weight,lw=0)
        for col,label,color,ls in [("y","實際","#282d29","-"),("base","三站共用","#72869d","--"),
                                   ("correction_2","＋2:1 修正","#397e69","-")]:
            ax.plot(g.hour,g[col],marker="o",ms=4,color=color,ls=ls,label=label)
        e=events.loc[events.model.eq("xgboost")&events.event_date.eq(str(date.date()))].set_index("method")
        ax.set(title=f"{date.date()}｜尖峰誤差 {e.loc['base','peak_abs']:.0f} → {e.loc['correction_2','peak_abs']:.0f}",
               xlabel="原始資料來源時段",ylabel="北門人次",xticks=range(18,24),ylim=(0,12500))
        ax.grid(axis="y",alpha=.14)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3)
    fig.suptitle("逐場比較：修正並非每場有利\n淡褐色＝與事前預定散場窗口重疊的來源時段；不代表量到的分鐘到站型態",fontsize=13)
    save(fig,figdir/"beimen_profiles.png")

    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    for ax,model in zip(axes,["ridge","xgboost"]):
        records=[r for r in corrections if r["model"]==model and r["under_ratio"]==2]
        for j,label,color in [(0,"散場後 0–60 分鐘","#397e69"),(1,"散場後 60–120 分鐘","#b38b5d")]:
            values=np.array([r["theta"][j] for r in records])*100
            b=ax.bar(np.arange(3)+(j-.5)*.3,values,.28,color=color,label=label)
            ax.bar_label(b,fmt="%+.1f%%",padding=3,fontsize=9)
        for y in [-25,25]:ax.axhline(y,color="#999999",ls=":",lw=1)
        ax.axhline(0,color="#777777",lw=.8)
        ax.set(xticks=range(3),xticklabels=["2024\n過去 6 場","2025\n過去 9 場","2026\n過去 12 場"],
               ylabel="修正係數（%）",title=MODEL[model],ylim=(-31,34))
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc="outside lower center",ncol=2)
    fig.suptitle("只從過去時間留出誤差學到的兩個修正係數（2:1）\n實際調整＝係數 × 該來源時段的窗口重疊比例；虛線為本案設定的 ±25% 上限",fontsize=12)
    save(fig,figdir/"correction_coefficients.png")

    display=s.copy();display.model=display.model.map(MODEL);display.method=display.method.map(METHOD)
    ann=annual.loc[annual.model.eq("xgboost") & annual.method.ne("correction_1")].copy()
    ann.method=ann.method.map(METHOD)
    coef=pd.DataFrame([dict(model=MODEL[r["model"]],year=r["cutoff_year"],events=len(r["eligible_dates"]),
                      first=100*r["theta"][0],second=100*r["theta"][1])
                      for r in corrections if r["under_ratio"]==2])
    report=f"""# 第十八版：三站共用模型＋少量散場修正

**已實作，但目前是取捨，不是全面改善。**三站 XGBoost 加 2:1 修正後，北門 MAE 491.32 → 492.99，RMSE 760.23 → 735.10；預定散場 2:1 誤差降低 6.4%，但實際尖峰絕對誤差增加 30.2%。保留原模型作主要比較基準，新版作候選情境，不依九場反覆結果宣稱已找到最佳模型。

## 1. 做了什麼

1. 北門、大橋頭、雙連繼續共用同一個 Ridge 或 XGBoost；保留車站、時段、車站×時段、週末、季節／年索引、預定結束訊號及原一般日基準，沒有新增當日 lag。
2. 每個年度先用更早年份選參數、擬合共用模型。產生 2021–2026 年逐年向前的預測；不用模型已經看過的訓練誤差。
3. 預測 2024／2025／2026 時，修正層分別只用過去 6／9／12 場「已知預定結束時間」活動的留出誤差。未知公告者不參與修正擬合。
4. 修正層只有兩個係數，分別對應北門預定結束後 0–60、60–120 分鐘；係數可正可負，總預測最多調整 ±25%。窗口外、其他站、結束時間未知者保留共用模型預測。
5. 用 L1 懲罰讓修正趨近零，低估罰 2 倍為主要實驗；另完整列出低估／高估等價的 1:1 比較。沒有再用當年結果挑參數。

這是**共用預測加受限制的殘差修正**，不是把真實最高小時切出來訓練另一棵樹，也不是前面三模型融合的權重重算。

## 2. 全部比較

同樣 2024–2026 九場北門、每場六個來源時段，54 列。每場等權，單位人次／來源時段，越低越好。

{table(display,["model","method","mae","rmse","peak_abs","peak_under","cost_2"],
["模型","版本","MAE","RMSE","尖峰絕對誤差","尖峰正向低估","散場 2:1 誤差"])}

![XGBoost 比較](figures/comparison.png)

- **MAE 幾乎沒變：**2:1 修正後增加 0.34%；不是全時段精準度改善。
- **RMSE 下降 3.3%：**全部六時段的大誤差略減，但不代表每場最高小時更準。
- **窗口低估減少、高估增加：**XGBoost 預定散場平均正向低估 384.61 → 301.12（−21.7%），高估 415.75 → 506.91（＋21.9%）。加權後的 2:1 損失才下降 6.4%。這是代理損失，不是實際列車調度成本。
- **真正最高小時沒有解決：**尖峰正向低估僅 469.07 → 461.52（−1.6%），尖峰絕對誤差反而 641.23 → 834.83（＋30.2%）。窗口低估和實際峰值低估必須分開看。
- **1:1 也沒有更好：**XGBoost MAE 545.62、尖峰誤差 1259.22；歷史誤差讓第二個窗口的修正偏向下調。
- Ridge 的 2:1 MAE 854.66 → 854.17，改善不足一人次；仍遠遜於 XGBoost。

## 3. 為什麼一些場次變差

{table(ann,["year","method","mae","rmse","peak_abs","cost_2"],
["驗證年","版本","MAE","RMSE","尖峰絕對誤差","散場 2:1 誤差"])}

![九場逐時圖](figures/beimen_profiles.png)

- **2024/8/10：**最高小時實際 11,101 人次，共用模型 9,273.80，修正後 8,450.39；低估 1,827.20 → 2,650.61。只看過去六場學到的第二段負修正，在這場大型活動沒有延續。
- **2026/8/15：**最高小時實際 10,904，共用模型 10,038.63，修正後 10,362.57；低估 865.37 → 541.43，有改善。
- **2026/8/5：**實際 4,642，共用模型已高估為 5,618.85，修正後 5,691.38，反而更高。
- 同一年度只用兩個係數，無法充分分辨一般場與大型場的散場速度、規模。這是與結果相容的解釋，未證明原因完全由該因素造成。

## 4. 學到了多少修正

{table(coef,["model","year","events","first","second"],
["模型","預測年","過去有效場數","第一段係數 %","第二段係數 %"])}

![修正係數](figures/correction_coefficients.png)

例如係數 +10%、窗口重疊只有 1/3 小時，該來源時段的預測只加約 3.33%，不是整小時都加 10%。2024 XGBoost 第一段碰到 +25% 上限，但**不能因此直接放寬上限**；當年尖峰已變差。上限、懲罰與至少六場皆為本案預設，不是論文保證的最佳值。

## 5. 時間隔離與資料性質

- 歷史 OOF 的選參也向前：預測 2023 的共用模型只能用 2019–2022 的封存候選驗證結果選參數，不能拿 2024 已選設定回頭產生「假 OOF」。
- 選參政策繼承 V11 四站核心候選庫；正式擬合則為目前三站＋原一般日基準。這輪沒有重新搜尋三站模型超參數。2024–2026 六個基準模型完整重現 V16。
- 兩個修正係數從過去年份的北門時間留出殘差學習；預測年度真值只用來評估。未知預定結束時間者不假設等於實際散場。
- **原始 D−1 可得性限制保留：**舊 OD 歷史發布時間尚未全部核實，事前公告也沒有完整歷史快照；沒有新增當日或事後天氣，但不能把這個原資料實驗称為已驗證可部署的 D−1 成果。
- 來源時段的精確鐘點定義仍待核實。窗口重疊只用作特徵／評分权重，沒有把小時人次按比例當成實測分鐘到站人流。
- 九場已反覆用於開發，不能把本次差異當成新未見資料的改善證據；沒有推論顯著性、信賴區間或自動部署。

## 6. 接下來採用什麼

- **主基準保留原三站 XGBoost。**2:1 修正可作降低散場低估的比較方案，但不宣稱尖峰更準。
- 要用於調度，還要檢查每場低估／高估的後果、分鐘到站形狀、行走延遲、方向比例、列車剩餘容量與班距限制；本次沒有以這個小時模型直接產生可執行班表。
- 若下一輪要調修正幅度或區分大型場，應先固定少量候選、使用過去時間驗證；另留新活動驗證。不能看到某場變差就單獨改其係數。

## 7. 文獻依據與本案延伸

- [Montero-Manso & Hyndman（2021）, Principles and Algorithms for Forecasting Groups of Time Series: Locality and Globality](https://arxiv.org/abs/2008.00444)：支持多序列共用模型的研究方向；不保證本案三站一定勝過單站或四站。
- [Breiman（1996）, Stacked Regressions](https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf)：借用「使用留出預測訓練第二層」的原則。原文以非負最小平方組合預測器；本案為時間向前、受限非對稱殘差修正，不是照原公式重現 stacking。
- [Gneiting（2011）, Making and Evaluating Point Forecasts](https://arxiv.org/abs/0912.0902)：評分損失要與決策目標一致。低估 2、高估 1 的絕對損失對應分位數目標 2/3；受限模型與小樣本不保證達到精確分位數校準。
- ±25%、L1=0.1、六場門檻、兩個窗口均為本案設計，沒有宣稱為上述文獻的定理或最佳設定。

## 8. 檔案與重現

~~~powershell
.\\.venv\\Scripts\\python.exe -X utf8 src/run_analysis_v18.py
.\\.venv\\Scripts\\python.exe -X utf8 src/verify_analysis_v18.py
.\\.venv\\Scripts\\python.exe -X utf8 src/report_analysis_v18.py
.\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s tests -p 'test_analysis_v*.py'
~~~

[設定](../../configs/analysis_v18.json)／[實作公式](../../process.md#29-第十八版共用模型少量散場修正)／[全部指標](tables/summary.csv)／[各年度](tables/annual.csv)／[逐場](tables/events.csv)／[係數](corrections.json)／[獨立驗證](independent_validation.json)。

- 12 次共用模型擬合、12 次修正擬合；另用不同 LP 表示方式獨立檢查 12 個最適目標值。
- 核對 80 個來源雜湊、7 個結果雜湊、12 個過去選參程序、78 列公開指標；最大預測差 {check['max_prediction_difference']:.2e}、LP 目標差 {check['max_lp_objective_difference']:.2e}。
- 新增 20 項測試，全系列 347 項通過。逐列資料在 .local/analysis_v18/；公開結果在 result/analysis_v18/，沒有另存 12 份大型模型二進位檔。
"""
    report=report.replace("称為","稱為").replace("权重","權重")
    (OUT/"report.md").write_text(report,encoding="utf8")
    print("Wrote report and 3 figures")


if __name__=="__main__":run()
