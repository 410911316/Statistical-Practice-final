"""Chinese learning-curve figures and report, based only on verified V17 outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis_v1.reporting import style, save
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"result/analysis_v17"
YEARS=[2024,2025,2026]
MODEL_LABEL={"ridge":"Ridge","xgboost":"XGBoost"}
PROTOCOL_LABEL={"event_subsample":"整場抽樣平均","oldest_first":"從最早場次增加","recent_first":"從最近場次增加"}


def table(frame,columns,titles):
    rows=["| "+" | ".join(titles)+" |","| "+" | ".join(["---"]*len(columns))+" |"]
    for record in frame.to_dict("records"):
        cells=[f"{record[c]:,.2f}" if isinstance(record[c],(float,np.floating)) else str(record[c]) for c in columns]
        rows.append("| "+" | ".join(cells)+" |")
    return "\n".join(rows)


def run():
    check=json.loads((OUT/"independent_validation.json").read_text("utf8"))
    assert check["passed"]
    a=pd.read_csv(OUT/"tables/curves.csv")
    m=a.loc[a.protocol.eq("event_subsample")]
    style();figdir=OUT/"figures";figdir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(14,8.7),constrained_layout=True)
    for i,model in enumerate(["ridge","xgboost"]):
        block=m.loc[m.model.eq(model)]
        upper=block.mae_p90.max()*1.12
        for ax,year in zip(axes[i],YEARS):
            g=block.loc[block.year.eq(year)]
            for scope,label,color,ls in [("train","訓練誤差","#8c959a","--"),("validation","較晚年度驗證誤差","#397e69","-")]:
                z=g.loc[g.scope.eq(scope)].sort_values("n_events")
                ax.plot(z.n_events,z.mae_mean,marker="o",label=label,color=color,ls=ls)
                ax.fill_between(z.n_events,z.mae_p10,z.mae_p90,color=color,alpha=.16)
                if scope=="validation":
                    last=z.iloc[-1]
                    ax.annotate(f"{last.mae_mean:.1f}",(last.n_events,last.mae_mean),
                                xytext=(-5,9),textcoords="offset points",ha="right",color=color)
            ax.set(title=f"{MODEL_LABEL[model]}｜驗證 {year} 年相同 3 場",
                   xlabel="用於擬合的完整活動場數",ylabel="北門 MAE（人次／來源時段）",
                   xticks=sorted(g.n_events.unique()),ylim=(0,upper))
            ax.grid(axis="y",alpha=.18)
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc="outside lower center",ncol=2)
    fig.suptitle("三站訓練、固定參數：增加完整活動場次的學習曲線\n陰影＝10 組過去活動抽樣的第 10–90 百分位；不是信賴區間，最右端只有 1 組全資料",fontsize=13)
    save(fig,figdir/"learning_mae.png")

    fig,axes=plt.subplots(1,3,figsize=(14,4.7),constrained_layout=True)
    for ax,year in zip(axes,YEARS):
        for model,color in [("ridge","#8c959a"),("xgboost","#397e69")]:
            z=m.loc[m.year.eq(year)&m.model.eq(model)&m.scope.eq("validation")].sort_values("n_events")
            ax.plot(z.n_events,z.peak_abs_mean,marker="o",color=color,label=MODEL_LABEL[model])
            ax.fill_between(z.n_events,z.peak_abs_p10,z.peak_abs_p90,color=color,alpha=.16)
        ax.set(title=f"{year} 年相同 3 場",xlabel="用於擬合的完整活動場數",
               ylabel="實際尖峰時段平均絕對誤差",xticks=sorted(z.n_events.unique()))
        ax.set_ylim(bottom=0);ax.grid(axis="y",alpha=.18)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc="outside lower center",ncol=2)
    fig.suptitle("尖峰誤差：增加資料對兩個模型的幫助不同\n尖峰只在評估時計算；陰影為抽樣組合範圍，不是未來預測區間",fontsize=13)
    save(fig,figdir/"learning_peak.png")

    fig,axes=plt.subplots(1,3,figsize=(14,4.7),constrained_layout=True)
    for ax,year in zip(axes,YEARS):
        for protocol,color,ls in [("event_subsample","#397e69","-"),("oldest_first","#72869d","--"),("recent_first","#b38b5d","-.")]:
            z=a.loc[a.year.eq(year)&a.model.eq("xgboost")&a.scope.eq("validation")&a.protocol.eq(protocol)].sort_values("n_events")
            ax.plot(z.n_events,z.mae_mean,marker="o",color=color,ls=ls,label=PROTOCOL_LABEL[protocol])
        ax.set(title=f"{year} 年相同 3 場",xlabel="用於擬合的完整活動場數",
               ylabel="北門驗證 MAE",xticks=sorted(z.n_events.unique()),ylim=(0,1400))
        ax.grid(axis="y",alpha=.18)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc="outside lower center",ncol=3)
    fig.suptitle("XGBoost：同樣場數，選哪些過去活動也會改變誤差\n兩種時間順序均只用驗證年之前資料；未依此挑選或刪除場次",fontsize=13)
    save(fig,figdir/"subset_order.png")

    changes=[]
    for year in YEARS:
        for model in ["ridge","xgboost"]:
            g=m.loc[m.year.eq(year)&m.model.eq(model)&m.scope.eq("validation")].sort_values("n_events")
            first,last=g.iloc[0],g.iloc[-1]
            changes.append(dict(year=year,model=MODEL_LABEL[model],full_events=int(last.n_events),
                six_mae=first.mae_mean,full_mae=last.mae_mean,
                improvement=100*(1-last.mae_mean/first.mae_mean),
                six_peak=first.peak_abs_mean,full_peak=last.peak_abs_mean))
    changes=pd.DataFrame(changes)
    changes.to_csv(OUT/"tables/endpoints.csv",index=False,encoding="utf-8-sig")
    allrows=m.loc[m.model.eq("xgboost")&m.scope.eq("validation")].copy()
    gaps=[]
    for year in YEARS:
        for model in ["ridge","xgboost"]:
            g=m.loc[m.year.eq(year)&m.model.eq(model)]
            n=g.n_events.max()
            final=g.loc[g.n_events.eq(n)].set_index("scope")
            gaps.append(dict(year=year,model=MODEL_LABEL[model],events=int(n),
                train_mae=final.loc["train","mae_mean"],validation_mae=final.loc["validation","mae_mean"]))
    gaps=pd.DataFrame(gaps)
    order=a.loc[a.model.eq("xgboost")&a.scope.eq("validation")].pivot(
        index=["year","n_events"],columns="protocol",values="mae_mean").reset_index()
    report=f"""# 第十七版：資料少是否限制北門預測——整場活動學習曲線

**結果支持「可用活動少、活動組合不足是限制之一」，但不足以把全部誤差歸因於筆數。**固定相同較晚年度活動驗證，XGBoost 由 6 場子集擴大到全部可用場次，三年的平均 MAE 都下降；Ridge 則較早停滯。2026 年 XGBoost 從 15 場到 18 場的平均改善只剩約 1.2%，而特定 15 場組合也可能比全部 18 場好。

這次完成學習曲線診斷，沒有選新模型、最佳訓練場數或新增一般日訓練；不把重複使用的九場稱為全新測試。

## 1. 比較結果

以下是**三站訓練、北門評估**。6 場一欄是 10 組不同過去活動子集的平均；全資料只擬合一次。各年度用同樣三場驗證，表格不可解讀成換年度就只增加了訓練筆數。

{table(changes,["year","model","full_events","six_mae","full_mae","improvement","six_peak","full_peak"],
       ["驗證年","模型","全部可用場數","6 場平均 MAE","全資料 MAE","MAE 改善 %","6 場尖峰誤差","全資料尖峰誤差"])}

單位人次／來源時段。尖峰誤差在各場實際最高人流的來源時段計算 |實際−預測|，再對三場平均；沒有用實際尖峰挑訓練資料或特徵。

![訓練與驗證學習曲線](figures/learning_mae.png)

### 圖怎麼讀

- 橫軸是完整活動場數，不是逐站逐小時的列數；每場三站各六時段，共 18 列。
- 灰線是北門訓練誤差，綠線是固定較晚年度的北門驗證誤差。
- 綠線下降表示在這個固定驗證組、既有資料與參數下，增加活動資料平均有幫助。
- 陰影是 10 組子集結果的第 10–90 百分位，表示「抽到哪些活動」的變動；不是信賴區間、不是未來預測區間。
- 最右端用全部過去活動，只有唯一一組；陰影收成一點是子集組合消失，**不是模型已沒有不確定性**。
- XGBoost 訓練誤差明顯低於驗證誤差，與過擬合、年度型態改變、未觀測資訊等情況都相容；本圖不能分辨各因素的因果貢獻。

## 2. 實驗設計

| 固定驗證年度 | 驗證活動 | 可用過去活動 | 比較的擬合場數 |
| --- | --- | ---: | --- |
| 2024 | 同年 3 場，北門 18 列 | 12 | 6、9、12 |
| 2025 | 同年 3 場，北門 18 列 | 15 | 6、9、12、15 |
| 2026 | 同年 3 場，北門 18 列 | 18 | 6、9、12、15、18 |

1. 使用 V16 原模型 none 輸入：車站、時段、原核心群組、預定結束訊號與既有回顧一般日基準。不增加當日 lag、不改特徵定義。
2. 固定北門、大橋頭、雙連三站；移除民權西路後再建立各子集的編碼、缺值填補與標準化。
3. 每年只從更早年份的完整活動中抽樣，不把同場不同站或小時拆到兩邊。
4. 主曲線各小場數抽 10 組；每次重複使用同一個過去活動隨機排列，6 場包含於 9 場，依此類推。Ridge 與 XGBoost 使用完全相同子集。
5. 另做「從最早活動增加」與「從最近活動增加」兩種固定日期順序，作組成敏感度比較。
6. 各年度凍結 V16 none 的原 Ridge λ 及 XGBoost 超參數，不依子集或這九場重新調參；模型種子固定。Ridge alpha=nλ，以維持平均損失中的懲罰強度。
7. 全資料終點在三條曲線共用，不重複擬合成 10 份。共 **222 個模型**，分別計算北門訓練與驗證指標。
8. 原三站全資料控制精確重現：三年合併北門 MAE 為 XGBoost **491.32**、Ridge **854.66**。這是重現，不是新改善。

**場數的含義：**曲線只縮減「參數擬合使用的活動場數」。凍結的超參數與背景特徵原本利用較廣的過去資料，並未重新限制為只來自該 6／9 場。因此這是固定既有設定下的擬合資料量敏感度，不是證明只蒐集六場全部資訊即可得到同樣結果。

## 3. XGBoost 每個場數的完整結果

{table(allrows,["year","n_events","n_fits","mae_mean","mae_p10","mae_p90","rmse_mean","peak_abs_mean","cost_2_mean"],
       ["驗證年","擬合場數","組數","平均 MAE","MAE 第10百分位","MAE 第90百分位","平均 RMSE","尖峰絕對誤差","散場2:1"])}

- 2024：6 → 12 場，平均 MAE 1126.49 → 784.95，改善約 30.3%。
- 2025：6 → 15 場，746.65 → 246.20，改善約 67.0%。
- 2026：6 → 18 場，661.21 → 442.82，改善約 33.0%；但 15 → 18 場只有 448.00 → 442.82。
- 以上改善不是相對原完整模型 491.32 的改善，而是相對刻意減少訓練資料的子集。
- 不外推「再加幾場就能降到多少」；沒有足夠資料估計可靠的學習曲線極限或所需場數。

![尖峰誤差曲線](figures/learning_peak.png)

Ridge 在 2026 的尖峰誤差約停在 3400 人次附近，增加活動沒有像 XGBoost 一樣下降。這提示模型表示能力、收縮設定或缺少關鍵訊號也可能限制表現；不能把所有問題交給增加筆數解決。

## 4. 哪些活動也重要

{table(order,["year","n_events","event_subsample","oldest_first","recent_first"],
       ["驗證年","擬合場數","10組抽樣平均 MAE","從最早增加 MAE","從最近增加 MAE"])}

同樣 6 場，預測 2025 時，最早六場 MAE **1067.53**、最近六場 **289.72**，顯示年代與活動組成可能很重要。但最近六場不在所有年度、所有指標都比較好，不能用九場驗證結果挑一個最佳截斷點。

2026 從最早 15 場增加到全部 18 場，MAE **396.33 → 442.82**；從最近 15 場增加到全部 18 場也是 **395.67 → 442.82**。雖然主曲線抽樣平均下降，特定加法仍會變差。

![不同加入順序](figures/subset_order.png)

日期順序同時改變新舊程度與活動類型；同年活動也可能相依。因此這只是組成敏感度，沒有控制實驗可聲稱年代或某場活動是原因，亦不據此刪除不利場次。

## 5. 全資料的訓練與驗證差距

{table(gaps,["year","model","events","train_mae","validation_mae"],
       ["驗證年","模型","訓練活動數","北門訓練 MAE","固定較晚年度 MAE"])}

訓練誤差描述已看過資料的擬合，不能取代預測誤差；訓練／驗證差距不單獨證明過擬合，也可能包含跨年分布差異。

## 6. 對下一步的意義

- 這輪支持擴充有代表性的活動案例作為候選方向；單純重複、插值或模擬出更多列，不會增加真實活动的資訊。
- 在新增足夠尖峰案例之前，將每場最高一筆抽出獨立建複雜模型會更缺資料；可將「共用模型＋簡單散場修正」保留為後續比較方案。
- 一般日訓練可補平常流量規律，但不能替代大型活動的尖峰資料；本輪未實作一般日背景模型。
- 本輪未根據驗證曲線挑最佳場數、時間窗口或重選參數，沒有更新排隊或列車調度；若日後將資料窗口作選擇項，須在更早年份內選，再找未見活動驗證。

## 7. 可得性與證據限制

- 原快照的歷史 OD 發布時點仍未完全核實，這輪不是已驗證可部署的 D−1 成效。
- 沒有加入活動當日 lag，沒有把當年真值放入訓練或選子集；來源時段的鐘點定義仍沿用待核實限制。
- 同年三場固定驗證，三年合计九場已反覆用於開發；10 次抽樣也没有新增驗證活動。
- 固定超參數來自原完整過去資料的設定，小子集結果有繼承的調參資訊；本輪不估計重新調參的最佳小樣本能力。
- 背景特徵仍按原歷史來源計算；未把它們也縮成同一活動子集。
- 子集陰影只描述既有活動組合，不包含未來活動、測量誤差或尚未蒐集的活動型態的不確定性。

## 8. 檔案、重現與檢核

~~~powershell
.\\.venv\\Scripts\\python.exe -X utf8 src/run_analysis_v17.py
.\\.venv\\Scripts\\python.exe -X utf8 src/verify_analysis_v17.py
.\\.venv\\Scripts\\python.exe -X utf8 src/report_analysis_v17.py
.\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s tests -p 'test_analysis_v*.py'
~~~

依賴 V15 輸入與 V16 凍結設定、參考預測及前代來源快照；缺檔或雜湊變動會停止。逐列預測放在忽略追蹤的 .local/analysis_v17；不另保存 222 份模型二進位檔，以設定、場次清單、固定種子重現。

- [設定](../../configs/analysis_v17.json)／[全部場次與参数](fits.json)／[逐次指標](tables/scores.csv)／[曲線摘要](tables/curves.csv)／[端點比較](tables/endpoints.csv)。
- [獨立檢核](independent_validation.json)：{check['source_hashes']} 個來源雜湊、222 模型重新擬合、444 列原始指標、144 列曲線摘要及 6 個全資料控制通過。
- 最大預測重算差 {check['max_prediction_difference']:.2e}，指標差 {check['max_metric_difference']:.2e}，前處理差 {check['max_preprocessing_difference']:.2e}。
- 另核對日期只來自過去年份、完整事件、巢狀子集、兩模型同樣資料、固定驗證場次、固定超參數與訓練子集標準化。
- 新增 12 項測試涵蓋子集抽取不依標籤、未来／排除場次的值擾動不影響擬合、全資料終點不重複計數及百分位手算。
- 詳細公式見 [process.md 第 28 節](../../process.md#28-第十七版整場活動學習曲線)。

## 9. 方法依據

- [scikit-learn，Learning curves](https://scikit-learn.org/stable/modules/learning_curve.html)：比較不同訓練樣本數的訓練與驗證誤差。本案改為以整場活動抽取，固定較晚年度驗證，未直接套用逐列隨機切分。
- [Hyndman & Athanasopoulos，Forecasting: Principles and Practice，時間向前驗證](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation)：驗證時間須晚於訓練。整年隔離、6／9／12 等場數及 10 次重複為本案設計，不是文獻建議的唯一或最佳設定。
- [Cawley & Talbot（2010），JMLR 11:2079–2107](https://jmlr.org/papers/v11/cawley10a.html)：模型選擇程序本身也可能過擬合。本輪保留完整曲線，不利用已看過的驗證結果宣稱最佳訓練場數。
"""
    for before,after in [("活动","活動"),("合计","合計"),("没有","沒有"),("参数","參數"),("未来","未來")]:
        report=report.replace(before,after)
    (OUT/"report.md").write_text(report,encoding="utf8")
    print(json.dumps({"report":str(OUT/"report.md"),"figures":3},ensure_ascii=False))


if __name__=="__main__":run()
