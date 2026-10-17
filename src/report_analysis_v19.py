"""Chinese report and figures for the verified two-station experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis_v1.reporting import style, save
from report_analysis_v16 import table
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/analysis_v19'
COHORTS={'four':'四站','three':'三站','two':'北門＋大橋頭'}
COLORS={'four':'#6b7884','three':'#49806c','two':'#bc713f'}
MODELS={'xgboost_w4':'XGBoost 散場加權 ×4','xgboost_w1':'XGBoost 未加權','ridge':'Ridge'}


def run():
    audit=json.loads((OUT/'independent_validation.json').read_text('utf8'))
    assert audit['passed']
    summary=pd.read_csv(OUT/'tables/summary.csv')
    annual=pd.read_csv(OUT/'tables/annual.csv')
    stations=pd.read_csv(OUT/'tables/stations.csv')
    corr=pd.read_csv(OUT/'tables/past_event_correlations.csv')
    peaks=pd.read_csv(OUT/'tables/actual_peak_predictions.csv')
    pred=pd.read_csv(ROOT/'.local/analysis_v19/predictions.csv',parse_dates=['event_date'])
    north=pred.loc[pred.station.eq('北門')]
    for f in (summary,annual,stations):
        f['範圍']=f.cohort.map(COHORTS)
        f['模型']=f.model.map(MODELS)
    main=summary.loc[summary.model.eq('xgboost_w4')]
    x=main.set_index('cohort')
    mae_change=100*(x.loc['two','mae']/x.loc['four','mae']-1)
    peak_gain=100*(1-x.loc['two','peak_abs']/x.loc['four','peak_abs'])
    style();(OUT/'figures').mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,key,label in zip(axes.flat,['mae','rmse','peak_abs','cost_2'],
            ['全六時段 MAE','全六時段 RMSE','實際尖峰時段平均絕對誤差','預定散場 2:1 誤差']):
        values=x.loc[list(COHORTS),key]
        bars=ax.bar(range(3),values,color=list(COLORS.values()),width=.6)
        ax.bar_label(bars,fmt='%.1f',padding=4,fontsize=11)
        ax.set_xticks(range(3),list(COHORTS.values()))
        ax.set_ylim(0,values.max()*1.2)
        ax.set(title=label,ylabel='人次／來源時段；越低越好')
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
    fig.suptitle('只用北門＋大橋頭：尖峰誤差較低，全時段 MAE 反而上升\n同九場 54 筆、相同年度參數與散場加權規則；回顧比較',fontsize=14)
    save(fig,OUT/'figures/comparison.png')

    fig,axes=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    for ax,(date,g) in zip(axes.flat,north.groupby('event_date')):
        g=g.sort_values('hour')
        ax.plot(g.hour,g.y,color='#222222',label='實際',marker='o',ms=3,lw=2)
        for cohort,ls in zip(COHORTS,['--',':','-.']):
            ax.plot(g.hour,g[f'{cohort}_xgboost_w4'],color=COLORS[cohort],ls=ls,label=COHORTS[cohort],marker='o',ms=3)
        ax.set(title=str(date.date()),xticks=range(18,24),xlabel='來源時段',ylabel='北門進站人次',ylim=(0,None))
        ax.grid(axis='y',alpha=.15)
    h,l=axes.flat[0].get_legend_handles_labels()
    fig.legend(h,l,loc='outside lower center',ncol=4)
    fig.suptitle('九場北門逐時預測｜固定參數、散場加權 XGBoost\n兩站版沒有每場改善；不以這九場重新挑選部署模型',fontsize=14)
    save(fig,OUT/'figures/beimen_profiles.png')

    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    c=corr.loc[corr.before_year.eq(2026)]
    for ax,field,title in zip(axes,['y','increment'],['活動日六小時總進站人流','扣除各站一般日基準後的增量']):
        g=c.loc[c.field.eq(field)]
        bars=ax.bar(range(3),g.spearman,color=['#438276','#738eaa','#b98b5e'],width=.55)
        ax.bar_label(bars,fmt='%.2f',padding=4)
        ax.set_xticks(range(3),g.station.str.replace('站',''))
        ax.set(title=title,ylabel='與北門的 Spearman 相關',ylim=(-.45,1.15))
        ax.axhline(0,color='#777777',lw=.8);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.12)
    fig.suptitle('原始人流相關低，不代表活動增量沒有共同變化\n只用 2026 年以前的 18 場；描述性診斷，未用此結果選模型',fontsize=14)
    save(fig,OUT/'figures/correlations.png')

    columns=['範圍','mae','rmse','peak_abs','cost_2']
    titles=['訓練車站','MAE','RMSE','尖峰絕對誤差','散場 2:1 誤差']
    annual_main=annual.loc[annual.model.eq('xgboost_w4')].sort_values(['year','cohort'])
    cwide=c.pivot(index='station',columns='field',values='spearman').reindex(['大橋頭站','雙連','民權西路']).reset_index()
    queue=json.loads((OUT/'queue_display_audit.json').read_text('utf8'))
    report=f'''# 第十九版：只用北門與大橋頭，預測是否更準？

**本輪未支持直接改成兩站。**現行加權 XGBoost 的北門 MAE 從 {x.loc['four','mae']:.2f} 增至 {x.loc['two','mae']:.2f}，增加 {mae_change:.1f}%；實際尖峰平均絕對誤差則從 {x.loc['four','peak_abs']:.2f} 降至 {x.loc['two','peak_abs']:.2f}，改善 {peak_gain:.1f}%。有取捨，整體誤差與散場 2:1 誤差仍變大；網站保留原模型。

## 1. 如何公平比較

- 四站：北門、大橋頭、雙連、民權西路；三站：北門、大橋頭、雙連；兩站：北門、大橋頭。來源名稱「大橋頭站」保持不變。
- 同一 21 場資料、18–23 六個來源時段；2024／2025／2026 分別用更早的 12／15／18 場訓練。同九場北門 54 筆為主要評估；次要比較共同兩站 108 筆。
- 依車站篩選訓練列，再做缺值處理、標準化、類別編碼、車站×時段交互作用；不先用四站估轉換再刪站。
- **主比較直接承接網站的 V14 frozen_w4。**另比較未加權 XGBoost 與 Ridge，讓本輪也能對照 V16 的原四／三站結果，避免拿不同模型設定混比。
- 每年度超參數固定為原封存設定，沒有新調參、融合或用這九場挑參數。加權版沿用北門預定散場權重，選定車站內每場平均權重為 1；Ridge 使用 alpha=n×λ，維持平均損失的懲罰強度。
- 輸入僅原先的一般日基準、日曆、事前預定結束訊號等；沒有當日前段人流、實測天氣、事後聲量或未來標籤。原歷史 OD／日曆發布時點仍未完全核實，這輪仍是回顧敏感度比較，不能新增宣稱為嚴格 D−1 可部署模型。
- 27 個模型擬合、15 組舊版預測控制均重現。四／三／兩站全資料分別 504／378／252 列，**獨立活動數都是 21，不會因逐站逐時列數變多就有更多活動**。

## 2. 現行加權 XGBoost：兩站沒有全面改善

{table(main,columns,titles)}

尖峰誤差是「每場實際最高人流時段」的絕對誤差再平均，只作事後評估，不以真實尖峰當輸入。散場 2:1 依事前預定結束後兩小時的重疊權重計算 2×低估＋高估，各場等權；不是實際候車分鐘或金錢。

![主要比較](figures/comparison.png)

兩站版散場正向低估由 {x.loc['four','window_under']:.2f} 降至 {x.loc['two','window_under']:.2f}，但高估由 {x.loc['four','window_over']:.2f} 升至 {x.loc['two','window_over']:.2f}，使散場 2:1 誤差仍惡化。不能只看低估減少就認定調度一定更好。

## 3. 各年度及個別活動

{table(annual_main,['year',*columns],['年度',*titles])}

兩站加權版在三個年度的整體 MAE 都高於四站版。2024/8/10 實際尖峰 11,101 人，四站預測 8,359，兩站 9,324，這一場的尖峰更接近；但 2026/8/5 實際尖峰 4,642，四站 4,725，兩站 5,702，反而高估更多。

![九場曲線](figures/beimen_profiles.png)

{table(peaks,['event_date','hour','y','four_xgboost_w4','three_xgboost_w4','two_xgboost_w4'],['活動日期','實際尖峰時段','實際','四站','三站','兩站'])}

## 4. 原未加權模型也比較

{table(summary.loc[summary.model.ne('xgboost_w4')],['模型',*columns],['模型',*titles])}

未加權 XGBoost 的北門 MAE 473.91 → 594.08，Ridge 845.44 → 868.10，也沒有因只用兩站而改善。這次固定參數是為隔離車站範圍，**不是證明兩站已調到最佳**。

次要比較大橋頭：加權 XGBoost MAE 四站 268.82、三站 225.46、兩站 247.65。兩站對大橋頭比四站好，但對北門較差；評估範圍不能隨模型改變，不能拿兩站平均直接對比四站平均。

## 5. 「另外兩站相關性不大」需要分兩件事

以預測 2026 年前可用的 18 場訓練活動為例，每場將六小時加總，再算與北門的 Spearman 相關：

{table(cwide,['station','y','increment'],['車站','原人流總量相關','扣除一般日基準後增量相關'])}

![車站關係](figures/correlations.png)

- 原始總量下，民權西路和北門的關係較弱且為負；扣掉各站一般日基準後，雙連與民權西路仍呈現共同活動增量。人流底量不同，不能只看總量就判定沒有資訊。
- 以上是同場的描述性關係，不是因果，也沒有把同場其他站的未來真實人流放入北門預測。全年共同趨勢與活動規模都可能使相關提高。
- **共用模型是在多站過去資料上學習，不是把另外三站當場實際人數當成三個預測變數。**移除車站會同時減少樣本、改變設計矩陣與樹分裂。相關係數不能取代配對向前驗證。
- 其他訓練截止年的診斷同樣保存在 [past_event_correlations.csv](tables/past_event_correlations.csv)；未以相關係數重新選車站。

## 6. 為什麼動畫進站時不滿載？

這是目前的載量及動畫假設，不是排隊理論規定列車必須留空：

1. 每班最大上限 K=2,220；預設本站下客後可再載 R=1,110，因此背景載量先假設 K−R=1,110（50%）。上游各站實際車上人數尚未建模。
2. 列車接近時仍顯示上述上車前假設；離站瞬間才上車 B=min(同方向候車 Q,R)，離站載量為 K−R+B。
3. 例如 2026/8/15、寬峰、往松山 21:02，模擬候車約 153 人，上車後約 1,263 人／2,220≈56.9%，不應為了讓畫面看起來擁擠而強制滿載。
4. 若同方向 Q>R，列車離站必滿載，剩餘旅客留到下一班。全部公開回放共 {queue['checked_train_records']:,} 筆班次情境中，{queue['records_with_left_behind']} 筆有離站後滯留；沒有「仍有同方向隊伍但未用完容量」的紀錄。
5. 紅色站點是雙向隊伍合計，單一列車只服務一個方向；列車離站後新到達的人也可能立即重新排隊。因此站紅與某列車未滿不一定矛盾。

目前是依班表批次上車的流體情境，不是把列車當 M/M/c 的平行櫃檯。未模擬車門吞吐、實際停站、拒絕上車、車廂分布或臨時加班車。前端已加上「上車前假設／離站後」字樣，不改容量來迎合期待。

要提高現實性，接下來優先校準：按班次／方向的既有車上載量與下車人數、分鐘進站量與方向比例、活動實際班表及有效可用容量。詳見[營運資料與官方來源](../../docs/metro_operations.md)；在這些資料尚缺時，保留均勻／寬峰／集中峰和剩餘容量敏感度，不能稱為實際即時人流復原。

## 7. 採用建議與重現

保留網站四站加權 XGBoost，兩站版留作候選／敏感度結果。若之後把訓練車站範圍也納入調參，需在更早年份的內層驗證選擇，再以未見活動評估，不能反覆在這九場挑最漂亮的結果。這次尚未完成新的兩站專屬調參，也不從探索結果推出「更多站永遠更好」。

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 src/run_analysis_v19.py
.\\.venv\\Scripts\\python.exe -X utf8 src/verify_analysis_v19.py
.\\.venv\\Scripts\\python.exe -X utf8 src/report_analysis_v19.py
.\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s tests -p test_analysis_v19.py -v
```

需要既有 V14／V16 本機快照；來源雜湊不一致會停止。模型及逐列預測留 `.local/analysis_v19`，公開表圖放 `result/analysis_v19`。[設定](../../configs/analysis_v19.json)／[擬合紀錄](fits.json)／[獨立計算檢核](independent_validation.json)／[列車滿載規則檢核](queue_display_audit.json)。檢核為另一程式重算，不宣稱由另一代理審查。
'''
    (OUT/'report.md').write_text(report,encoding='utf8')
    print('Wrote report.md and three figures.')


if __name__=='__main__':run()
