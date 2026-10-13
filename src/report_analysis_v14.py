"""Reproducible Chinese reporting for independently verified V14 results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis_v1.reporting import style,save

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/analysis_v14'
LABELS={
 'archived_474':'原固定參數＋一般日基準','frozen_w2':'原參數＋散場訓練 2 倍','frozen_w4':'原參數＋散場訓練 4 倍',
 'local_fixed_w1':'新學習率／樹數格點，不早停','local_fixed_w2':'新格點不早停＋訓練 2 倍',
 'local_fixed_w4':'新格點不早停＋訓練 4 倍','local_fixed_wselected':'新格點不早停＋歷史選倍率',
 'local_es_w1':'新學習率格點＋early stopping','local_es_w2':'Early stopping＋訓練 2 倍',
 'local_es_w4':'Early stopping＋訓練 4 倍','local_es_wselected':'Early stopping＋歷史選倍率',
 'archived_retuned2':'舊 180 組依散場 2:1 調參','archived_retuned_mae':'舊 180 組改依全時段 MAE 調參'}
ORDER=list(LABELS)

def table(frame,columns,labels=None):
    lines=['| '+' | '.join(labels or map(str,columns))+' |','| '+' | '.join(['---']*len(columns))+' |']
    for r in frame.to_dict('records'):
        lines.append('| '+' | '.join(f'{r[c]:,.2f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in columns)+' |')
    return '\n'.join(lines)

def run():
    check=json.loads((OUT/'independent_validation.json').read_text('utf-8'));assert check['passed']
    s=pd.read_csv(OUT/'tables/summary.csv').set_index('method')
    annual=pd.read_csv(OUT/'tables/annual.csv')
    p=pd.read_csv(ROOT/'.local/analysis_v14/predictions.csv',parse_dates=['event_date'])
    selections=json.loads((OUT/'selections.json').read_text('utf-8'))
    style();figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    selected=['archived_474','frozen_w2','frozen_w4','local_fixed_w1','local_es_w1','local_es_w4','local_es_wselected']
    short=['原版\n474','原參數\n訓練 2 倍','原參數\n訓練 4 倍','新格點\n不早停','新格點\n早停 1 倍','新格點\n早停 4 倍','早停\n歷史選倍率']
    colors=['#888888','#91aaa0','#477a67','#b49c76','#537e99','#466073','#a96965']
    fig,axes=plt.subplots(1,3,figsize=(18,5.3),constrained_layout=True)
    for ax,metric,title in zip(axes,['mae','rmse','cost_2'],['全六小時 MAE','全六小時 RMSE','散場 2×低估＋高估']):
        bars=ax.bar(range(len(selected)),s.loc[selected,metric],color=colors)
        ax.bar_label(bars,fmt='%.0f',padding=3,fontsize=10);ax.set_xticks(range(len(selected)),short,fontsize=9)
        ax.set_ylim(0,s.loc[selected,metric].max()*1.19);ax.set_title(title);ax.set_ylabel('人次／小時')
    fig.suptitle('原資料回顧比較｜相同九場北門、54 個小時\n訓練仍保留四站；歷史資料發布時間尚未完全核實',fontsize=14)
    save(fig,figdir/'comparison.png')

    fig,axes=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    for ax,(date,g) in zip(axes.flat,p.groupby('event_date')):
        g=g.sort_values('hour')
        for col,label,color,ls in [('y','實際','#222222','-'),('archived_474','原版 474','#999999',':'),
          ('frozen_w4','原參數＋訓練 4 倍','#477a67','-'),('local_es_w1','早停 1 倍','#537e99','--'),
          ('local_es_w4','早停 4 倍','#a47842','-.')]:
            ax.plot(g.hour,g[col],marker='o',ms=3,lw=1.6,label=label,color=color,ls=ls)
        ax.set_title(str(date.date()));ax.set_xticks(range(18,24));ax.set_xlabel('小時')
        ax.set_ylabel('進站需求代理（人次／小時）');ax.set_ylim(bottom=0)
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=5)
    fig.suptitle('北門九場逐時曲線｜平均改善不等於每場尖峰都改善',fontsize=15)
    save(fig,figdir/'beimen_profiles.png')

    fig,axes=plt.subplots(1,3,figsize=(14.5,5.2),constrained_layout=True)
    for ax,metric,title in zip(axes,['mae','rmse','peak_abs'],['全六小時 MAE','全六小時 RMSE','實際尖峰小時絕對誤差']):
        for j,(methods,label,color) in enumerate([(['archived_474','frozen_w2','frozen_w4'],'固定原參數','#477a67'),
          ([f'local_fixed_w{w}' for w in [1,2,4]],'新格點不早停','#b49c76'),
          ([f'local_es_w{w}' for w in [1,2,4]],'新格點＋早停','#537e99')]):
            y=s.loc[methods,metric].to_numpy()
            bars=ax.bar(np.arange(3)+(j-1)*.26,y,width=.24,label=label,color=color)
            ax.bar_label(bars,fmt='%.0f',padding=3,fontsize=9)
        ax.set_xticks(range(3),['1 倍','2 倍','4 倍']);ax.set_xlabel('北門事前散場窗口的相對訓練權重')
        ax.set_title(title);ax.set_ylabel('人次／小時')
        ax.set_ylim(0,ax.get_ylim()[1]*1.1)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3)
    fig.suptitle('权重越大不一定越準｜1、2、4 倍完整呈現；尖峰僅於評估時定位'.replace('权','權'),fontsize=14)
    save(fig,figdir/'weight_ablation.png')

    fig,axes=plt.subplots(1,3,figsize=(14,4.8),constrained_layout=True)
    for ax,year in zip(axes,[2024,2025,2026]):
        row=next(r for r in selections if r['year']==year and r['method']=='local_es_w1')
        stop=row['stopping'];curve=np.array(stop['curve']);n=stop['trees']
        ax.plot(np.arange(1,len(curve)+1),curve,color='#537e99');ax.axvline(n,color='#477a67',ls='--')
        ax.scatter([n],[curve[n-1]],color='#477a67',zorder=3)
        ax.annotate(f'選 {n} 棵',(n,curve[n-1]),xytext=(-5,17),textcoords='offset points',ha='right',color='#477a67')
        ax.set_title(f'預測 {year}｜用 {year-1} 年決定停止點');ax.set_xlabel('累積樹數');ax.set_ylabel('過去年份北門 MAE')
    fig.suptitle('Early stopping 不看當年真值｜找停止點後，以全部更早年度重訓\n圖為不加權版本：深度 4、學習率 0.1、葉門檻 1',fontsize=14)
    save(fig,figdir/'early_stopping.png')

    report=s.loc[ORDER].reset_index();report.insert(1,'方法',report.method.map(LABELS))
    main=report.set_index('method').loc[selected].reset_index()
    years=annual.pivot(index='method',columns='year',values='mae').loc[['archived_474','frozen_w4','local_es_w1','local_es_w4']].reset_index()
    years['方法']=years.method.map(LABELS)
    rows=[]
    for r in selections:
        if r['method'] in ['local_es_w1','local_es_w4','local_es_wselected']:
            rows.append({'年':r['year'],'方法':LABELS[r['method']],'倍率':r['multiplier'],
              '深度':r['params']['max_depth'],'學習率':r['params']['learning_rate'],'葉門檻':r['params']['min_child_weight'],
              '樹數':r['params']['n_estimators'],'停止用年度':r['stopping']['stop_start'][:4],'過去OOF場數':r['past_events']})
    pt=pd.DataFrame(rows);pt.to_csv(OUT/'tables/selected_parameters.csv',encoding='utf-8-sig',index=False)
    gain=100*(1-s.loc['frozen_w4','mae']/s.loc['archived_474','mae'])
    esgain=100*(1-s.loc['local_es_w1','mae']/s.loc['archived_474','mae'])
    references="""- [XGBoost 官方：sklearn 介面與 Early Stopping](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html)：用驗證集監控停止點；本文另要求整年度向前切分及選定樹數後重訓。
- [XGBoost 官方：XGBRegressor.fit](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor.fit)：支援逐列 sample_weight。1、2、4 倍與每活動總權重固定，是本案實驗設計，不是文獻保證的最佳倍率。
- [Cawley & Talbot（2010），On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://jmlr.org/papers/v11/cawley10a.html)：選參數本身可能過度配合驗證資料，增加候選不保證泛化更好。"""
    body=f"""# 第十四版：原資料的 XGBoost 調參、early stopping 與北門散場訓練加權

## 1. 結論與適用範圍

原 MAE **473.91** 已逐列重現。只改北門散場訓練權重為 4 倍，MAE **440.28**，降低 **{gain:.1f}%**；擴充學習率搭配 early stopping、不加權，MAE **450.55**，降低 **{esgain:.1f}%**。兩項都值得保留為下一輪候選。

**尚不能宣布尖峰問題解決，也沒有換上線模型。** 原參數 4 倍版在 6/9 場 MAE 較低，但平均實際尖峰正向低估量 **556.60 → 658.01** 反而增加；尖峰絕對誤差 **749.02 → 747.14** 幾乎不變。全六小時平均不能代表散場容量已足夠。

本次回到 V12 原 21 場、四站 504 列，並非 V13 版本篩選後的 17 場。一般日基準與部分歷史資料發布時間尚未核實，因此是**原資料回顧敏感度實驗**，不能稱為完整符合活動前一天可得條件的模型。沒有新加入事後天氣、Google 熱度或當場實測輸入。九場測試活動已反覆檢視，不是新的獨立測試。

## 2. 完整誤差比較

{table(report,['方法','mae','rmse','cost_2','peak_under','peak_abs'],['方法','MAE','RMSE','散場 2:1 誤差','尖峰正向低估','尖峰絕對誤差'])}

單位人次／小時，越低越好；每場六小時，活動等權。尖峰按實際最大值定位，只用於評估。散場指標依預定結束後兩小時與整小時的重疊長度加權，2:1 是研究假設，不是貨幣成本。未知結束時間的評估退回六小時；本次九場皆有已知窗口。訓練未知窗口則不額外加權。

![比較](figures/comparison.png)
![權重完整消融](figures/weight_ablation.png)

### 不是每年都改善

{table(years,['方法',2024,2025,2026],['方法','2024 MAE','2025 MAE','2026 MAE'])}

改善主要集中於 2024；表中新方法在 2025、2026 的年度 MAE 均未勝過原版。原參數 2 倍的散場 2:1 誤差 **1137.02** 比原參數 4 倍 **1202.98** 低。早停 4 倍的 RMSE **669.27** 比原參數 4 倍 **690.91** 低，但 MAE 稍高。不同目的有取捨，不能只报一個數字。

## 3. 這次怎麼調參

- **原對照：**固定原版深度／學習率／樹數等參數，不加權重訓並重現 473.91；另只换 2、4 倍訓練權重，隔離加權效果。
- **新格點不早停：**深度 2、3、4 × 學習率 0.01、0.03、0.05、0.1 × 葉門檻 1、6 × 樹數 100、250、500、1000，共 96 組／倍率。
- **新格點加早停：**相同 24 種深度／學習率／葉門檻組合，最多 1000 棵；驗證 MAE 連續 50 輪未改善即停止，取已觀察曲線的最低點。固定 L2=1、L1=0、資料與特徵抽樣=1。本輪是局部比較，沒有宣稱找到全域最佳參數。
- **停止點只用過去：**預測 2026 時，先用截至 2024 的四站訓練、2025 北門驗證，決定樹數；再用截至 2025 的四站按該樹數重訓。2026 真值只用於最後評估；編碼器也只在各自訓練集擬合。
- **形狀參數只用過去 OOF：**逐年模擬上述流程，2024／2025／2026 各用過去 10／13／16 場的北門全六小時 MAE 選候選。早期 2019 停止層訓練不足兩場，依事前規則用 250 棵，不拿當年真值決定停止。
- **完整呈現倍率：**1、2、4 倍為事先列出的敏感度組，另實作只依過去 OOF 自動選倍率；沒有用當年結果挑倍率。

{table(pt,['年','方法','倍率','深度','學習率','葉門檻','樹數','停止用年度','過去OOF場數'])}

不加權早停版選 **186、818、63** 棵，不是每年一律 100 或 250。43/504 條有效停止曲線碰到 1000 棵上限，不能假定每組都自然停止；上表外層選中的早停設定皆未碰上限。曲線屬停止層，正式模型再以全部過去資料重訓。

![停止曲線](figures/early_stopping.png)

新格點不早停 MAE **606.71**；舊 180 組改以 MAE 選也只有 **537.79**。原參數其實已包含於先前 180 組候選，不能說以前漏放原最佳參數。選擇由較早活動決定，其排名未必適用後來的大型活動；增加格點不保證進步。

## 4. 與以前只用北門訓練有何不同

以前 V10 把其他三站移除，504 列減為 126 列；當時四站 XGBoost MAE 642.59、北門專用 813.60，確實沒有改善。V10 用「煙火＋天氣」輸入，與本輪原基準版不同，這兩個數字只交代過去結論，不能和 474／440 直接作公平排名。

本輪保留四站全部資料，只改各列在訓練損失中的重要性。令 q 為北門該小時與事前預定結束後兩小時的重疊比例；其他站 q=0。原權重 r=1+(m−1)q，再將每場總權重正規化為 24。完整落在窗口內的北門小時，相對其他列為 m 倍。**沒有使用實際尖峰決定權重**；未知結束時間不額外加權。

sample_weight 讓同樣大小的高估和低估都更重要，XGBoost 仍學加權平方誤差。它**不是低估罰 2 倍的非對稱損失，也不是三模型融合權重**。本輪未重算融合。

### 為什麼不能看完結果就自動選 4 倍

只依過去 OOF 選倍率的早停版 MAE **592.92**，沒有改善。2024 過去資料選 2 倍（過去 MAE 1082.76 比 1 倍 1088.69 只好約 6），但當年結果很差；2025、2026 選 1 倍。「4 倍在這九場較好」與「能事先穩定選對倍率」是兩件事，不可用當年最小 MAE 回頭挑倍率。

## 5. 尖峰與下一步

![九場曲線](figures/beimen_profiles.png)

- 2024/8/10 21 時實際 **11,101**：原版 **8,413**；原參數 4 倍 **8,359**，低估沒有改善；早停 4 倍 **9,166**，仍低估約 **1,935**。
- 2026/8/15 21 時實際 **10,904**：原版 **10,101**；早停 4 倍 **9,821**，反而更差。不能從一場改善推廣到所有大型場次。
- **保留不同角色：**原版作對照；原參數 4 倍作全時段 MAE 候選；早停 4 倍作大誤差／尖峰候選；原參數 2 倍保留作散場 2:1 決策目標對照。沒有證實唯一全面最佳模型。
- 下一步先固定規則，在 V13 已做版本截止篩選的資料上另作同條件比較，再用未參與開發的新活動驗證。調度端還要同時比較低估與過度加班成本，不能把最低 MAE 當容量保證。
- 現有逐時進站需求代理，仍需分鐘到達、方向比例、行走延遲與列車剩餘容量；不能直接換算真實候車分鐘或班表。

## 6. 重現與檢核

[process 第 25 節](../../process.md#25-第十四版原資料的早停與北門散場訓練加權)／[設定](../../configs/analysis_v14.json)／[年度表](tables/annual.csv)／[各場表](tables/events.csv)／[獨立檢核](independent_validation.json)。

獨立重算 27 次選擇，檢查 504 條停止曲線、72 次歷史不足備援與 24 個快取，重載 33 個正式模型、重算 169 列指標。預測最大差 {check['max_prediction_difference']:.2e}，指標最大差 {check['max_metric_difference']:.2e}；原 473.909516 重現。來源與模型都有 SHA-256 紀錄，使用 XGBoost 3.1.3。分析系列 285 項測試通過，含本版新增 15 項；其中直接擾動目標年度真值，驗證停止點與預測完全不變。

## 7. 方法依據

{references}
"""
    (OUT/'report.md').write_text(body.replace('报','報').replace('换','換'),'utf-8')
    intro=f"""## 最新進度：原資料調參、Early stopping 與北門散場加權（2026-10-13）

**原 MAE 473.91 已重現。保留原參數、只提高北門散場訓練權重到 4 倍，降到 440.28（改善 {gain:.1f}%）；新學習率格點搭配 early stopping、不加權，降到 450.55。**

保留四站訓練，比較同樣 2024–2026 九場北門、54 個小時。本次回到 V12 原資料，歷史發布時間尚未完全核實，屬回顧敏感度實驗；**不可當成已驗證 D−1 可部署模型，也不可和下方 V13 不同資料版本直接排名。**

{table(main,['方法','mae','rmse','cost_2'],['方法','MAE','RMSE','散場 2:1 誤差'])}

- **與以前北門專用不同：**保留其他三站，只提高北門事前預定結束後兩小時的訓練重要性。每場總權重相同；不用實際尖峰決定權重。
- **早停怎麼做：**用更早年份訓練、最近過去年份驗證決定樹數，再以全部過去資料重訓。不加權版選 186／818／63 棵；當年真值不參與停止或選參數。
- **尖峰仍未解決：**原參數 4 倍版 MAE 較好，但尖峰平均絕對誤差僅 749.02 → 747.14，尖峰正向低估反而增加。早停 4 倍 RMSE 較低，仍非每場都改善。
- **自動選倍率也會選錯：**只依過去 OOF 選 1／2／4 倍，早停版 MAE 592.92；不能看完九場後挑 4 倍，就說已證明未來最佳。改善主要集中於 2024，九場已反覆用於開發。

![調參與訓練權重比較](result/analysis_v14/figures/comparison.png)

[完整結果與解讀](result/analysis_v14/report.md)／[九場逐時圖](result/analysis_v14/figures/beimen_profiles.png)／[1、2、4 倍比較](result/analysis_v14/figures/weight_ablation.png)／[停止曲線](result/analysis_v14/figures/early_stopping.png)／[實作公式](process.md#25-第十四版原資料的早停與北門散場訓練加權)／[獨立檢核](result/analysis_v14/independent_validation.json)。

方法依據：[XGBoost 官方 Early stopping](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html)、[sample_weight](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor.fit)、[Cawley & Talbot（2010），模型選擇過擬合](https://jmlr.org/papers/v11/cawley10a.html)。倍率與窗口是本案假設，文獻不保證一定改善。

"""
    path=ROOT/'README.md';old=path.read_text('utf-8')
    old=old.replace('## 最新進度：一般日基準＋活動增量與低估 2 倍融合（2026-10-13）','## 第十三版紀錄：一般日基準＋活動增量與低估 2 倍融合（2026-10-13）',1)
    marker='## 最新進度：原資料調參、Early stopping 與北門散場加權（2026-10-13）'
    if marker in old:
        start=old.index(marker);end=old.index('## 第十三版紀錄：',start);old=old[:start]+old[end:]
    first,rest=old.split('\n',1);path.write_text(first+'\n\n'+intro+rest.lstrip(),'utf-8')
    print('V14 report, README, selected parameters and four figures written.')

if __name__=='__main__':run()
