"""Generate Chinese reporting from completed, independently checked V13 outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis_v1.reporting import style,save

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/analysis_v13';PRIVATE=ROOT/'.local/analysis_v13'
LABELS={
 'archived_474':'舊版固定參數（僅參考）','baseline_only':'只用一般日基準',
 'core_ridge':'核心 Ridge','core_gam':'核心 GAM','core_xgboost':'核心 XGBoost',
 'direct_ridge':'直接預測 Ridge','direct_gam':'直接預測 GAM','direct_xgboost':'直接預測 XGBoost',
 'increment_ridge':'基準＋增量 Ridge','increment_gam':'基準＋增量 GAM','increment_xgboost':'基準＋增量 XGBoost',
 'equal':'增量模型等權融合','fusion_mae':'全時段 MAE 融合',
 'fusion_window1':'散場 1:1 融合','fusion_window2':'散場 2:1 融合'}


def table(frame,columns,labels=None):
    labels=labels or columns
    lines=['| '+' | '.join(labels)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame.to_dict('records'):
        values=[]
        for c in columns:
            value=row[c]
            values.append(f'{value:,.2f}' if isinstance(value,(float,np.floating)) else str(value))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def run():
    check=json.loads((OUT/'independent_validation.json').read_text('utf-8'));assert check['passed']
    s=pd.read_csv(OUT/'tables/summary.csv').set_index('method');p=pd.read_csv(PRIVATE/'predictions.csv',parse_dates=['event_date'])
    a=pd.read_csv(OUT/'tables/availability.csv');weights=json.loads((OUT/'weights.json').read_text('utf-8'))
    style();figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    selected=['archived_474','direct_xgboost','increment_xgboost','equal','fusion_mae','fusion_window1','fusion_window2']
    short=['舊版 474\n僅參考','直接\nXGBoost','基準＋增量\nXGBoost','等權\n融合','全時段\nMAE 融合','散場\n1:1 融合','散場\n2:1 融合']
    colors=['#b0b0b0','#65889b','#497c68','#50565c','#8c80a4','#b79a63','#b25c51']
    fig,axes=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
    for ax,metric,title in zip(axes,['mae','rmse','cost_2'],['全六小時 MAE','全六小時 RMSE','散場低估 2 倍誤差']):
        bars=ax.bar(np.arange(len(selected)),s.loc[selected,metric],color=colors)
        ax.bar_label(bars,fmt='%.0f',padding=3);ax.set_xticks(np.arange(len(selected)),short,fontsize=9)
        ax.set_title(title);ax.set_ylabel('人次／小時');ax.set_ylim(0,s.loc[selected,metric].max()*1.18)
        ax.axvline(.5,color='#999999',ls=':',lw=1)
    fig.suptitle('北門九場比較｜2024–2026 年、18–23 時\n舊版資料版本未核實；其餘共用版本篩選資料與固定核心參數',fontsize=14)
    save(fig,figdir/'comparison.png')
    fig,axes=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    for ax,(date,g) in zip(axes.flat,p.groupby('event_date')):
        g=g.sort_values('hour')
        for col,label,color,ls in [('y','實際','#222222','-'),('archived_474','舊版 474','#aaaaaa',':'),
           ('direct_xgboost','新版直接預測','#65889b','--'),('increment_xgboost','基準＋增量 XGB','#497c68','-'),
           ('fusion_window2','散場 2:1 融合','#b25c51','-')]:
            ax.plot(g.hour,g[col],label=label,color=color,ls=ls,marker='o',markersize=3,lw=1.5)
        ax.set_title(str(date.date()));ax.set_xticks(range(18,24));ax.set_xlabel('小時');ax.set_ylabel('進站需求代理（人次／小時）')
        ax.set_ylim(bottom=0)
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=5)
    fig.suptitle('北門九場逐時曲線｜以事前資料預測，實際值只用於評估',fontsize=15)
    save(fig,figdir/'beimen_profiles.png')
    fig,axes=plt.subplots(1,3,figsize=(13,5.5),constrained_layout=True)
    for ax,year in zip(axes,[2024,2025,2026]):
        records=[next(r for r in weights if r['year']==year and r['method']==method) for method in ['fusion_mae','fusion_window1','fusion_window2']]
        bottom=np.zeros(3)
        for model,color in zip(['ridge','gam','xgboost'],['#65889b','#b79a63','#497c68']):
            v=np.array([r['weights'][model] for r in records]);bars=ax.bar(range(3),v,bottom=bottom,color=color,label=model)
            for j,z in enumerate(v):
                if z>=.045:ax.text(j,bottom[j]+z/2,f'{z:.1%}',ha='center',va='center',color='white',fontsize=10)
            bottom+=v
        ax.set_title(f"{year} 年｜過去 {records[0]['past_events']} 場 OOF");ax.set_ylim(0,1)
        ax.set_xticks(range(3),['全時段 MAE','散場 1:1','散場 2:1']);ax.set_ylabel('融合權重')
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3)
    fig.suptitle('增量模型融合：權重僅由預測年度以前的時間留出預測求得',fontsize=14)
    save(fig,figdir/'fusion_weights.png')
    fig,axes=plt.subplots(3,3,figsize=(14,9),constrained_layout=True)
    for ax,(date,g) in zip(axes.flat,p.groupby('event_date')):
        g=g.sort_values('hour');b=g.available_baseline.to_numpy();delta=g.raw_delta_xgboost.to_numpy()
        ax.bar(g.hour,b,color='#c3d3cd',label='一般日基準');ax.bar(g.hour,delta,bottom=b,color='#497c68',alpha=.8,label='預測活動增量')
        ax.plot(g.hour,g.y,color='#222222',marker='o',ms=3,label='實際總人流')
        ax.axhline(0,color='#999999',lw=.6);ax.set_title(str(date.date()));ax.set_xticks(range(18,24));ax.set_ylabel('人次／小時')
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3)
    fig.suptitle('XGBoost 分解｜總預測＝一般日基準＋活動增量（增量可為負）',fontsize=14)
    save(fig,figdir/'baseline_increment.png')
    report=s.reset_index();report['方法']=report.method.map(LABELS)
    weight_table=[]
    for r in weights:
        if r['year']>=2024:weight_table.append({'年':r['year'],'方法':LABELS[r['method']],**{m:f"{r['weights'][m]:.2%}" for m in ['ridge','gam','xgboost']},'過去OOF場數':r['past_events']})
    wt=pd.DataFrame(weight_table)
    inc=s.loc['increment_xgboost'];direct=s.loc['direct_xgboost'];mix=s.loc['fusion_window2']
    delta=100*(inc.mae/direct.mae-1)
    conclusion=f"相同資料與參數下，XGBoost 改成基準＋增量，MAE {direct.mae:.2f} → {inc.mae:.2f}（{'降低' if delta<0 else '增加'} {abs(delta):.1f}%）。散場 2:1 融合 MAE {mix.mae:.2f}，沒有勝過單一增量 XGBoost。"
    reference="[官方 OD 目錄](https://data.taipei/dataset/detail?id=63f31c7e-7fc3-418b-bd82-b95158755b4d)、[Azure 版本與分段校驗定義](https://learn.microsoft.com/en-us/rest/api/storageservices/get-blob)、[Breiman（1996）stacking](https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf)、[Gneiting（2011）損失與決策目標](https://arxiv.org/abs/0912.0902)、[FPP3 時間向前驗證](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation)"
    text=f'''# 第十三版：一般日基準＋活動增量與低估 2 倍融合

## 1. 實際結果

{conclusion}

**本輪判斷：**保留增量 XGBoost 作新版主要比較候選；不因「罰低估」就採用融合。散場 2:1 融合相對同窗口 1:1，低估量 {s.loc['fusion_window1','window_under']:.2f} → {mix.window_under:.2f}、高估量 {s.loc['fusion_window1','window_over']:.2f} → {mix.window_over:.2f}；2:1 誤差 {s.loc['fusion_window1','cost_2']:.2f} → {mix.cost_2:.2f}，但全時段 MAE {s.loc['fusion_window1','mae']:.2f} → {mix.mae:.2f}。等權融合的 MAE {s.loc['equal','mae']:.2f}、2:1 誤差 {s.loc['equal','cost_2']:.2f} 也都優於 2:1 學權重，因此本次沒有支持直接換上該組權重。

舊版固定參數 MAE **473.91** 已用原預測逐列重現，但不能作為相同 D−1 可得條件的控制組。新版的直接／增量模型共用相同資料、核心參數及輸入；兩者差別為學習總人流或學習偏離基準的量。

{table(report,['方法','mae','rmse','cost_2','peak_under','peak_abs'],['方法','MAE','RMSE','散場 2:1 誤差','實際尖峰正向低估','實際尖峰絕對誤差'])}

單位人次／小時；全部以活動等權計算。正向低估在高估時記 0，必須連同绝對誤差及高估量判斷。散場指標依事前預定結束後兩小時與整小時的重疊加權，**沒有把小時資料當成分鐘觀測**。2 倍是研究假設，非貨幣成本。

![誤差比較](figures/comparison.png)
![九場逐時](figures/beimen_profiles.png)

## 2. 一般日資料何時能用

- D−1 截止為活動前一天 18:00（臺北）。同星期、前 7–56 天，保留目前官方檔案版本時間早於截止的最近四日，至少兩日，平均得到同站同時段基準。
- 排除既有活動登錄日期。未取得事前版本的補班／假日日曆，因此本次不用該日曆來挑配對日；這是「同星期一般日代理基準」，可能仍含補班或其他特殊日。資料排除登錄本身仍是回顧整理。
- 例如 2025 年 7 月資料的目前版本在 8/14 更新，預測 8/6 時不能使用。2017–2020 年現存 OD 版本於 2021 年更新，無法證實舊版本內容，因此四場排除。**21 場減為 17 場、408 個四站逐時列**；外層仍為相同的 2024–2026 九場北門、54 列。
- Last-Modified 是「目前物件版本時間」，不是首次公開日，也不是歷史存取權限的完整證明。已完成版本截止檢查；仍不能宣稱完整歷史公開可得性全部核實。
- 每年訓練僅用更早年度，且目標 OD 版本已在該年 1/1 前存在。公告沿用事前來源帳本；未知結束時間保留未知。沒有加入實測天氣、事後 Google 熱度或當場人流。
- 官方下載全部逐 4 MiB 核對交易 MD5、範圍、ETag；部分整檔 MD5 屬性與現取內容不一致，保留差異紀錄，不用該屬性取代逐段驗證。

詳見 [每場基準日期](tables/availability.csv)、[官方來源與版本帳本](../../docs/d1_od_version_sources.json)。本輪不變更 SQL 或上線模型。

## 3. 模型與權重

令 B 為事前一般日基準，D＝Y−B。Ridge、GAM、XGBoost 分別學 D，再用 max(0, B＋預測 D) 回到人次。**D 可以小於 0**，只有總預測才截為非負。直接控制組用相同設計矩陣學 Y，基準 B 在兩組都作輸入。

核心參數只在更早年份 OOF 用散場 2:1 選取；加 B 與改預測增量後都固定該組參數，不再調參。這沿用舊版「固定核心參數」的比較方式，但因可用訓練歷史改變，重新取得合格歷史的核心參數；**不是把舊版所有參數值照搬，故不把 474 與新版差異全歸因於增量模型。**

融合使用三個增量模型的總人流 OOF 預測，限制權重非負、總和為 1。最小化「2×低估量＋1×高估量」，以線性規劃求解；另保留全六小時 MAE 與相同散場窗口的 1:1 控制，分清換窗口與換低估懲罰。所有融合權重只用預測年度之前的 OOF。

{table(wt,['年','方法','ridge','gam','xgboost','過去OOF場數'])}

![年度融合權重](figures/fusion_weights.png)
![基準與增量](figures/baseline_increment.png)

## 4. 判讀與限制

- 一般日基準是預測參考，不是經因果識別的「無活動人流」；增量也不能直接稱活動的因果效果。
- 凸融合無法超出三模型預測的範圍；若三者都低估同一尖峰，改權重也補不回超出最高基模型的部分。低估罰 2 倍不能保證測試期每場都減少低估。
- 2025、2026 年散場 2:1 權重均選 Ridge 100%；這是過去 OOF 目標的解，不代表 Ridge 在當年最準。2026 年因此失去 XGBoost 成分，MAE 比散場 1:1 融合差。權重本身也有估計誤差。
- 2024–2026 九場已反覆用於開發；本次是探索性回測，沒有未見活動驗證，不自動把最低 MAE 當成已證實的新部署模型。
- 候車與列車調度仍需分鐘到達曲線、行走延遲、方向分配、列車到站與剩餘容量。現在只提供逐時需求代理，不輸出真實等候分鐘或最適班表。

## 5. 檢核、檔案與文獻

[獨立檢核](independent_validation.json)：核對 {check['verified_official_months']} 個官方月份、重建 {check['reconstructed_baseline_events']} 場基準、重載 {check['reloaded_fits']} 個模型、獨立求解 {check['independently_solved_fusions']} 次融合，以及重算 {check['metric_rows']} 列指標。完整公式與執行步驟在 [process 第 24 節](../../process.md#24-第十三版一般日基準活動增量與非對稱融合)。

分析系列 270 項測試通過，含本版新增 12 項，涵蓋截止、未發布標籤、負增量、最後才截斷、時間 OOF 與非對稱融合。

{reference}。

Breiman 支持以留出預測建立組合；本文另採時間切分、和為 1 約束與非對稱絕對誤差，並非原文平方誤差的完全重現。一般日均值＋殘差預測、56 天與最多四日是本案事先設定，文獻不保證它一定勝過直接模型。
'''
    (OUT/'report.md').write_text(text.replace('参数','參數').replace('绝對','絕對'),'utf-8')
    small=report.loc[report.method.isin(selected)]
    intro=f'''## 最新進度：一般日基準＋活動增量與低估 2 倍融合（2026-10-13）

**{conclusion}**

同樣比較 2024–2026 九場北門 18–23 時；新版四站訓練，依官方 OD 版本时间篩選一般日與訓練標籤。下表越低越好，單位人次／小時。

{table(small,['方法','mae','rmse','cost_2'],['方法','MAE','RMSE','散場 2:1 誤差'])}

- **舊版 474 只作參考：**原資料缺發布核實；新版排除 2017–2020 四場、改用截止前已存在的官方 OD 版本，21 場縮為 17 場。新版直接／增量組則共用相同資料與核心參數，才能比較分解方式。
- **模型怎麼做：**先算同站同時段、同星期且截止前可用的前 56 天均值，再預測活動日與它的差。增量允許負值，最後總人流才限制非負；它不是活動因果效果。
- **權重怎麼做：**只用過去年份的北門 OOF，分別最小化全時段 MAE、散場 1:1 與散場 2:1。2 倍是研究假設；本輪沒有自動替換上線模型。
- **沒有全面改善：**散場 2:1 融合的低估量比 1:1 少，但高估增加、全時段 MAE 變差；等權融合 MAE {s.loc['equal','mae']:.2f}、2:1 誤差 {s.loc['equal','cost_2']:.2f} 也都比它低。2025、2026 的 2:1 權重都是 Ridge 100%，不能把過去最適權重當成未來必勝。
- **資料界線：**目前版本時間早於截止已檢查，但沒有完整歷史公開權限證據；同星期代理基準尚未排除所有補班／節日。沒有加入事後天氣、聲量或當場實測輸入。九場已重複檢視，仍需未見活動驗證。

![一般日基準與活動增量比較](result/analysis_v13/figures/comparison.png)

[完整結果與三模型比較](result/analysis_v13/report.md)／[九場曲線](result/analysis_v13/figures/beimen_profiles.png)／[基準＋增量圖](result/analysis_v13/figures/baseline_increment.png)／[年度權重圖](result/analysis_v13/figures/fusion_weights.png)／[公式與重現步驟](process.md#24-第十三版一般日基準活動增量與非對稱融合)／[獨立檢核](result/analysis_v13/independent_validation.json)。

方法依據：{reference}。

'''
    path=ROOT/'README.md';old=path.read_text('utf-8')
    old=old.replace('## 最新進度：場次定位、熱度、一般日與天氣拆分（2026-10-13）','## 第十二版紀錄：場次定位、熱度、一般日與天氣拆分（2026-10-13）',1)
    marker='## 最新進度：一般日基準＋活動增量與低估 2 倍融合（2026-10-13）'
    if marker in old:
        start=old.index(marker);end=old.index('## 第十二版紀錄：',start);old=old[:start]+old[end:]
    first,rest=old.split('\n',1);path.write_text(first+'\n\n'+intro.replace('时间','時間')+rest.lstrip(),'utf-8')
    print(conclusion)


if __name__=='__main__':run()
