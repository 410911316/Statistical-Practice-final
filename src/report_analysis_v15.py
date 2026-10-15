"""Chinese report/figures for the completed and independently checked V15 run."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis_v1.reporting import style, save

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'result/analysis_v15'
LABELS={'none':'原模型','origin_w1':'進站 1 時段','origin_w3':'進站 3 時段','origin_w5':'進站 5 時段',
        'both_w1':'進站＋目的站 1 時段','both_w3':'進站＋目的站 3 時段','both_w5':'進站＋目的站 5 時段'}
ORDER=list(LABELS)
COLORS=['#777777','#387c67','#759680','#b7c6bd','#496f91','#8198ae','#b2c1d0']


def table(df, columns, titles):
    rows=['| '+' | '.join(titles)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for record in df.to_dict('records'):
        cells=[f'{record[c]:,.2f}' if isinstance(record[c],(float,np.floating)) else str(record[c]) for c in columns]
        rows.append('| '+' | '.join(cells)+' |')
    return '\n'.join(rows)


def run():
    check=json.loads((OUT/'independent_validation.json').read_text('utf8'))
    assert check['passed']
    s=pd.read_csv(OUT/'tables/summary.csv')
    a=pd.read_csv(OUT/'tables/annual.csv')
    c=pd.read_csv(OUT/'tables/conservative.csv')
    p=pd.read_csv(ROOT/'.local/analysis_v15/predictions.csv',parse_dates=['event_date'])
    for df in [s,a,c]:df['label']=df.variant.map(LABELS)
    x=s.loc[s.model.eq('xgboost')].set_index('variant').loc[ORDER].reset_index()
    r=s.loc[s.model.eq('ridge')].set_index('variant').loc[ORDER].reset_index()
    style();figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    short=['原模型','進站\n1 時段','進站\n3 時段','進站\n5 時段','兩方向\n1 時段','兩方向\n3 時段','兩方向\n5 時段']
    fig,axes=plt.subplots(1,3,figsize=(17,5.1),constrained_layout=True)
    for ax,metric,title in zip(axes,['mae','rmse','peak_abs'],['全部來源時段 MAE','全部來源時段 RMSE','實際尖峰時段絕對誤差']):
        bars=ax.bar(range(7),x[metric],color=COLORS)
        ax.bar_label(bars,fmt='%.0f',padding=3,fontsize=10);ax.set_xticks(range(7),short,fontsize=9)
        ax.set_ylim(0,x[metric].max()*1.2);ax.set_title(title);ax.set_ylabel('人次／來源時段')
    fig.suptitle('XGBoost｜凍結原參數，只加入活動前段人流\n相同九場北門、54 筆；當天可取得資料仍是假設',fontsize=14)
    save(fig,figdir/'comparison.png')

    fig,axes=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    lines=[('y','實際','#222222','-'),('xgboost_none','原模型','#888888',':'),
           ('xgboost_origin_w1','進站 1 時段','#387c67','-'),
           ('xgboost_origin_w5','進站 5 時段','#a68050','--'),
           ('xgboost_both_w5','進站＋目的站 5 時段','#496f91','-.')]
    for ax,(date,g) in zip(axes.flat,p.groupby('event_date')):
        g=g.sort_values('hour')
        for name,label,color,ls in lines:ax.plot(g.hour,g[name],marker='o',ms=3,lw=1.5,label=label,color=color,ls=ls)
        ax.set_title(str(date.date()));ax.set_xticks(range(18,24));ax.set_xlabel('來源時段（鐘點定義待核實）')
        ax.set_ylabel('北門進站人次');ax.set_ylim(bottom=0)
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=5)
    fig.suptitle('名義 18:00 更新後凍結輸入｜全程不使用來源時段 18–23 的實際人流',fontsize=14)
    save(fig,figdir/'beimen_profiles.png')

    fig,axes=plt.subplots(1,2,figsize=(13,5.6),constrained_layout=True)
    for ax,model,label in zip(axes,['ridge','xgboost'],['Ridge','XGBoost']):
        g=a.loc[a.model.eq(model)]
        for variant,color in zip(ORDER,COLORS):
            z=g.loc[g.variant.eq(variant)].sort_values('year')
            ax.plot(z.year,z.mae,marker='o',color=color,label=LABELS[variant],lw=1.6)
        ax.set_xticks([2024,2025,2026]);ax.set_title(label);ax.set_ylabel('年度 MAE（人次／來源時段）')
        ax.set_ylim(bottom=0)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=4,fontsize=9)
    fig.suptitle('年度結果不一致｜所有預設窗口完整呈現，未按測試結果選窗口',fontsize=14)
    save(fig,figdir/'annual.png')

    fig,axes=plt.subplots(1,2,figsize=(12,5.5),constrained_layout=True)
    for ax,metric,title in zip(axes,['mae','rmse'],['MAE','RMSE']):
        for j,variant in enumerate(['none','origin_w1']):
            vals=[float(x.loc[x.variant.eq(variant),metric].iloc[0]),
                  float(c.loc[c.model.eq('xgboost')&c.variant.eq(variant),metric].iloc[0])]
            bars=ax.bar(np.arange(2)+(j-.5)*.33,vals,width=.31,color=COLORS[j],label=LABELS[variant])
            ax.bar_label(bars,fmt='%.1f',padding=3)
        ax.set_xticks([0,1],['來源時段 18–23\n九場 × 6＝54 筆','來源時段 19–23\n九場 × 5＝45 筆'])
        ax.set_title(title);ax.set_ylabel('人次／來源時段');ax.set_ylim(0,ax.get_ylim()[1]*1.15)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2)
    fig.suptitle('排除來源時段 18 的敏感度檢查｜兩個比較範圍不可混用基準',fontsize=14)
    save(fig,figdir/'target_window.png')

    cols=['label','mae','rmse','cost_2','peak_abs','peak_under']
    titles=['輸入版本','MAE','RMSE','散場 2:1 誤差','尖峰絕對誤差','尖峰正向低估']
    xt=table(x,cols,titles);rt=table(r,cols,titles)
    annual=a.loc[a.model.eq('xgboost')].pivot(index='variant',columns='year',values='mae').loc[ORDER].reset_index()
    annual['label']=annual.variant.map(LABELS)
    ct=c.loc[c.model.eq('xgboost')]
    peaks=p.loc[p.groupby('event_date').y.idxmax(),['event_date','hour','y','xgboost_none','xgboost_origin_w1']].copy()
    peaks.event_date=peaks.event_date.dt.strftime('%Y-%m-%d')
    report=f"""# 第十五版：活動當日前段人流是否改善北門預測

**結果：加入前段人流沒有全面改善。最近 1 個來源時段的進站量，MAE 473.91 → 481.77（惡化 1.7%），但 RMSE、散場 2:1 誤差與尖峰絕對誤差略有改善。3／5 個時段及加入目的站旅次量，XGBoost 的 MAE 都更差。**

## 1. 先釐清資料與時間

- **原始資料確實有「目的站北門」的逐來源時段旅次量。**它是把各起站、目的站為北門的 OD 人次加總；本次確實納入實驗。
- 北門「進站」表示從北門刷卡搭捷運；「目的站北門」表示搭捷運到北門的旅次。兩者不是同一方向，後者也不等於參加煙火的人數。
- 來源 CSV 是「日期、時段、進站、出站、人次」。既有程式在同一「時段」上分別彙總起站、迄站；本次查到的官方說明沒有交代時段按進站還是出站計時，也未確認它標示區間起點或終點。因此保留名稱 **destination_count_same_source_bin**，不寫成已核實的實際刷卡出站時刻。
- 原研究是 D−1 預測；本輪改為**假設當天拿得到前段人流的回顧更新實驗**。官方月更新 OD 並不是即時 API。原 V12 歷史資料發布證據也仍有缺口。
- 名義截止 18:00，固定只取來源時段 **13–17**，不使用 18–23 真值。若時段標示起點，它才對應 13:00–18:00；目前不能把這個對應說成已核實。
- 為處理來源時段 18 可能標示區間終點的問題，另完整報告 **19–23（45 筆）**，在「起點／終點」兩種常見解讀下都位於名義 18:00 之後；這仍不能取代資料實際到手時間的查核。

## 2. 怎麼做

1. 保留原 21 場 × 四站 × 六個目標來源時段，共 504 列。四站各自取得自己的前段人流，預測北門時使用北門的值。
2. 依年向前：2024 用更早 12 場訓練；2025 用更早 15 場；2026 用更早 18 場。相同九場北門 54 列作回顧比較。
3. **Ridge 與 XGBoost 都凍結 V12 各年度原固定參數。**原模型、進站 1／3／5 時段、進站＋目的站 1／3／5 時段，共七種版本；42 次最終擬合。本輪不增加搜尋、不早停、不改樣本權重、不新做融合。
4. 輸入為最近一時段數量、窗口累積量、首尾平均增量、較同站同來源時段一般日多出多少。1 時段不重複放相同累積量，也不定義趨勢。
5. 一般日沿用既有選定清單（每個活動車站 2–4 天），全部早於活動日；每一參考日要有完整前段來源時段，不用 0 填缺資料。
6. 前段快照在六個未來目標列保持不變。18:00 預測較晚時段，不填入 19／20 時的實際值。標準化只用當折訓練資料。
7. 所有窗口事先列定、完整公布，沒有用九場測試誤差選「最適窗口」。每場相同六列，所以全列 MAE 等於活動等權 MAE；尖峰只在評估時定位。

## 3. XGBoost 結果

{xt}

單位人次／來源時段。尖峰絕對誤差＝各場實際最高人流時段的預測絕對誤差再平均；尖峰正向低估＝max(實際−預測,0) 再平均。散場 2:1＝事前結束窗口加權的 2×低估＋高估，沿用原版定義。

![比較](figures/comparison.png)

- 最近 1 時段：散場 2:1 誤差 **1231.28 → 1130.13（降低 8.2%）**；尖峰絕對誤差 **749.02 → 707.42（降低 5.6%）**。
- 但尖峰正向低估 **556.60 → 596.33**，低估風險沒有隨平均絕對誤差一起降低。
- 2024/8/10 實際尖峰 11101，原版 8413.09，加入 1 時段變 **8085.17**；2026/8/15 實際 10904，原版 10101.25，變 **9872.16**。仍不能認為大型場尖峰已解決。

## 4. 不是每年都改善

{table(annual,['label',2024,2025,2026],['輸入版本','2024 MAE','2025 MAE','2026 MAE'])}

最近 1 時段在 2024／2025 的年度 MAE 改善，2026 變差，抵銷前兩年的好處。九場已被反覆檢視，這些是探索性結果，不是全新未見測試。

![年度](figures/annual.png)

## 5. Ridge 對照

{rt}

Ridge 加入兩方向前段量後，RMSE、尖峰絕對誤差與散場成本降低，卻提高全時段 MAE；both_w5 的平均偏差由 −396.87 變成 +248.82。這是誤差分布改變，不能只挑較好的指標宣稱全面改善。

## 6. 只看較晚來源時段 19–23

{table(ct,['label','mae','rmse'],['輸入版本','MAE（45 筆）','RMSE（45 筆）'])}

相同 45 筆上，最近 1 時段 MAE **532.96 → 513.46（改善 3.7%）**；不能拿 513.46 和全六時段的 473.91 直接比較。這個排除來源時段 18 的檢查在跑結果前就已寫入設定，目的是處理時間標示不明，並非看完結果再刪掉不好的時段。

![較晚時段](figures/target_window.png)

## 7. 九場曲線與尖峰

![逐場曲線](figures/beimen_profiles.png)

{table(peaks,['event_date','hour','y','xgboost_none','xgboost_origin_w1'],['活動日','實際尖峰來源時段','實際人次','原 XGBoost','加入進站 1 時段'])}

## 8. 目前能下的結論與下一步

- 已驗證：這份資料與原固定參數下，單純增加 3／5 個時段或目的站旅次量，沒有降低 XGBoost 的 MAE；短窗口有局部價值。
- **尚未證明變差原因。**新增特徵每場只增加一份前段訊號，不會把 21 場變成 504 個獨立活動。前段仍含一般通勤；搭到北門不代表一定去活動或同站離開。這些是合理解釋，尚非因果驗證。
- 固定參數能隔離「加輸入」的效果，但不代表已找到適合新特徵的最適模型；本輪沒有重新調參，不能據此宣稱前段資料永遠無用。
- 保留原模型，最近 1 時段版本列為候選；下一輪若要繼續，可在**過去年份內**選窗口、正則化及樹複雜度，或用前段超額人流校正原預測。須凍結規則後用新活動評估，不可直接用這九場挑最好結果。
- 尚不能接即時排隊部署：要取得有實際到手時間的進出站資料；還需確認分鐘到達形狀、行走延遲、搭乘方向與列車剩餘容量。目的站小時總量不能直接當分鐘到達率。

## 9. 檔案、重現與檢核

~~~powershell
.\\.venv\\Scripts\\python.exe -X utf8 src/run_analysis_v15.py
.\\.venv\\Scripts\\python.exe -X utf8 src/verify_analysis_v15.py
.\\.venv\\Scripts\\python.exe -X utf8 src/report_analysis_v15.py
.\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s tests -p 'test_analysis_v*.py'
~~~

需先有原 V12／V14 私有快照與其校驗來源、station_hourly.parquet、一般日清單；全新 clone 不包含這些原始資料。輸入、1680 個特徵值、逐列預測及模型保存在忽略追蹤的 .local/analysis_v15/；本目錄保存四圖、表格、模型設定及報告。執行後不自動換上網站模型。

獨立檢核：{check['source_hashes']} 個來源／程式雜湊、84 份活動車站快照、{check['feature_values_checked']} 個特徵值、{check['models_reloaded']} 個模型、{check['metric_rows']} 列指標；原 XGBoost 473.9095 及原 Ridge 均重現。最大特徵差 {check['max_feature_error']:.2e}，最大預測差 {check['max_prediction_error']:.2e}。細節見 [independent_validation.json](independent_validation.json)。

## 10. 方法依據與界線

- [臺北市官方分時 OD 資料目錄](https://data.taipei/dataset/detail?id=63f31c7e-7fc3-418b-bd82-b95158755b4d)：OD 逐時資料、每月更新。目的站統計存在，但本次查閱說明未明示時段刷卡基準。
- [scikit-learn：Lagged features for time series forecasting](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html)：示範落後值、移動統計及依時間切分。**1／3／5 小時與 18:00 是本案實驗設定，不是文獻保證的最優值。**
- [Hyndman & Athanasopoulos，FPP3 §5.10](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation)：向前回測以過去預測未來；本案按整年度隔離，同年其他活動不進訓練。
- [Cawley & Talbot（2010），JMLR](https://jmlr.org/papers/v11/cawley10a.html)：選模本身也有過擬合風險。本案反覆觀察九場，不將其標成獨立確認實驗。
"""
    (OUT/'report.md').write_text(report,encoding='utf8')
    print(OUT/'report.md')


if __name__=='__main__':run()
