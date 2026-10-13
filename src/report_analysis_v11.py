"""Render the completed, fixed v11 experiment without changing its selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis_v1.reporting import style, save
from compare_fusion_losses import markdown

ROOT=Path(__file__).resolve().parents[1]
LABELS={'ridge':'Ridge','gam':'GAM','xgboost':'XGBoost'}


def run():
    out=ROOT/'result/analysis_v11';figdir=out/'figures';figdir.mkdir(exist_ok=True)
    summary=pd.read_csv(out/'tables/summary.csv')
    choices=json.loads((out/'selections.json').read_text('utf-8'))
    style()
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for i,variant in enumerate(['both','core']):
        frame=summary.loc[summary.variant.eq(variant)].set_index('method')
        for j,(metric,title) in enumerate([('mae','18–23 時 MAE'),('cost_2','散場加權誤差：2 × 低估 + 高估')]):
            ax=axes[i,j];x=np.arange(3)
            for offset,prefix,label,color in [(-1,'old_','原模型','#A7ADB0'),(0,'original_r2_','原網格＋新評分','#658778'),(1,'expanded_r2_','擴大網格＋新評分','#AA794D')]:
                values=[frame.loc[prefix+m,metric] for m in LABELS]
                bars=ax.bar(x+offset*.25,values,.25,label=label,color=color)
                ax.bar_label(bars,fmt='%.0f',fontsize=8,padding=3)
            ax.set(xticks=x,xticklabels=list(LABELS.values()),ylabel='人次／小時（加權）',title=f"{'煙火＋天氣' if variant=='both' else '核心輸入'}｜{title}")
            ax.set_ylim(0,max(p.get_height() for p in ax.patches)*1.2);ax.grid(axis='y',alpha=.15)
    axes[0,0].legend(fontsize=8,loc='upper right')
    fig.suptitle('北門 2024–2026 九場｜評分方式與搜尋範圍分開比較\n已反覆檢視的外層活動，屬探索性回顧；數值越低越好',fontsize=13)
    save(fig,figdir/'tuning_ablation.png')

    fig,axes=plt.subplots(1,3,figsize=(14,4.8),constrained_layout=True)
    for ax,model in zip(axes,LABELS):
        f=summary.loc[(summary.variant=='both')&(summary.model==model)&(summary.grid=='expanded')].sort_values('ratio')
        x=np.arange(3)
        ax.bar(x,2*f.window_under,color='#AA794D',label='低估貢獻 × 2')
        ax.bar(x,f.window_over,bottom=2*f.window_under,color='#658778',label='高估貢獻 × 1')
        for pos,total in zip(x,f.cost_2):ax.text(pos,total+50,f'{total:.0f}',ha='center',fontsize=9)
        ax.set(title=LABELS[model],xticks=x,xticklabels=['1:1','2:1','3:1'],xlabel='選參數時的低估：高估懲罰',ylabel='共同用 2:1 評估的誤差')
        ax.set_ylim(0,f.cost_2.max()*1.18);ax.grid(axis='y',alpha=.15)
    axes[0].legend(fontsize=9)
    fig.suptitle('懲罰比敏感度｜煙火＋天氣、擴大網格、相同九場\n三種選法都用相同 2:1 尺度比較，避免把不同目標值直接比大小',fontsize=12)
    save(fig,figdir/'penalty_sensitivity.png')

    p=pd.read_csv(ROOT/'.local/analysis_v11/both_predictions.csv',parse_dates=['event_date'])
    fig,axes=plt.subplots(3,3,figsize=(14,10),sharex=True,constrained_layout=True)
    for ax,(date,e) in zip(axes.flat,p.groupby('event_date')):
        e=e.sort_values('hour')
        for col,label,color,ls in [('y','實際','#171717','-'),('old_xgboost','原 XGBoost','#9A9A9A','--'),('expanded_r2_xgboost','新 XGBoost','#AA794D','-'),('expanded_r2_ridge','新 Ridge','#58748B',':'),('expanded_r2_gam','新 GAM','#658778','-.')]:
            ax.plot(e.hour,e[col],label=label,color=color,ls=ls,marker='o',markersize=3,lw=1.6)
        ax.set(title=f'{date:%Y-%m-%d}',xticks=range(18,24),xlabel='來源小時',ylabel='北門起站人次／小時')
        ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('北門全部九場｜原 XGBoost 與 2:1 擴大調參後三模型\n小時級曲線，不代表已觀測到分鐘到達形狀',fontsize=13)
    save(fig,figdir/'beimen_tuned_profiles.png')

    columns=['method','mae','rmse','cost_2','window_under','window_over','peak_under','peak_abs']
    parts=['# 第十一版：散場優先調參與分鐘到站資料依據','日期：2026-10-13。研究比較已完成；本報告的改善只指同一批回顧式九場，不是新活動驗證。',
           '## 1. 先看結果',
           '**Ridge 的低估下降，但 XGBoost 未改善；GAM 改變很小。擴大搜尋不是這次 Ridge 進步的主要原因。**',
           '主分析固定「煙火＋天氣」，仍使用四站訓練，只用北門預定展演結束後兩小時的加權誤差挑參數。下表 old 為舊模型；original 為原網格、新評分；expanded 為擴大網格、新評分。',
           markdown(summary.loc[(summary.variant=='both')&((summary.grid=='old')|summary.ratio.eq(2)),columns]),
           '- Ridge：MAE 1117.39 → 1000.13（下降 10.5%），散場 2:1 誤差 4112.39 → 3386.89（下降 17.6%）。散場低估由 1993.02 降至 1495.71，但高估由 126.36 增至 395.46。',
           '- 原網格只改評分的 Ridge 已達 MAE 982.30、2:1 誤差 3317.20；擴大網格反而略退步，不能把改善歸功於搜尋更多參數。',
           '- GAM：MAE 1108.64 → 1116.66；2:1 誤差 3341.43 → 3306.29（下降約 1.1%），沒有實質解決尖峰問題。',
           '- XGBoost：MAE 642.59 → 666.07；2:1 誤差 1595.61 → 1639.36（增加約 2.7%）。原網格與擴大網格在 2:1 選到相同設定，新增 90 組沒有勝出。',
           '- 因此保留原四站 XGBoost 作主要候選；本次結果不支持用新調參版取代它。沒有重新估融合權重，也沒有修改網站使用的模型。',
           '![搜尋範圍與評分比較](figures/tuning_ablation.png)',
           '## 2. 怎麼調參',
           '1. 固定 D−1 候選資訊、四站訓練與時間切分；core／both 分開報告，不用外層真值選輸入版本。\n2. 每個驗證年度只用更早年度訓練；整場活動、同年所有活動留在同一側。\n3. 每組參數產生過去北門的時間留出預測。依預定展演結束後 0–120 分鐘與來源小時的重疊比例評分，先場內加權再活動等權。\n4. 主評分為 2×低估＋1×高估；另以 1:1、3:1 各自重新選參數。\n5. 每個外層年度只用該年以前的留出誤差選參數，再用該年前全部四站資料重訓，預測該年。\n6. 三種懲罰比都報 MAE、RMSE、相同 2:1 尺度、低估／高估及實際尖峰同時刻誤差。',
           '| 模型 | 搜尋範圍 | 用途 |\n|---|---|---|\n| Ridge | 25 個 λ，10⁻⁴ 至 10² | 控制係數收縮；過小可能不穩定，過大可能欠擬合 |\n| GAM | 基底 4／5／7 × 平滑懲罰 5 組 × 全係數懲罰 4 組，共 60 組 | 同時調曲線彈性與穩定性；保留原訓練支持度檢查 |\n| XGBoost | 原 90 組＋固定種子抽取 90 組，共 180 組 | 聯合比較深度、樹數、學習率、葉門檻、抽樣與正則化 |',
           '**這次改的是驗證選參數的目標。** 三模型訓練仍用原本平方損失與正則化，沒有改成分量迴歸、沒有直接把預測乘 2。故 2:1 不保證每一場的預測都上升，也不能宣稱已估出 2/3 分位數。2 倍是使用者指定的研究偏好，還不是實測調度成本；高估的 1 倍懲罰也不是班次或高估的硬上限。',
           '完整網格、公式、日期缺漏與重現見 [process.md 第 22 節](../../process.md#22-第十一版散場優先調參與分鐘資料依據)。設定清單：[config](../../configs/analysis_v11.json)。',
           '## 3. 年度選出的參數（擴大網格、2:1）']
    for variant in ['both','core']:
        rows=[]
        for r in choices:
            if r['variant']==variant and r['grid']=='expanded' and r['ratio']==2:
                rows.append({'年度':r['outer_year'],'模型':LABELS[r['model']],'過去驗證場次':r['inner_events'],'設定':'; '.join(f'{k}={v:g}' for k,v in r['params'].items())})
        parts += [f'### {variant}',markdown(pd.DataFrame(rows).sort_values(['年度','模型']))]
    parts += ['參數可以跨年不同，因為可用的過去驗證場次增加；不能從哪個年度剛好好看反向改選參數。',
              '## 4. 1:1、2:1、3:1 敏感度',
              markdown(summary.loc[(summary.variant=='both')&(summary.grid=='expanded'),columns]),
              'GAM 的 2:1 與 3:1 結果相同；XGBoost 的 1:1 與 2:1 相同。3:1 的 XGBoost 在九場反而惡化，顯示加重懲罰並不等於跨年穩定改善，不能把主設定事後換成最好看的懲罰比。',
              '![懲罰敏感度](figures/penalty_sensitivity.png)',
              '## 5. 核心輸入對照與完整逐場曲線',
              markdown(summary.loc[(summary.variant=='core')&((summary.grid=='old')|summary.ratio.eq(2)),columns]),
              '核心輸入 XGBoost：原 MAE 504.20，原網格＋新評分 485.43，擴大網格＋新評分 555.12。較低 MAE 不一定有較小散場低估；三者 2:1 誤差依序為 1027.67、1042.57、1244.25。沒有用此外層結果自動切換輸入版本。',
              '![全部九場曲線](figures/beimen_tuned_profiles.png)',
              '## 6. 排隊理論：分鐘型態不能從逐時人次直接辨認',
              '現有資料可預測一小時有多少起站人次，但無法分辨這些人平均到站或集中在某十分鐘。已找到特殊活動的五分鐘實測研究、臺灣捷運十五分鐘實測圖，以及北捷 API 申請管道；尚未找到可直接下載的大稻埕活動日北門歷史一／五分鐘進站計數。',
              '下一步優先取得北門彙總刷卡計數、管制時段、列車時刻與剩餘容量；在取得前只能比較均勻／寬峰／集中峰情境，不能標成已估計的真實到達分布。完整來源、資料需求與銜接公式見 [分鐘到站依據](../../docs/minute_arrivals_evidence.md)。',
              '## 7. 資料性質與檢核',
              '- 21 場、四站、18–23 時，共 504 列是整體來源；每折只訓練較早年份。共同外層為 2024–2026 各三場，九場、北門 54 列。不是 504 個獨立活動。',
              '- 預定結束未知三場：2018-08-18、2023-07-05、2023-08-09，採六小時等權備援；2018 不進參數驗證，故各年度過去評分有兩場備援。九場外層都已知預定結束。',
              '- 重疊比例只對整小時的預測誤差加權，沒有將觀測人次裁切成已知散場分鐘人數。若想要真正的兩小時散場人次，仍需要更細時間資料。',
              '- 新增事後變數為零；D−1 發布時間限制仍沿用：OD 發布時間、日曆版本沒有完全核實，標記 retrospective_non_deployable。九場已被反覆查看，這輪屬探索性，不能聲稱是全新獨立測試。',
              '- [獨立驗證](independent_validation.json)：4,240 次候選擬合、48 個年度折、108 次選參數、56 個不同模型重載、546 列指標重算，數值核對通過；全專案 332 項測試通過（新增 16 項）。這是程式正確性檢查，不是泛化保證。',
              '完整數據：[整體](tables/summary.csv)／[年度](tables/annual.csv)／[逐場](tables/events.csv)／[參數與所有候選分數](selections.json)。',
              '## 8. 參考依據',
              '- [Gneiting（2011）, Making and Evaluating Point Forecasts](https://arxiv.org/abs/0912.0902)：損失函數應與預測目的相符；本案 2:1 數值與散場權重是研究設定。',
              '- [Bergstra & Bengio（2012）, Random Search for Hyper-Parameter Optimization](https://jmlr.org/papers/v13/bergstra12a.html)：有限預算的隨機搜尋依據；本次固定抽取 90 組，不保證全域最優。',
              '- [Cawley & Talbot（2010）](https://jmlr.org/papers/v11/cawley10a.html)：驗證準則本身也可能過擬合；擴大搜尋後外層惡化並不矛盾。',
              '- [FPP3：時間序列交叉驗證](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation)：以過去預測未來的評估方式。',
              '- [Ridge 官方參數](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)、[XGBoost 官方參數](https://xgboost.readthedocs.io/en/stable/parameter.html)。GAM 延續第七版經檢核的全係數懲罰設計，僅新增基底數候選。',
              '- 分鐘到站的官方來源與 Santanam 等（2021）文獻另見 [資料依據文件](../../docs/minute_arrivals_evidence.md)。']
    (out/'report.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    print(out/'report.md')


if __name__=='__main__':run()
