"""Public figures use retrospective held-out predictions, never fitted labels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plot_style import style,save
LABELS={'ridge':'Ridge','gam':'GAM','xgboost':'XGBoost 未加權','xgboost_weighted':'XGBoost 散場加權'}
COLORS=['#879eac','#8a9270','#c79d67','#377b6a']
def render(pred,data,models,cfg,out):
    style();figdir=out/'figures';figdir.mkdir(exist_ok=True)
    # Compare observed and predicted mean for each full event, not averaged peaks.
    means=pred.groupby('event_date')[['y',*LABELS]].mean()
    means.to_csv(out/'tables/event_means.csv',encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(12,4.7),constrained_layout=True)
    for i,(key,label,color) in enumerate(zip(means.columns,['實際',*LABELS.values()],['#333333',*COLORS])):
        ax.bar(np.arange(9)+(i-2)*.16,means[key],width=.15,label=label,color=color)
    ax.set(xticks=np.arange(9),xticklabels=means.index.strftime('%Y/%m/%d'),ylabel='北門每來源時段平均進站人次',
        title='每場活動六時段平均：實際與凍結模型預測')
    ax.tick_params(axis='x',rotation=25);ax.legend(ncol=5,loc='upper left',fontsize=9)
    ax.set_ylim(0,means.max().max()*1.2);save(fig,figdir/'event_means.png')
    e=pred.xgboost_weighted-pred.y
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    bins=np.arange(np.floor(e.min()/500)*500,np.ceil(e.max()/500)*500+501,500)
    axes[0].hist(e,bins=bins,color='#438276',edgecolor='white')
    axes[0].axvline(0,color='#444444',ls='--')
    axes[0].set(title='最終 XGBoost 誤差直方圖',xlabel='預測 − 實際（人次；負數為低估）',ylabel='來源時段筆數')
    for year,color in zip([2024,2025,2026],['#716690','#438276','#c49b59']):
        group=pred.loc[pred.year.eq(year)]
        axes[1].scatter(group.y,group.xgboost_weighted,color=color,s=25,label=str(year))
    axes[1].legend(title='評估年度',loc='upper left',fontsize=9)
    top=max(pred.y.max(),pred.xgboost_weighted.max())*1.05
    axes[1].plot([0,top],[0,top],ls='--',c='#555555')
    axes[1].set(xlim=(0,top),ylim=(0,top),xlabel='實際進站人次',ylabel='預測進站人次',title='預測對實際；虛線為完全一致')
    fig.suptitle('同九場北門 54 個來源時段｜回顧式驗證，不是新的獨立測試',fontsize=11)
    save(fig,figdir/'prediction_diagnostics.png')
    # North-only training panel prevents station composition from driving correlations.
    train=data.loc[data.year.lt(2026)&data.station.eq('北門')]
    columns=['hour','is_weekend','day_of_year','year_index','retrospective_baseline_56d',
             'planned_end_0_60_beimen','planned_end_60_120_beimen']
    names=['來源時段','週末','年內日期','年度趨勢','一般日基準','散場第 1 小時重疊','散場第 2 小時重疊']
    corr=train[columns].corr(method='spearman')
    corr.to_csv(out/'tables/feature_correlations.csv',encoding='utf-8-sig')
    counts=train[columns].notna().astype(int).T @ train[columns].notna().astype(int)
    counts.to_csv(out/'tables/feature_correlation_pair_counts.csv',encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(9,7),constrained_layout=True)
    im=ax.imshow(corr,cmap='RdBu_r',vmin=-1,vmax=1)
    for i in range(len(columns)):
        for j in range(len(columns)):
            val=corr.iloc[i,j]
            ax.text(j,i,'—' if pd.isna(val) else f'{val:.2f}',ha='center',va='center',
                    color='white' if abs(val)>.55 else '#222222',fontsize=10)
    ax.set(xticks=range(7),yticks=range(7),xticklabels=names,yticklabels=names,
        title='輸入變數 Spearman 相關｜北門、2026 前 18 場訓練資料\n重複時段不是獨立活動；未含當年測試與 y，不解讀為因果')
    ax.tick_params(axis='x',rotation=40);fig.colorbar(im,ax=ax,shrink=.75,label='Spearman ρ')
    save(fig,figdir/'feature_correlations.png')
    # Keep nine chronological profiles for peak/timing interpretation.
    fig,axes=plt.subplots(3,3,figsize=(13,9),constrained_layout=True)
    peak=[]
    for ax,(date,g) in zip(axes.flat,pred.groupby('event_date')):
        ax.plot(g.hour,g.y,c='#222222',marker='o',ms=3,label='實際')
        ax.plot(g.hour,g.xgboost_weighted,c='#438276',ls='--',marker='o',ms=3,label='最終 XGBoost')
        k=g.y.idxmax();row=g.loc[k]
        peak.append(dict(event_date=str(date.date()),hour=int(row.hour),actual=float(row.y),
                         predicted=float(row.xgboost_weighted),error=float(row.xgboost_weighted-row.y)))
        ax.set(title=str(date.date()),xticks=range(18,24),xlabel='來源時段',ylabel='北門進站人次',ylim=(0,None))
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2)
    fig.suptitle('最終模型九場逐時對照｜尖峰低估及時段偏移仍須保留')
    save(fig,figdir/'hourly_profiles.png')
    pd.DataFrame(peak).to_csv(out/'tables/actual_peaks.csv',index=False,encoding='utf-8-sig')
    # GAM components and tree split importance are predictive descriptions only.
    curves=models[2026,'gam'].effect_curves(data.loc[data.year.lt(2026)])
    curves.to_csv(out/'tables/gam_components_2026.csv',index=False,encoding='utf-8-sig')
    tree=models[2026,'xgboost_weighted']
    imp=pd.DataFrame({'feature':tree.encoder.names,'gain_importance':tree.estimator.feature_importances_}).sort_values('gain_importance',ascending=False)
    imp.to_csv(out/'tables/tree_importance_2026.csv',index=False,encoding='utf-8-sig')
    metrics=pd.read_csv(out/'tables/metrics.csv')
    table='| 模型 | 北門 MAE | RMSE | 尖峰平均絕對誤差 |\n|---|---:|---:|---:|\n'
    table+='\n'.join(f"| {LABELS[r.model]} | {r.mae:.2f} | {r.rmse:.2f} | {r.peak_abs:.2f} |" for r in metrics.itertuples())
    report=f'''# 最終人流模型結果

保留四站散場加權 XGBoost，Ridge、GAM 為比較模型；未加權 XGBoost 為訓練加權的對照。這是整理原結果，沒有新調參或把當年答案放回訓練。

{table}

- 資料為 21 場活動、四站、18–23 六個來源時段；504 列不是 504 個獨立活動。
- 預測 2024／2025／2026 時，分別僅用過去 12／15／18 場擬合；主要評分為同九場北門 54 列。
- 現有歷史 OD 與日曆的原始發布時間未完全核實，因此是回顧敏感度研究，不能稱已驗證嚴格 D−1 部署。
- 九場已反覆用於開發。保留乘數 4 的決定受這些外層比較影響，尚需新活動驗證，不能把 440.28 稱為無偏泛化保證。
- 基礎參數原先在較早年度的留出預測上選取；一般日基準加入後保留原參數。細節及完整格點保存在 [調參紀錄](tuning_record.json)。
- 四站減至兩站的比較留在 [車站範圍摘要](tables/station_scope_comparison.csv)，不再保留逐版入口。

## 圖表怎麼讀

![每場平均](figures/event_means.png)

每一組是同一場活動六時段平均。這是分組長條圖；平均接近不代表每小時或最高峰也接近。

![誤差直方圖](figures/prediction_diagnostics.png)

直方圖負值表示低估，正值表示高估；每格 500 人次。散點越靠近虛線越準，遠離虛線表示該時段誤差大。54 列同屬九場，不能當獨立樣本做一般推論。

![逐時](figures/hourly_profiles.png)

最嚴重尖峰低估仍在 2024/8/10：實際 11,101、預測約 8,359，少估約 2,742（24.7%）。候車需保留需求與剩餘容量敏感度。

![變數相關](figures/feature_correlations.png)

熱力圖為 2026 前的北門訓練資料，採兩兩非缺值 Spearman；相應筆數另存表格。它是七項代表性輸入的描述，不是超參數相關、因果效果或刪變數規則。車站類別、其他站散場交互項與缺值旗標未列入此七欄圖。

## 公式、係數與模型細節

- Ridge 的完整數值公式、訓練均數／尺度、補值、參考車站與時段：[JSON](ridge_equation_2026.json)；[係數表](tables/ridge_coefficients_2026.csv)。使用 2026 評估折，即僅訓練至 2025，不是用 2026 再擬合的因果模型。
- GAM 的逐項平滑／線性設定與解算檢核：[擬合紀錄](fits.json)；[中心化曲線](tables/gam_components_2026.csv)。本案一般日基準是線性項，不能宣稱所有連續變數都畫成曲線。
- XGBoost：[分裂增益重要度](tables/tree_importance_2026.csv)。只描述 2026 折訓練模型如何使用特徵，不是因果影響或新樣本上的重要度保證。
- [年度指標](tables/annual_metrics.csv)／[各場真實尖峰](tables/actual_peaks.csv)／[逐欄重現檢核](validation.json)／[歷史來源](provenance.json)。
- 等權／學權重融合曾試驗但未採用；不把不同輸入或不同調參目標的歷史融合數字混進這張相同資料的結果表。

[README](../../README.md)／[完整實作流程與公式](../../process.md)
'''
    (out/'report.md').write_text(report,encoding='utf8')
