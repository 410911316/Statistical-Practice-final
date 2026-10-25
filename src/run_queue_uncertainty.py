"""Monte Carlo waiting-time intervals around the frozen Beimen forecasts.

Unlike V1 (run_queue_scenarios.py), this reads observed y, but only to form
actual/forecast ratios of the OTHER eight events: an event never borrows its
own error. Each draw also samples an assumed minute shape, Poisson arrivals,
direction share and remaining capacity, then reports pooled FIFO waits.
"""
import json
from datetime import date

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metro_operations import ROOT,sha,collect
from queue_scenarios import shaped_arrivals
from queue_waits import fifo_wait_pieces,summarize_wait_pieces

LABELS={'uniform':'均勻','broad':'寬峰（σ=20 分）','concentrated':'集中峰（σ=5 分）'}
COLORS={'uniform':'#467f88','broad':'#bd9758','concentrated':'#ae5360'}
NORTH,SOUTH='往松山','往新店／台電大樓'


def load_cohort(path,expected_hash,prediction,truth,hours):
    """Return (forecasts, actual/forecast ratios) as event × hour tables."""
    if sha(path)!=expected_hash: raise ValueError('Frozen model predictions changed')
    data=pd.read_csv(path,usecols=['event_date','station','hour',prediction,truth])
    if len(data)!=54 or set(data.station)!={'北門'} or data.event_date.nunique()!=9:
        raise ValueError('Expected frozen nine-event Beimen cohort')
    forecasts=data.pivot(index='event_date',columns='hour',values=prediction)
    truths=data.pivot(index='event_date',columns='hour',values=truth)
    if list(forecasts.columns)!=list(hours) or forecasts.isna().any().any() or truths.isna().any().any():
        raise ValueError('Incomplete hourly cohort')
    if (forecasts<=0).any().any(): raise ValueError('Ratios need positive forecasts')
    return forecasts,truths/forecasts


def donor_ratios(ratios,event,rng):
    """Draw another event's six hourly ratios together, keeping within-event correlation."""
    donors=[e for e in ratios.index if e!=event]
    donor=donors[rng.integers(len(donors))]
    return donor,ratios.loc[donor].to_numpy()


def beimen_trains(tables,day,first_hour):
    """Official Beimen departures (minutes after first_hour) for each direction."""
    trains={}
    for table in (t for t in tables['metro_timetable'] if t['station_label']=='北門'):
        north='松山' in table['direction_text']
        used=[r for r in tables['metro_departure'] if r['source_id']==table['source_id']
              and r['day_type']==day and r['departure_minute']>=first_hour*60]
        # Same guard as V1: short turns would need destination-compatible boarding.
        if any(r['terminal']!=('松山' if north else '新店') for r in used):
            raise ValueError('Evening short turns require destination-compatible boarding model')
        trains[NORTH if north else SOUTH]=sorted(r['departure_minute']-first_hour*60 for r in used)
    if set(trains)!={NORTH,SOUTH}: raise ValueError('Expected two Beimen directions')
    return trains


def simulate_draw(hourly,shape,sigma,center,share,spaces,trains,delay,rng=None,
                  quantiles=(.5,.9,.95),thresholds=()):
    """One scenario; rng=None keeps the deterministic V1 fluid arrivals."""
    rate=shaped_arrivals(hourly,shape,sigma,center,delay)
    if rng is None:
        arrivals={NORTH:rate*share,SOUTH:rate*(1-share)}
    else:
        # Poisson per minute, then binomial thinning = independent Poisson by direction.
        total=rng.poisson(rate);north=rng.binomial(total,share)
        arrivals={NORTH:north.astype(float),SOUTH:(total-north).astype(float)}
    pieces,row=[],{}
    cleared,residual,left_behind,worst_queue=True,0.,0,0.
    for direction,counts in arrivals.items():
        times=trains[direction]
        queue,p=fifo_wait_pieces(counts,times,[spaces]*len(times))
        pieces.append(p)
        cleared&=queue['cleared'];residual+=queue['residual_queue']
        left_behind+=sum(t['left_behind']>1e-7 for t in queue['train_records'])
        worst_queue=max(worst_queue,queue['max_queue'])
        row[f'mean_wait_{direction}']=queue['mean_wait_minutes']
    row.update(cleared=cleared,residual_queue=residual,demand=float(sum(a.sum() for a in arrivals.values())),
               train_departures_with_left_behind=left_behind,max_direction_queue=worst_queue)
    if cleared:
        waits=summarize_wait_pieces(np.vstack(pieces),quantiles,thresholds)
        row.update(mean_wait_minutes=waits['mean_wait_minutes'],max_wait_minutes=waits['max_wait_minutes'])
        row.update({f'p{round(q*100)}_wait_minutes':v for q,v in waits['quantiles'].items()})
        row.update({f'share_wait_over_{t:g}_min':v for t,v in waits['share_waiting_longer_than'].items()})
    return row


def interval(series):
    s=series.dropna()
    return pd.Series({'q05':s.quantile(.05),'median':s.median(),'q95':s.quantile(.95)})


def main():
    cfg=json.loads((ROOT/'configs/queue_uncertainty_v1.json').read_text(encoding='utf-8'))
    hours=range(cfg['first_hour'],cfg['last_hour']+1)
    forecasts,ratios=load_cohort(ROOT/cfg['prediction_path'],cfg['prediction_sha256'],
                                 cfg['prediction_column'],cfg['truth_column'],hours)
    snapshot=ROOT/'.local/transit_reference'/cfg['operations_snapshot']
    tables=json.loads((snapshot/'normalized.json').read_text(encoding='utf-8'))
    if tables!=collect(snapshot): raise ValueError('Normalized data differ from source PDF/CSV parsing')
    for s in tables['metro_source_snapshot']:
        if sha(snapshot/s['local_file'])!=s['content_sha256']: raise ValueError('Official source snapshot changed')
    maximum=tables['metro_capacity_reference'][0]['max_train_persons']
    delay,q,th=cfg['platform_delay_minutes'],cfg['wait_quantiles'],cfg['wait_thresholds_minutes']
    shapes=list(cfg['shapes'])
    out=ROOT/'result/queue_uncertainty_v1';(out/'figures').mkdir(parents=True,exist_ok=True)
    v1=pd.read_csv(ROOT/'result/queue_dispatch_v1/event_summary.csv')
    rows=[];max_v1_gap=0.
    streams=np.random.SeedSequence(cfg['seed']).spawn(len(forecasts))
    for stream,(event,hourly) in zip(streams,forecasts.iterrows()):
        weekday=date.fromisoformat(event).weekday()
        if weekday not in [2,5]: raise ValueError('Event needs explicit holiday/service-day mapping')
        trains=beimen_trains(tables,'weekday' if weekday==2 else 'saturday',cfg['first_hour'])
        hourly=hourly.to_numpy()
        # Anchor: ratio 1, fluid arrivals and V1 settings must reproduce V1 exactly.
        for shape in shapes:
            for fraction in (.25,.5,.75):
                ref=simulate_draw(hourly,shape,cfg['shapes'][shape],30,.5,maximum*fraction,trains,delay)
                expected=v1[(v1.event_date==event)&(v1['shape']==shape)&(v1.remaining_fraction==fraction)]
                max_v1_gap=max(max_v1_gap,abs(ref['mean_wait_minutes']-expected.mean_wait_minutes.iloc[0]))
        rng=np.random.default_rng(stream)
        for draw in range(cfg['draws_per_event']):
            donor,ratio=donor_ratios(ratios,event,rng)
            shape=shapes[rng.integers(len(shapes))]
            center=int(rng.choice(cfg['peak_centers_minutes'])) if shape!='uniform' else 30
            share=rng.uniform(*cfg['northbound_share_range'])
            fraction=rng.uniform(*cfg['remaining_capacity_fraction_range'])
            row=simulate_draw(hourly*ratio,shape,cfg['shapes'][shape],center,share,maximum*fraction,
                              trains,delay,rng,q,th)
            rows.append(dict(event_date=event,draw=draw,donor_event=donor,shape=shape,center_minute=center,
                             northbound_share=share,remaining_fraction=fraction,
                             forecast_demand=float(hourly.sum()),demand_ratio=row['demand']/hourly.sum(),**row))
    if max_v1_gap>1e-9: raise AssertionError(f'Deterministic anchor differs from V1 by {max_v1_gap}')
    draws=pd.DataFrame(rows);draws.to_csv(out/'draws.csv',index=False,encoding='utf-8-sig')
    metrics=['mean_wait_minutes','p95_wait_minutes']
    per_event=[]
    for event,g in draws.groupby('event_date'):
        row={'event_date':event,'draws':len(g),'censored_share':1-g.cleared.mean(),
             'share_draws_with_left_behind':(g.train_departures_with_left_behind>0).mean(),
             'share_draws_p95_over_10_min':(g.p95_wait_minutes>10).mean()}
        for m in metrics:row.update({f'{m}_{k}':v for k,v in interval(g[m]).items()})
        per_event.append(row)
    per_event=pd.DataFrame(per_event);per_event.to_csv(out/'event_intervals.csv',index=False,encoding='utf-8-sig')
    # Factor bins are for reading sensitivity, not a causal decomposition.
    draws['capacity_bin']=pd.cut(draws.remaining_fraction,[.25,5/12,7/12,.75],include_lowest=True,
                                 labels=['25–42%','42–58%','58–75%'])
    # Tertiles within each event: pooled tertiles mix in Saturday's longer headways.
    draws['demand_bin']=draws.groupby('event_date').demand_ratio.transform(
        lambda s:pd.qcut(s,3,labels=False)).map(dict(enumerate(['需求偏低三分位','中三分位','需求偏高三分位'])))
    draws['demand_bin']=pd.Categorical(draws.demand_bin,['需求偏低三分位','中三分位','需求偏高三分位'],ordered=True)
    factors=[]
    for factor,label in [('shape','到達形狀'),('capacity_bin','剩餘容量'),('demand_bin','需求／預測比')]:
        for level,g in draws.groupby(factor,observed=True):
            factors.append(dict(factor=label,level=LABELS.get(level,level),draws=len(g),
                                **{f'p95_{k}':v for k,v in interval(g.p95_wait_minutes).items()},
                                share_left_behind=(g.train_departures_with_left_behind>0).mean()))
    factors=pd.DataFrame(factors);factors.to_csv(out/'factor_summary.csv',index=False,encoding='utf-8-sig')

    plt.rcParams.update({'font.family':'Microsoft JhengHei','axes.unicode_minus':False,'figure.dpi':140})
    fig,axes=plt.subplots(1,2,figsize=(11,5.6),sharey=True,layout='constrained')
    order=per_event.sort_values('event_date',ascending=False).reset_index(drop=True)
    for ax,m,title in zip(axes,metrics,['平均候車（分鐘）','P95 候車（分鐘）']):
        lo,mid,hi=order[f'{m}_q05'],order[f'{m}_median'],order[f'{m}_q95']
        ax.hlines(order.index,lo,hi,color='#467f88',linewidth=2)
        ax.plot(mid,order.index,'o',color='#467f88',markersize=7,markeredgecolor='white',markeredgewidth=1.5)
        for i,(a,b,c) in enumerate(zip(lo,mid,hi)):
            ax.annotate(f'{b:.1f}',(c,i),xytext=(5,-3),textcoords='offset points',fontsize=8,color='#555')
        ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.set_xlim(left=0)
    axes[0].set_yticks(order.index,order.event_date)
    fig.suptitle(f'北門九場：每場 {cfg["draws_per_event"]:,} 次情境的 5–95% 區間（點＝中位數）\n'
                 '情境混合區間，不是已校準的實際候車預測區間',fontsize=11)
    fig.savefig(out/'figures/event_intervals.png');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,4.6),sharey=True,layout='constrained')
    panels=[('shape',[s for s in shapes],[LABELS[s] for s in shapes],'到達形狀'),
            ('capacity_bin',['25–42%','42–58%','58–75%'],['25–42%','42–58%','58–75%'],'每班剩餘空間／2,220 人'),
            ('demand_bin',['需求偏低三分位','中三分位','需求偏高三分位'],['偏低','中間','偏高'],'模擬需求／模型預測（各場內三分位）')]
    for ax,(factor,levels,ticks,title) in zip(axes,panels):
        data=[draws.loc[draws[factor]==l,'p95_wait_minutes'].dropna() for l in levels]
        box=ax.boxplot(data,whis=(5,95),showfliers=False,patch_artist=True,widths=.5)
        for patch in box['boxes']:patch.set(facecolor='#dfe9ea',edgecolor='#467f88')
        for med in box['medians']:med.set(color='#2b4f55',linewidth=2)
        ax.set_xticks(range(1,len(ticks)+1),ticks);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('P95 候車（分鐘）')
    fig.suptitle('哪個假設最影響 P95 候車？（箱＝四分位，鬚＝5–95%；九場所有情境合併）',fontsize=11)
    fig.savefig(out/'figures/factor_effects.png');plt.close(fig)

    overall={m:interval(draws[m]).to_dict() for m in metrics}
    lines=['# 北門候車不確定性模擬（第一版）','',
    '**固定 V14 四站 XGBoost 的北門預測，沒有重新訓練。區間是「情境混合」：結合其他場次的預測誤差、假設的分鐘形狀、隨機到達與假設剩餘容量。不是已校準的實際候車預測區間。**','',
    '## 1. 每次模擬抽什麼','',
    f'每場 {cfg["draws_per_event"]:,} 次，共 {len(draws):,} 次；亂數種子 {cfg["seed"]}，重跑結果相同。','',
    '| 不確定性 | 抽法 | 依據 |','|---|---|---|',
    '| 小時人數 | 預測 × 另一場的「實際／預測」六時段比例（整組一起抽） | 九場留出預測；**絕不使用本場自己的實際值** |',
    '| 分鐘形狀 | 均勻／寬峰／集中峰各 1/3；峰中心第 15／30／45 分 | V1 假設，未經分鐘資料校準 |',
    '| 隨機到達 | 每分鐘 Poisson，再以二項分布拆方向 | 到達彼此獨立的假設 |',
    f'| 方向比例 | 往松山 {cfg["northbound_share_range"][0]:.0%}–{cfg["northbound_share_range"][1]:.0%} 均勻 | V1 敏感度範圍 |',
    f'| 剩餘容量 | 2,220 人的 {cfg["remaining_capacity_fraction_range"][0]:.0%}–{cfg["remaining_capacity_fraction_range"][1]:.0%} 均勻 | V1 敏感度範圍，非量測值 |','',
    '等待時間以累積到達與累積上車曲線的水平距離計算（先到先上），兩方向合併後取 P95。',
    f'檢核：比例＝1、無隨機到達、V1 設定時，九場 × 三形狀 × 三容量的平均候車與 V1 最大差 {max_v1_gap:.1e} 分。','',
    '## 2. 結果','',
    '| 指標 | 5% | 中位數 | 95% |','|---|---:|---:|---:|']
    for m,label in zip(metrics,['平均候車（分）','P95 候車（分）']):
        lines.append(f'| {label} | {overall[m]["q05"]:.2f} | {overall[m]["median"]:.2f} | {overall[m]["q95"]:.2f} |')
    lines+=['',f'- 有乘客跨班滯留的情境：{(draws.train_departures_with_left_behind>0).mean():.1%}；P95 超過 10 分鐘：{(draws.p95_wait_minutes>10).mean():.1%}；末班後仍未清空：{1-draws.cleared.mean():.2%}。',
    '- 沒有人被留下時，P95 約等於 0.95 個班距，所以有底線：平日場約 5.8 分、週六場約 7 分（週六晚間班距較長）。超過底線的長尾都來自跨班滯留。',
    '','![各場區間](figures/event_intervals.png)','',
    '| 活動 | 平均候車 中位數（5–95%） | P95 候車 中位數（5–95%） | 跨班滯留機率 |','|---|---:|---:|---:|']
    for r in per_event.itertuples():
        lines.append(f'| {r.event_date} | {r.mean_wait_minutes_median:.2f}（{r.mean_wait_minutes_q05:.2f}–{r.mean_wait_minutes_q95:.2f}） | '
                     f'{r.p95_wait_minutes_median:.2f}（{r.p95_wait_minutes_q05:.2f}–{r.p95_wait_minutes_q95:.2f}） | {r.share_draws_with_left_behind:.1%} |')
    lines+=['','## 3. 哪個假設影響最大','','![因子比較](figures/factor_effects.png)','',
    '| 因子 | 水準 | P95 中位數 | P95 95% | 跨班滯留機率 |','|---|---|---:|---:|---:|']
    for r in factors.itertuples():
        lines.append(f'| {r.factor} | {r.level} | {r.p95_median:.2f} | {r.p95_q95:.2f} | {r.share_left_behind:.1%} |')
    lines+=['','分組只供閱讀敏感度；因子同時抽樣，不是因果分解。需求比例在各場內切三分位，避免平日／週六班距差異混入。','',
    '## 4. 界線','',
    '- 預測誤差只來自這九場，而且九場在開發時反覆使用，誤差可能被低估；區間很可能偏窄。',
    '- 比例依「同一時段」借用，另一場的尖峰時段可能不同。',
    '- 分鐘形狀、方向比例、剩餘容量與 5 分鐘步行延遲都仍是假設；上車瞬間完成，不含車門吞吐與月台管制。',
    '- 使用 2026/8/30 起的一般班表，不是當年活動加班表；不是歷史候車重建。',
    '','逐次結果：[draws.csv](draws.csv)；逐場：[event_intervals.csv](event_intervals.csv)；因子：[factor_summary.csv](factor_summary.csv)。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit={'model_refitted':False,'uses_observed_y':'other_events_ratios_only','predictions_sha256':sha(ROOT/cfg['prediction_path']),
           'normalized_snapshot_sha256':sha(snapshot/'normalized.json'),'seed':cfg['seed'],'draws':len(draws),
           'max_v1_anchor_gap_minutes':max_v1_gap,'censored_share':float(1-draws.cleared.mean()),
           'implementation_sha256':{p:sha(ROOT/p) for p in ['configs/queue_uncertainty_v1.json','src/run_queue_uncertainty.py',
                                    'src/queue_waits.py','src/queue_scenarios.py','src/queueing.py']},
           'outputs':{str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='audit.json'}}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(overall).T.to_string());print(json.dumps({k:v for k,v in audit.items() if k not in ('outputs','implementation_sha256')},indent=2))


if __name__=='__main__':main()
