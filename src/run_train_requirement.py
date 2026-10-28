"""Beimen trains needed per 15-minute slot versus the current timetable (add or reduce).

Scenarios are sampled as in run_dispatch_optimization.py. Headways are chosen
on one set of draws and both the timetable and the resulting schedule are
scored on another. Only Beimen boardings are modelled: added trains are a
lower bound, reductions are candidates until line-wide section loads are checked.
"""
import json
import sys
from datetime import date

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
import numpy as np
import pandas as pd

from dispatch_optimization import DirectionQueue
from metro_operations import ROOT,sha,collect
from run_dispatch_optimization import clock,evaluate_plan,sample_scenarios
from run_queue_uncertainty import NORTH,SOUTH,beimen_trains,load_cohort
from train_requirement import required_headways,schedule_from_headways,trains_per_slot


def main():
    config=sys.argv[1] if len(sys.argv)>1 else 'configs/train_requirement_v1.json'
    cfg=json.loads((ROOT/config).read_text(encoding='utf-8'))
    hours=range(cfg['first_hour'],cfg['last_hour']+1)
    forecasts,ratios=load_cohort(ROOT/cfg['prediction_path'],cfg['prediction_sha256'],
                                 cfg['prediction_column'],cfg['truth_column'],hours)
    snapshot=ROOT/'.local/transit_reference'/cfg['operations_snapshot']
    tables=json.loads((snapshot/'normalized.json').read_text(encoding='utf-8'))
    if tables!=collect(snapshot): raise ValueError('Normalized data differ from source PDF/CSV parsing')
    for s in tables['metro_source_snapshot']:
        if sha(snapshot/s['local_file'])!=s['content_sha256']: raise ValueError('Official source snapshot changed')
    maximum=tables['metro_capacity_reference'][0]['max_train_persons']
    slot,end=cfg['slot_minutes'],cfg['service_end_minutes']
    slots=end//slot
    arrival_end=len(hours)*60+cfg['platform_delay_minutes']
    if end<arrival_end+max(cfg['headway_options_minutes']):
        raise ValueError('Service window must leave a last train after the final arrivals')
    out=ROOT/cfg['output_dir'];(out/'figures').mkdir(parents=True,exist_ok=True)
    streams=np.random.SeedSequence(cfg['seed']).spawn(2*len(forecasts))
    rows,evals=[],[]
    for i,(event,hourly) in enumerate(forecasts.iterrows()):
        weekday=date.fromisoformat(event).weekday()
        if weekday not in [2,5]: raise ValueError('Event needs explicit holiday/service-day mapping')
        day='weekday' if weekday==2 else 'saturday'
        timetable=beimen_trains(tables,day,cfg['first_hour'])
        hourly=hourly.to_numpy()
        opt_arr,opt_spaces=sample_scenarios(hourly,ratios,event,cfg,maximum,np.random.default_rng(streams[2*i]),
                                            cfg['optimization_draws_per_event'])
        ev_arr,ev_spaces=sample_scenarios(hourly,ratios,event,cfg,maximum,np.random.default_rng(streams[2*i+1]),
                                          cfg['evaluation_draws_per_event'])
        required={}
        for d in (NORTH,SOUTH):
            queue=DirectionQueue(opt_arr[d],opt_spaces,end)
            headways,shares,feasible=required_headways(queue,slots,slot,end,cfg['headway_options_minutes'],
                                                       cfg['left_behind_epsilon'])
            required[d]=schedule_from_headways(headways,slot,end)
            need=trains_per_slot(required[d],slot,slots);now=trains_per_slot(timetable[d],slot,slots)
            for s in range(slots):
                rows.append(dict(event_date=event,day_type=day,direction=d,slot_start=clock(s*slot,cfg['first_hour']),
                                 slot_index=s,headway_minutes=headways[s],required_trains=int(need[s]),
                                 timetable_trains=int(now[s]),delta=int(need[s]-now[s]),
                                 opt_left_behind_share=shares[s],feasible=feasible[s]))
        none={NORTH:[],SOUTH:[]}
        for plan,trains in [('現行班表',timetable),('建議班表',required)]:
            evals.append(dict(event_date=event,day_type=day,plan=plan,
                              trains_in_window=sum(int(trains_per_slot(trains[d],slot,slots).sum()) for d in (NORTH,SOUTH)),
                              **evaluate_plan(ev_arr,ev_spaces,ev_spaces,trains,none,cfg['p95_target_minutes'])))
        print(event,'done',flush=True)
    req=pd.DataFrame(rows);req.to_csv(out/'requirement.csv',index=False,encoding='utf-8-sig')
    ev=pd.DataFrame(evals);ev.to_csv(out/'evaluation.csv',index=False,encoding='utf-8-sig')

    plt.rcParams.update({'font.family':'Microsoft JhengHei','axes.unicode_minus':False,'figure.dpi':140})
    events=sorted(req.event_date.unique())
    labels=[clock(s*slot,cfg['first_hour']) for s in range(slots)]
    span=max(1,int(req.delta.abs().max()))
    cmap=LinearSegmentedColormap.from_list('delta',['#3b6fb6','#ececec','#b6403b'])
    fig,axes=plt.subplots(2,1,figsize=(13,8.6),layout='constrained')
    for ax,d in zip(axes,(NORTH,SOUTH)):
        g=req[req.direction==d].pivot(index='event_date',columns='slot_index',values='delta').loc[events]
        image=ax.imshow(g.to_numpy(),cmap=cmap,norm=TwoSlopeNorm(0,-span,span),aspect='auto')
        bad=req[(req.direction==d)&~req.feasible]
        for r in bad.itertuples():
            ax.text(r.slot_index,events.index(r.event_date),'×',ha='center',va='center',fontsize=9,color='#222')
        for (y,x),v in np.ndenumerate(g.to_numpy()):
            if v!=0 and not ((bad.event_date==events[y])&(bad.slot_index==x)).any():
                ax.text(x,y,f'{v:+d}',ha='center',va='center',fontsize=6.5,color='#222')
        ax.set_yticks(range(len(events)),[f'{e}（{"六" if date.fromisoformat(e).weekday()==5 else "三"}）' for e in events])
        ax.set_xticks(range(0,slots,2),labels[::2]);ax.set_title(f'{d}：建議班數減現行班數（每 15 分鐘）')
        ax.set_xticks(np.arange(-.5,slots,1),minor=True);ax.set_yticks(np.arange(-.5,len(events),1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1.5);ax.tick_params(which='minor',length=0)
    bar=fig.colorbar(image,ax=axes,shrink=.6,label='班數差（紅＝要加班，藍＝減班候選）')
    fig.suptitle(f'北門每 15 分鐘所需班數 vs 現行一般班表（班距 {min(cfg["headway_options_minutes"])}–{max(cfg["headway_options_minutes"])} 分；'
                 f'跨班滯留機率不超過 {cfg["left_behind_epsilon"]:.0%}）\n×＝最短班距仍無法達標。只看北門上車：減班為候選，需全線站間負荷確認',fontsize=11)
    fig.savefig(out/'figures/requirement_heatmap.png');plt.close(fig)

    summary=req.groupby(['event_date','day_type','direction']).agg(
        required=('required_trains','sum'),timetable=('timetable_trains','sum'),
        add_slots=('delta',lambda s:int((s>0).sum())),reduce_slots=('delta',lambda s:int((s<0).sum())),
        infeasible_slots=('feasible',lambda s:int((~s).sum()))).reset_index()
    summary.to_csv(out/'summary.csv',index=False,encoding='utf-8-sig')
    lines=['# 北門每 15 分鐘所需班數（第一版）','',
    '**只看北門上車需求，從現行一般班表出發判斷加班或減班。加班數是下限；減班是「候選」，要用全線站間負荷確認後才能採用。**','',
    '## 1. 方法','',
    '| 項目 | 設定 |','|---|---|',
    f'| 範圍 | 北門雙向，18:00–{clock(end,cfg["first_hour"])}，每 {slot} 分鐘一段 |',
    f'| 決策 | 每段一個班距，候選 {min(cfg["headway_options_minutes"])}–{max(cfg["headway_options_minutes"])} 分鐘（上限為營運政策假設，下限為最小間隔假設） |',
    f'| 規則 | 依時間順序，每段選最長班距，使該段發車的班次「有人被留下」的情境 ≤ {cfg["left_behind_epsilon"]:.0%} |',
    f'| 為何能控制候車 | 沒有人被留下時，每人最多等一個班距；班距 ≤ {max(cfg["headway_options_minutes"])} 分即保證 P95 ≤ {cfg["p95_target_minutes"]} 分 |',
    f'| 需求 | 與候車不確定性模擬相同的情境（他場誤差、分鐘形狀、Poisson、方向、剩餘空間）；{cfg["optimization_draws_per_event"]} 個情境決定班距 |',
    f'| 評估 | 另外 {cfg["evaluation_draws_per_event"]} 個獨立情境，比較現行班表與建議班表 |','',
    '因為以 95% 情境為準，等同用需求的高分位數規劃；這對減班特別重要（模型對尖峰平均少估約 24.7%）。','',
    '## 2. 結果','','![所需班數熱圖](figures/requirement_heatmap.png)','',
    f'| 活動 | 方向 | 18:00–{clock(end,cfg["first_hour"])} 建議班數 | 現行班數 | 要加班的時段數 | 減班候選時段數 | 最短班距仍不足 |','|---|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.event_date} | {r.direction} | {r.required} | {r.timetable} | {r.add_slots} | {r.reduce_slots} | {r.infeasible_slots} |')
    lines+=['',f'獨立情境評估（P95 候車超過 {cfg["p95_target_minutes"]} 分鐘的機率）：','',
    '| 活動 | 現行：班數 | 現行：P95 中位數（95%） | 現行：超標機率 | 建議：班數 | 建議：P95 中位數（95%） | 建議：超標機率 |','|---|---:|---:|---:|---:|---:|---:|']
    for event,g in ev.groupby('event_date'):
        a=g[g.plan=='現行班表'].iloc[0];b=g[g.plan=='建議班表'].iloc[0]
        lines.append(f'| {event} | {a.trains_in_window} | {a.p95_wait_median:.2f}（{a.p95_wait_q95:.2f}） | {a.share_p95_over_target:.1%} | '
                     f'{b.trains_in_window} | {b.p95_wait_median:.2f}（{b.p95_wait_q95:.2f}） | {b.share_p95_over_target:.1%} |')
    lines+=['','## 3. 界線','',
    '- **只看北門**：一班車服務整條線，北門車廂空不代表其他區段空。減班候選需用全線起訖資料算最大站間負荷後再決定。',
    '- 最長班距 10 分、最短 2 分、5% 門檻、剩餘空間、分鐘形狀與方向比例都是假設；建議班表是條件解。',
    '- 建議班表是均勻班距，沒有考慮車隊、司機、折返、出庫與號誌限制，也沒有和其他時段的班表銜接。',
    '- 依時間順序逐段決定是啟發式；預測誤差只來自九場，且九場在開發時反覆使用。',
    '','逐段結果：[requirement.csv](requirement.csv)；摘要：[summary.csv](summary.csv)；評估：[evaluation.csv](evaluation.csv)。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit={'model_refitted':False,'uses_observed_y':'other_events_ratios_only','predictions_sha256':sha(ROOT/cfg['prediction_path']),
           'normalized_snapshot_sha256':sha(snapshot/'normalized.json'),'seed':cfg['seed'],
           'implementation_sha256':{p:sha(ROOT/p) for p in [config,'src/run_train_requirement.py','src/train_requirement.py',
                                    'src/dispatch_optimization.py','src/run_dispatch_optimization.py','src/run_queue_uncertainty.py',
                                    'src/queue_waits.py','src/queue_scenarios.py','src/queueing.py']},
           'outputs':{str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='audit.json'}}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False))


if __name__=='__main__':main()
