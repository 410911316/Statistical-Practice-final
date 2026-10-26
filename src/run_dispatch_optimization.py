"""Extra Beimen trains: optimize on sampled scenarios, evaluate on independent draws.

Scenarios are sampled exactly like run_queue_uncertainty.py (other events'
actual/forecast ratios, assumed minute shapes, Poisson arrivals, direction
share, remaining capacity). Plans are chosen on one set of draws and scored
on another, so the reported improvement is not the optimizer's own sample.
"""
import json
import sys
from datetime import date

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dispatch_optimization import DirectionQueue,optimize_insertions
from metro_operations import ROOT,sha,collect
from queue_scenarios import shaped_arrivals
from queue_waits import fifo_wait_pieces,summarize_wait_pieces
from run_queue_uncertainty import NORTH,SOUTH,beimen_trains,donor_ratios,interval,load_cohort


def sample_scenarios(hourly,ratios,event,cfg,maximum,rng,count):
    """Return ({direction: scenarios × minutes arrivals}, remaining spaces per scenario)."""
    shapes=list(cfg['shapes']);arrivals={NORTH:[],SOUTH:[]};spaces=[]
    for _ in range(count):
        _,ratio=donor_ratios(ratios,event,rng)
        shape=shapes[rng.integers(len(shapes))]
        center=int(rng.choice(cfg['peak_centers_minutes'])) if shape!='uniform' else 30
        share=rng.uniform(*cfg['northbound_share_range'])
        spaces.append(maximum*rng.uniform(*cfg['remaining_capacity_fraction_range']))
        total=rng.poisson(shaped_arrivals(hourly*ratio,shape,cfg['shapes'][shape],center,cfg['platform_delay_minutes']))
        north=rng.binomial(total,share)
        arrivals[NORTH].append(north);arrivals[SOUTH].append(total-north)
    return {d:np.array(v,dtype=float) for d,v in arrivals.items()},np.array(spaces)


def evaluate_plan(arrivals,spaces,extra_spaces,trains,extras,threshold):
    """Pooled FIFO waits per evaluation scenario for one plan."""
    rows=[]
    for s in range(len(spaces)):
        pieces,cleared,left=[],True,0
        for d in (NORTH,SOUTH):
            times=sorted(list(trains[d])+list(extras[d]))
            capacities=[extra_spaces[s] if t in extras[d] else spaces[s] for t in times]
            queue,p=fifo_wait_pieces(arrivals[d][s],times,capacities)
            pieces.append(p);cleared&=queue['cleared']
            left+=sum(t['left_behind']>1e-7 for t in queue['train_records'])
        row={'cleared':cleared,'left_behind':left,'mean_wait_minutes':np.nan,'p95_wait_minutes':np.nan}
        if cleared:
            w=summarize_wait_pieces(np.vstack(pieces),[.95])
            row.update(mean_wait_minutes=w['mean_wait_minutes'],p95_wait_minutes=w['quantiles'][.95])
        rows.append(row)
    rows=pd.DataFrame(rows)
    # Uncleared draws count as exceeding the target rather than being dropped.
    over=(rows.p95_wait_minutes>threshold)|~rows.cleared
    return {**{f'mean_wait_{k}':v for k,v in interval(rows.mean_wait_minutes).items()},
            **{f'p95_wait_{k}':v for k,v in interval(rows.p95_wait_minutes).items()},
            'share_left_behind':(rows.left_behind>0).mean(),'share_p95_over_target':over.mean(),
            'censored_share':1-rows.cleared.mean()}


def clock(minute,first_hour):
    total=first_hour*60+int(minute)
    return f'{total//60%24:02d}:{total%60:02d}'


def main():
    config=sys.argv[1] if len(sys.argv)>1 else 'configs/dispatch_optimization_v1.json'
    cfg=json.loads((ROOT/config).read_text(encoding='utf-8'))
    extra=cfg['extra_train_capacity']
    if extra!='same_as_regular' and not (isinstance(extra,dict) and 0<extra.get('remaining_fraction',0)<=1):
        raise ValueError('extra_train_capacity must be same_as_regular or {"remaining_fraction": (0, 1]}')
    hours=range(cfg['first_hour'],cfg['last_hour']+1)
    forecasts,ratios=load_cohort(ROOT/cfg['prediction_path'],cfg['prediction_sha256'],
                                 cfg['prediction_column'],cfg['truth_column'],hours)
    snapshot=ROOT/'.local/transit_reference'/cfg['operations_snapshot']
    tables=json.loads((snapshot/'normalized.json').read_text(encoding='utf-8'))
    if tables!=collect(snapshot): raise ValueError('Normalized data differ from source PDF/CSV parsing')
    for s in tables['metro_source_snapshot']:
        if sha(snapshot/s['local_file'])!=s['content_sha256']: raise ValueError('Official source snapshot changed')
    maximum=tables['metro_capacity_reference'][0]['max_train_persons']
    target,window=cfg['p95_target_minutes'],tuple(cfg['insertion_window_minutes'])
    out=ROOT/cfg.get('output_dir','result/dispatch_optimization_v1');(out/'figures').mkdir(parents=True,exist_ok=True)
    streams=np.random.SeedSequence(cfg['seed']).spawn(2*len(forecasts))
    plan_rows,eval_rows=[],[]
    for i,(event,hourly) in enumerate(forecasts.iterrows()):
        weekday=date.fromisoformat(event).weekday()
        if weekday not in [2,5]: raise ValueError('Event needs explicit holiday/service-day mapping')
        day='weekday' if weekday==2 else 'saturday'
        trains=beimen_trains(tables,day,cfg['first_hour'])
        if max(max(t) for t in trains.values())>cfg['horizon_minutes']: raise ValueError('Horizon too short')
        hourly=hourly.to_numpy()
        # Separate streams: optimization draws never reappear in evaluation.
        opt_arr,opt_spaces=sample_scenarios(hourly,ratios,event,cfg,maximum,np.random.default_rng(streams[2*i]),
                                            cfg['optimization_draws_per_event'])
        ev_arr,ev_spaces=sample_scenarios(hourly,ratios,event,cfg,maximum,np.random.default_rng(streams[2*i+1]),
                                          cfg['evaluation_draws_per_event'])
        opt_extra,ev_extra=((opt_spaces,ev_spaces) if extra=='same_as_regular' else
                            (np.full(len(opt_spaces),maximum*extra['remaining_fraction']),
                             np.full(len(ev_spaces),maximum*extra['remaining_fraction'])))
        queues={d:DirectionQueue(opt_arr[d],opt_spaces,cfg['horizon_minutes'],opt_extra) for d in (NORTH,SOUTH)}
        for separation in cfg['min_separation_minutes']:
            plans=optimize_insertions(queues,trains,cfg['max_extra_trains'],separation,window,
                                      cfg['residual_penalty_minutes'])
            for n,extras in enumerate(plans):
                for d in (NORTH,SOUTH):
                    plan_rows.append(dict(event_date=event,day_type=day,min_separation=separation,n_extra=n,
                                          direction=d,inserted=' '.join(clock(m,cfg['first_hour']) for m in extras[d])))
                eval_rows.append(dict(event_date=event,day_type=day,min_separation=separation,n_extra=n,
                                      **evaluate_plan(ev_arr,ev_spaces,ev_extra,trains,extras,target)))
            print(event,separation,'done',flush=True)
    plans=pd.DataFrame(plan_rows);plans.to_csv(out/'plans.csv',index=False,encoding='utf-8-sig')
    evaluation=pd.DataFrame(eval_rows);evaluation.to_csv(out/'evaluation.csv',index=False,encoding='utf-8-sig')
    needed=[]
    for (event,separation),g in evaluation.groupby(['event_date','min_separation']):
        ok=g[g.share_p95_over_target<=cfg['max_exceed_probability']]
        needed.append(dict(event_date=event,min_separation=separation,
                           baseline_share_p95_over_target=g[g.n_extra==0].share_p95_over_target.iloc[0],
                           min_extra_trains=int(ok.n_extra.min()) if len(ok) else None))
    needed=pd.DataFrame(needed);needed.to_csv(out/'needed_trains.csv',index=False,encoding='utf-8-sig')

    main_sep=cfg['min_separation_minutes'][0]
    plt.rcParams.update({'font.family':'Microsoft JhengHei','axes.unicode_minus':False,'figure.dpi':140})
    fig,axes=plt.subplots(1,2,figsize=(11.5,5),layout='constrained')
    base=evaluation[evaluation.min_separation==main_sep]
    # Label each panel at the end where its lines separate: left at n=0, right at n=max.
    for ax,col,title,end in [(axes[0],'share_p95_over_target',f'P95 候車超過 {target} 分鐘的機率',0),
                             (axes[1],'p95_wait_q95','P95 候車的 95 百分位（分鐘）',-1)]:
        for event,g in base.groupby('event_date'):
            g=g.sort_values('n_extra');heavy=g.share_p95_over_target.iloc[0]>cfg['max_exceed_probability']
            ax.plot(g.n_extra,g[col],marker='o',markersize=5,linewidth=2 if heavy else 1,
                    color='#ae5360' if heavy else '#b9c4c6',zorder=3 if heavy else 2)
            if heavy:
                ax.annotate(event,(g.n_extra.iloc[end],g[col].iloc[end]),xytext=(-6 if end==0 else 6,0),
                            textcoords='offset points',ha='right' if end==0 else 'left',va='center',fontsize=8,color='#444')
        ax.set_title(title);ax.set_xlabel('加開班次（兩方向合計）')
        ax.set_xticks(range(cfg['max_extra_trains']+1))
        ax.set_xlim(*((-1.4,cfg['max_extra_trains']+.3) if end==0 else (-.3,cfg['max_extra_trains']+1.4)))
        ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    axes[0].axhline(cfg['max_exceed_probability'],color='#555',linestyle='--',linewidth=1)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    fig.suptitle(f'北門加開班次與候車（最小間隔 {main_sep} 分；以獨立 {cfg["evaluation_draws_per_event"]} 次情境評估）\n'
                 '紅線＝不加開時超過 5% 門檻的場次；灰線＝其他場次。情境假設，非實際可行班表',fontsize=11)
    fig.savefig(out/'figures/extra_trains_tradeoff.png');plt.close(fig)

    capacity_text=('與該情境的一般班次剩餘空間相同（保守；空車折返可能更多）' if extra=='same_as_regular' else
                   f'固定為 2,220 人的 {extra["remaining_fraction"]:.0%}（接近空車的樂觀界線；往新店方向自松山發車途經多站，未必成立）')
    seps=cfg['min_separation_minutes']
    lines=[f'# {cfg.get("report_title","北門加開班次最佳化（第一版）")}','',
    '**在假設情境下求「加開幾班、插在哪一分鐘」的啟發式解；用另一批獨立情境評估。不是可直接上線的班表：車隊、司機、折返、出庫就緒與號誌最小班距都未核實。**','',
    '## 1. 問題設定','',
    '| 項目 | 設定 |','|---|---|',
    '| 決策 | 每場加開 0–{0} 班（兩方向合計），每班插在哪個整數分鐘 |'.format(cfg['max_extra_trains']),
    f'| 限制 | 與同方向任何一班至少相隔 {"／".join(map(str,cfg["min_separation_minutes"]))} 分鐘（假設值，另做敏感度）；只能在 18:00 後 0–{window[1]} 分鐘內插入 |',
    f'| 加開列車容量 | {capacity_text} |',
    f'| 目標 | {cfg["optimization_draws_per_event"]} 個最佳化情境的平均總等待人分鐘最小；末班仍未上車者每人計 {cfg["residual_penalty_minutes"]} 分鐘懲罰 |',
    '| 解法 | 逐班貪婪插入，再逐班移動到更好位置直到無法改善；小型問題已與窮舉最佳解比對一致 |',
    f'| 評估 | 另外 {cfg["evaluation_draws_per_event"]} 個獨立情境，計算平均候車、P95 與「P95 超過 {target} 分」機率 |','',
    '情境抽法與候車不確定性模擬相同：其他場次的實際／預測比例、三種分鐘形狀、Poisson 到達、方向比例與剩餘容量。','',
    f'## 2. 結果（最小間隔 {main_sep} 分）','','![加開班次取捨](figures/extra_trains_tradeoff.png)','',
    f'| 活動 | 不加開：P95>{target} 分機率 | 最少加開幾班可降到 ≤{cfg["max_exceed_probability"]:.0%} |'+(f' 間隔 {seps[-1]} 分時 |' if len(seps)>1 else ''),
    '|---|---:|---:|'+('---:|' if len(seps)>1 else '')]
    for event,g in needed.groupby('event_date'):
        a=g[g.min_separation==main_sep].iloc[0];b=g[g.min_separation==seps[-1]].iloc[0]
        fmt=lambda v:'0' if v==0 else (f'{int(v)} 班' if pd.notna(v) else f'>{cfg["max_extra_trains"]} 班')
        lines.append(f'| {event} | {a.baseline_share_p95_over_target:.1%} | {fmt(a.min_extra_trains)} |'+(f' {fmt(b.min_extra_trains)} |' if len(seps)>1 else ''))
    lines+=['',f'各場加開位置（最小間隔 {main_sep} 分）：','','| 活動 | 加開數 | 往松山 | 往新店 | 平均候車中位數 | P95 中位數（95%） | P95>{0} 分機率 |'.format(target),'|---|---:|---|---|---:|---:|---:|']
    for r in base.sort_values(['event_date','n_extra']).itertuples():
        p=plans[(plans.event_date==r.event_date)&(plans.min_separation==main_sep)&(plans.n_extra==r.n_extra)].set_index('direction').inserted
        lines.append(f'| {r.event_date} | {r.n_extra} | {p[NORTH] or "—"} | {p[SOUTH] or "—"} | {r.mean_wait_median:.2f} | '
                     f'{r.p95_wait_median:.2f}（{r.p95_wait_q95:.2f}） | {r.share_p95_over_target:.1%} |')
    lines+=['','## 3. 界線','',
    '- 最小間隔、加開列車容量、分鐘形狀、方向比例與剩餘容量都是假設；結果是「在這些假設下」的條件解。',
    '- 加開列車只算本站一個方向的上車，沒有算回程、車隊與司機，也沒有檢查是否能從機廠或折返站及時到達（見 V1 候選發車站行車時間）。',
    '- 啟發式解法不保證全域最佳；不同場次的加開數彼此獨立求解，沒有共用車隊限制。',
    '- 預測誤差只來自九場，且九場在開發時反覆使用，風險可能被低估。',
    '','計畫：[plans.csv](plans.csv)；評估：[evaluation.csv](evaluation.csv)；最少班數：[needed_trains.csv](needed_trains.csv)。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit={'model_refitted':False,'uses_observed_y':'other_events_ratios_only','predictions_sha256':sha(ROOT/cfg['prediction_path']),
           'normalized_snapshot_sha256':sha(snapshot/'normalized.json'),'seed':cfg['seed'],
           'implementation_sha256':{p:sha(ROOT/p) for p in [config,'src/run_dispatch_optimization.py',
                                    'src/dispatch_optimization.py','src/run_queue_uncertainty.py','src/queue_waits.py',
                                    'src/queue_scenarios.py','src/queueing.py']},
           'outputs':{str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='audit.json'}}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(needed.to_string(index=False))


if __name__=='__main__':main()
