"""Run three arrival shapes with official schedules and explicit capacity assumptions."""
import json
from datetime import date

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from queueing import simulate_bulk_queue
from metro_operations import ROOT,sha,collect
from queue_scenarios import shaped_arrivals

LABELS={'uniform':'均勻','broad':'寬峰（σ=20 分）','concentrated':'集中峰（σ=5 分）'}
COLORS={'uniform':'#467f88','broad':'#bd9758','concentrated':'#ae5360'}


def prepare_forecasts(path,expected_hash,column):
    if sha(path)!=expected_hash: raise ValueError('Frozen model predictions changed')
    # Deliberately do not load truth, event peak labels, or alternative models.
    data=pd.read_csv(path,usecols=['event_date','station','hour',column])
    if len(data)!=54 or set(data.station)!={'北門'} or data.event_date.nunique()!=9:
        raise ValueError('Expected frozen nine-event Beimen cohort')
    for _,d in data.groupby('event_date'):
        if sorted(d.hour.tolist())!=list(range(18,24)): raise ValueError('Incomplete hourly cohort')
    return data


def main():
    cfg=json.loads((ROOT/'configs/queue_dispatch_v1.json').read_text(encoding='utf-8'))
    if (cfg['first_hour'],cfg['last_hour'])!=(18,23):
        raise ValueError('Frozen V14 cohort covers 18-23 only; do not relabel its hours')
    if any(not 0<s<1 for s in cfg['northbound_shares']): raise ValueError('Invalid direction share')
    if any(not 0<f<=1 for f in cfg['remaining_capacity_fractions']): raise ValueError('Invalid remaining capacity')
    path=ROOT/cfg['prediction_path'];data=prepare_forecasts(path,cfg['prediction_sha256'],cfg['prediction_column'])
    snapshot=ROOT/'.local/transit_reference'/cfg['operations_snapshot']
    tables=json.loads((snapshot/'normalized.json').read_text(encoding='utf-8'))
    if tables != collect(snapshot): raise ValueError('Normalized data differ from source PDF/CSV parsing')
    # Check raw source bytes before using their parsed schedules.
    for s in tables['metro_source_snapshot']:
        if sha(snapshot/s['local_file'])!=s['content_sha256']: raise ValueError('Official source snapshot changed')
    beimentables=[t for t in tables['metro_timetable'] if t['station_label']=='北門']
    maximum=tables['metro_capacity_reference'][0]['max_train_persons']
    out=ROOT/'result/queue_dispatch_v1';(out/'figures').mkdir(parents=True,exist_ok=True)
    rows,examples,arrival_examples=[],{},{}
    max_conservation=0.
    for event,g in data.groupby('event_date'):
        counts=g.sort_values('hour')[cfg['prediction_column']].to_numpy()
        weekday=date.fromisoformat(event).weekday()
        # All nine events are Wednesdays or Saturdays. Do not infer holiday policy generally.
        if weekday not in [2,5]: raise ValueError('Event needs explicit holiday/service-day mapping')
        day='weekday' if weekday==2 else 'saturday'
        for shape,sigma in cfg['shapes'].items():
            centers=[30] if shape=='uniform' else cfg['peak_centers_minutes']
            for center in centers:
                arrivals=shaped_arrivals(counts,shape,sigma,center,cfg['platform_delay_minutes'])
                if event==cfg['display_event'] and center==30:arrival_examples[shape]=arrivals
                for share in cfg['northbound_shares']:
                    for fraction in cfg['remaining_capacity_fractions']:
                        for table in beimentables:
                            north='松山' in table['direction_text']
                            direction='往松山' if north else '往新店／台電大樓'
                            share_d=share if north else 1-share
                            trains=sorted(r['departure_minute']-cfg['first_hour']*60
                                          for r in tables['metro_departure']
                                          if r['source_id']==table['source_id'] and r['day_type']==day
                                          and r['departure_minute']>=cfg['first_hour']*60)
                            used=[r for r in tables['metro_departure'] if r['source_id']==table['source_id'] and r['day_type']==day and r['departure_minute']>=1080]
                            # Current evening Beimen tables have no southbound short turns.
                            # Refuse to pool incompatible terminals if future source changes.
                            expected='松山' if north else '新店'
                            if any(r['terminal']!=expected for r in used):
                                raise ValueError('Evening short turns require destination-compatible boarding model')
                            q=simulate_bulk_queue(arrivals*share_d,trains,[maximum*fraction]*len(trains))
                            max_conservation=max(max_conservation,abs(q['conservation_error']))
                            rows.append(dict(event_date=event,day_type=day,shape=shape,center_minute=center,
                                             northbound_share=share,remaining_fraction=fraction,direction=direction,
                                             available_spaces=maximum*fraction,forecast_peak_hour=int(g.sort_values('hour').iloc[np.argmax(counts)]['hour']),
                                             forecast_demand=q['total_arrivals'],train_count=q['train_count'],
                                             mean_wait_minutes=q['mean_wait_minutes'],max_queue=q['max_queue'],
                                             residual_queue=q['residual_queue'],waiting_person_minutes=q['waiting_person_minutes'],
                                             train_departures_with_left_behind=sum(t['left_behind']>1e-7 for t in q['train_records']),
                                             last_departure_minute=max(trains)+1080,cleared=q['cleared']))
                            if event==cfg['display_event'] and center==30 and share==.5 and fraction==.5:
                                examples[(shape,direction)]=q
    results=pd.DataFrame(rows);results.to_csv(out/'scenarios.csv',index=False,encoding='utf-8-sig')
    base=results[(results.northbound_share==.5)&(results.center_minute==30)]
    cohorts=[]
    for keys,g in base.groupby(['event_date','shape','remaining_fraction']):
        cohorts.append(dict(event_date=keys[0],shape=keys[1],remaining_fraction=keys[2],
                            mean_wait_minutes=g.waiting_person_minutes.sum()/g.forecast_demand.sum() if g.cleared.all() else np.nan,
                            max_direction_queue=g.max_queue.max(),residual_queue=g.residual_queue.sum()))
    cohort=pd.DataFrame(cohorts);cohort.to_csv(out/'event_summary.csv',index=False,encoding='utf-8-sig')
    summary=cohort.groupby(['shape','remaining_fraction']).agg(mean_wait_minutes=('mean_wait_minutes','mean'),
                    worst_direction_queue=('max_direction_queue','max'),max_residual=('residual_queue','max'),events=('event_date','count')).reset_index()
    summary.to_csv(out/'summary.csv',index=False,encoding='utf-8-sig')
    plt.rcParams.update({'font.family':'Microsoft JhengHei','axes.unicode_minus':False,'figure.dpi':140})
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
    for shape in cfg['shapes']:
        arr=arrival_examples[shape];axes[0].plot(18+(np.arange(len(arr))+.5)/60,arr,label=LABELS[shape],color=COLORS[shape])
        q=examples[(shape,'往新店／台電大樓')];trace=q['trace']
        # Duplicate x at service instants draws the pre/post boarding jump.
        xx=[];yy=[]
        for r in trace:
            xx.extend([18+r['time_minutes']/60]*2);yy.extend([r['queue_before_service'],r['queue_after_service']])
        axes[1].plot(xx,yy,color=COLORS[shape],label=LABELS[shape])
    axes[0].set_title('相同逐時預測人數，只改變最高預測時段內的抵達形狀')
    axes[0].set_ylabel('雙向抵達人次／分鐘');axes[0].legend(ncol=3,fontsize=9)
    axes[1].set_title('往新店方向：每班剩餘 1,110 人、雙向各 50%、站內延遲 5 分鐘')
    axes[1].set_ylabel('候車人數');axes[1].set_xlabel('時刻（24 表示次日 00:00）')
    for ax in axes:ax.grid(alpha=.18);ax.set_xlim(18,24.85)
    fig.suptitle('北門｜2024-08-10 預測需求 × 2026-08-30 一般週六班表\n情境假設示範，非活動當日實測候車',fontsize=12)
    fig.savefig(out/'figures/arrival_queue_comparison.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for shape in cfg['shapes']:
        g=summary[summary['shape']==shape].sort_values('remaining_fraction')
        ax.plot(g.remaining_fraction*100,g.mean_wait_minutes,marker='o',color=COLORS[shape],label=LABELS[shape])
        for r in g.itertuples():ax.annotate(f'{r.mean_wait_minutes:.2f}',(r.remaining_fraction*100,r.mean_wait_minutes),xytext=(5,-13 if shape=='uniform' else 6),textcoords='offset points',fontsize=9,color=COLORS[shape])
    ax.set(xlabel='每班剩餘空間／2,220 人上限（%）',ylabel='平均候車（分鐘）',title='北門九場預測需求：每場等權平均候車情境\n方向各 50%、峰中心在該小時第 30 分鐘；不是已校準的實際等候時間')
    ax.set_xticks([25,50,75]);ax.grid(alpha=.2);ax.legend();fig.savefig(out/'figures/capacity_sensitivity.png');plt.close(fig)
    example=cohort[(cohort.event_date==cfg['display_event'])&(cohort.remaining_fraction==.5)]
    lines=['# 北門候車情境與四站營運資料（第一版）','',
    '**已執行：固定 V14 四站 XGBoost 的北門預測，沒有重新訓練。這是現行一般班表的壓力情境，不是歷史活動候車重建。**','',
    '## 1. 本次實際整理的資料','',
    '- 四站共 10 份官方方向 PDF、3 種日型、7,118 個離站時刻；民權西路分紅線／橘線，保留區間車與蘆洲／迴龍終點。',
    '- 187 筆官方相鄰站資料，推算 20 條候選發車站→研究車站路徑；重複候選依營運路線保留。',
    '- 八份班表自 2026/8/30 生效；橘線往南勢角的兩份表自 2026/10/1 生效。九場需求來自 2024–2026/8 的封存預測，故只是假設採用目前班表。',
    '- 已寫入 MySQL 七張新表；重跑不重複新增。原分析表 74,880 列不變，逐欄讀回核對通過。',
    '', '## 2. 三種到達形狀及容量比較','',
    '只調整模型**預測最高的小時**：均勻分布、截斷常態寬峰（σ=20 分鐘）、集中峰（σ=5 分鐘）。每小時總量完全相同，其他小時均勻。不是從真值挑尖峰，也不是已估出的分鐘分布。',
    '', '每班剩餘空間假設為整列上限 2,220 人的 25%／50%／75%（555／1,110／1,665 人）；這些比例不是量測值。方向拆分另測 40:60、50:50、60:40，峰中心另測第 15／30／45 分鐘。站內步行延遲固定 5 分鐘，仍待校準。',
    '', '| 到達情境 | 剩餘空間 25% 平均候車 | 50% | 75% |','|---|---:|---:|---:|']
    for shape in cfg['shapes']:
        vals=[summary[(summary['shape']==shape)&(summary.remaining_fraction==f)].iloc[0].mean_wait_minutes for f in [.25,.5,.75]]
        lines.append(f'| {LABELS[shape]} | '+' | '.join(f'{v:.2f} 分' for v in vals)+' |')
    lines+=['','以上是九場各自雙向依人數加權後，再對九場等權平均。均已清空才報完整平均；沒有把未上車者刪掉。',
    '', '![容量比較](figures/capacity_sensitivity.png)','',
    f'例：{cfg["display_event"]}，每班剩餘 1,110 人：','',
    '| 情境 | 雙向平均候車 | 最擁擠方向的最大候車人數 |','|---|---:|---:|']
    for r in example.itertuples():lines.append(f'| {LABELS[r.shape]} | {r.mean_wait_minutes:.2f} 分 | {r.max_direction_queue:,.0f} |')
    lines+=['','![同人數不同到達形狀](figures/arrival_queue_comparison.png)','',
    '## 3. 候選發車站到四站的時間','',
    '**下表為候選站已發車後的區間行駛＋中間站停靠時間。不是接到調度命令到抵達的完整時間，也不表示候選站當時有備車或可插車。**',
    'CSV 只列一個方向；反向採相同區間行駛時間的假設。停站欄依起站為 0 的序列解讀為 stationA 停靠，仍需營運方確認。沒有把乘客跨線轉乘當成列車調車路徑。','',
    '| 抵達站／路線 | 候選發車站 | 估計行車含中途停靠 | 反向對稱假設 |','|---|---|---:|---|']
    seen=set()
    for r in tables['metro_dispatch_path']:
        key=(r['station_label'],r['line_code'],r['origin'])
        if key in seen:continue
        seen.add(key);lines.append(f'| {r["station_label"]}／{r["line_code"]} | {r["origin"]} | {r["travel_seconds"]/60:.2f} 分 | '+('是' if r['reverse_assumed'] else '否')+' |')
    lines+=['','完整調度提前量：**整備／司機就緒＋出庫接軌＋等待插車＋上述行車時間**，必要時另加折返。未知部分在 DB 留 NULL，沒有當作 0。',
    '', '## 4. 目前可說與不可說的結論','',
    f'- 共執行 {len(results):,} 組單方向情境；全部清空：{bool(results.cleared.all())}；最大人數守恆誤差 {max_conservation:.2e}。',
    '- 相同逐時人數仍可能有不同候車：尖峰落在兩班列車之間且超過剩餘容量時，會發生跨班滯留。集中峰不保證在所有班表相位都最差，故另外保留峰位置敏感度。',
    '- 模擬只追蹤 18–23 時的預測進站群體及其步行尾端；初始候車為 0，沒有額外轉乘、00 時後新進站人流、離隊或改道。終班後剩餘會保留，不偷偷增加清空列車。',
    '- 用離站時刻近似瞬間完成上車，不含車門吞吐、實際停站延長與月台管制。北門晚間這批班表無台電大樓區間車；若未來出現會停止執行，要求先做目的地相容配車。',
    '- 本版不是 M/M/c，也沒有宣稱最佳加班車數或實際能調度；車隊、司機、折返、號誌最小班距、道岔衝突與月台容量尚未核實。',
    '- 小時 OD 時間語義及歷史發布時點仍有原研究限制；分鐘分配是情境假設。既有預測的歷史 D−1 稽核界線不因本版而消失。',
    '', '車上人數估法、官方來源與仍缺資料：[營運資料與實作界線](../../docs/metro_operations.md)。',
    '逐組結果：[scenarios.csv](scenarios.csv)；逐場：[event_summary.csv](event_summary.csv)；來源及 DB 核對：[import_audit.json](import_audit.json)。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit={'model_refitted':False,'predictions_sha256':sha(path),'normalized_snapshot_sha256':sha(snapshot/'normalized.json'),
           'implementation_sha256':{p:sha(ROOT/p) for p in ['configs/queue_dispatch_v1.json','src/run_queue_scenarios.py','src/queue_scenarios.py','src/metro_operations.py','src/queueing.py']},
           'scenarios':len(results),'max_conservation_error':max_conservation,'all_cleared':bool(results.cleared.all()),
           'max_residual_queue':float(results.residual_queue.max()),'uses_observed_y':False,
           'official_timetable_effective_from':sorted({s['effective_from'] for s in tables['metro_source_snapshot'] if s['source_kind']=='general_timetable'}),
           'historical_reconstruction':False,'outputs':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='scenario_audit.json'}}
    (out/'scenario_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print(json.dumps({k:v for k,v in audit.items() if k!='outputs'},indent=2))


if __name__=='__main__':main()
