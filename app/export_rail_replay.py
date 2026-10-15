"""Publish animation traces from the same fluid queue kernel as the report.

Only the existing public forecast/timetable snapshot is read. No fitting or
database access. Direction shares=50:50, peak center=30, walking delay=5 min.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from analysis_v1.queueing import simulate_bulk_queue
from queue_scenarios import shaped_arrivals


def build_rail_replay(output):
    source = output / 'research.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    maximum = data['queue']['assumptions']['max_train_persons']
    replay = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), maximum=maximum,
                  share_north=.5, center_minute=30, walk_delay=5,
                  station='北門', timetable_effective='2026-08-30', events={})
    for event in sorted({r['event_date'] for r in data['model']['hourly']}):
        hourly = sorted((r for r in data['model']['hourly'] if r['event_date']==event), key=lambda r:r['hour'])
        counts = [r['predicted'] for r in hourly]
        cases={}
        for shape,sigma in [('uniform',None),('broad',20),('concentrated',5)]:
            arrivals = shaped_arrivals(counts,shape,sigma,30,5)*.5
            for fraction in [.25,.5,.75]:
                case={}
                for north in [True,False]:
                    direction='往松山' if north else '往新店／台電大樓'
                    saved=next(r for r in data['queue']['scenarios'] if r['event_date']==event and r['shape']==shape
                               and float(r['remaining_fraction'])==fraction and r['direction']==direction
                               and float(r['center_minute'])==30 and float(r['northbound_share'])==.5)
                    table=next(r for r in data['operations']['timetable'] if r['station_label']=='北門'
                               and ('松山' in r['direction_text'])==north)
                    departures=sorted(int(r['departure_minute'])-1080 for r in data['operations']['departure']
                                      if r['source_id']==table['source_id'] and r['day_type']==saved['day_type']
                                      and int(r['departure_minute'])>=1080)
                    q=simulate_bulk_queue(arrivals,departures,[maximum*fraction]*len(departures))
                    for key in ['max_queue','mean_wait_minutes','residual_queue','forecast_demand']:
                        actual=q['total_arrivals'] if key=='forecast_demand' else q[key]
                        if abs(actual-float(saved[key]))>1e-6: raise ValueError(f'Replay disagrees with report: {event}/{shape}/{key}')
                    # Every official departure is at an integer minute. Store post-boarding
                    # queues; interpolate only arrivals between minutes, never through a drop.
                    horizon=int(q['horizon_minutes'])
                    lookup={int(r['time_minutes']):r for r in q['trace']}
                    queues=[]
                    for minute in range(horizon+1):
                        value=lookup[minute]['queue_after_service'] if minute in lookup else queues[-1]
                        queues.append(round(value,6))
                    case['north' if north else 'south']=dict(queue=queues,arrivals=[round(float(x),6) for x in arrivals],
                        trains=[dict(time=r['time_minutes'],boarded=round(r['boarded'],6),before=round(r['waiting_before'],6),left=round(r['left_behind'],6)) for r in q['train_records']],
                        mean_wait=q['mean_wait_minutes'], max_queue=q['max_queue'], residual=q['residual_queue'])
                cases[f'{shape}-{fraction}']=case
        replay['events'][event]=cases
    path=output/'rail-replay.json'
    path.write_bytes((json.dumps(replay,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8'))
    return path


if __name__=='__main__':
    output=ROOT/'app/frontend/public/data'
    path=build_rail_replay(output)
    manifest_path=output/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['outputs'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_bytes((json.dumps(manifest,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8'))
    print(f'Exported 162 direction traces; {path.stat().st_size:,} bytes; all match archived report.')
