"""Export only allowlisted public aggregates; never copy a database or .local tree.

Run from the repository root: python -m app.export_snapshot
Pages builds use these checked-in snapshots, without MySQL or credentials.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from app.backend.database import Repository
from app.backend.main import FLOW_FIELDS, TIME_SEMANTICS
from app.export_rail_replay import build_rail_replay

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'app/frontend/public/data'
STATIONS = ['北門', '大橋頭站', '雙連', '民權西路']
EVENT_FIELDS = ['event_date', 'event_name', 'event_status', 'announced_show_start_time', 'fireworks_duration_seconds']


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def select_fields(row, fields):
    return {key: row.get(key) for key in fields}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n', encoding='utf-8')


def export(repo=None, output=OUTPUT):
    repo = repo or Repository()
    repo.health()
    events = [select_fields(e, EVENT_FIELDS) for e in repo.events()]
    flows = []
    for event in events:
        flows.extend(select_fields(row, FLOW_FIELDS) for row in repo.flows(date.fromisoformat(event['event_date']), STATIONS, 18, 24))
    keys = [(r['event_date'], r['station'], r['hour_start']) for r in flows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate hourly observation')
    metadata = dict(generated_at=datetime.now(timezone.utc).isoformat(), source='公開彙總快照（匯出自 MySQL）',
                    time_semantics=TIME_SEMANTICS, stations=STATIONS, start_hour=18, end_hour_exclusive=24,
                    row_count=len(flows), missing_values='null／不完整不等於 0')
    observations = dict(events=events, rows=flows, metadata=metadata)
    qroot = ROOT / 'result/queue_dispatch_v1'
    mroot = ROOT / 'result/forecast/tables'
    config = json.loads((ROOT / 'configs/queue_dispatch_v1.json').read_text(encoding='utf-8'))
    prediction = ROOT / config['prediction_path']
    if hashlib.sha256(prediction.read_bytes()).hexdigest() != config['prediction_sha256']:
        raise ValueError('Frozen predictions changed; review before publication')
    hourly = [dict(event_date=r['event_date'], hour=int(r['hour']), actual=float(r['y']), predicted=float(r[config['prediction_column']])) for r in read_csv(prediction)]
    if len(hourly) != 54 or len({r['event_date'] for r in hourly}) != 9:
        raise ValueError('Expected nine-event frozen cohort')
    scenarios = read_csv(qroot / 'scenarios.csv')
    if len(scenarios) != 1134:
        raise ValueError('Unexpected queue cohort')
    research = dict(generated_at=metadata['generated_at'], model=dict(name='四站 XGBoost · 散場加權',
                    hourly=hourly, events=read_csv(mroot/'website_events.csv'), metrics=read_csv(mroot/'website_summary.csv')[0],
                    selection_note='已反覆比較的九場開發驗證；不是全新獨立測試。'),
                    queue=dict(scenarios=scenarios, summary=read_csv(qroot/'summary.csv'), assumptions=dict(
                        platform_delay_minutes=config['platform_delay_minutes'], max_train_persons=2220, cars=6,
                        shapes=config['shapes'], historical_reconstruction=False)),
                    operations={name:read_csv(qroot/f'metro_{name}.csv') for name in ['source_snapshot','station_line','timetable','departure','dispatch_path','capacity_reference']})
    # Nothing is written until the database read and frozen model checks succeed.
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'observations.json', observations)
    write_json(output/'research.json', research)
    build_rail_replay(output)
    for event in events:
        with (output/f"flows-{event['event_date']}.csv").open('w', encoding='utf-8-sig', newline='') as stream:
            writer=csv.DictWriter(stream, fieldnames=FLOW_FIELDS)
            writer.writeheader()
            writer.writerows(r for r in flows if r['event_date']==event['event_date'])
    for name in ['arrival_queue_comparison.png','capacity_sensitivity.png']:
        shutil.copyfile(qroot/'figures'/name, output/name)
    shutil.copyfile(qroot/'scenarios.csv', output/'queue-scenarios.csv')
    write_json(output/'manifest.json', dict(**metadata, model=config['prediction_column'], model_refitted=False,
        prediction_sha256=config['prediction_sha256'], queue_scenarios=len(scenarios),
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print(json.dumps(dict(events=len(events), flow_rows=len(flows), forecast_rows=len(hourly), scenarios=len(scenarios)), ensure_ascii=False))


if __name__=='__main__':
    export()
