"""Official general-service snapshots; never silently historical event schedules."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import pdfplumber
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DB = "taipei_metro_analysis"
STATIONS = {"105": "北門", "128": "大橋頭站", "054": "雙連", "055": "民權西路"}
CODES = {("105", "G"): "G13", ("128", "O"): "O12", ("054", "R"): "R12",
         ("055", "R"): "R13", ("055", "O"): "O11"}
LINK_URL = "https://data.taipei/api/frontstage/tpeod/dataset/resource.download?rid=c32998fa-c9b2-4324-9016-e19fbe0815f1"
LINK_PAGE = "https://data.taipei/dataset/detail?id=513e97fe-6a98-4a0d-b7dc-11122c8638d4"
CAPACITY_URL = "https://www.dorts.gov.taipei/News_Content.aspx?n=2A66A485FACB0D5B&s=F32662CFF6FE4D80"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(folder, name, url):
    path = folder / name
    if not path.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def source(path, url, kind, effective=None):
    digest = sha(path)
    return dict(source_id=hashlib.sha256((url + digest).encode()).hexdigest(), source_url=url,
                content_sha256=digest, source_kind=kind, effective_from=effective,
                retrieved_at_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                local_file=path.name)


def minute_of_service_day(hour, minute):
    if hour not in list(range(6, 24)) + [0] or not 0 <= minute < 60:
        raise ValueError("Unsupported timetable hour/minute")
    return (24 if hour == 0 else hour) * 60 + minute


def parse_timetable(path, destinations):
    """Read three day-type columns and PDF underline geometry, not text alone.

    Fail closed when the layout/notes change. Underlines identify short turns
    and the Luzhou branch. 00:xx belongs to the preceding service day (24:xx).
    """
    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) != 1:
            raise ValueError("Expected a one-page official timetable")
        pg = pdf.pages[0]
        text = pg.extract_text()
        for required in ['Weekdays', 'Saturday', 'Sunday', '離站時間', '大型活動']:
            if required not in text:
                raise ValueError(f"Not an approved general timetable: {required}")
        match = re.search(r'(\d{3})\.(\d{1,2})\.(\d{1,2})生效', text)
        if not match:
            raise ValueError("Missing official effective date")
        effective = datetime(int(match[1])+1911, int(match[2]), int(match[3])).date().isoformat()
        words = pg.extract_words()
        note_top = min(w['top'] for w in words if w['text'].startswith('註'))
        digits = [w for w in words if re.fullmatch(r'\d{2}', w['text']) and 90 < w['top'] < note_top]
        first_top = min(w['top'] for w in digits)
        top_row = sorted([w for w in digits if abs(w['top']-first_top) < 1], key=lambda w:w['x0'])
        # The hour columns start each equal-width table; numeric minutes start >20 pt later.
        headers = sorted([w for w in words if w['text']=='Hour'], key=lambda w:w['x0'])
        if len(headers)!=3: raise ValueError('Expected three hour column headers')
        anchors = [min(top_row,key=lambda w:abs(w['x0']-h['x0']))['x0'] for h in headers]
        if len(destinations) not in (1, 2):
            raise ValueError("Unknown terminal layout")
        if len(destinations) == 2 and '加註底線' not in text:
            raise ValueError("Missing short-turn/branch legend")
        underlines = [r for r in pg.rects if 0 < r['height'] < 1.5 and 7 < r['width'] < 15]
        result, consumed = [], 0
        for col, day in enumerate(['weekday', 'saturday', 'sunday_holiday']):
            left, right = anchors[col]-1, (anchors[col+1]-1 if col < 2 else pg.width)
            group = [w for w in digits if left <= w['x0'] < right]
            hours = sorted([w for w in group if abs(w['x0']-anchors[col]) < 2], key=lambda w:w['top'])
            if [int(w['text']) for w in hours] != list(range(6, 24))+[0]:
                raise ValueError(f"Missing/extra hour rows: {path.name} {day}")
            for idx, hour in enumerate(hours):
                end = hours[idx+1]['top']-1 if idx+1 < len(hours) else note_top
                minutes = sorted([w for w in group if w['x0'] > anchors[col]+15 and hour['top']-1 <= w['top'] < end], key=lambda w:(w['top'],w['x0']))
                values = [int(w['text']) for w in minutes]
                if not values or values != sorted(set(values)) or max(values)>59:
                    raise ValueError(f"Invalid minutes: {path.name} {day} {hour['text']}")
                consumed += 1+len(minutes)
                for word in minutes:
                    underline = any(abs(r['x0']-word['x0']) < 1 and abs(r['x1']-word['x1']) < 1
                                    and -1 < r['top']-word['bottom'] < 4 for r in underlines)
                    terminal = destinations[0] if underline or len(destinations)==1 else destinations[1]
                    result.append(dict(day_type=day, departure_minute=minute_of_service_day(int(hour['text']),int(word['text'])),
                                       terminal=terminal, is_underlined=int(underline)))
        if consumed != len(digits) or (len(destinations)==2 and not any(r['is_underlined'] for r in result)):
            raise ValueError("Not all numeric tokens/terminal markers reconciled")
        return effective, result


def parse_links(path):
    records = list(csv.reader(io.StringIO(path.read_bytes().decode('cp950'))))
    expected = ['SeqNo','stationbus','stationA','stationB','traveltime','SeqNo','stoptime','UpdateTime','EffectiveDate']
    if records[0] != expected:
        raise ValueError("Link CSV schema changed")
    out=[]
    for r in records[1:]:
        if not r: continue
        if len(r)!=9: raise ValueError("Invalid link CSV row")
        name=lambda v:v.removeprefix('捷運').removesuffix('站')
        out.append(dict(link_no=int(r[0]), route_name=r[1], station_a=name(r[2]), station_b=name(r[3]),
                        run_seconds=int(r[4]), route_seq=int(r[5]), dwell_at_a_seconds=int(r[6]),
                        effective_date=datetime.strptime(r[8],'%Y%m%d').date().isoformat()))
    for route in {r['route_name'] for r in out}:
        edges=[r for r in out if r['route_name']==route]
        if [r['route_seq'] for r in edges] != list(range(1,len(edges)+1)):
            raise ValueError("Route sequence gap")
        if edges[0]['dwell_at_a_seconds'] != 0 or any(a['station_b']!=b['station_a'] for a,b in zip(edges,edges[1:])):
            raise ValueError("Route/dwell alignment changed")
        if any(r['run_seconds']<=0 or r['dwell_at_a_seconds']<0 for r in edges):
            raise ValueError("Invalid durations")
    return out


def dispatch_paths(links):
    """Candidate service origins; NOT verified available spare trains.

    Published CSV contains one direction per route. Reverse movements use the
    same interstation times as an explicit symmetry assumption. Source dwell
    is aligned to station A (first terminal=0); inferred from the row sequence.
    """
    routes={'松山-新店':('G',['北門']), '松山-台電大樓':('G',['北門']),
            '淡水-廣慈/奉天宮':('R',['雙連','民權西路']), '北投-大安':('R',['雙連','民權西路']),
            '南勢角-蘆洲':('O',['大橋頭','民權西路']), '南勢角-迴龍':('O',['大橋頭','民權西路'])}
    out=[]
    for route,(line,targets) in routes.items():
        edges=[r for r in links if r['route_name']==route]
        if not edges: raise ValueError(f"Missing route {route}")
        nodes=[r['station_a'] for r in edges]+[edges[-1]['station_b']]
        dwell={r['station_a']:r['dwell_at_a_seconds'] for r in edges}
        for origin in [nodes[0],nodes[-1]]:
            for target in targets:
                i,j=nodes.index(origin),nodes.index(target)
                run=sum(e['run_seconds'] for e in edges[min(i,j):max(i,j)])
                stop=sum(dwell[n] for n in nodes[min(i,j)+1:max(i,j)])
                path=nodes[i:j+1] if i<j else nodes[j:i+1][::-1]
                out.append(dict(route_name=route,origin=origin,station_label='大橋頭站' if target=='大橋頭' else target,
                                line_code=line,path_text=' → '.join(path),run_seconds=run,intermediate_dwell_seconds=stop,
                                travel_seconds=run+stop,reverse_assumed=int(i>j),ready_seconds=None,
                                depot_exit_seconds=None,insertion_wait_seconds=None,total_dispatch_seconds=None))
    return out


def collect(folder):
    folder.mkdir(parents=True,exist_ok=True)
    sources, mappings, timetables, departures = {}, {}, [], []
    for web_id,label in STATIONS.items():
        url=f'https://web.metro.taipei/pages2026/WebStation/{web_id}/7'
        page=fetch(folder,f'{web_id}.html',url)
        s=source(page,url,'station_page');sources[s['source_id']]=s
        links={urljoin(url,a['href']).split('?')[0]:a.get_text(' ',strip=True)
               for a in BeautifulSoup(page.read_bytes(),'html.parser').find_all('a',href=True)
               if 'timetables/' in a['href'].lower() and '.pdf' in a['href'].lower()}
        if len(links)!=(4 if web_id=='055' else 2): raise ValueError('Expected station directions changed')
        for pdfurl,title in links.items():
            path=fetch(folder,pdfurl.rsplit('/',1)[-1].lower(),pdfurl)
            matches=re.findall(r'([GRO])\d{2}(.+?)站',title)
            line=matches[0][0]; destinations=[m[1] for m in matches]
            effective,rows=parse_timetable(path,destinations)
            s=source(path,pdfurl,'general_timetable',effective);sources[s['source_id']]=s
            mappings[(label,line)]=dict(station_label=label,line_code=line,station_code=CODES[(web_id,line)],web_id=web_id)
            timetables.append(dict(source_id=s['source_id'],station_label=label,line_code=line,
                                   direction_text=title,is_event_specific=0))
            departures.extend(dict(source_id=s['source_id'],**r) for r in rows)
    p=fetch(folder,'link_times.csv',LINK_URL);links=parse_links(p)
    s=source(p,LINK_URL,'link_times');sources[s['source_id']]=s
    link_source=s['source_id']
    routes=dispatch_paths(links)
    links=[dict(source_id=link_source,**r) for r in links]
    routes=[dict(source_id=link_source,**r) for r in routes]
    p=fetch(folder,'capacity.html',CAPACITY_URL)
    text=BeautifulSoup(p.read_bytes(),'html.parser').get_text(' ',strip=True)
    if not all(v in text.replace(',','') for v in ['370','2220']): raise ValueError('Capacity evidence changed')
    s=source(p,CAPACITY_URL,'capacity_reference');sources[s['source_id']]=s
    capacities=[dict(source_id=s['source_id'],train_class='high_capacity_6_car',cars=6,
                     max_persons_per_car=370,max_train_persons=2220,is_remaining_capacity=0)]
    return {'metro_source_snapshot':list(sources.values()),'metro_station_line':list(mappings.values()),
            'metro_timetable':timetables,'metro_departure':departures,'metro_link_time':links,
            'metro_dispatch_path':routes,'metro_capacity_reference':capacities}


# Each downloaded source is immutable; composite PK prevents mixing directions/day types.
SCHEMA = {
'metro_source_snapshot': ('source_id', '''source_id CHAR(64) PRIMARY KEY,source_url TEXT NOT NULL,
 content_sha256 CHAR(64) NOT NULL,source_kind VARCHAR(32) NOT NULL,effective_from DATE NULL,
 retrieved_at_utc DATETIME NOT NULL,local_file VARCHAR(128) NOT NULL'''),
'metro_station_line': ('station_label,line_code', '''station_label VARCHAR(64) NOT NULL,line_code VARCHAR(2) NOT NULL,
 station_code VARCHAR(6) NOT NULL,web_id CHAR(3) NOT NULL,PRIMARY KEY(station_label,line_code),
 FOREIGN KEY(station_label) REFERENCES analysis_station_label(station)'''),
'metro_timetable': ('source_id', '''source_id CHAR(64) PRIMARY KEY,station_label VARCHAR(64) NOT NULL,
 line_code VARCHAR(2) NOT NULL,direction_text VARCHAR(128) NOT NULL,is_event_specific BOOLEAN NOT NULL,
 FOREIGN KEY(source_id) REFERENCES metro_source_snapshot(source_id),
 FOREIGN KEY(station_label,line_code) REFERENCES metro_station_line(station_label,line_code),CHECK(is_event_specific=0)'''),
'metro_departure': ('source_id,day_type,departure_minute', '''source_id CHAR(64) NOT NULL,
 day_type VARCHAR(24) NOT NULL,departure_minute SMALLINT UNSIGNED NOT NULL,
 terminal VARCHAR(64) NOT NULL,is_underlined BOOLEAN NOT NULL,PRIMARY KEY(source_id,day_type,departure_minute),
 FOREIGN KEY(source_id) REFERENCES metro_timetable(source_id),
 CHECK(day_type IN ('weekday','saturday','sunday_holiday')),CHECK(departure_minute BETWEEN 360 AND 1499),CHECK(is_underlined IN (0,1))'''),
'metro_link_time': ('source_id,link_no', '''source_id CHAR(64) NOT NULL,link_no INT NOT NULL,route_name VARCHAR(64) NOT NULL,
 station_a VARCHAR(64) NOT NULL,station_b VARCHAR(64) NOT NULL,run_seconds INT NOT NULL,route_seq INT NOT NULL,
 dwell_at_a_seconds INT NOT NULL,effective_date DATE NOT NULL,PRIMARY KEY(source_id,link_no),
 UNIQUE(source_id,route_name,route_seq),FOREIGN KEY(source_id) REFERENCES metro_source_snapshot(source_id),
 CHECK(run_seconds>0),CHECK(dwell_at_a_seconds>=0)'''),
'metro_dispatch_path': ('source_id,route_name,origin,station_label', '''source_id CHAR(64) NOT NULL,
 route_name VARCHAR(64) NOT NULL,origin VARCHAR(64) NOT NULL,station_label VARCHAR(64) NOT NULL,line_code VARCHAR(2) NOT NULL,
 path_text TEXT NOT NULL,run_seconds INT NOT NULL,intermediate_dwell_seconds INT NOT NULL,travel_seconds INT NOT NULL,
 reverse_assumed BOOLEAN NOT NULL,ready_seconds INT NULL,depot_exit_seconds INT NULL,
 insertion_wait_seconds INT NULL,total_dispatch_seconds INT NULL,
 PRIMARY KEY(source_id,route_name,origin,station_label),FOREIGN KEY(source_id) REFERENCES metro_source_snapshot(source_id),
 FOREIGN KEY(station_label,line_code) REFERENCES metro_station_line(station_label,line_code),
 CHECK(travel_seconds=run_seconds+intermediate_dwell_seconds)'''),
'metro_capacity_reference': ('source_id,train_class', '''source_id CHAR(64) NOT NULL,train_class VARCHAR(64) NOT NULL,
 cars INT NOT NULL,max_persons_per_car INT NOT NULL,max_train_persons INT NOT NULL,is_remaining_capacity BOOLEAN NOT NULL,
 PRIMARY KEY(source_id,train_class),FOREIGN KEY(source_id) REFERENCES metro_source_snapshot(source_id),
 CHECK(max_train_persons=cars*max_persons_per_car),CHECK(is_remaining_capacity=0)'''),
}


def schema_sql():
    return f'USE {DB};\n'+ '\n'.join(f'CREATE TABLE IF NOT EXISTS {t} ({ddl}) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;' for t,(_,ddl) in SCHEMA.items())


def sql_value(value):
    if value is None: return 'NULL'
    if isinstance(value,int): return str(value)
    return "CONVERT(X'"+str(value).encode('utf-8').hex()+"' USING utf8mb4)"


def import_sql(tables):
    """Atomic insert-only data import. Abort on conflicting existing PK values."""
    statements=[f'USE {DB};', 'CREATE TEMPORARY TABLE metro_guard(label VARCHAR(128),ok INT NOT NULL CHECK(ok=1));']
    for table,rows in tables.items():
        if not rows: raise ValueError('Empty import table')
        statements.append(f'CREATE TEMPORARY TABLE incoming_{table} LIKE {table};')
        cols=','.join(rows[0])
        for i in range(0,len(rows),250):
            vals=','.join('('+','.join(sql_value(r[c]) for c in rows[0])+')' for r in rows[i:i+250])
            statements.append(f'INSERT INTO incoming_{table} ({cols}) VALUES {vals};')
    statements.append('START TRANSACTION;')
    for table,rows in tables.items():
        keys=SCHEMA[table][0].split(',');cols=list(rows[0])
        on=' AND '.join(f'd.{c}=s.{c}' for c in keys)
        # A re-download of identical bytes may have a new retrieval timestamp.
        same=' AND '.join(f'd.{c}<=>s.{c}' for c in cols if c!='retrieved_at_utc')
        statements.append(f"INSERT INTO metro_guard SELECT '{table}_conflict',COUNT(*)=0 FROM {table} d JOIN incoming_{table} s ON {on} WHERE NOT({same});")
        statements.append(f"INSERT INTO {table} ({','.join(cols)}) SELECT {','.join('s.'+c for c in cols)} FROM incoming_{table} s LEFT JOIN {table} d ON {on} WHERE d.{keys[0]} IS NULL;")
        statements.append(f'SET @added_{table}=ROW_COUNT();')
        statements.append(f"INSERT INTO metro_guard SELECT '{table}_reconciled',COUNT(*)={len(rows)} FROM {table} d JOIN incoming_{table} s ON {on} WHERE {same};")
    statements.extend(['COMMIT;',"SELECT JSON_OBJECT("+','.join(f"'{t}',@added_{t}" for t in tables)+');'])
    return '\n'.join(statements)
