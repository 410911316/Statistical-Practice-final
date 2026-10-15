import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {scenarioRows, waitValue, departureTime} from '../src/lib/research.ts';
const data=JSON.parse(readFileSync(new URL('../public/data/research.json',import.meta.url),'utf8'));

test('every selectable case resolves all three shapes, without swapping event or direction',()=>{
  const dates=[...new Set(data.model.hourly.map(r=>r.event_date))];
  for(const date of dates) for(const center of [15,30,45]) for(const share of [.4,.5,.6]) for(const capacity of [.25,.5,.75]) for(const direction of ['往松山','往新店／台電大樓']) {
    const rows=scenarioRows(data,date,center,share,capacity,direction);
    assert.equal(rows.length,3);
    assert.ok(rows.every(r=>r && r.event_date===date && r.direction===direction));
    assert.equal(+rows[0].center_minute,30);
    assert.ok(rows.every(r=>+r.available_spaces===2220*capacity));
    assert.ok(Math.max(...rows.map(r=>+r.forecast_demand))-Math.min(...rows.map(r=>+r.forecast_demand))<1e-6);
  }
});
test('never presents a partial or missing wait as a complete average',()=>{
  assert.equal(waitValue(undefined),null);
  assert.equal(waitValue({cleared:'False',mean_wait_minutes:3}),null);
  assert.equal(waitValue({cleared:'True',mean_wait_minutes:''}),null);
  assert.equal(waitValue({cleared:'True',mean_wait_minutes:'0'}),0);
});
test('after-midnight trains retain the service-day meaning',()=>{
  assert.equal(departureTime(1439),'23:59');
  assert.equal(departureTime(1440),'次日 00:00');
  assert.equal(departureTime(1497),'次日 00:57');
});
