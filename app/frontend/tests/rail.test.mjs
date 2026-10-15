import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import {
  congestionColor,
  queueAt,
  trainLoad,
  visibleTrains,
  peakMinute,
  replayTime,
} from "../src/lib/rail.ts";

const source = readFileSync(
  new URL("../public/data/research.json", import.meta.url),
);
const research = JSON.parse(source);
const replay = JSON.parse(
  readFileSync(
    new URL("../public/data/rail-replay.json", import.meta.url),
    "utf8",
  ),
);
const near = (a, b, epsilon = 1e-3) =>
  assert.ok(Math.abs(a - b) < epsilon, `${a} != ${b}`);

test("unknown values remain grey and the continuous scale caps at red", () => {
  for (const value of [null, NaN, Infinity, -1])
    assert.equal(congestionColor(value), "#777b85");
  assert.equal(congestionColor(0), "hsl(120 72% 52%)");
  assert.equal(congestionColor(0.5), "hsl(60 72% 52%)");
  assert.equal(congestionColor(1), "hsl(0 72% 52%)");
  assert.equal(congestionColor(3), congestionColor(1));
});

test("queue grows between minutes but boarding is an instantaneous drop", () => {
  const trace = { queue: [0, 10, 0, 10], arrivals: [10, 10, 10] };
  near(queueAt(trace, 1.999), 19.99);
  assert.equal(queueAt(trace, 2), 0);
  assert.equal(queueAt(trace, 2.5), 5);
  for (const t of [-1, NaN, 4]) assert.equal(queueAt(trace, t), null);
});

test("train colour uses total onboard load, not just passengers boarding at Beimen", () => {
  const train = { time: 10, boarded: 555, before: 700, left: 145 };
  const before = trainLoad(train, 9.99, 555, 2220);
  assert.equal(before.onboard, 1665);
  assert.equal(before.ratio, 0.75);
  assert.equal(before.full, false);
  const after = trainLoad(train, 10, 555, 2220);
  assert.equal(after.onboard, 2220);
  assert.equal(after.full, true);
  assert.equal(congestionColor(after.ratio), "hsl(0 72% 52%)");
  assert.equal(trainLoad({ ...train, boarded: 0 }, 10, 2220, 2220).ratio, 0);
  assert.throws(() => trainLoad(train, 10, 3000, 2220));
});

test("illustrated trains enter, stop at station position, and leave at bounded positions", () => {
  const trace = { trains: [{ time: 10, boarded: 1, before: 1, left: 0 }] };
  assert.equal(visibleTrains(trace, 7).length, 0);
  assert.equal(visibleTrains(trace, 8)[0].progress, 0);
  assert.equal(visibleTrains(trace, 10)[0].progress, 0.5);
  assert.equal(visibleTrains(trace, 12)[0].progress, 1);
  assert.equal(visibleTrains(trace, 13).length, 0);
  assert.equal(replayTime(360), "次日 00:00:00");
  assert.equal(replayTime(404), "次日 00:44:00");
});

test("all 162 replay traces reproduce report, obey capacity and conserve passengers", () => {
  assert.equal(
    replay.source_sha256,
    createHash("sha256").update(source).digest("hex"),
  );
  assert.equal(Object.keys(replay.events).length, 9);
  let cases = 0;
  for (const [date, scenarios] of Object.entries(replay.events)) {
    const prediction = research.model.hourly
      .filter((r) => r.event_date === date)
      .reduce((s, r) => s + r.predicted, 0);
    for (const [key, pair] of Object.entries(scenarios)) {
      const [shape, fractionText] = key.split("-"),
        fraction = +fractionText;
      for (const [direction, trace] of Object.entries(pair)) {
        cases++;
        const saved = research.queue.scenarios.find(
          (r) =>
            r.event_date === date &&
            r.shape === shape &&
            +r.remaining_fraction === fraction &&
            +r.center_minute === 30 &&
            +r.northbound_share === 0.5 &&
            r.direction ===
              (direction === "north" ? "往松山" : "往新店／台電大樓"),
        );
        assert.ok(saved);
        near(trace.mean_wait, +saved.mean_wait_minutes);
        near(trace.max_queue, +saved.max_queue);
        near(trace.residual, +saved.residual_queue);
        const arrivals = trace.arrivals.reduce((s, a) => s + a, 0);
        near(arrivals, prediction / 2);
        near(
          trace.trains.reduce((s, t) => s + t.boarded, 0) + trace.residual,
          arrivals,
        );
        const services = new Map(trace.trains.map((t) => [t.time, t]));
        for (let i = 1; i < trace.queue.length; i++) {
          const before = trace.queue[i - 1] + (trace.arrivals[i - 1] || 0);
          const train = services.get(i);
          near(trace.queue[i], before - (train?.boarded || 0));
          assert.ok(trace.queue[i] >= 0);
          if (train) {
            near(train.before, before);
            near(train.boarded, Math.min(before, replay.maximum * fraction));
            near(train.left, trace.queue[i]);
            assert.ok(train.boarded <= replay.maximum * fraction + 1e-6);
          }
        }
      }
      const t = peakMinute(pair.north, pair.south);
      assert.ok(t >= 0 && t < pair.north.queue.length);
      const q = queueAt(pair.north, t) + queueAt(pair.south, t);
      assert.ok(Number.isFinite(q));
    }
  }
  assert.equal(cases, 162);
});
