export type RailDirection = "north" | "south";
export interface ReplayTrain {
  time: number;
  boarded: number;
  before: number;
  left: number;
}
export interface DirectionTrace {
  queue: number[];
  arrivals: number[];
  trains: ReplayTrain[];
  mean_wait: number;
  max_queue: number;
  residual: number;
}
export interface RailReplay {
  maximum: number;
  source_sha256: string;
  timetable_effective: string;
  events: Record<string, Record<string, Record<RailDirection, DirectionTrace>>>;
}

/** Unknown is grey; 0 is green, 0.5 yellow and >=1 red. */
export function congestionColor(ratio: number | null): string {
  if (ratio == null || !Number.isFinite(ratio) || ratio < 0) return "#777b85";
  return `hsl(${120 * (1 - Math.min(1, ratio))} 72% 52%)`;
}
export function queueAt(trace: DirectionTrace, minute: number): number | null {
  if (!Number.isFinite(minute) || minute < 0 || minute > trace.queue.length - 1)
    return null;
  const i = Math.floor(minute);
  return trace.queue[i] + (trace.arrivals[i] || 0) * (minute - i);
}
export function trainLoad(
  train: ReplayTrain,
  minute: number,
  remaining: number,
  maximum: number,
) {
  if (remaining < 0 || remaining > maximum || maximum <= 0)
    throw new Error("Invalid train capacity");
  const onboard = Math.min(
    maximum,
    maximum - remaining + (minute >= train.time ? train.boarded : 0),
  );
  return {
    onboard,
    ratio: onboard / maximum,
    full: onboard >= maximum - 1e-6,
    after: minute >= train.time,
  };
}
/** Approach/departure positions are illustration only, not GPS or fitted runtime. */
export function visibleTrains(trace: DirectionTrace, minute: number) {
  return trace.trains
    .filter((t) => Math.abs(t.time - minute) <= 2)
    .map((t) => ({ ...t, progress: (minute - t.time + 2) / 4 }));
}
export function peakMinute(
  north: DirectionTrace,
  south: DirectionTrace,
): number {
  let peak = 0,
    time = 0;
  const end = Math.min(north.queue.length, south.queue.length) - 1;
  for (let i = 1; i <= end; i++) {
    const t = i - 0.001;
    const total = (queueAt(north, t) || 0) + (queueAt(south, t) || 0);
    if (total > peak) {
      peak = total;
      time = t;
    }
  }
  return time;
}
export function replayTime(minute: number) {
  const seconds = Math.max(0, Math.floor((1080 + minute) * 60 + 1e-7));
  return `${seconds >= 86400 ? "次日 " : ""}${String(Math.floor(seconds / 3600) % 24).padStart(2, "0")}:${String(Math.floor(seconds / 60) % 60).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}
