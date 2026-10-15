export interface EventRecord {
  event_date: string;
  event_name: string;
  event_status: string;
  announced_show_start_time: string | null;
  fireworks_duration_seconds: number | null;
}

export interface FlowRow {
  event_date: string;
  hour_start: string;
  station: string;
  origin_count: number | null;
  destination_count_same_source_bin: number | null;
  origin_complete: boolean | number;
  destination_complete: boolean | number;
  baseline_origin_mean_28d: number | null;
  baseline_n_days: number;
}

export interface FlowResponse {
  rows: FlowRow[];
  metadata: { source: string; time_semantics: string; [key: string]: unknown };
}

export const STATIC_MODE = import.meta.env.VITE_DATA_MODE === "snapshot";
interface Snapshot {
  events: EventRecord[];
  rows: FlowRow[];
  metadata: FlowResponse["metadata"];
}
let snapshotPromise: Promise<Snapshot> | undefined;
let snapshotCache: Snapshot | undefined;
export function readSnapshot(): Promise<Snapshot> {
  if (!snapshotPromise)
    snapshotPromise = fetch(`${import.meta.env.BASE_URL}data/observations.json`)
      .then((r) => {
        if (!r.ok) throw new Error("公開快照讀取失敗，請重新整理。");
        return r.json();
      })
      .then((data) => {
        snapshotCache = data;
        return data;
      })
      .catch((e) => {
        snapshotPromise = undefined;
        throw e;
      });
  return snapshotPromise;
}

export async function api<T>(path: string, signal?: AbortSignal): Promise<T> {
  if (STATIC_MODE) {
    const snapshot = await readSnapshot();
    signal?.throwIfAborted();
    const url = new URL(path, "https://snapshot.invalid");
    if (url.pathname === "/events") return { events: snapshot.events } as T;
    if (url.pathname === "/flows") {
      const stations = (url.searchParams.get("stations") || "").split(",");
      const start = Number(url.searchParams.get("start_hour") || 18);
      const end = Number(url.searchParams.get("end_hour") || 24);
      const rows = snapshot.rows.filter(
        (r) =>
          r.event_date === url.searchParams.get("event_date") &&
          stations.includes(r.station) &&
          hourOf(r.hour_start) >= start &&
          hourOf(r.hour_start) < end,
      );
      return {
        rows,
        metadata: { ...snapshot.metadata, row_count: rows.length },
      } as T;
    }
    throw new Error("公開快照未提供這個資料項目。");
  }
  const response = await fetch(`/api${path}`, { signal });
  if (!response.ok) {
    let message = "暫時無法讀取資料，請確認本機資料庫與網站服務。";
    try {
      const data = await response.json();
      if (typeof data.detail === "string") message = data.detail;
    } catch {
      /* Keep a user-readable message for network and proxy failures. */
    }
    throw new Error(message);
  }
  return response.json();
}

export const numberText = (value: number | null | undefined) =>
  value == null || !Number.isFinite(value)
    ? "—"
    : new Intl.NumberFormat("zh-TW", { maximumFractionDigits: 0 }).format(
        value,
      );

export const hourOf = (value: string) => Number(value.slice(11, 13));

/** A missing or incomplete source bin must not silently become zero. */
export function completeTotal(
  rows: FlowRow[],
  field: "origin_count" | "destination_count_same_source_bin",
  expected = 6,
): number | null {
  const complete =
    field === "origin_count" ? "origin_complete" : "destination_complete";
  if (
    rows.length !== expected ||
    rows.some((row) => row[field] == null || !row[complete])
  )
    return null;
  return rows.reduce((sum, row) => sum + (row[field] as number), 0);
}

export function csvUrl(date: string, stations: string[]) {
  if (STATIC_MODE && snapshotCache) {
    const fields: (keyof FlowRow)[] = [
      "event_date",
      "hour_start",
      "station",
      "origin_count",
      "destination_count_same_source_bin",
      "origin_complete",
      "destination_complete",
      "baseline_origin_mean_28d",
      "baseline_n_days",
    ];
    const rows = snapshotCache.rows.filter(
      (r) => r.event_date === date && stations.includes(r.station),
    );
    const escape = (v: unknown) =>
      v == null ? "" : '"' + String(v).replaceAll('"', '""') + '"';
    const csv = [
      fields.join(","),
      ...rows.map((r) => fields.map((f) => escape(r[f])).join(",")),
    ].join("\r\n");
    return "data:text/csv;charset=utf-8," + encodeURIComponent("\uFEFF" + csv);
  }
  if (STATIC_MODE) return `${import.meta.env.BASE_URL}data/flows-${date}.csv`;
  return `/api/flows.csv?${new URLSearchParams({ event_date: date, stations: stations.join(","), start_hour: "18", end_hour: "24" })}`;
}
