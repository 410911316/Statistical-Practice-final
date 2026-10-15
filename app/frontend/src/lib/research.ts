export const publicFile = (name: string) =>
  `${import.meta.env.BASE_URL}data/${name}`;
export type Numeric = number | string;
export interface Scenario {
  event_date: string;
  shape: string;
  center_minute: Numeric;
  northbound_share: Numeric;
  remaining_fraction: Numeric;
  direction: string;
  available_spaces: Numeric;
  mean_wait_minutes: Numeric | null;
  max_queue: Numeric;
  forecast_demand: Numeric;
  residual_queue: Numeric;
  forecast_peak_hour: Numeric;
  cleared: string;
}
export interface Research {
  generated_at: string;
  model: {
    name: string;
    metrics: Record<string, string>;
    selection_note: string;
    hourly: {
      event_date: string;
      hour: number;
      actual: number;
      predicted: number;
    }[];
    events: Record<string, string>[];
  };
  queue: {
    scenarios: Scenario[];
    summary: Record<string, string>[];
    assumptions: {
      platform_delay_minutes: number;
      max_train_persons: number;
      cars: number;
    };
  };
  operations: Record<string, Record<string, string>[]>;
}
export const SHAPES = [
  { id: "uniform", name: "均勻" },
  { id: "broad", name: "寬峰" },
  { id: "concentrated", name: "集中峰" },
];
export function scenarioRows(
  data: Research,
  event: string,
  center: number,
  share: number,
  capacity: number,
  direction: string,
) {
  return SHAPES.map((shape) =>
    data.queue.scenarios.find(
      (r) =>
        r.event_date === event &&
        r.shape === shape.id &&
        +r.center_minute === (shape.id === "uniform" ? 30 : center) &&
        +r.northbound_share === share &&
        +r.remaining_fraction === capacity &&
        r.direction === direction,
    ),
  );
}
export const waitValue = (r: Scenario | undefined) =>
  !r ||
  !["True", "true"].includes(r.cleared) ||
  r.mean_wait_minutes == null ||
  r.mean_wait_minutes === ""
    ? null
    : +r.mean_wait_minutes;
export const departureTime = (minute: number) =>
  `${minute >= 1440 ? "次日 " : ""}${String(Math.floor(minute / 60) % 24).padStart(2, "0")}:${String(minute % 60).padStart(2, "0")}`;
