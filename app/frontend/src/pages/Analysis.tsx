import { useMemo, useState } from "react";
import { ArrowDownToLine, ArrowUpRight, Check, RefreshCw } from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Button } from "@/components/ui/button";
import {
  completeTotal,
  csvUrl,
  hourOf,
  numberText,
  type FlowRow,
} from "../lib/api";
import { STATIONS } from "../data/geography";
import "./Analysis.css";

interface AnalysisProps {
  date: string;
  rows: FlowRow[];
  loading: boolean;
  error: string | null;
  retry: () => void;
}

const HOURS = [18, 19, 20, 21, 22, 23];

function initialStation() {
  try {
    const saved = localStorage.getItem("dadaocheng-analysis-station");
    return STATIONS.some((station) => station.id === saved)
      ? (saved as string)
      : "北門";
  } catch {
    return "北門";
  }
}

export function Analysis({ date, rows, loading, error, retry }: AnalysisProps) {
  const [stationId, setStationId] = useState(initialStation);
  const station = STATIONS.find((item) => item.id === stationId) ?? STATIONS[0];
  const stationRows = useMemo(
    () =>
      rows.filter(
        (row) =>
          row.event_date === date &&
          row.station === station.id &&
          HOURS.includes(hourOf(row.hour_start)),
      ),
    [date, rows, station.id],
  );
  const hourly = useMemo(
    () =>
      HOURS.map((hour) => {
        const row = stationRows.find(
          (item) => hourOf(item.hour_start) === hour,
        );
        return {
          hour,
          time: `${hour}:00`,
          origin: row?.origin_complete ? row.origin_count : null,
          destination: row?.destination_complete
            ? row.destination_count_same_source_bin
            : null,
          baseline:
            row && row.baseline_n_days > 0
              ? row.baseline_origin_mean_28d
              : null,
          baselineDays: row?.baseline_n_days ?? 0,
          complete: !!(
            row?.origin_complete &&
            row.destination_complete &&
            row.origin_count != null &&
            row.destination_count_same_source_bin != null
          ),
        };
      }),
    [stationRows],
  );

  const coverage = hourly.filter((row) => row.complete).length;
  const isReady = !loading && !error && !!date;
  const entryTotal = isReady
    ? completeTotal(stationRows, "origin_count")
    : null;
  const destinationTotal = isReady
    ? completeTotal(stationRows, "destination_count_same_source_bin")
    : null;
  const noRows = isReady && !stationRows.length;

  function selectStation(id: string) {
    setStationId(id);
    try {
      localStorage.setItem("dadaocheng-analysis-station", id);
    } catch {
      /* Private browsing can disable storage. */
    }
  }

  return (
    <div className="analysis-page">
      <section className="analysis-intro">
        <div>
          <div className="eyebrow">DATA EXPLORER</div>
          <h1>分析資料</h1>
          <p className="muted">
            從一座車站、一個時段，觀察活動當天的人流變化。
          </p>
        </div>
        {isReady && stationRows.length > 0 && (
          <a
            className="analysis-download"
            download={`人流-${date}-${station.id}.csv`}
            href={csvUrl(date, [station.id])}
          >
            <ArrowDownToLine size={15} aria-hidden="true" /> 下載 CSV
          </a>
        )}
      </section>

      <div className="analysis-controls">
        <div className="analysis-stations" aria-label="選擇分析車站">
          {STATIONS.map((item) => (
            <Button
              key={item.id}
              type="button"
              variant={station.id === item.id ? "secondary" : "ghost"}
              className={`analysis-station-button ${station.id === item.id ? "is-selected" : ""}`}
              aria-pressed={station.id === item.id}
              onClick={() => selectStation(item.id)}
            >
              {item.name}
              {station.id === item.id && <Check size={12} aria-hidden="true" />}
            </Button>
          ))}
        </div>
        <span className="analysis-period num">
          18:00 — 24:00 <span>臺北時間</span>
        </span>
      </div>

      {error ? (
        <div className="panel analysis-error" role="alert">
          <div>
            <h2>資料暫時無法讀取</h2>
            <p className="muted">{error}</p>
          </div>
          <Button variant="outline" onClick={retry}>
            <RefreshCw size={14} aria-hidden="true" />
            重新讀取
          </Button>
        </div>
      ) : (
        <>
          <section
            className="analysis-stats"
            aria-label="選定時段資料摘要"
            aria-busy={loading}
          >
            <div className="panel analysis-stat">
              <div className="analysis-stat-label">
                起站人次 <ArrowUpRight size={16} aria-hidden="true" />
              </div>
              <div className="analysis-stat-value num">
                {numberText(entryTotal)}
                <span>人次</span>
              </div>
              <p>由 {station.name} 站出發的 OD 加總</p>
            </div>
            <div className="panel analysis-stat">
              <div className="analysis-stat-label">
                目的站人次{" "}
                <span className="analysis-small-badge">同來源時段</span>
              </div>
              <div className="analysis-stat-value num">
                {numberText(destinationTotal)}
                <span>人次</span>
              </div>
              <p>以 {station.name} 站為目的站的 OD 加總</p>
            </div>
            <div className="panel analysis-stat">
              <div className="analysis-stat-label">
                時段完整度 <span className="analysis-coverage-dot" />
              </div>
              <div className="analysis-stat-value num">
                {isReady ? coverage : "—"}
                <span>/ 6 時段</span>
              </div>
              <p>
                {loading
                  ? "正在讀取資料…"
                  : coverage === 6
                    ? "六個來源時段皆有完整資料"
                    : "缺漏或不完整時段不視為零"}
              </p>
            </div>
          </section>

          <section className="panel analysis-chart-panel" aria-busy={loading}>
            <div className="analysis-chart-heading">
              <div>
                <div className="eyebrow">HOURLY FLOW</div>
                <h2>逐時人流</h2>
              </div>
              <div className="analysis-legend">
                <span>
                  <i className="legend-origin" />
                  起站人次
                </span>
                <span>
                  <i className="legend-destination" />
                  目的站人次（同來源時段）
                </span>
                <span>
                  <i className="legend-baseline" />
                  一般日起站基準
                </span>
              </div>
            </div>
            <div className="analysis-chart-body">
              {loading ? (
                <div className="empty-state" role="status">
                  <RefreshCw className="analysis-loading-icon" size={20} />
                  正在讀取逐時資料…
                </div>
              ) : noRows || !date ? (
                <div className="empty-state">
                  {date
                    ? "此活動與車站沒有可用的逐時資料。"
                    : "選擇活動日期後，即可查看人流資料。"}
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%" minHeight={270}>
                  <ComposedChart
                    data={hourly}
                    margin={{ top: 16, right: 14, left: 2, bottom: 6 }}
                    accessibilityLayer
                  >
                    <defs>
                      <linearGradient
                        id="origin-flow-fill"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="0%"
                          stopColor="#e7e5e4"
                          stopOpacity={0.22}
                        />
                        <stop
                          offset="100%"
                          stopColor="#e7e5e4"
                          stopOpacity={0.01}
                        />
                      </linearGradient>
                    </defs>
                    <CartesianGrid
                      stroke="#2b2d30"
                      strokeDasharray="3 5"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="time"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#8e929b", fontSize: 11 }}
                      tickMargin={15}
                    />
                    <YAxis
                      axisLine={false}
                      tickLine={false}
                      width={52}
                      tick={{ fill: "#8e929b", fontSize: 11 }}
                      tickFormatter={(value) => numberText(Number(value))}
                      domain={[0, "auto"]}
                    />
                    <Tooltip
                      cursor={{ stroke: "#74777e", strokeDasharray: "3 3" }}
                      content={({ active, payload, label }) =>
                        active && payload?.length ? (
                          <div className="analysis-tooltip">
                            <strong className="num">
                              {date} · {label}
                            </strong>
                            {payload.map((item) => (
                              <div key={String(item.dataKey)}>
                                <span>{item.name}</span>
                                <b className="num">
                                  {numberText(
                                    typeof item.value === "number"
                                      ? item.value
                                      : null,
                                  )}
                                </b>
                              </div>
                            ))}
                          </div>
                        ) : null
                      }
                    />
                    <Area
                      type="linear"
                      dataKey="origin"
                      name="起站人次"
                      stroke="#ecebea"
                      strokeWidth={2.3}
                      fill="url(#origin-flow-fill)"
                      connectNulls={false}
                      isAnimationActive={false}
                      activeDot={{
                        r: 4,
                        fill: "#fafafa",
                        stroke: "#101114",
                        strokeWidth: 2,
                      }}
                    />
                    <Line
                      type="linear"
                      dataKey="destination"
                      name="目的站人次（同來源時段）"
                      stroke="#8d939d"
                      strokeWidth={1.8}
                      dot={{ r: 2.5, fill: "#8d939d", strokeWidth: 0 }}
                      connectNulls={false}
                      isAnimationActive={false}
                    />
                    <Line
                      type="linear"
                      dataKey="baseline"
                      name="一般日起站基準"
                      stroke="#74706a"
                      strokeWidth={1.6}
                      strokeDasharray="5 5"
                      dot={false}
                      connectNulls={false}
                      isAnimationActive={false}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              )}
            </div>
            <p className="analysis-chart-note">
              目的站人次沿用原始 OD
              的來源時段，尚不能解讀為實際出站小時。一般日基準採既有資料的 28
              日比較設定；缺值保留空白。
            </p>
          </section>

          <section className="panel analysis-table-panel" aria-busy={loading}>
            <div className="analysis-table-heading">
              <div>
                <div className="eyebrow">SOURCE DETAILS</div>
                <h2>看見每一筆數字</h2>
              </div>
              <span className="muted">
                {station.name} · {date || "尚未選擇日期"}
              </span>
            </div>
            <div className="analysis-table-scroll">
              <table>
                <caption className="sr-only">
                  {station.name} 站 {date} 18 時至 24 時人流與一般日基準
                </caption>
                <thead>
                  <tr>
                    <th scope="col">來源時段</th>
                    <th scope="col">起站人次</th>
                    <th scope="col">
                      目的站人次<span>同來源時段</span>
                    </th>
                    <th scope="col">一般日起站均值</th>
                    <th scope="col">基準天數</th>
                    <th scope="col">完整度</th>
                  </tr>
                </thead>
                <tbody>
                  {hourly.map((row) => (
                    <tr key={row.hour}>
                      <th scope="row" className="num">
                        {row.time} — {row.hour + 1}:00
                      </th>
                      <td className="num">
                        {numberText(isReady ? row.origin : null)}
                      </td>
                      <td className="num">
                        {numberText(isReady ? row.destination : null)}
                      </td>
                      <td className="num">
                        {numberText(isReady ? row.baseline : null)}
                      </td>
                      <td className="num">
                        {isReady ? row.baselineDays : "—"}
                      </td>
                      <td>
                        <span
                          className={`analysis-row-status ${isReady && row.complete ? "is-complete" : ""}`}
                        >
                          {loading
                            ? "讀取中"
                            : row.complete && isReady
                              ? "完整"
                              : "缺漏／不完整"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <p className="analysis-deferred-note">
            本頁呈現逐時觀測；預測與候車成果請看「模型與候車」。
          </p>
        </>
      )}
    </div>
  );
}
