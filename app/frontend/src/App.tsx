import { lazy, Suspense, useEffect, useState } from "react";
import {
  BrowserRouter,
  HashRouter,
  useLocation,
  NavLink,
  Route,
  Routes,
  Link,
} from "react-router-dom";
import {
  Activity,
  ArrowDownToLine,
  ArrowUpRight,
  ChevronDown,
  Database,
  ExternalLink,
  Layers2,
  Map as MapIcon,
  Orbit,
  Pause,
  Play,
  RotateCcw,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  api,
  STATIC_MODE,
  completeTotal,
  csvUrl,
  hourOf,
  numberText,
  type EventRecord,
  type FlowResponse,
  type FlowRow,
} from "@/lib/api";
import { STATIONS, SOURCE_LINKS } from "@/data/geography";
import { FlowMap } from "@/components/FlowMap";
const Analysis = lazy(() =>
  import("@/pages/Analysis").then((module) => ({ default: module.Analysis })),
);

const Research = lazy(() =>
  import("@/pages/Research").then((m) => ({ default: m.ResearchPage })),
);
const RailSimulation = lazy(() =>
  import("@/components/RailSimulation").then((m) => ({
    default: m.RailSimulation,
  })),
);

function useSaved<T>(
  key: string,
  fallback: T,
  valid: (value: unknown) => boolean,
) {
  const [value, setValue] = useState<T>(() => {
    try {
      const v = JSON.parse(
        localStorage.getItem(`dadaocheng.v1.${key}`) || "null",
      );
      return valid(v) ? (v as T) : fallback;
    } catch {
      return fallback;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(`dadaocheng.v1.${key}`, JSON.stringify(value));
    } catch {
      /* Storage optional. */
    }
  }, [key, value]);
  return [value, setValue] as const;
}
const range = (a: number, b: number) => (v: unknown) =>
  typeof v === "number" && Number.isFinite(v) && v >= a && v <= b;
const timeText = (m: number) =>
  `${String(18 + Math.floor(m / 60)).padStart(2, "0")}:${String(Math.floor(m % 60)).padStart(2, "0")}`;
const statuses: Record<string, string> = {
  held: "已舉行",
  cancelled: "取消",
  postponed: "延期",
};

function Workspace() {
  const researchPage = useLocation().pathname === "/research";
  const [viewMode, setViewMode] = useState<"rail" | "walk">("rail");
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [date, setDate] = useSaved(
    "date",
    "",
    (v) => typeof v === "string" && /^\d{4}-\d{2}-\d{2}$/.test(v),
  );
  const [rows, setRows] = useState<FlowRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [revision, setRevision] = useState(0);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [minute, setMinute] = useSaved("minute", 150, range(0, 359));
  const [speed, setSpeed] = useSaved("speed", 5, (v) =>
    [1, 5, 15].includes(v as number),
  );
  const [density, setDensity] = useSaved("density", 36, range(12, 72));
  const [opacity, setOpacity] = useSaved("opacity", 75, range(20, 100));
  const [labels, setLabels] = useSaved(
    "labels",
    true,
    (v) => typeof v === "boolean",
  );
  const [selected, setSelected] = useSaved("station", "北門", (v) =>
    STATIONS.some((s) => s.id === v),
  );
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    setEvents([]);
    setRows([]);
    setConnected(false);
    setPlaying(false);
    api<{ events: EventRecord[] }>("/events", controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return;
        const ordered = [...result.events].sort((a, b) =>
          a.event_date.localeCompare(b.event_date),
        );
        setEvents(ordered);
        setConnected(true);
        setDate((current) =>
          ordered.some((e) => e.event_date === current)
            ? current
            : [...ordered].reverse().find((e) => e.event_status === "held")
                ?.event_date ||
              ordered.at(-1)?.event_date ||
              "",
        );
        if (!result.events.length) {
          setLoading(false);
          setError("目前資料庫沒有活動日期。");
        }
      })
      .catch((e) => {
        if (!controller.signal.aborted) {
          setError(e.message);
          setLoading(false);
          setConnected(false);
          setRows([]);
        }
      });
    return () => controller.abort();
  }, [revision, setDate]);
  useEffect(() => {
    if (!date || !events.some((e) => e.event_date === date)) return;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    setRows([]);
    setPlaying(false);
    const query = new URLSearchParams({
      event_date: date,
      stations: STATIONS.map((s) => s.id).join(","),
      start_hour: "18",
      end_hour: "24",
    });
    api<FlowResponse>(`/flows?${query}`, controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return;
        setRows(result.rows);
        setConnected(true);
        setLoading(false);
      })
      .catch((e) => {
        if (!controller.signal.aborted) {
          setError(e.message);
          setLoading(false);
          setConnected(false);
        }
      });
    return () => controller.abort();
  }, [date, events]);
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () => setMinute((m) => Math.min(359, m + speed / 5)),
      200,
    );
    return () => clearInterval(timer);
  }, [playing, speed, setMinute]);
  useEffect(() => {
    if (minute >= 359) setPlaying(false);
  }, [minute]);
  const retry = () => setRevision((n) => n + 1);
  const currentEvent = events.find((e) => e.event_date === date);
  const currentHour = 18 + Math.floor(minute / 60);
  const station = STATIONS.find((s) => s.id === selected) || STATIONS[0];
  const currentRow = rows.find(
    (r) => r.station === selected && hourOf(r.hour_start) === currentHour,
  );
  const totals = STATIONS.map((s) =>
    completeTotal(
      rows.filter((r) => r.station === s.id),
      "origin_count",
    ),
  );
  const combined = totals.every((n) => n != null)
    ? totals.reduce<number>((a, b) => a + (b ?? 0), 0)
    : null;
  const completeBins = rows.filter(
    (r) => r.origin_complete && r.destination_complete,
  ).length;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link to="/" className="brand" aria-label="回人流展示首頁">
          <span className="brand-mark">
            <Orbit size={27} />
          </span>
          <span>
            稻埕觀測室<small>DADAOCHENG OBSERVATORY</small>
          </span>
        </Link>
        <div className="workspace-label">
          統計實務 · 研究專案<span>01</span>
        </div>
        <nav aria-label="主要導覽">
          <NavLink to="/" end>
            <MapIcon size={18} />
            人流展示<span className="nav-number">01</span>
          </NavLink>
          <NavLink to="/analysis">
            <Activity size={18} />
            資料分析<span className="nav-number">02</span>
          </NavLink>
          <NavLink to="/research">
            <Layers2 size={18} />
            模型與候車<span className="nav-number">03</span>
          </NavLink>
        </nav>
        <div className="sidebar-note">
          <span className="eyebrow">THE RESEARCH</span>
          <p>
            從一場煙火，
            <br />
            看見城市的流動。
          </p>
          <span className="muted">大稻埕活動 × 捷運人流</span>
        </div>
        <div className="sidebar-bottom">
          <div className="private-label">
            <ShieldCheck size={15} />
            {STATIC_MODE ? "公開研究成果" : "團隊研究空間"}
          </div>
          <p>每個人的操作獨立保留在自己的瀏覽器。</p>
          <button
            onClick={() => setSourcesOpen(!sourcesOpen)}
            className="source-link"
          >
            資料來源與說明
            <ArrowUpRight size={14} />
          </button>
          <div className="version">
            成果展示<span>v0.2</span>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            大稻埕煙火研究<span>/</span>
            <span className="breadcrumb-last">人流觀測</span>
          </div>
          <div className="topbar-right">
            <span className={`connection ${connected ? "is-connected" : ""}`}>
              <i />
              {connected
                ? STATIC_MODE
                  ? "公開快照已載入"
                  : "MySQL 已連線"
                : loading
                  ? "資料連接中"
                  : STATIC_MODE
                    ? "快照未載入"
                    : "資料庫未連線"}
            </span>
            <div className="avatar">研</div>
          </div>
        </header>
        <main className="main-content">
          {!researchPage && (
            <div className="context-bar">
              <div className="context-label">
                <span className="eyebrow">EVENT EXPLORER</span>
                <span className="context-dot">/</span>
                <span>活動觀測</span>
              </div>
              <div className="date-control">
                <label htmlFor="event-date">活動日期</label>
                <div className="select-wrap">
                  <select
                    id="event-date"
                    value={date}
                    onChange={(e) => setDate(e.target.value)}
                    disabled={!events.length}
                  >
                    {!events.length && <option value="">讀取中…</option>}
                    {[...events].reverse().map((e) => (
                      <option key={e.event_date} value={e.event_date}>
                        {e.event_date} ·{" "}
                        {statuses[e.event_status] || e.event_status}
                      </option>
                    ))}
                  </select>
                  <ChevronDown size={14} />
                </div>
                <Badge variant="outline">18:00–24:00</Badge>
              </div>
            </div>
          )}
          {sourcesOpen && (
            <div className="source-panel panel">
              <div>
                <strong>資料來源與使用方式</strong>
                <button
                  aria-label="關閉資料說明"
                  onClick={() => setSourcesOpen(false)}
                >
                  ×
                </button>
              </div>
              <p>
                人流來自已整理的北捷逐時
                OD。目的站人次沿用來源資料的時間格，尚未確認為實際出站時間。地圖是活動代表位置與車站入口；路徑與粒子是未經校準的示意，不代表實際足跡、導航或人數。底圖需連網載入。
              </p>
              <div className="source-links">
                {SOURCE_LINKS.map((l) => (
                  <a key={l.url} href={l.url} target="_blank" rel="noreferrer">
                    {l.label}
                    <ExternalLink size={12} />
                  </a>
                ))}
              </div>
            </div>
          )}
          {!researchPage &&
            currentEvent &&
            currentEvent.event_status !== "held" && (
              <div className="status-note">
                所選活動狀態為「
                {statuses[currentEvent.event_status] ||
                  currentEvent.event_status}
                」。下列為該日期車站運量，不能解讀成活動散場人流。
              </div>
            )}
          <Routes>
            <Route
              path="/"
              element={
                <>
                  <div className="page-heading">
                    <div>
                      <div className="eyebrow">URBAN MOVEMENT / 城市流動</div>
                      <h1>即時地圖資訊</h1>
                      <p>從散場步行到列車候車，探索大稻埕的人流與運輸情境。</p>
                    </div>
                    <Badge variant="outline" className="prototype-badge">
                      <span className="badge-dot" />
                      {viewMode === "rail" ? "列車情境回放" : "散場路徑示意"}
                    </Badge>
                  </div>
                  <div className="view-mode-switch" aria-label="展示模式">
                    <Button
                      variant={viewMode === "rail" ? "default" : "outline"}
                      aria-pressed={viewMode === "rail"}
                      onClick={() => {
                        setViewMode("rail");
                        setPlaying(false);
                      }}
                    >
                      列車與候車
                    </Button>
                    <Button
                      variant={viewMode === "walk" ? "default" : "outline"}
                      aria-pressed={viewMode === "walk"}
                      onClick={() => {
                        setViewMode("walk");
                        setPlaying(false);
                      }}
                    >
                      步行流向
                    </Button>
                  </div>
                  {error && (
                    <div role="alert" className="error-banner">
                      <Database size={17} />
                      <span>{error}</span>
                      <Button variant="outline" size="sm" onClick={retry}>
                        重新連線
                      </Button>
                    </div>
                  )}
                  {viewMode === "rail" ? (
                    <Suspense
                      fallback={
                        <div className="empty-state">正在載入列車回放…</div>
                      }
                    >
                      <RailSimulation date={date} onDateChange={setDate} />
                    </Suspense>
                  ) : (
                    <>
                      <div className="metrics-row">
                        <Metric
                          label="觀測車站"
                          value="04"
                          suffix="站"
                          note="北門 · 大橋頭 · 雙連 · 民權西路"
                        />
                        <Metric
                          label="四站進站合計"
                          value={loading ? "…" : numberText(combined)}
                          suffix="人次"
                          note="18–24 時，全體旅客，非活動人數"
                        />
                        <Metric
                          label={`${station.name}・${currentHour} 時進站`}
                          value={
                            loading
                              ? "…"
                              : numberText(
                                  currentRow?.origin_complete
                                    ? currentRow.origin_count
                                    : null,
                                )
                          }
                          suffix="人次"
                          note={
                            STATIC_MODE
                              ? "公開快照・原始時段彙整"
                              : "MySQL 原始時段彙整"
                          }
                        />
                        <Metric
                          label="完整資料時段"
                          value={loading ? "…" : `${completeBins} / 24`}
                          suffix="筆"
                          note="4 站 × 6 小時；缺值不視為零"
                        />
                      </div>
                      <div className="map-layout">
                        <section
                          className="map-panel panel"
                          aria-label="大稻埕散場流向地圖"
                        >
                          <div className="panel-header">
                            <div>
                              <span className="eyebrow">MOVEMENT MAP</span>
                              <h2>大稻埕散場流向</h2>
                            </div>
                            <span className="map-type">
                              <Layers2 size={14} />
                              步行示意圖層
                            </span>
                          </div>
                          <FlowMap
                            selected={selected}
                            onSelect={setSelected}
                            minute={minute}
                            density={density}
                            opacity={opacity / 100}
                            labels={labels}
                          />
                          <div className="timeline">
                            <div className="timeline-head">
                              <div className="playback-controls">
                                <Button
                                  size="icon"
                                  className="play-button"
                                  aria-label={playing ? "暫停播放" : "開始播放"}
                                  onClick={() => {
                                    if (minute >= 359) setMinute(0);
                                    setPlaying(!playing);
                                  }}
                                >
                                  {playing ? (
                                    <Pause size={16} />
                                  ) : (
                                    <Play size={16} />
                                  )}
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  aria-label="回到起點"
                                  onClick={() => {
                                    setMinute(0);
                                    setPlaying(false);
                                  }}
                                >
                                  <RotateCcw size={16} />
                                </Button>
                                <span className="current-time num">
                                  {timeText(minute)}
                                </span>
                                <span className="timeline-date">
                                  台北時間 · 時間格播放
                                </span>
                              </div>
                              <select
                                className="speed-select"
                                aria-label="播放速度"
                                value={speed}
                                onChange={(e) =>
                                  setSpeed(Number(e.target.value))
                                }
                              >
                                <option value={1}>1× 播放</option>
                                <option value={5}>5× 播放</option>
                                <option value={15}>15× 播放</option>
                              </select>
                            </div>
                            <Slider
                              aria-label="觀測時間"
                              value={[minute]}
                              min={0}
                              max={359}
                              step={1}
                              onValueChange={(v) => {
                                setMinute(v[0]);
                                setPlaying(false);
                              }}
                            />
                            <div className="time-labels">
                              {[
                                "18:00",
                                "19:00",
                                "20:00",
                                "21:00",
                                "22:00",
                                "23:00",
                                "24:00",
                              ].map((t) => (
                                <span key={t}>{t}</span>
                              ))}
                            </div>
                          </div>
                        </section>
                        <aside className="destinations panel">
                          <div className="panel-header">
                            <div>
                              <span className="eyebrow">DESTINATIONS</span>
                              <h2>周邊觀測車站</h2>
                            </div>
                            <span className="small-tag">{currentHour}:00</span>
                          </div>
                          <p className="destination-intro">
                            選擇車站，查看路徑與該時段的真實進站人次。
                          </p>
                          <div className="station-list">
                            {STATIONS.map((s, i) => {
                              const row = rows.find(
                                (r) =>
                                  r.station === s.id &&
                                  hourOf(r.hour_start) === currentHour,
                              );
                              const count = row?.origin_complete
                                ? row.origin_count
                                : null;
                              const max = Math.max(
                                1,
                                ...rows
                                  .filter(
                                    (r) =>
                                      hourOf(r.hour_start) === currentHour &&
                                      r.origin_complete,
                                  )
                                  .map((r) => r.origin_count || 0),
                              );
                              return (
                                <button
                                  key={s.id}
                                  onClick={() => setSelected(s.id)}
                                  className={`station-card ${selected === s.id ? "is-selected" : ""}`}
                                >
                                  <div className="station-top">
                                    <span className="station-number num">
                                      0{i + 1}
                                    </span>
                                    <div>
                                      <strong>{s.name}</strong>
                                      <span>{s.label}</span>
                                    </div>
                                    <ArrowUpRight size={16} />
                                  </div>
                                  <div className="station-value">
                                    <span className="num">
                                      {loading ? "…" : numberText(count)}
                                    </span>
                                    <small>進站人次</small>
                                  </div>
                                  <div className="station-track">
                                    <span
                                      style={{
                                        width: `${count == null ? 0 : (count / max) * 100}%`,
                                      }}
                                    />
                                  </div>
                                </button>
                              );
                            })}
                          </div>
                          <div className="route-description">
                            <span className="eyebrow">SELECTED ROUTE</span>
                            <p>{station.routeDescription}</p>
                            <small>
                              簡化道路示意；未核對當日水門與交通管制。
                            </small>
                          </div>
                        </aside>
                      </div>
                      <section className="settings-panel panel">
                        <div className="settings-title">
                          <Settings2 size={18} />
                          <div>
                            <h2>展示設定</h2>
                            <p>只改變視覺呈現，不改變觀測人次。</p>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => {
                              setDensity(36);
                              setOpacity(75);
                              setLabels(true);
                              setSpeed(5);
                            }}
                          >
                            重設
                          </Button>
                        </div>
                        <div className="settings-fields">
                          <div className="setting">
                            <label>
                              粒子密度<span className="num">{density}</span>
                            </label>
                            <Slider
                              aria-label="粒子密度"
                              min={12}
                              max={72}
                              step={4}
                              value={[density]}
                              onValueChange={(v) => setDensity(v[0])}
                            />
                            <small>視覺粒子，不對應旅客數量</small>
                          </div>
                          <div className="setting">
                            <label>
                              路線清晰度<span className="num">{opacity}%</span>
                            </label>
                            <Slider
                              aria-label="路線清晰度"
                              min={20}
                              max={100}
                              step={5}
                              value={[opacity]}
                              onValueChange={(v) => setOpacity(v[0])}
                            />
                            <small>調整地圖上的路徑透明度</small>
                          </div>
                          <div className="setting setting-switch">
                            <label htmlFor="labels">
                              車站名稱
                              <Switch
                                id="labels"
                                checked={labels}
                                onCheckedChange={setLabels}
                              />
                            </label>
                            <small>顯示車站與代表入口標籤</small>
                          </div>
                        </div>
                      </section>
                      <div className="bottom-note">
                        <SlidersHorizontal size={15} />
                        <span>
                          本頁為路徑示意；預測與候車情境請看「模型與候車」。路徑不是實測足跡。
                        </span>
                        <a
                          download={`四站人流-${date}.csv`}
                          href={
                            date
                              ? csvUrl(
                                  date,
                                  STATIONS.map((s) => s.id),
                                )
                              : undefined
                          }
                          aria-disabled={!date || !!error}
                          onClick={(e) => {
                            if (!date || error) e.preventDefault();
                          }}
                        >
                          <ArrowDownToLine size={14} />
                          匯出觀測資料
                        </a>
                      </div>
                    </>
                  )}
                </>
              }
            />
            <Route
              path="/analysis"
              element={
                <Suspense
                  fallback={<div className="empty-state">正在載入資料頁…</div>}
                >
                  <Analysis
                    date={date}
                    rows={rows}
                    loading={loading}
                    error={error}
                    retry={retry}
                  />
                </Suspense>
              }
            />
            <Route
              path="/research"
              element={
                <Suspense
                  fallback={
                    <div className="empty-state">正在載入研究成果…</div>
                  }
                >
                  <Research />
                </Suspense>
              }
            />
            <Route
              path="*"
              element={
                <div className="empty-state">
                  <h1>找不到這個頁面</h1>
                  <Link to="/">返回人流展示</Link>
                </div>
              }
            />
          </Routes>
          <footer className="footer">
            <span>大稻埕煙火人流研究</span>
            <span>
              觀測資料與模擬示意分開呈現<span className="footer-dot">·</span>
              Asia / Taipei
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}
function Metric({
  label,
  value,
  suffix,
  note,
}: {
  label: string;
  value: string;
  suffix: string;
  note: string;
}) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <div>
        <strong className="num">{value}</strong>
        <span className="metric-unit">{suffix}</span>
      </div>
      <small>{note}</small>
    </div>
  );
}
export default function App() {
  const Router = STATIC_MODE ? HashRouter : BrowserRouter;
  return (
    <Router>
      <Workspace />
    </Router>
  );
}
