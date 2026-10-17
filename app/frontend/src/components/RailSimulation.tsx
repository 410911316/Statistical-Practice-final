import { useEffect, useMemo, useRef, useState } from "react";
import {
  Pause,
  Play,
  RotateCcw,
  TrainFront,
  Users,
  Crosshair,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { publicFile } from "@/lib/research";
import { numberText } from "@/lib/api";
import {
  congestionColor,
  peakMinute,
  queueAt,
  replayTime,
  trainLoad,
  visibleTrains,
  type RailReplay,
  type RailDirection,
} from "@/lib/rail";
import "./RailSimulation.css";

export function RailSimulation({
  date,
  onDateChange,
}: {
  date: string;
  onDateChange: (date: string) => void;
}) {
  const [data, setData] = useState<RailReplay>();
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    fetch(publicFile("rail-replay.json"), { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error("列車情境資料載入失敗。");
        return r.json();
      })
      .then(setData)
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      });
    return () => controller.abort();
  }, [attempt]);
  if (error)
    return (
      <div className="panel rail-empty" role="alert">
        <p>{error}</p>
        <Button
          onClick={() => {
            setError("");
            setAttempt((x) => x + 1);
          }}
        >
          重新載入
        </Button>
      </div>
    );
  if (!data)
    return (
      <div className="panel rail-empty" aria-busy="true">
        正在載入列車與候車情境…
      </div>
    );
  if (!data.events[date])
    return (
      <div className="panel rail-empty">
        <TrainFront size={28} />
        <h2>這一天尚無封存的北門預測</h2>
        <p>不以零人或其他場次代替。請選擇已完成驗證的活動：</p>
        <div className="rail-dates">
          {Object.keys(data.events).map((d) => (
            <Button variant="outline" key={d} onClick={() => onDateChange(d)}>
              {d}
            </Button>
          ))}
        </div>
      </div>
    );
  return <RailPlayback key={date} data={data} date={date} />;
}

function RailPlayback({ data, date }: { data: RailReplay; date: string }) {
  const diagramScroll = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const element = diagramScroll.current;
    if (!element) return;
    const center = () => {
      element.scrollLeft = Math.max(
        0,
        (element.scrollWidth - element.clientWidth) / 2,
      );
    };
    const observer = new ResizeObserver(center);
    observer.observe(element);
    center();
    return () => observer.disconnect();
  }, []);
  const [shape, setShape] = useState("broad");
  const [fraction, setFraction] = useState(0.5);
  const [threshold, setThreshold] = useState(1000);
  const [minute, setMinute] = useState(180);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [selectedTrain, setSelectedTrain] = useState<{
    direction: RailDirection;
    time: number;
  } | null>(null);
  const playback = useRef({ minute, horizon: 420, speed });
  const current = data.events[date][`${shape}-${fraction}`];
  const horizon =
    Math.max(current.north.queue.length, current.south.queue.length) - 1;
  useEffect(() => {
    playback.current = { minute, horizon, speed };
  }, [minute, horizon, speed]);
  useEffect(() => {
    if (!playing) return;
    let previous = performance.now();
    const timer = window.setInterval(() => {
      const now = performance.now(),
        delta = ((now - previous) / 1000) * playback.current.speed;
      previous = now;
      const next = Math.min(
        playback.current.horizon,
        playback.current.minute + delta,
      );
      playback.current.minute = next;
      setMinute(next);
      if (next >= playback.current.horizon) setPlaying(false);
    }, 100);
    return () => window.clearInterval(timer);
  }, [playing]);
  const queues = {
    north: queueAt(
      current.north,
      Math.min(minute, current.north.queue.length - 1),
    )!,
    south: queueAt(
      current.south,
      Math.min(minute, current.south.queue.length - 1),
    )!,
  };
  const total = queues.north + queues.south;
  const stationColor = congestionColor(total / threshold);
  const remaining = data.maximum * fraction;
  const peak = useMemo(
    () => peakMinute(current.north, current.south),
    [current],
  );
  const detail =
    selectedTrain &&
    current[selectedTrain.direction].trains.find(
      (t) => t.time === selectedTrain.time,
    );
  function seek(t: number) {
    setMinute(Math.min(horizon, Math.max(0, t)));
    setPlaying(false);
  }
  return (
    <section className="rail-panel panel" aria-label="列車載客與車站候車模擬">
      <div className="rail-heading">
        <div>
          <span className="eyebrow">RAIL & QUEUE / 情境回放</span>
          <h2>看見列車載量，也看見候車壓力。</h2>
          <p>{date} 預測需求 × 一般班表 · 北門高運量六節編組</p>
        </div>
        <span className="rail-simulation-tag">模擬估計，非即時監控</span>
      </div>
      <div className="rail-controls">
        <label>
          到達形狀
          <select
            value={shape}
            onChange={(e) => {
              setShape(e.target.value);
              setSelectedTrain(null);
            }}
          >
            <option value="uniform">均勻</option>
            <option value="broad">寬峰 · σ 20 分</option>
            <option value="concentrated">集中峰 · σ 5 分</option>
          </select>
        </label>
        <label>
          到北門下客後可再載
          <select
            value={fraction}
            onChange={(e) => {
              setFraction(+e.target.value);
              setSelectedTrain(null);
            }}
          >
            {[0.25, 0.5, 0.75].map((f) => (
              <option key={f} value={f}>
                {numberText(data.maximum * f)} 人／班
              </option>
            ))}
          </select>
        </label>
        <label>
          車站紅色門檻（雙向合計）
          <select
            value={threshold}
            onChange={(e) => setThreshold(+e.target.value)}
          >
            {[500, 1000, 2000, 3000].map((n) => (
              <option value={n} key={n}>
                {numberText(n)} 人
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="rail-status-strip">
        <span>
          <Users size={15} />
          北門目前候車{" "}
          <strong style={{ color: stationColor }} data-testid="queue-total">
            {numberText(total)} 人
          </strong>
        </span>
        <span>
          往松山 {numberText(queues.north)}／往新店 {numberText(queues.south)}
        </span>
        <span>高運量上限 {numberText(data.maximum)} 人／列</span>
      </div>
      <div className="rail-diagram-scroll" ref={diagramScroll}>
        <svg
          className="rail-diagram"
          viewBox="0 0 1000 380"
          role="group"
          aria-label="北門雙向列車示意；列車依載客率、北門依候車人數變色，其他站灰色待建模"
        >
          <defs>
            <pattern
              id="rail-grid"
              width="30"
              height="30"
              patternUnits="userSpaceOnUse"
            >
              <circle cx="1" cy="1" r=".7" fill="#333a3c" />
            </pattern>
          </defs>
          <rect width="1000" height="380" fill="url(#rail-grid)" />
          <rect
            x="449"
            y="46"
            width="102"
            height="179"
            rx="18"
            fill={stationColor}
            fillOpacity=".06"
            stroke={stationColor}
            strokeOpacity=".45"
          />
          <text
            x="500"
            y="32"
            textAnchor="middle"
            fill="#e9eeec"
            fontSize="16"
            fontWeight="600"
          >
            北門 G13
          </text>
          <text x="70" y="32" fill="#b3c3bd" fontSize="12">
            往松山 →
          </text>
          <text x="930" y="242" textAnchor="end" fill="#b3c3bd" fontSize="12">
            ← 往新店
          </text>
          {(["north", "south"] as const).map((direction, i) => {
            const y = i === 0 ? 88 : 183;
            const trains = visibleTrains(current[direction], minute);
            const next = current[direction].trains.find((t) => t.time > minute);
            return (
              <g key={direction}>
                <line
                  x1="70"
                  y1={y}
                  x2="930"
                  y2={y}
                  stroke="#384d45"
                  strokeWidth="8"
                />
                <line
                  x1="70"
                  y1={y}
                  x2="930"
                  y2={y}
                  stroke="#a6beb3"
                  strokeWidth="2"
                />
                {[250, 750].map((x) => (
                  <path
                    key={x}
                    d={
                      i === 0
                        ? `M ${x - 6} ${y - 5} l 7 5 -7 5`
                        : `M ${x + 6} ${y - 5} l -7 5 7 5`
                    }
                    fill="none"
                    stroke="#97b5a5"
                    strokeWidth="2"
                  />
                ))}
                <circle
                  cx="500"
                  cy={y}
                  r="10"
                  fill="#131918"
                  stroke={stationColor}
                  strokeWidth="4"
                />
                <text
                  x={i === 0 ? 70 : 930}
                  y={i === 0 ? 117 : 161}
                  textAnchor={i === 0 ? "start" : "end"}
                  fill="#9fa7a3"
                  fontSize="11"
                >
                  {next
                    ? `下一班 ${replayTime(next.time).slice(0, -3)}`
                    : "本時段末班已離站"}
                </text>
                {trains.map((train) => {
                  const load = trainLoad(
                    train,
                    minute,
                    remaining,
                    data.maximum,
                  );
                  const color = congestionColor(load.ratio);
                  const x =
                    70 + 860 * (i === 0 ? train.progress : 1 - train.progress);
                  const phase = load.after ? "離站後" : "上車前假設";
                  const name = `${direction === "north" ? "往松山" : "往新店"} ${replayTime(train.time)}，${phase}，${load.full ? "滿載" : `估計載客率 ${Math.floor(load.ratio * 100)}%`}`;
                  return (
                    <g
                      key={train.time}
                      transform={`translate(${x} ${y})`}
                      className="rail-train"
                      role="button"
                      tabIndex={0}
                      aria-label={name}
                      data-full={load.full}
                      data-load-color={color}
                      onClick={() =>
                        setSelectedTrain({ direction, time: train.time })
                      }
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          setSelectedTrain({ direction, time: train.time });
                        }
                      }}
                    >
                      <title>
                        {name}；{load.after ? "上車後" : "下客後、上車前"}約{" "}
                        {numberText(load.onboard)} 人；點選明細
                      </title>
                      <rect
                        x="-49"
                        y="-16"
                        width="98"
                        height="32"
                        rx="7"
                        fill="#101715"
                        stroke={color}
                        strokeWidth={load.full ? 3 : 2}
                      />
                      {[0, 1, 2, 3, 4, 5].map((car) => (
                        <g key={car}>
                          <rect
                            x={-42 + car * 14}
                            y="-9"
                            width="12"
                            height="18"
                            rx="2"
                            fill={color}
                            fillOpacity=".65"
                          />
                          <rect
                            x={-40 + car * 14}
                            y="-6"
                            width="8"
                            height="5"
                            rx="1"
                            fill="#101715"
                          />
                        </g>
                      ))}
                      <path
                        d={
                          i === 0 ? "M 44 -4 l 4 4 -4 4" : "M -44 -4 l -4 4 4 4"
                        }
                        stroke="#fff"
                        fill="none"
                      />
                      <text
                        y={i === 0 ? -26 : 34}
                        textAnchor="middle"
                        fill={color}
                        fontSize="12"
                        fontWeight="600"
                      >
                        {phase}{" "}
                        {load.full
                          ? "滿載 100%"
                          : `${Math.floor(load.ratio * 100)}%`}
                      </text>
                    </g>
                  );
                })}
              </g>
            );
          })}
          <text
            x="500"
            y="139"
            textAnchor="middle"
            fill={stationColor}
            fontSize="17"
            fontWeight="600"
          >
            {numberText(total)} 人候車
          </text>
          <text
            x="500"
            y="242"
            textAnchor="middle"
            fill="#8a9490"
            fontSize="10"
          >
            接近／離開位置為動畫示意，不是列車 GPS 或站間行車時間
          </text>
          <line x1="70" y1="267" x2="930" y2="267" stroke="#303536" />
          {["大橋頭", "雙連", "民權西路"].map((name, i) => (
            <g
              key={name}
              transform={`translate(${185 + i * 315} 306)`}
              aria-label={`${name}待建模，無候車估計`}
            >
              <circle
                r="10"
                fill="#202527"
                stroke={congestionColor(null)}
                strokeWidth="3"
              />
              <text x="22" y="5" fill="#a9adaf" fontSize="14">
                {name}
              </text>
              <text x="22" y="29" fill="#82878c" fontSize="11">
                待建模 · 未知，不當作 0 人
              </text>
            </g>
          ))}
        </svg>
      </div>
      <div className="rail-playback">
        <div className="rail-playback-head">
          <div>
            <Button
              size="icon"
              aria-label={playing ? "暫停列車回放" : "播放列車回放"}
              onClick={() => {
                if (minute >= horizon) setMinute(0);
                setPlaying(!playing);
              }}
            >
              {playing ? <Pause size={16} /> : <Play size={16} />}
            </Button>
            <Button
              size="icon"
              variant="ghost"
              aria-label="列車回放回到起點"
              onClick={() => seek(0)}
            >
              <RotateCcw size={16} />
            </Button>
            <strong className="num">{replayTime(minute)}</strong>
          </div>
          <div>
            <Button variant="outline" size="sm" onClick={() => seek(peak)}>
              <Crosshair size={14} />
              跳至候車尖峰
            </Button>
            <select
              aria-label="列車回放速度"
              value={speed}
              onChange={(e) => setSpeed(+e.target.value)}
            >
              <option value={0.2}>1 秒＝0.2 分鐘</option>
              <option value={1}>1 秒＝1 分鐘</option>
              <option value={3}>1 秒＝3 分鐘</option>
            </select>
          </div>
        </div>
        <input
          type="range"
          aria-label="列車回放分鐘"
          min={0}
          max={horizon}
          step={0.01}
          value={minute}
          onChange={(e) => seek(+e.target.value)}
        />
        <div className="rail-time-labels">
          <span>18:00</span>
          <span
            style={{
              position: "absolute",
              left: `${(180 / horizon) * 100}%`,
              transform: "translateX(-50%)",
            }}
          >
            21:00
          </span>
          <span>{replayTime(horizon).slice(0, -3)}</span>
        </div>
      </div>
      <div className="rail-legends">
        <div>
          <strong>列車載客率</strong>
          <span className="rail-gradient" />
          <span>0% 綠 → 50% 黃 → 100% 紅（滿載）</span>
        </div>
        <div>
          <strong>車站雙向候車</strong>
          <span className="rail-gradient" />
          <span>
            0 人綠 → {numberText(threshold / 2)} 人黃 → ≥{numberText(threshold)}{" "}
            人紅
          </span>
        </div>
        <p>
          上車前顯示固定的背景載量假設，離站瞬間才加入北門上車人數。同方向隊伍超過可再載空間時，離站必為滿載；站點顯示的是雙向候車合計。
        </p>
        <p>
          車站門檻可調，僅供視覺比較，並非月台安全容量。灰色＝尚無模型。尚未取得實際到站載量或模擬停站上車過程。
        </p>
      </div>
      {selectedTrain && detail && (
        <div className="rail-train-detail" aria-live="polite">
          <strong>
            {selectedTrain.direction === "north" ? "往松山" : "往新店"} ·{" "}
            {replayTime(detail.time).slice(0, -3)} 離站班次
          </strong>
          <span>
            下客後既有載量（假設）{numberText(data.maximum - remaining)} ＋
            北門上車（模擬）{numberText(detail.boarded)} ＝ 上車後{" "}
            {numberText(data.maximum - remaining + detail.boarded)} 人
          </span>
          <span>本班離站後該方向仍候車 {numberText(detail.left)} 人</span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSelectedTrain(null)}
          >
            關閉明細
          </Button>
        </div>
      )}
      <details className="rail-assumptions">
        <summary>這些顏色怎麼算？</summary>
        <p>
          列車顏色＝（下客後既有載量＋本班上車人數）÷ {numberText(data.maximum)}
          。上車前僅顯示下客後的既有載量假設，離站時瞬間完成上車；每列車的背景載量獨立設定，未推估跨站載量傳遞。
        </p>
        <p>
          北門顏色＝當下雙向候車合計 ÷
          你設定的紅色門檻。三種形狀維持相同逐時總量，只改最高預測小時；方向各
          50%、峰中心第 30 分鐘、站內步行 5
          分鐘。逐分鐘隊伍與每班上車數由原候車程式輸出，未另造動畫數字。
        </p>
        <p>
          採 {data.timetable_effective}{" "}
          起一般班表；歷史需求搭配現行班表是情境比較。沒有實測車上載量、即時列車位置或已校準月台人數。其他三站依你的設定保持灰色。
        </p>
      </details>
    </section>
  );
}
