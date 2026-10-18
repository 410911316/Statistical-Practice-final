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
        <div className="rail-station-title">
          <span className="rail-code">G13</span>
          <h2>北門</h2>
          <span>高運量六節編組</span>
        </div>
        <span className="rail-simulation-tag">模擬估計，非即時監控</span>
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
      <div className="rail-stage" role="group" aria-label="北門雙向列車示意">
        {(["north", "south"] as const).map((direction) => {
          const next = current[direction].trains.find((t) => t.time > minute);
          return (
            <div className="rail-lane" key={direction}>
              <div className="rail-lane-heading">
                <strong>
                  {direction === "north" ? "往松山 →" : "← 往新店"}
                </strong>
                <span>
                  {next
                    ? `下一班 ${replayTime(next.time).slice(0, -3)}`
                    : "本時段末班已離站"}
                </span>
              </div>
              <div className="rail-track">
                <div
                  className="rail-platform"
                  style={{ borderColor: stationColor }}
                >
                  <span style={{ color: stationColor }}>
                    {numberText(queues[direction])} 人候車
                  </span>
                </div>
                {visibleTrains(current[direction], minute).map((train) => {
                  const load = trainLoad(
                    train,
                    minute,
                    remaining,
                    data.maximum,
                  );
                  const color = congestionColor(load.ratio);
                  const phase = load.after ? "離站後" : "上車前假設";
                  const name = `${direction === "north" ? "往松山" : "往新店"} ${replayTime(train.time)}，${phase}，${load.full ? "滿載" : `估計載客率 ${Math.floor(load.ratio * 100)}%`}`;
                  return (
                    <button
                      key={train.time}
                      className="rail-train"
                      aria-label={name}
                      data-full={load.full}
                      data-load-color={color}
                      aria-pressed={
                        selectedTrain?.direction === direction &&
                        selectedTrain.time === train.time
                      }
                      style={{
                        left: `${100 * (direction === "north" ? train.progress : 1 - train.progress)}%`,
                        color,
                      }}
                      title={`${name}；約 ${numberText(load.onboard)} 人；點選明細`}
                      onClick={() =>
                        setSelectedTrain({ direction, time: train.time })
                      }
                    >
                      <span className="rail-train-load">
                        {load.full
                          ? "滿載"
                          : `${Math.floor(load.ratio * 100)}%`}
                        <small>{phase}</small>
                      </span>
                      <span className="rail-cars" aria-hidden="true">
                        {[0, 1, 2, 3, 4, 5].map((car) => (
                          <i key={car} />
                        ))}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
        <p className="rail-stage-note">
          點選列車查看載量明細 · 移動位置僅為示意
        </p>
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
      <details className="rail-settings">
        <summary>
          <span>情境設定</span>
          <span>
            {shape === "uniform"
              ? "均勻"
              : shape === "broad"
                ? "寬峰"
                : "集中峰"}{" "}
            · 可再載 {numberText(remaining)} 人／班
          </span>
        </summary>
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
      </details>
      <div className="rail-unmodeled" aria-label="其他車站建模狀態">
        <span>其他車站</span>
        {["大橋頭", "雙連", "民權西路"].map((name) => (
          <span key={name}>
            <i />
            {name} <small>待建模</small>
          </span>
        ))}
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
