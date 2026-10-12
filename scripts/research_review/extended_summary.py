"""Combine the two extensions with a fixed seven-question Holm adjustment.

This is follow-up exploration after the original results were observed. The
family does not include every earlier or future analysis in this project.
Untestable hypotheses keep null scientific p-values and use 1 only internally
to preserve the seven planned slots. No source data or database is modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re


def holm(p_values: list[float | None]) -> list[float | None]:
    slots = [1.0 if p is None else float(p) for p in p_values]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in slots):
        raise ValueError("p-values must be finite in [0,1], or None for not tested")
    order = sorted(range(len(slots)), key=slots.__getitem__)
    adjusted = [0.0] * len(slots)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(slots) - rank) * slots[index])
        adjusted[index] = min(1.0, running)
    return [None if p is None else adjusted[i] for i, p in enumerate(p_values)]


def self_test() -> None:
    assert all(math.isclose(a, b) for a, b in zip(
        holm([.01, .02, .03, .04]), [.04, .06, .06, .06]))
    assert holm([None, .02]) == [None, .04]
    assert holm([1, 0, .02]) == [1, 0, .04]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path, base: Path | None = None) -> dict:
    base = base or root / "results/research_review/2026-10-05"
    paths = [base / "station_time/summary.json", base / "weather_regression/summary.json"]
    station, weather = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    tests = []

    def add(name, label, n, raw_p, method):
        tests.append(dict(id=name, question=label, n=n, raw_p=raw_p, method=method,
                          status="not_tested" if raw_p is None else "exploratory_tested"))

    omnibus = station["station_omnibus"]
    add("station_omnibus", "四站增幅的整體比較", omnibus["n_blocks"],
        omnibus["permutation"]["p_plus_one"], "Friedman statistic, within-event Monte Carlo permutation")
    for comparison in ["大橋頭站", "雙連", "民權西路"]:
        rows = [r for r in station["station_pairwise"] if r["comparison"] == comparison]
        if len(rows) != 1:
            raise ValueError(f"Expected one preselected comparison: {comparison}")
        row = rows[0]
        add("beimen_vs_" + comparison, "北門增幅與" + comparison + "比較",
            row["n_nonzero"], row["p_two_sided_exact"], "two-sided paired sign test")
    late = station["beimen_late_share"]["event_sign_test"]
    add("beimen_late_share", "北門晚間後段占比與基準比較", late["n_nonzero"],
        late["p_two_sided_exact"], "two-sided paired sign test")
    labels = {"temperature_six_label_mean_c": "氣溫與北門相對增幅",
              "wind_six_label_mean_m_s": "風速與北門相對增幅"}
    for key, label in labels.items():
        rows = [r for r in weather["weather_correlations"] if r["variable"] == key]
        if len(rows) != 1:
            raise ValueError(f"Expected one weather hypothesis: {key}")
        row = rows[0]
        if row.get("status") == "exploratory_tested":
            if set(row["dates"]) - set(station["coverage"]["paired_dates"]):
                raise ValueError("Weather analysis includes dates outside the fixed primary sample")
            raw_p = row["permutation_p"]
        else:
            raw_p = None
        add(key, label, row.get("n", 0), raw_p, "Spearman, absolute-rho Monte Carlo permutation")
    if len(tests) != 7:
        raise ValueError("The extension family must retain exactly seven planned slots")
    adjusted = holm([r["raw_p"] for r in tests])
    for row, adjusted_p in zip(tests, adjusted):
        row["holm_seven_p"] = adjusted_p
        row["internal_adjustment_slot"] = 1.0 if row["raw_p"] is None else row["raw_p"]
    return {
        "family": "seven fixed questions in this follow-up extension only",
        "interpretation": "exploratory; Holm does not repair event dependence, exchangeability, confounding or prior outcome-informed topic selection",
        "not_tested_policy": "raw and adjusted p remain null; 1 is used internally only to preserve the planned family size",
        "holm_source": "https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html",
        "tests": tests,
        "findings": {
            "paired_events": station["coverage"]["paired_events"],
            "held_events": station["coverage"]["held_dates"],
            "friedman_Q": omnibus["friedman_Q"], "kendall_W": omnibus["kendall_W"],
            "station_descriptive": station["station_descriptive"],
            "peak_descriptive": station["peak_and_time_descriptive"],
            "weather": weather["weather_correlations"],
            "regression": weather["baseline_regression"],
        },
        "sources": [{"path": p.relative_to(root).as_posix(), "sha256": sha(p)} for p in paths],
        "script_sha256": sha(Path(__file__)),
    }


def p_text(value) -> str:
    return "未檢定" if value is None else f"{value:.6g}"


def report(data: dict) -> str:
    findings = data["findings"]
    north = next(r for r in findings["station_descriptive"] if r["station"] == "北門")
    peak = next(r for r in findings["peak_descriptive"] if r["station"] == "北門")
    regression = findings["regression"]
    n, held_n = findings["paired_events"], findings["held_events"]
    ordering = "、".join(r["station"] for r in sorted(findings["station_descriptive"], key=lambda r: r["mean_rank"], reverse=True))
    rank_text = ("每場排序均為" if math.isclose(findings["kendall_W"], 1) else "平均排名順序為") + ordering
    weather_tests = [r for r in data["tests"] if r["id"] in ("temperature_six_label_mean_c", "wind_six_label_mean_m_s")]
    weather_evidence = any(r["holm_seven_p"] is not None and r["holm_seven_p"] < .05 for r in weather_tests)
    weather_text = ("至少一項達探索性校正門檻，仍需看效應與假設" if weather_evidence else "本次未取得清楚的單調關聯證據，不能解釋為天氣沒有影響")
    worse = regression["loyo_mae_log1p"] >= regression["loyo_intercept_only_mae_log1p"]
    fit_comparison = "未降低" if worse else "降低了"
    table = "\n".join(f"| {r['question']} | {r['n']} | {p_text(r['raw_p'])} | {p_text(r['holm_seven_p'])} |" for r in data["tests"])
    weather = "\n".join(
        f"- {'氣溫' if r['variable'].startswith('temperature') else '風速'}：n={r['n']}，"
        + (f"Spearman ρ={r['rho']:.3f}。" if r.get('rho') is not None else "未符合檢定條件。")
        for r in findings["weather"])
    return f"""# 擴充統計：四站差異、時段集中、天氣關聯與簡單回歸

整理日期：2026-10-05。前一版回答活動日是否高於歷史基準；本輪新增四站比較、時段結構、天氣關聯及單一變數回歸。這是在看過前輪趨勢後進行的探索，並非獨立驗證或全研究預先註冊。

## 分析與主要發現

| 問題 | 方法與樣本 | 本次結果 |
| --- | --- | --- |
| 四站相對基準的增幅是否不同？ | 同{n}場的 Friedman、場內置換與 Kendall W | Q={findings['friedman_Q']:.1f}，W={findings['kendall_W']:.2f}；{rank_text} |
| 北門是否更常高於其他站？ | 北門對另外三站的配對符號檢定 | 各組正負場數與幅度見四站分報告；保留同場配對關係 |
| 活動人流是否更集中在晚間後段？ | {n}場，北門21:00–24:00占18:00–24:00比例，對照同場基準 | 各場占比再平均：觀測{north['observed_late_share_mean']:.2%}、基準{north['baseline_late_share_mean']:.2%}，平均相差{north['late_share_difference_mean'] * 100:.2f}個百分點 |
| 單小時尖峰有多集中？ | 全{held_n}場已舉行活動，描述峰值時段與占比 | 北門{peak['peak_includes_21_events']}/{held_n}場最高在21時來源分箱；最大一小時占六小時總量的中位數{peak['peak_share_median']:.2%} |
| 氣溫與風速是否和增幅有關？ | 北門固定樣本，Spearman配對置換與多重比較校正；各項場數見下表 | {weather_text} |
| 一般日流量能解釋多少活動日流量？ | 單一OLS：log(觀測+1)對log(基準+1) | 樣本內R²={regression['r_squared_in_sample']:.3f}；相較只用訓練平均，整年留出{fit_comparison}MAE |

四站比較的「增幅」定義為每場 `log(觀測總量+1)−log(基準總量+1)`，比較的是相對基準變化，並非四站原始總量排序。Kendall W描述跨場排名的一致程度，不是回歸的解釋變異比例。

上述時間是原始OD來源分箱。沒有實際展演結束、月台到達或出閘時間，不能把21:00–24:00直接命名為真實散場時段；也不能從人次推導候車長度。

## 七個檢定一起看

各模組另附局部校正，但本輪跨題比較統一使用下表的 **七項 Holm 校正**。OLS係數的條件p與診斷不列主要七項，也不據此另外挑選顯著結論。校正範圍不包含前輪與未來所有研究嘗試。

| 本輪問題 | 場數／非零配對數 | 原始p | 七項Holm校正p |
| --- | ---: | ---: | ---: |
{table}

四站整體置換有199,999次，原始p=0.000005是加一法的模擬解析度下限，不是精確尾機率，也不是p=0。符號檢定檢查正負方向機率，不是平均差值大小。年度敏感度見分報告。

Holm能處理多重比較，但前提仍是各檢定p值適用；它無法修復同年度／共用基準的相依、車站可交換性、時間對齊或選樣問題。因此這些仍是有條件的探索結果，不作因果或普遍母體推論。方法依據：[R官方p.adjust說明](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)。不可檢定項保留空值；程式只在校正內部以1保留原定七個位置，並不把1寫成觀測p。

## 天氣與回歸怎麼解讀

{weather}

天氣使用臺北主測站18、19、20、21、22、23六個整點標籤的平均，必須六點都可用。風速原值是整點前10分鐘平均；此摘要不是連續六小時平均風速。23:59未核實標籤排除，雨量T保留微量意義，雲量只盤點而未跨觀測方法強做推論。

簡單回歸斜率為{regression['slope']:.3f}，條件95%區間為[{regression['slope_ci95'][0]:.3f}, {regression['slope_ci95'][1]:.3f}]。R²只描述本樣本擬合；基準本身也是估計值，同年相依與殘差診斷限制仍在。

按整年留出的{regression['loyo_n']}筆結果，log1p尺度MAE為{regression['loyo_mae_log1p']:.3f}，同批fold只用訓練平均的MAE為{regression['loyo_intercept_only_mae_log1p']:.3f}；RMSE分別{regression['loyo_rmse_log1p']:.3f}與{regression['loyo_intercept_only_rmse_log1p']:.3f}。這個單變數模型相較簡單基準{fit_comparison}MAE。訓練可能包含較晚年份，所以此檢查也不是未來預測回測。

## 文件、圖表與重跑

- [四站與時段完整報告](station_time_statistics.md)：配對資料、尖峰／占比圖、逐年敏感度與方法假設。
- [天氣與回歸完整報告](weather_regression_statistics.md)：氣象品質、散布圖、回歸診斷及留年結果。
- [本輪獨立審查](review_extended_statistics.md)：原始資料重算、程式檢查與修正狀態。
- [七項校正結果](../../results/research_review/2026-10-05/extended_statistics/tests.csv)／[機器可讀彙整](../../results/research_review/2026-10-05/extended_statistics/summary.json)。

```powershell
uv sync --group notebook --group analysis --all-extras
.\\.venv\\Scripts\\python.exe -X utf8 scripts/research_review/station_time_statistics.py
.\\.venv\\Scripts\\python.exe -X utf8 scripts/research_review/weather_regression_statistics.py --reuse-snapshot
.\\.venv\\Scripts\\python.exe -X utf8 scripts/research_review/extended_summary.py
```

天氣重現需本機私有快照；首次擷取方式見分報告。各模組保存來源雜湊、方法、隨機種子和數值結果。多做檢定未增加獨立活動場數；下一步仍應先補一般日基準、核對天氣時間與官方日曆，再訂少量解釋變數與按時間的回測。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-base", type=Path, help="Folder containing station_time and weather_regression outputs")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "results/research_review/2026-10-05/extended_statistics"
    report_path = args.report or root / "docs/research_review/extended_statistics.md"
    self_test()
    data = build(root, args.input_base)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "tests.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data["tests"][0]))
        writer.writeheader()
        writer.writerows(data["tests"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep links correct when a reviewer writes a report into a private folder.
    canonical_docs = root / "docs/research_review"
    def rewrite_link(match):
        link = match.group(1)
        if link.startswith(("https://", "http://", "#")):
            return match.group(0)
        if link.startswith("../../results/research_review/2026-10-05/extended_statistics/"):
            target = output / Path(link).name
        else:
            target = canonical_docs / link
        relative = os.path.relpath(target.resolve(), report_path.parent.resolve()).replace("\\", "/")
        return f"]({relative})"
    content = re.sub(r"\]\(([^)]+)\)", rewrite_link, report(data))
    report_path.write_text(content, encoding="utf-8")
    print(json.dumps({"tests": data["tests"], "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
