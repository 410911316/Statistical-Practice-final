"""Create a slide-ready annual finance reference chart without treating gaps as zero.

The stack is an arithmetic comparison, not an audited expenditure breakdown:
publicity required + (original legal budget - publicity required) = budget.
Unknown publicity years retain only an unsplit, hatched original-budget bar.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "results/research_review/2026-10-07-refresh/data_inventory"
    / "年度預算與宣導經費.csv"
)
OUT = ROOT / "result/presentation"
D = Decimal


def decimal_text(value: Decimal, digits: int = 2) -> str:
    rounded = value.quantize(D(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return f"{rounded:,.{digits}f}"


def read_records(path: Path, start: int, end: int) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if start <= int(row["年度"]) <= end
        ]
    rows.sort(key=lambda row: int(row["年度"]))
    years = [int(row["年度"]) for row in rows]
    if years != list(range(start, end + 1)):
        raise ValueError("Each requested year must occur exactly once.")
    records = []
    for row in rows:
        budget = D(row["原列法定活動預算_元"])
        value = row["年度宣導案所需經費_元"].strip()
        publicity = D(value) if value else None
        if budget <= 0 or (publicity is not None and not 0 <= publicity <= budget):
            raise ValueError("This reference stack requires 0 <= publicity <= budget.")
        difference = budget - publicity if publicity is not None else None
        ratio = publicity / budget * 100 if publicity is not None else None
        if publicity is not None:
            if publicity + difference != budget:
                raise ValueError("Stack must reconcile to the original budget.")
            reported = row["宣導所需經費占原列法定預算_百分比"].strip()
            if reported and abs(D(reported) - ratio) > D("0.005"):
                raise ValueError("Source ratio does not agree with original amounts.")
        records.append({
            "year": int(row["年度"]),
            "budget": budget,
            "publicity": publicity,
            "difference": difference,
            "ratio": ratio,
            "budget_scope": row["預算範圍說明"],
            "budget_url": row["預算來源網址"],
            "publicity_source": row["宣導經費來源"],
        })
    return records


def build_chart(records: list[dict], target: Path, font: Path) -> None:
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=str(font)).get_name(),
        "axes.unicode_minus": False,
        "svg.fonttype": "path",
        "font.size": 14,
    })
    bg, ink, muted = "#F4F1E9", "#292E2B", "#737C75"
    green, orange, gray = "#668777", "#D89B70", "#CBD0CC"
    fig = plt.figure(figsize=(16, 9), dpi=200, facecolor=bg)
    ax = fig.add_axes((0.077, 0.30, 0.893, 0.51), facecolor=bg)
    first, last = records[0]["year"], records[-1]["year"]
    fig.text(0.075, 0.944, f"{first}–{last} 年度活動預算與宣導經費",
             fontsize=26, fontweight="bold", color=ink, va="top")
    fig.text(0.075, 0.878, "堆疊為金額對照示意；每根總高＝原列法定活動預算",
             fontsize=15, color=muted, va="top")
    ax.set_ylim(0, 1400)
    ax.set_xlim(-0.65, len(records) - 0.35)
    ax.yaxis.set_major_locator(MultipleLocator(200))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    ax.set_ylabel("金額（萬元）", color=ink, labelpad=14)
    ax.set_xticks(range(len(records)), [str(r["year"]) for r in records])
    ax.tick_params(axis="both", length=0, colors=ink, labelsize=14, pad=10)
    ax.grid(axis="y", color="#DDDCD4", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#AAB0A8")
    for index, row in enumerate(records):
        total = float(row["budget"] / 10_000)
        if row["publicity"] is None:
            ax.bar(index, total, width=0.59, color=gray,
                   edgecolor="#AAB2AB", hatch="///", linewidth=0.6, zorder=3)
            ax.text(index, total / 2, "宣導\n未核實", ha="center",
                    va="center", color=ink, fontsize=14, linespacing=1.5)
        else:
            required = float(row["publicity"] / 10_000)
            difference = float(row["difference"] / 10_000)
            ax.bar(index, required, width=0.59, color=orange, zorder=3)
            ax.bar(index, difference, bottom=required, width=0.59,
                   color=green, zorder=3)
            ax.text(index, required / 2,
                    decimal_text(row["publicity"] / 10_000),
                    ha="center", va="center", fontsize=14, color=ink)
        suffix = "*" if row["year"] == 2024 else ""
        ax.text(index, total + 30, f"{total:,.0f}{suffix}",
                ha="center", va="bottom", fontsize=16,
                color=ink, fontweight="bold")
    handles = [
        Patch(facecolor=orange, label="宣導案所需經費（對照值）"),
        Patch(facecolor=green, label="原列預算減宣導所需（算術差額）"),
        Patch(facecolor=gray, edgecolor="#AAB2AB", hatch="///",
              label="原列預算（宣導未核實，未拆分）"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.188),
               ncol=3, frameon=False, fontsize=12.5,
               handlelength=1.7, columnspacing=2.0, labelcolor=ink)
    notes = [
        "2017–2021 年宣導所需金額未核實，缺值不當成 0；「所需」不等於實際支出。",
        "綠色僅為兩數相減，未確認同一預算科目；不代表其他支出、剩餘預算或已核實的預算組成。",
        "* 2024 年預算包含河岸童樂會移入 300 萬元；各年度預算範圍須一併考量。",
        "資料：專案年度經費表（2026-10-07 快照）；金額來源與口徑另附文字檔。",
    ]
    for y, line in zip((0.157, 0.124, 0.091, 0.052), notes):
        fig.text(0.075, y, line, fontsize=11.3, color=muted, va="top")
    # Keep the slide ratio and all source/missingness notes in the export.
    fig.savefig(target.with_suffix(".png"), dpi=200, facecolor=bg)
    fig.savefig(target.with_suffix(".svg"), facecolor=bg)
    plt.close(fig)


def build_smooth_area(records: list[dict], target: Path, font: Path) -> None:
    """Smooth cumulative boundaries, retaining annual anchors and valid bands."""
    if len(records) < 2 or any(row["publicity"] is None for row in records):
        raise ValueError("Smooth area needs at least two fully observed annual records.")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=str(font)).get_name(),
        "axes.unicode_minus": False,
        "svg.fonttype": "path",
        "font.size": 14,
    })
    years = np.array([row["year"] for row in records], dtype=float)
    total = np.array([float(row["budget"] / 10_000) for row in records])
    required = np.array([float(row["publicity"] / 10_000) for row in records])
    # Interpolate cumulative boundaries, not independent band thicknesses.
    # PCHIP preserves the monotone total and avoids an invented overshooting peak.
    total_curve = PchipInterpolator(years, total, extrapolate=False)
    publicity_curve = PchipInterpolator(years, required, extrapolate=False)
    x = np.unique(np.r_[np.linspace(years[0], years[-1], 601), years])
    top, bottom = total_curve(x), publicity_curve(x)
    if not (np.all(bottom >= 0) and np.all(top >= bottom)):
        raise ValueError("Interpolated boundaries cross or become negative.")
    if not (np.allclose(total_curve(years), total)
            and np.allclose(publicity_curve(years), required)):
        raise ValueError("Interpolation must preserve every observed annual amount.")

    bg, ink, muted = "#F4F1E9", "#292E2B", "#737C75"
    green, orange = "#668777", "#D89B70"
    fig = plt.figure(figsize=(16, 9), dpi=200, facecolor=bg)
    ax = fig.add_axes((0.077, 0.29, 0.893, 0.53), facecolor=bg)
    fig.text(0.075, 0.946,
             f'{records[0]["year"]}–{records[-1]["year"]} 年度活動預算與宣導經費',
             fontsize=26, fontweight="bold", color=ink, va="top")
    fig.text(0.075, 0.88, "上緣＝原列法定活動預算；百分比＝宣導案所需經費 ÷ 原列預算",
             fontsize=15, color=muted, va="top")
    ax.set_xlim(years[0] - 0.08, years[-1] + 0.08)
    ax.set_ylim(0, (np.ceil(total.max() / 200) + 1) * 200)
    ax.yaxis.set_major_locator(MultipleLocator(200))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:,.0f}"))
    ax.set_ylabel("金額（萬元）", color=ink, labelpad=14)
    ax.set_xticks(years, [str(int(year)) for year in years])
    ax.tick_params(axis="both", length=0, colors=ink, labelsize=15, pad=11)
    ax.grid(axis="y", color="#DDDCD4", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#AAB0A8")
    ax.fill_between(x, 0, bottom, color=orange, alpha=0.77, zorder=2)
    ax.fill_between(x, bottom, top, color=green, alpha=0.50, zorder=2)
    ax.plot(x, top, color=green, linewidth=2.5, zorder=4)
    ax.plot(x, bottom, color=orange, linewidth=2.5, zorder=4)
    ax.scatter(years, total, s=45, facecolor=bg, edgecolor=green,
               linewidth=1.8, zorder=5)
    ax.scatter(years, required, s=45, facecolor=bg, edgecolor=orange,
               linewidth=1.8, zorder=5)
    for row, year, budget, publicity in zip(records, years, total, required):
        suffix = "*" if int(year) == 2024 else ""
        ax.annotate(f"{budget:,.0f}{suffix}", (year, budget),
                    xytext=(0, 13), textcoords="offset points",
                    ha="center", va="bottom", color=ink,
                    fontsize=16, fontweight="bold")
        ax.annotate(f"{publicity:,.2f}", (year, publicity),
                    xytext=(0, -17), textcoords="offset points",
                    ha="center", va="top", color=ink, fontsize=15)
        ax.annotate(f'{decimal_text(row["ratio"])}%', (year, publicity),
                    xytext=(0, 13), textcoords="offset points",
                    ha="center", va="bottom", color="#8A4824",
                    fontsize=16, fontweight="bold")
    handles = [
        Patch(facecolor=orange, alpha=0.77, label="宣導案所需經費（對照值）"),
        Patch(facecolor=green, alpha=0.50, label="原列預算減宣導所需（算術差額）"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.187),
               ncol=2, frameon=False, fontsize=14,
               handlelength=1.8, columnspacing=2.5, labelcolor=ink)
    notes = [
        "曲線僅為年度點間的視覺插值，圓點為年度資料；不表示年內支出走勢。「所需」不等於實際支出。",
        "綠色僅為兩數相減，未確認同一預算科目；不代表其他支出、剩餘預算或已核實的預算組成。",
        "* 2024 年預算包含河岸童樂會移入 300 萬元。資料：專案年度經費表（2026-10-07 快照）。",
    ]
    for y, line in zip((0.143, 0.103, 0.063), notes):
        fig.text(0.075, y, line, fontsize=11.3, color=muted, va="top")
    fig.savefig(target.with_suffix(".png"), dpi=200, facecolor=bg)
    fig.savefig(target.with_suffix(".svg"), facecolor=bg)
    plt.close(fig)
    print("Verified smooth anchors, nonnegative areas and noncrossing boundaries.")


def export_copy(records: list[dict], target: Path, source: Path) -> None:
    first, last = records[0]["year"], records[-1]["year"]
    lines = [
        f"{first}–{last} 年度活動預算與宣導經費",
        "金額單位：萬元",
        "",
        "年度｜原列法定預算｜宣導案所需經費｜比例",
    ]
    for row in records:
        budget = decimal_text(row["budget"] / 10_000)
        publicity = (
            decimal_text(row["publicity"] / 10_000)
            if row["publicity"] is not None else "未核實"
        )
        ratio = decimal_text(row["ratio"]) + "%" if row["ratio"] is not None else "—"
        lines.append(f'{row["year"]}｜{budget}｜{publicity}｜{ratio}')
    lines.extend([
        "",
        "比例＝年度宣導案所需經費 ÷ 原列法定活動預算 × 100%。",
        "以未四捨五入的原始金額計算；表內金額以萬元取小數兩位。",
        "年度金額不分攤到每場活動；所需經費不等於實際支出。",
        "堆疊中的差額僅為算術對照，未確認同科目支應，不是其他支出或剩餘預算。",
        "2024 年預算包含河岸童樂會移入 300 萬元，跨年範圍有差異。",
        "",
        "以下為資料追溯，不需貼在投影片正文：",
        f"本機來源：{source.as_posix()}",
    ])
    for row in records:
        lines.extend([
            f'\n{row["year"]} 年：原列預算 {row["budget"]:,} 元；'
            + (
                f'宣導所需 {row["publicity"]:,} 元。'
                if row["publicity"] is not None else "宣導所需未核實。"
            ),
            f'預算範圍：{row["budget_scope"]}',
            f'預算來源：{row["budget_url"]}',
            f'宣導來源：{row["publicity_source"] or "目前快照未提供金額來源"}',
        ])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--start", type=int, default=2017)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--style", choices=("bar", "smooth-area"), default="bar")
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msjh.ttc"))
    args = parser.parse_args()
    records = read_records(args.source, args.start, args.end)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"annual_finance_{args.start}_{args.end}"
    chart_suffix = "_smooth_stacked_area" if args.style == "smooth-area" else "_stacked"
    builder = build_smooth_area if args.style == "smooth-area" else build_chart
    builder(records, args.output_dir / f"{stem}{chart_suffix}", args.font)
    export_copy(records, args.output_dir / f"{stem}_canva.txt", args.source)
    print(f"Years: {len(records)}; known publicity: "
          f"{sum(r['publicity'] is not None for r in records)}")
    for suffix in (chart_suffix + ".png", chart_suffix + ".svg", "_canva.txt"):
        print(args.output_dir / (stem + suffix))


if __name__ == "__main__":
    main()
