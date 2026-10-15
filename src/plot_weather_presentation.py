"""Redraw the existing 19-event weather analysis for a presentation; no new model fit."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/research_review/2026-10-07-refresh/four_week_primary/weather_regression"
OUTPUT = ROOT / "result/presentation"


def run():
    data = pd.read_csv(SOURCE / "primary_weather_merged.csv")
    archived = pd.read_csv(SOURCE / "weather_correlations.csv").set_index("variable")
    assert len(data) == 19 and data.event_date.nunique() == 19
    assert data.station.eq("北門").all() and (data.baseline_origin_window > 0).all()
    assert data.six_exact_labels_complete.eq(True).all()
    ratio = data.origin_count_window / data.baseline_origin_window
    assert np.isfinite(ratio).all()
    variables = [
        ("temperature_six_label_mean_c", "實測氣溫", "六個時點的平均氣溫（°C）", (23.5, 33.5), [24, 26, 28, 30, 32]),
        ("wind_six_label_mean_m_s", "實測風速", "六筆風速觀測的平均（m/s）", (0.5, 4.7), [1, 2, 3, 4]),
    ]
    font = Path("C:/Windows/Fonts/msjh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        "font.family": [font_manager.FontProperties(fname=str(font)).get_name(), "DejaVu Sans"],
        "font.size": 13, "axes.unicode_minus": False, "svg.fonttype": "path",
        "text.color": "#303731", "axes.labelcolor": "#303731",
        "xtick.color": "#505953", "ytick.color": "#505953",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    fig.subplots_adjust(left=.092, right=.98, bottom=.31, top=.80, wspace=.17)
    fig.suptitle("這 19 場中，天氣與人流倍率的單調關聯不明顯",
                 x=.53, y=.98, fontsize=20, fontweight="bold")
    fig.text(.53, .89, "每個點＝一場活動　｜　3 倍＝該場晚間進站人次為一般日基準的 3 倍",
             ha="center", fontsize=12.5, color="#63766a")
    checks = []
    for ax, (column, title, xlabel, limits, ticks) in zip(axes, variables):
        rho = data[column].rank().corr(ratio.rank())
        old_rho = data[column].rank().corr(data.response_log1p_difference.rank())
        # Raw activity/base ratio is not algebraically identical to the old +1 log
        # response; explicitly verify that all ranks match before reusing rho/p.
        np.testing.assert_array_equal(ratio.rank(), data.response_log1p_difference.rank())
        assert abs(rho-old_rho) < 1e-12
        assert abs(rho-float(archived.loc[column, "rho"])) < 1e-12
        ax.scatter(data[column], ratio, s=94, color="#628875", edgecolor="#365346",
                   linewidth=.75, alpha=.90, zorder=3)
        ax.set_title(f"{title}  ·  Spearman ρ = {rho:.2f}", fontsize=16, pad=13, fontweight="bold")
        ax.set_xlabel(xlabel, labelpad=12)
        ax.set_xlim(*limits); ax.set_xticks(ticks)
        ax.set_ylim(.75, 6.2); ax.set_yticks([1, 2, 3, 4, 5, 6])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f} 倍"))
        ax.grid(axis="y", color="#cad1ca", lw=.65, alpha=.65, zorder=0)
        ax.axhline(1, color="#9aab9d", lw=1.1, ls="--", zorder=1)
        ax.tick_params(length=0, pad=7)
        for side in ["bottom", "left"]:
            ax.spines[side].set_color("#a4b2a6")
        ax.text(.98, .075, "1 倍＝一般日基準", transform=ax.transAxes,
                ha="right", color="#758575", fontsize=11)
        checks.append({"variable": column, "n": len(data), "rho": float(rho),
                       "unchanged_rank_order": True,
                       "original_holm_two_tests_p": float(archived.loc[column, "holm_two_tests_p"])})
    axes[0].set_ylabel("北門晚間進站人流／一般日基準", labelpad=12)
    fig.text(.092, .125, "探索使用實測天氣；事前預測須使用截止前已發布的預報。",
             fontsize=13.5, fontweight="bold", color="#4e725e")
    fig.text(.092, .065, "19 場／2017–2026；未控制年度與活動規模。不能據此判定天氣沒有影響。",
             fontsize=11, color="#687169")
    fig.text(.092, .020, "資料：CWA 臺北站、北捷 OD、10/7 四週基準分析。風速各筆為前 10 分鐘平均的觀測。",
             fontsize=9.5, color="#7c837d")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "svg"]:
        fig.savefig(OUTPUT/f"weather_vs_flow_canva.{suffix}", dpi=250, transparent=True)
    plt.close(fig)
    (OUTPUT/"weather_vs_flow_check.json").write_text(json.dumps({
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [SOURCE/"primary_weather_merged.csv", SOURCE/"weather_correlations.csv"]},
        "display_response": "origin_count_window / baseline_origin_window",
        "source_response": "log1p(origin_count_window) - log1p(baseline_origin_window)",
        "ratios_range": [float(ratio.min()), float(ratio.max())],
        "rank_checks": checks,
        "new_inference": False,
        "measurement_window": "legacy source labels 18..23",
        "same_year_dependence": "No new independent-event claim or confidence interval",
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    run()
