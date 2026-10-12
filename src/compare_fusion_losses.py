"""Compare MAE/MSE fusion on fixed, audited v8 chronological predictions.

Run from the project root: python -X utf8 src/compare_fusion_losses.py
No SQL access, new features, base-model retuning or deployment is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_v1.reporting import style, save
from analysis_v6.reporting import VARIANTS, KEYS
from analysis_v9.fusion import MODELS, DISPLAY_MODELS, compare_variant, evaluate, aggregate_metrics

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"ridge": "Ridge", "gam": "GAM", "xgboost": "XGBoost", "gam_equal": "等權融合",
          "mae_fusion": "MAE 學權重", "mse_fusion": "MSE 學權重", "y": "實際"}
COLORS = {"ridge": "#87949A", "gam": "#729784", "xgboost": "#B38D4D", "gam_equal": "#B2B2B2",
          "mae_fusion": "#4B7192", "mse_fusion": "#9D5752", "y": "#202020"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified(root):
    audit_path = root / "result/analysis_v8/independent_validation.json"
    validation = json.loads(audit_path.read_text(encoding="utf-8"))
    if validation["status"] != "passed":
        raise ValueError("v8 independent validation did not pass")
    expected = {item["path"]: item["sha256"] for item in validation["files"]}
    frames, weights, audits, manifests, files = {}, [], {}, {}, []
    outer_reference = meta_reference = None
    for variant in VARIANTS:
        paths = [f".local/analysis_v8/{variant}/{name}.csv" for name in
                 ("outer_predictions", "meta_predictions", "analysis_input")]
        paths += [f"result/analysis_v8/{variant}/{name}.json" for name in ("folds", "run_manifest")]
        for rel in paths:
            path = root / rel
            if not path.exists():
                raise FileNotFoundError(f"Missing v8 cache: {rel}; reproduce v8 first")
            digest = sha(path)
            if expected.get(rel) != digest:
                raise ValueError(f"v8 cache changed after independent audit: {rel}")
            files.append({"path": rel, "sha256": digest})
        outer, meta, inputs = [pd.read_csv(root / path) for path in paths[:3]]
        folds, manifest = [json.loads((root / path).read_text(encoding="utf-8")) for path in paths[3:]]
        manifest_outer = manifest["outer_test"]
        if (manifest_outer["rows"] != len(outer) or manifest_outer["events"] != outer.event_date.nunique()
                or manifest_outer["years"] != sorted(outer.outer_year.unique())):
            raise ValueError("Source manifest disagrees with outer panel")
        paired, weight, check = compare_variant(outer, meta, folds, inputs)
        outer_keys = paired[KEYS + ["y", "train_end", "outer_year"]]
        meta_keys = meta.sort_values(["outer_year", *KEYS])[["outer_year", *KEYS, "y", "base_train_end"]].reset_index(drop=True)
        if outer_reference is not None:
            if not outer_keys.equals(outer_reference) or not meta_keys.equals(meta_reference):
                raise ValueError("Four variants do not share identical test/OOF labels and cutoffs")
        else:
            outer_reference, meta_reference = outer_keys, meta_keys
        frames[variant], audits[variant], manifests[variant] = paired, check, manifest
        weights.append(weight.assign(variant=variant))
    files.append({"path": str(audit_path.relative_to(root)).replace("\\", "/"), "sha256": sha(audit_path)})
    return frames, pd.concat(weights, ignore_index=True), audits, manifests, files


def markdown(frame):
    out = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    for row in frame.itertuples(index=False, name=None):
        cells = [f"{v:,.2f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def figures(frames, summary, weights, directory):
    style()
    directory.mkdir(parents=True, exist_ok=True)
    both = summary.loc[summary.variant.eq("both")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.3), constrained_layout=True)
    for row, scope in enumerate(("all", "beimen")):
        values = both.loc[both.scope.eq(scope)].set_index("model").loc[list(DISPLAY_MODELS)]
        for col, metric in enumerate(("mae", "rmse")):
            ax = axes[row, col]
            bars = ax.bar(range(6), values[metric], color=[COLORS[m] for m in DISPLAY_MODELS])
            ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=10)
            ax.set_xticks(range(6), [LABELS[m].replace(" ", "\n") for m in DISPLAY_MODELS], fontsize=9)
            ax.set(ylabel="人次／站／小時", ylim=(0, values[metric].max() * 1.22),
                   title=f"{'四站' if scope == 'all' else '北門'}｜{metric.upper()}（越低越好）")
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
    fig.suptitle("固定煙火＋天氣組｜只改融合損失；同 9 場時間留出活動\n基模型調參維持 MAE；探索性回顧比較", fontsize=14)
    save(fig, directory / "loss_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, loss in zip(axes, ("mae", "mse")):
        part = weights.loc[weights.variant.eq("both") & weights.loss.eq(loss)].sort_values("outer_year")
        bottom = np.zeros(len(part))
        for model in MODELS:
            vals = part[model].to_numpy()
            ax.bar(part.outer_year.astype(str), vals, bottom=bottom, color=COLORS[model], label=LABELS[model], width=.6)
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v > .06:
                    ax.text(i, b + v/2, f"{v:.1%}", ha="center", va="center", color="white", fontsize=11)
            bottom += vals
        ax.set(title=f"{loss.upper()} 學權重", xlabel="外層測試年度", ylabel="權重", ylim=(0, 1.03))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("每年度只用較早 OOF 學權重｜同一權重套用四站、全部時段")
    save(fig, directory / "fusion_weights.png")

    frame = frames["both"].loc[frames["both"].station.eq("北門")]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5), constrained_layout=True, sharey=True)
    for ax, (date, part) in zip(axes.flat, frame.groupby("event_date")):
        part = part.sort_values("hour")
        for model in ("y", "xgboost", "mae_fusion", "mse_fusion"):
            ax.plot(part.hour, part[model], marker="o" if model == "y" else None,
                    linestyle="-" if model in ("y", "mse_fusion") else "--",
                    color=COLORS[model], label=LABELS[model], lw=1.8)
        ax.set(title=f"{date:%Y-%m-%d}", xticks=range(18, 24), xlabel="來源小時", ylabel="起站人次")
        ax.grid(alpha=.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.suptitle("北門全部 9 場外層預測｜煙火＋天氣｜非分鐘月台到達率", fontsize=14)
    save(fig, directory / "beimen_profiles.png")


def report(summary, events, weights, output):
    both = summary.loc[summary.variant.eq("both")]
    primary = both.loc[both.scope.eq("beimen")].set_index("model")
    a, b = primary.loc["mae_fusion"], primary.loc["mse_fusion"]
    interpretation = ("本次北門 MAE、RMSE 均以 MAE 學權重較低；MSE 並沒有改善整體尖峰低估。"
                      if a.mae < b.mae and a.rmse < b.rmse and a.peak_underprediction < b.peak_underprediction
                      else "MAE、RMSE 與尖峰指標應分別判讀，不能只憑訓練目標決定勝者。")
    sections = ["# 第九版：MAE 與 MSE 融合權重的公平比較", "",
                f"**{interpretation}** 這是相同九場的探索性結果，不表示 MAE 在所有情境都比 MSE 好。", "",
                "固定第八版 Ridge、GAM、XGBoost 的訓練、調參與預測，只更換融合權重的損失函數。主要組仍為煙火＋天氣，其他三組完整列出。這不是重新以 MSE 調整所有基模型。", "",
                "- 外層：2024–2026 年各 3 場，共 9 場、四站 × 18–23 時 = 216 列；北門 54 列。",
                "- 每個年度只用更早的時間 OOF 同時擬合 MAE、MSE 權重；相同非負、總和為 1 限制。",
                "- 每場活動等權，活動內站時等權。MSE 的單位為人次平方；展示 RMSE，與 MAE 都是人次／站／小時。",
                "- RMSE = sqrt(各活動 MSE 的平均)，不是各活動 RMSE 的平均。在同一樣本及加權定義下，最小化 MSE 與 RMSE 的權重相同。", ""]
    for scope, name in (("all", "四站"), ("beimen", "北門")):
        part = both.loc[both.scope.eq(scope)].copy()
        part["model"] = part.model.map(LABELS)
        table = part[["model", "mae", "rmse", "peak_underprediction", "peak_absolute_error", "peak_hour_mae"]]
        table.columns = ["方法", "MAE", "RMSE", "尖峰正向低估", "尖峰絕對誤差", "尖峰時刻差（小時）"]
        sections += [f"## {name}結果：煙火＋天氣", "", markdown(table), ""]
    sections += ["尖峰正向低估：每站每場以實際最高人次的那個小時，計算 max(實際−同小時預測, 0)，高估記 0；另列絕對誤差，避免只報低估而掩蓋高估。尖峰時刻差比較各自最高小時，並列最高時取較早小時。", "",
                 "![MAE、RMSE 比較](figures/loss_comparison.png)", "",
                 "## 每年度的融合權重", ""]
    w = weights.loc[weights.variant.eq("both"), ["outer_year", "loss", *MODELS, "oof_events", "oof_mae", "oof_rmse"]].copy()
    for model in MODELS:
        w[model] = w[model].map(lambda value: f"{value:.2%}")
    w.columns = ["外層年度", "目標", "Ridge", "GAM", "XGBoost", "OOF 場數", "OOF MAE", "OOF RMSE"]
    sections += [markdown(w), "", "OOF 成績是用來學權重的訓練損失，不能當外層泛化成績。不同目標只能保證在自己目標的訓練 OOF 上不更差，不能保證未來或北門較準。", "",
                 "![年度權重](figures/fusion_weights.png)", "", "## 四組配對差異", "",
                 "下表是 MSE 融合 − MAE 融合；負數表示 MSE 融合誤差較低。主要組不因本表結果改選。", ""]
    differences = []
    for (variant, scope), part in summary.groupby(["variant", "scope"], sort=False):
        p = part.set_index("model")
        differences.append({"輸入": VARIANTS[variant], "範圍": "四站" if scope == "all" else "北門",
                            "ΔMAE": p.loc["mse_fusion", "mae"] - p.loc["mae_fusion", "mae"],
                            "ΔRMSE": p.loc["mse_fusion", "rmse"] - p.loc["mae_fusion", "rmse"],
                            "Δ尖峰正向低估": p.loc["mse_fusion", "peak_underprediction"] - p.loc["mae_fusion", "peak_underprediction"]})
    sections += [markdown(pd.DataFrame(differences)), "", "## 北門逐場結果", "",
                 "![九場北門曲線](figures/beimen_profiles.png)", ""]
    paired = events.loc[events.variant.eq("both") & events.scope.eq("beimen") & events.model.isin(["mae_fusion", "mse_fusion"])]
    a = paired.loc[paired.model.eq("mae_fusion")].set_index("event_date")
    b = paired.loc[paired.model.eq("mse_fusion")].set_index("event_date")
    diff = pd.DataFrame({"活動": a.index.strftime("%Y-%m-%d"), "MAE 融合 MAE": a.mae.to_numpy(),
                         "MSE 融合 MAE": b.mae.to_numpy(), "MAE 融合 RMSE": np.sqrt(a.mse.to_numpy()),
                         "MSE 融合 RMSE": np.sqrt(b.mse.to_numpy())})
    annual = aggregate_metrics(paired, ("year", "model"))[["year", "model", "mae", "rmse", "peak_underprediction"]]
    annual["model"] = annual.model.map(LABELS)
    annual.columns = ["年度", "方法", "MAE", "RMSE", "尖峰正向低估"]
    sections += [markdown(diff), "", "### 北門分年度：保留相反結果", "", markdown(annual), "",
                 "年度成績不可用來事後替同年度挑模型；每年度權重在看到當年度真值前已由過去 OOF 固定。", "",
                 "## 方法、文獻與限制", "",
                 "- [Breiman（1996）, Stacked Regressions](https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf)：支持以交叉驗證預測做非負最小平方融合；本案另固定總和為 1 與時間向前切分，不宣稱全部細節照搬原文。",
                 "- [Ersoy 等, Stacking Regression for Time-Series（2023 作者稿，2024 章節）](https://doi.org/10.1007/978-3-031-35763-3_7)：第 3 節比較 L1/L2 stacking，式 (2) 的 L1 使用非負、總和為 1 的絕對誤差目標。[作者稿](https://pure.hw.ac.uk/ws/portalfiles/portal/91472074/Schaffer_Stacking_paper_ECONVN2023.pdf)。",
                 "- [Gneiting（2011）, Making and Evaluating Point Forecasts](https://arxiv.org/abs/0912.0902)：平方損失與條件平均數、絕對損失與條件中位數相應；有限樣本、受限制的融合不保證恰好等於兩者。",
                 "- MAE 或 RMSE 較低都是這九場的描述結果，尚無新未見測試集支持最終選模；同一活動的 24 列不是獨立活動，未做逐列顯著性檢定。",
                 "- 沿用 D−1 18:00 事前節目／預報輸入；歷史 OD 標籤發布時間及日曆版本仍未完全核實，狀態維持 retrospective_non_deployable。沒有新加入當日觀測或實際散場特徵。",
                 "- 融合權重是全站、全時段共同估計；MSE 同時重罰高估與低估，不保證降低每場尖峰低估。凸組合不能超過該列三個基模型的最高預測。",
                 "- 接排隊前還需方向比例、分鐘到達形狀、步行／入站延遲、班距及剩餘容量。這次未產生候車時間或調度效果。", "",
                 "重現：`python -X utf8 src/compare_fusion_losses.py`。需本機 `.local/analysis_v8/` 已驗證快照；公開的彙整圖表不包含逐筆私有預測。完整實作見根目錄 process.md 第 20 節。", ""]
    (output / "report.md").write_text("\n".join(sections), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    frames, weights, audits, manifests, sources = load_verified(root)
    summaries, event_tables = [], []
    for variant, frame in frames.items():
        summary, events = evaluate(frame)
        summaries.append(summary.assign(variant=variant))
        event_tables.append(events.assign(variant=variant))
    summary, events = pd.concat(summaries, ignore_index=True), pd.concat(event_tables, ignore_index=True)
    annual = aggregate_metrics(events, ("variant", "scope", "year", "model"))
    output, private = root / "result/analysis_v9", root / ".local/analysis_v9"
    for path in (output / "tables", private):
        path.mkdir(parents=True, exist_ok=True)
    for name, frame in (("summary", summary), ("event_metrics", events), ("annual_metrics", annual), ("weights", weights)):
        frame.to_csv(output / "tables" / f"{name}.csv", index=False, encoding="utf-8-sig")
    for variant, frame in frames.items():
        frame.to_csv(private / f"{variant}_paired_predictions.csv", index=False, encoding="utf-8-sig")
    figures(frames, summary, weights, output / "figures")
    report(summary, events, weights, output)
    code_paths = [Path(__file__), root / "src/analysis_v9/fusion.py", root / "src/analysis_v5/models.py",
                  root / "src/analysis_v6/reporting.py", root / "src/analysis_v5/reporting.py", root / "src/analysis_v1/reporting.py"]
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "analysis_version": "v9",
                "status": "exploratory_cached_outer_comparison", "primary_variant": "both", "fusion_models": list(MODELS),
                "changed": "fusion_loss_only_MAE_vs_MSE", "base_tuning_loss": "MAE_unchanged",
                "event_weighting": "equal_events_then_equal_rows_within_event",
                "d1_deployable": False, "d1_limitations": manifests["both"]["data"]["audit"],
                "source_files": sources, "checks": audits,
                "code": [{"path": str(p.relative_to(root)).replace("\\", "/"), "sha256": sha(p)} for p in code_paths]}
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.loc[summary.variant.eq("both"), ["scope", "model", "mae", "rmse", "peak_underprediction"]].to_string(index=False))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
