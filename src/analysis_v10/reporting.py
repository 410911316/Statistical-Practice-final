"""Chinese reporting of completed Beimen ablations; no model selection here."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_v1.reporting import save, style
from analysis_v10.experiment import MODELS, summarize
from compare_fusion_losses import markdown

LABELS = {
    "shared_ridge": "四站訓練 Ridge", "shared_gam": "四站訓練 GAM", "shared_xgboost": "四站訓練 XGBoost",
    "shared_equal": "四站訓練＋等權融合", "shared_fusion_all": "四站訓練＋四站學權重",
    "shared_fusion_beimen": "四站訓練＋北門學權重", "shared_gate_all": "四站權重＋採用機制",
    "shared_gate_beimen": "北門權重＋採用機制", "local_ridge": "北門重訓 Ridge", "local_gam": "北門重訓 GAM",
    "local_xgboost": "北門重訓 XGBoost", "local_equal": "北門重訓＋等權融合",
    "local_fusion_beimen": "北門重訓＋北門融合", "local_gate_beimen": "北門重訓＋採用機制"}
PLOT_METHODS = ["shared_xgboost", "shared_fusion_all", "shared_fusion_beimen", "shared_gate_beimen",
                "local_xgboost", "local_fusion_beimen", "local_gate_beimen"]


def comparison_plot(scores, path):
    data = scores.set_index("model").loc[PLOT_METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    colors = ["#B18D4F", "#A5A5A5", "#547D94", "#3F746C", "#BE8377", "#947A9D", "#535A67"]
    for ax, metric in zip(axes, ("mae", "rmse")):
        bars = ax.barh(range(len(data)), data[metric], color=colors)
        ax.set_yticks(range(len(data)), [LABELS[m] for m in PLOT_METHODS], fontsize=10)
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.1f", padding=4)
        ax.set(xlabel="人次／小時", title=f"北門 {metric.upper()}（越低越好）", xlim=(0, data[metric].max()*1.18))
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("同九場、54 個北門站時觀測｜煙火＋天氣｜探索性時間留出\n保留四站訓練的 XGBoost，仍是本次表現最好的候選", fontsize=13)
    save(fig, path)


def complementarity_plot(pairs, path):
    part = pairs.loc[pairs.variant.eq("both") & pairs.layer.eq("past_oof") & pairs.outer_year.eq(2026)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), constrained_layout=True)
    im = None
    for ax, regime, title in zip(axes, ("shared", "local"), ("四站訓練", "北門重訓")):
        matrix = np.eye(3)
        for row in part.loc[part.regime.eq(regime)].itertuples():
            i, j = MODELS.index(row.model_a), MODELS.index(row.model_b)
            matrix[i, j] = matrix[j, i] = row.error_correlation
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=["Ridge", "GAM", "XGBoost"],
               yticklabels=["Ridge", "GAM", "XGBoost"], title=title)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=15, color="white" if abs(matrix[i,j])>.55 else "black")
    fig.colorbar(im, ax=list(axes), shrink=.85, label="活動等權 Pearson 誤差相關")
    fig.suptitle("2026 年之前的北門 OOF｜14 場、84 列\n誤差＝預測−實際；只作描述，不把逐時列當獨立樣本推論", fontsize=12)
    save(fig, path)


def profile_plot(frame, path):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True, sharey=True)
    specs = [("y", "實際", "#171717", "-"), ("shared_xgboost", "四站 XGBoost／採用機制", "#B18D4F", "--"),
             ("shared_fusion_beimen", "四站訓練＋北門權重", "#547D94", "--"),
             ("local_xgboost", "北門重訓 XGBoost", "#BE8377", ":"),
             ("local_gate_beimen", "北門重訓＋採用機制", "#786181", "-")]
    for ax, (date, event) in zip(axes.flat, frame.groupby("event_date")):
        event = event.sort_values("hour")
        for method, name, color, linestyle in specs:
            ax.plot(event.hour, event[method], color=color, linestyle=linestyle, label=name,
                    marker="o" if method=="y" else None, lw=1.8)
        ax.set(title=f"{date:%Y-%m-%d}", xticks=range(18,24), xlabel="來源小時", ylabel="起站人次")
        ax.grid(alpha=.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=10)
    fig.suptitle("北門九場同樣本比較｜沒有省略表現不好的年份", fontsize=14)
    save(fig, path)


def metric_table(scores):
    part = scores[["model", "mae", "rmse", "peak_underprediction", "peak_absolute_error", "peak_hour_mae"]].copy()
    part["model"] = part.model.map(LABELS)
    part.columns = ["方法", "MAE", "RMSE", "尖峰正向低估", "尖峰絕對誤差", "尖峰小時差"]
    return markdown(part)


def build_report(frames, scores, events, pairs, help_table, decisions, output):
    style()
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    primary = scores.loc[scores.variant.eq("both")]
    p = primary.set_index("model")
    comparison_plot(primary, directory / "beimen_focus_comparison.png")
    complementarity_plot(pairs, directory / "error_complementarity.png")
    profile_plot(frames["both"], directory / "beimen_focus_profiles.png")
    improvement = 100*(1-p.loc["shared_fusion_beimen", "mae"]/p.loc["shared_fusion_all", "mae"])
    report = ["# 第十版：誤差互補、融合採用機制與北門專用模型", "",
              f"**只用北門學權重，融合 MAE 由 {p.loc['shared_fusion_all','mae']:.2f} 降至 {p.loc['shared_fusion_beimen','mae']:.2f}（改善 {improvement:.1f}%）。北門重訓 XGBoost 為 {p.loc['local_xgboost','mae']:.2f}，四站訓練 XGBoost 為 {p.loc['shared_xgboost','mae']:.2f}。**", "",
              "研究分成兩個實驗：①固定四站基模型，只改融合權重的學習對象；②連基模型都只用北門重新訓練。兩者再各加入由過去驗證決定是否採用融合的機制。主要輸入維持煙火＋天氣。", "",
              "## 1. 比較條件", "",
              "- 同 21 場可用活動；四站基模型輸入 504 列，北門專用模型 126 列。活動數沒有增加，站時列不是獨立場次。",
              "- 同一批 2024–2026 年九場測試，全部方法只評北門 18–23 時的 54 列。目標為起站人次，不是月台瞬間人數。",
              "- 北門模型沿用相同候選資訊、90 組 XGBoost 網格、Ridge/GAM 網格、訓練期補值與支持門檻，各自在過去年度內重新調參。原 Trainer 也計算 OLS/Lasso，但本版僅比較指定三模型。",
              "- 單站資料的站別常數、站別×時段退化項與其他站的結束訊號常數，由原訓練期編碼器移除；保留小時及北門的預定展演結束訊號。不是硬塞全零交互項。",
              "- 新增輸入仍限 D−1 18:00 可核對的公告／預報；歷史 OD 標籤發布時間及日曆版本未完全核實，仍屬 retrospective_non_deployable。", "",
              "## 2. 同九場北門結果", "", metric_table(primary), "",
              "尖峰正向低估是在每場實際最高小時算 max(實際−同小時預測,0)，高估記 0；因此另報絕對誤差。RMSE 是活動 MSE 平均後開根號。各場先平均六小時，再活動等權。", "",
              "![北門聚焦比較](figures/beimen_focus_comparison.png)", "",
              "## 3. 模型錯誤有沒有互補？", "",
              "下面固定展示 2026 年之前的 14 場、84 列北門 OOF；每個基模型預測該年時都只使用更早資料。2024／2025 年截止及外層事後描述另外存表；不把不同外層中重複的 OOF 活動混合計算。", ""]
    comp = pairs.loc[pairs.variant.eq("both") & pairs.layer.eq("past_oof") & pairs.outer_year.eq(2026)].copy()
    comp["regime"] = comp.regime.map({"shared":"四站訓練", "local":"北門重訓"})
    comp["opposite_sign_fraction"] = comp.opposite_sign_fraction.map(lambda v:f"{v:.1%}")
    view = comp[["regime", "model_a", "model_b", "error_correlation", "within_event_error_correlation", "opposite_sign_fraction"]]
    view.columns = ["訓練範圍", "模型 A", "模型 B", "誤差相關", "去除場均誤差後相關", "一高估一低估比例"]
    report += [markdown(view), "", "![誤差相關](figures/error_complementarity.png)", "",
               "相關不等於融合效益。除了方向，還要看誤差幅度。下表額外檢查 XGBoost 低估時，其他模型是否更接近真值，以及兩模型各半是否改善；這是過去誤差診斷，不以每列真值挑未來模型。", ""]
    help_view = help_table.loc[help_table.variant.eq("both") & help_table.layer.eq("past_oof") & help_table.outer_year.eq(2026)].copy()
    help_view["regime"] = help_view.regime.map({"shared":"四站訓練", "local":"北門重訓"})
    for column in ["candidate_closer_when_xgb_under", "pair_blend_better_when_xgb_under"]:
        help_view[column] = help_view[column].map(lambda v:f"{v:.1%}")
    help_view = help_view[["regime", "candidate", "xgb_under_rows", "candidate_closer_when_xgb_under", "pair_blend_better_when_xgb_under", "pair_blend_mae_minus_xgb"]]
    help_view.columns = ["訓練範圍", "搭配模型", "XGB 低估列數", "搭配模型更準比例", "各半混合更準比例", "全 OOF 各半 MAE−XGB"]
    report += [markdown(help_view), "", "上述條件比例先給活動等權，再條件化於 XGBoost 低估列；正負誤差乘積小於 0 才算方向相反，誤差為 0 不算。最後一欄正數代表各半混合較差。沒有假定 84 列獨立，也未提供不適當的相關顯著性檢定。", "",
               "## 4. 融合採用機制怎麼決定？", "",
               "1. 對外層年度 Y，只收 Y 以前的時間 OOF。",
               "2. 再按 OOF 年度向前切：至少四場更早 OOF 才能學權重，預測較晚完整年度。",
               "3. 僅用北門比較這些較晚年度的活動等權 MAE；至少三場、涵蓋一個年度才作採用判斷。融合嚴格較低才採用，平手或不足時選 XGBoost。沒有搜尋門檻。",
               "4. 決策固定後，以全部過去 OOF 重估最終權重，再預測外層年度。權重擬合成績沒有拿來代替融合採用的驗證成績。", "",
               "四場／三場／一年度是本次事前固定的實作門檻，並非文獻保證足夠的樣本量。2024 年只有 2021、2022 年共兩場 OOF 可用於驗證 2023，未達四場，所以沒有合格的額外驗證，沿用 XGBoost。2025 年採用判斷有三場，2026 年有六場，證據仍少。", ""]
    rows, weight_rows = [], []
    for d in decisions:
        if d["variant"] != "both":
            continue
        name = "北門重訓" if d["regime"] == "local" else ("四站權重" if d["fit_scope"] == "all" else "北門權重")
        rows.append({"版本":name,"預測年度":d["outer_year"],"採用":d["chosen"],"驗證場數":d["validation_events"],
                     "驗證 XGB MAE": d["xgb_validation_mae"] if d["xgb_validation_mae"] is not None else "不足",
                     "驗證融合 MAE": d["fusion_validation_mae"] if d["fusion_validation_mae"] is not None else "不足"})
        weight_rows.append({"版本":name,"年度":d["outer_year"],**{m:f"{v:.2%}" for m,v in d["final_weights"].items()}})
    report += [markdown(pd.DataFrame(rows)), "", "這張表為過去的採用驗證成績，不是第 2 節外層成績。原四站模型的兩個機制都採 XGBoost，所以最後成績等於 XGBoost，不能說是『融合超越 XGBoost』。北門重訓的機制在 2025、2026 年採融合，但九場合併 MAE 反而高於該版單獨 XGBoost；選擇機制也可能泛化失敗，保留這個反例。", "",
               "### 最終過去 OOF 權重（若採融合時使用）", "", markdown(pd.DataFrame(weight_rows)), "",
               "## 5. 年度與逐場比較", ""]
    annual = summarize(events.loc[events.variant.eq("both") & events.model.isin(PLOT_METHODS)], ("year", "model"))
    annual["model"] = annual.model.map(LABELS)
    annual = annual[["year","model","mae","rmse"]]
    annual.columns = ["年度","方法","MAE","RMSE"]
    supplementary = scores.loc[scores.model.isin(["shared_xgboost", "shared_fusion_all", "shared_fusion_beimen", "shared_gate_beimen"])].pivot(index="variant", columns="model", values="mae")
    supplementary = supplementary.reindex(["core", "fireworks", "weather", "both"])[["shared_xgboost", "shared_fusion_all", "shared_fusion_beimen", "shared_gate_beimen"]]
    supplementary.index = ["基本", "煙火", "天氣", "煙火＋天氣"]
    supplementary.columns = ["四站 XGBoost", "四站權重", "北門權重", "北門權重＋採用機制"]
    supplementary = supplementary.rename_axis("輸入").reset_index()
    report += [markdown(annual), "", "![九場完整曲線](figures/beimen_focus_profiles.png)", "",
               "### 其他輸入組的補充檢查", "", markdown(supplementary), "",
               "上表均為同九場北門 MAE，只重估四站基模型的權重及採用機制；完整北門重訓限於主要煙火＋天氣組。沒有按這張外層表重新挑特徵組合。", "",
               "## 6. 建議與限制", "",
               "- 目前保留四站訓練的 XGBoost 為主要候選。只聚焦北門學融合權重，能降低原融合的誤差，但仍未優於此基準。",
               "- 僅用北門重訓沒有更準。資料列變少、失去跨站資訊及訓練後的設計／最佳參數改變，都是可能原因；這次實驗不能把差異唯一歸因於某一項。相同 XGBoost 葉節點列數門檻，在單站資料下也可能對應不同的活動支持。",
               "- 採用機制是可重現的事前程序，不保證每次不輸。不要看外層結果後改門檻或指定某年採融合。",
               "- 九場測試已反覆檢視，本版為探索性方法比較；需要新的活動作最後驗證，沒有重新宣稱獨立的泛化保證。",
               "- 現在仍是逐時起站人次預測；排隊／調度須另補方向、分鐘到達形狀、步行延遲、班距與剩餘容量。", "",
               "## 7. 文獻與重現", "",
               "- [Breiman（1996）, Stacked Regressions](https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf)：OOF 預測的非負融合；本次 MAE、活動等權及北門採用門檻是另外明定的設計。",
               "- [Cawley & Talbot（2010）, On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://jmlr.org/papers/v11/cawley10a.html)：模型選擇準則也可能過度擬合，因此採用決策與外層評分分開。",
               "- [Hyndman & Athanasopoulos, FPP3 §5.10](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation)：時間向前驗證只能使用過去資料。沒有文獻保證本案的三模型、四場或三場門檻最適。", "",
               "重跑：`python -X utf8 src/run_analysis_v10.py`；若北門專用模型已完成且來源／程式雜湊相同，可加 `--reuse-local`。需要第八版私有快照。公開表格在 `tables/`，私有逐筆預測與模型在根目錄 `.local/analysis_v10/`。詳細方法見 [process.md 第 21 節](../../process.md#21-第十版誤差互補融合採用與北門專用重訓)。", "",
               "驗證：`python -X utf8 src/verify_analysis_v10.py`，重新載入六個年度模型組核對預測，獨立建立 54 個線性規劃、重算 27 個採用決策與 38 列摘要；[檢核紀錄](independent_validation.json) 記錄差值及雜湊。22 項新增單元測試涵蓋時間洩漏、採用機制失敗案例、條件互補率及尖峰定義。", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
