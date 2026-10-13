"""Chinese, reproducible reports and static figures for completed V12 ablations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from analysis_v1.reporting import style, save
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "core": "核心", "finale": "＋壓軸", "event_type": "＋壓軸／特別場",
    "temperature": "＋溫度", "wind": "＋風速", "cloud": "＋雲量", "weather": "＋三項天氣",
    "fire_weather": "＋煙火／三項天氣", "event_type_weather": "＋場次／三項天氣",
    "sensitivity_baseline": "＋一般日基準（敏感度）", "sensitivity_google": "＋上月熱度（敏感度）"}


def table(d):
    columns = list(d.columns)
    rows = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in d.itertuples(index=False, name=None):
        vals = [f"{v:,.2f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def google_analysis(frame, output):
    g = pd.read_csv(output / "tables/google_monthly_audit.csv")
    g["month"] = pd.to_datetime(g.month)
    g["year"] = g.month.dt.year
    annual = g.groupby("year").agg(months=("month", "size"), index_mean_mid=("midpoint", "mean"),
             index_mean_lower=("lower", "mean"), index_mean_upper=("upper", "mean"),
             index_peak=("midpoint", "max")).reset_index()
    event = frame.loc[frame.station.eq("北門")].groupby(["year", "event_date"]).y.sum().reset_index()
    totals = event.groupby("year").agg(events=("event_date", "size"),
                 mean_event_18_24_trips=("y", "mean")).reset_index()
    annual = annual.merge(totals, on="year", validate="one_to_one")
    annual["full_year"] = annual.months.eq(12)
    annual.to_csv(output / "tables/google_annual_descriptive.csv", index=False, encoding="utf-8-sig")
    a = annual.loc[annual.full_year]
    result = dict(n_years=len(a), years=a.year.astype(int).tolist(),
                  pearson=float(pearsonr(a.index_mean_mid, a.mean_event_18_24_trips).statistic),
                  spearman=float(spearmanr(a.index_mean_mid, a.mean_event_18_24_trips).statistic),
                  metric="arithmetic mean of 12 monthly Google indexes vs mean observed Beimen 18-24 trips per event",
                  status="descriptive_only_query_metadata_unverified_no_causal_or_D1_claim",
                  partial_2026_excluded=True, p_values_not_reported=True,
                  censored_lower_and_upper_are_bounds_not_confidence_intervals=True)
    (output / "google_descriptive.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", "utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    axes[0].plot(g.month, g.midpoint, color="#52756e", lw=1.5)
    axes[0].set(title="117 個月的原檔相對熱度", ylabel="相對指數（0–100），不是搜尋人數",
                xlabel="月份；<1 僅以 0.5 畫圖")
    axes[1].scatter(a.index_mean_mid, a.mean_event_18_24_trips, color="#52756e", s=32)
    for r in a.itertuples():
        axes[1].annotate(str(r.year), (r.index_mean_mid, r.mean_event_18_24_trips), xytext=(4,4), textcoords="offset points", fontsize=8)
    axes[1].margins(.15)
    axes[1].set(title=f"9 個完整年度的描述比較：r={result['pearson']:.2f}",
                xlabel="全年 12 個月指數的算術平均", ylabel="北門每場平均 18–24 時起站人次")
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle("Google 月指數可供描述；查詢條件與當時版本未核實，尚不能證明預測力")
    save(fig, output / "figures/google_descriptive.png")
    return annual, result


def build():
    output = ROOT / "result/analysis_v12"
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    style()
    s = pd.read_csv(output / "tables/summary.csv")
    pred = pd.read_csv(ROOT / ".local/analysis_v12/predictions.csv", parse_dates=["event_date"])
    data = pd.read_csv(ROOT / ".local/analysis_v12/analysis_input.csv", parse_dates=["event_date"])
    cfg = json.loads((ROOT / "configs/analysis_v12.json").read_text("utf-8"))
    primary = [v for v in cfg["variants"] if v not in cfg["sensitivity_only"]]
    x = s.loc[s.model.eq("xgboost") & s.control.eq("retuned")].set_index("variant").loc[list(cfg["variants"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), constrained_layout=True)
    for ax, metric, title in zip(axes, ["mae", "cost_2"], ["北門逐時 MAE", "散場低估 2:1 加權誤差"]):
        values = x.loc[primary, metric]
        bars = ax.barh(np.arange(len(values)), values, color=["#555555"] + ["#70958d"] * (len(values)-1))
        ax.set_yticks(np.arange(len(values)), [LABELS[v] for v in primary])
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.0f", padding=3)
        ax.set_xlim(0, values.max()*1.2)
        ax.set(xlabel="人次／小時；越低越好", title=title)
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("相同九場北門比較：XGBoost，各組只用過去年度重新選參數")
    save(fig, figures / "feature_comparison.png")

    weather = ["core", "temperature", "wind", "cloud", "weather"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True)
    for ax, model in zip(axes, ["ridge", "gam", "xgboost"]):
        for offset, control, label, color in [(-.18,"fixed_core","固定核心參數","#aaaaaa"),
                                               (.18,"retuned","過去年份重選參數","#52756e")]:
            d = s.loc[s.model.eq(model) & s.control.eq(control)].set_index("variant").loc[weather]
            bars = ax.bar(np.arange(5)+offset, d.mae, width=.35, color=color, label=label)
            ax.bar_label(bars, fmt="%.0f", fontsize=7, rotation=90, padding=2)
        ax.set_xticks(np.arange(5), [LABELS[v] for v in weather], rotation=28, ha="right")
        ax.set(title={"ridge":"Ridge","gam":"GAM","xgboost":"XGBoost"}[model], ylabel="北門逐時 MAE")
        ax.set_ylim(0, ax.get_ylim()[1]*1.18)
        ax.grid(axis="y", alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle("天氣拆開比較：灰色隔離新增輸入的影響，綠色再允許調參")
    save(fig, figures / "weather_ablation.png")

    p = pred.loc[pred.model.eq("xgboost") & pred.control.eq("retuned")]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for ax, date in zip(axes.flat, sorted(p.event_date.unique())):
        f = p.loc[p.event_date.eq(date)]
        truth = f.loc[f.variant.eq("core")].sort_values("hour")
        ax.plot(truth.hour, truth.y, "o-", color="#222222", label="實際", lw=2)
        for v, color in [("core","#999999"),("event_type","#52756e"),("weather","#a77154")]:
            t = f.loc[f.variant.eq(v)].sort_values("hour")
            ax.plot(t.hour, t.prediction, "--" if v=="core" else "-", color=color, label=LABELS[v], lw=1.5)
        ax.set(title=f"{pd.Timestamp(date):%Y-%m-%d}", xticks=list(range(18,24)), xlabel="來源小時", ylabel="北門起站人次")
        ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("九場逐時曲線：場次標記、天氣與核心輸入（XGBoost）")
    save(fig, figures / "beimen_profiles.png")

    fig, ax = plt.subplots(figsize=(10,4.4), constrained_layout=True)
    sens = ["core","sensitivity_baseline","sensitivity_google"]
    for offset, model, color in [(-.25,"ridge","#aaaaaa"),(0,"gam","#52756e"),(.25,"xgboost","#555555")]:
        d=s.loc[s.model.eq(model)&s.control.eq("retuned")].set_index("variant").loc[sens]
        bars=ax.bar(np.arange(3)+offset,d.mae,width=.24,color=color,label=model)
        ax.bar_label(bars,fmt="%.0f",padding=3)
    ax.set_xticks(np.arange(3),[LABELS[v] for v in sens])
    ax.set(ylabel="北門逐時 MAE",title="回顧性敏感度：一般日／Google 未通過當時發布證據，不能作上線結果")
    ax.set_ylim(0,ax.get_ylim()[1]*1.16)
    ax.legend(ncol=3)
    save(fig,figures/"unverified_sensitivity.png")
    annual, google = google_analysis(data,output)

    display = s.loc[s.control.eq("retuned")].pivot(index="variant",columns="model",values="mae").loc[list(cfg["variants"])]
    display=display.rename(index=LABELS,columns={"ridge":"Ridge MAE","gam":"GAM MAE","xgboost":"XGBoost MAE"}).reset_index().rename(columns={"variant":"加入的資訊"})
    case = pred.loc[pred.model.eq("xgboost") & pred.event_date.eq("2025-08-30") & pred.hour.eq(21)]
    case_table=case.pivot(index="variant",columns="control",values="prediction").loc[primary]
    case_table.columns=["固定核心參數","過去年度重選參數"]
    case_table["重選後誤差（預測−10,251）"]=case_table["過去年度重選參數"]-10251
    case_table=case_table.rename(index=LABELS).reset_index().rename(columns={"variant":"輸入"})
    case_table.to_csv(output/"tables/2025_finale_21h.csv",index=False,encoding="utf-8-sig")
    baseline=pd.read_csv(output/"tables/baseline_availability.csv")
    source_doc=json.loads((ROOT/"docs/d1_event_type_sources.json").read_text("utf-8"))
    typ=pd.DataFrame(source_doc["records"])
    year_table=annual[["year","months","index_mean_mid","index_mean_lower","index_mean_upper","events","mean_event_18_24_trips"]].copy()
    year_table.columns=["年度","月份數","月指數平均","低值界","高值界","已納入活動數","北門每場六小時平均人次"]
    value = lambda variant, control="retuned": float(s.loc[
        s.variant.eq(variant)&s.model.eq("xgboost")&s.control.eq(control), "mae"].iloc[0])
    text=f"""# 第十二版：場次定位、Google 熱度、一般日基準與天氣拆分

日期：2026-10-13。這一輪已實際重跑三個模型；不是只有列出建議。

**本輪判讀：**

1. 壓軸／特別場已補入，但 XGBoost MAE 由核心 {value('core'):.2f} 變成 {value('event_type'):.2f}，整體沒有改善。Ridge 有小幅改善，不能推論所有模型都無法利用場次資訊。
2. 三項天氣各自加入的 XGBoost MAE 為溫度 {value('temperature'):.2f}、風速 {value('wind'):.2f}、雲量 {value('cloud'):.2f}，一起為 {value('weather'):.2f}，均高於核心。2025/8/30 的尖峰低估在溫度與風速單項就出現，雲量單項沒有同樣大幅低估；仍須兼看其他八場。
3. 一般日基準固定參數時降至 {value('sensitivity_baseline','fixed_core'):.2f}，重選參數卻為 {value('sensitivity_baseline'):.2f}。有後續研究價值，但改善不穩定且歷史發布時間未核實。
4. 前月 Google 熱度敏感度為 {value('sensitivity_google'):.2f}，沒有改善核心；年度資料可供描述，不能用同月／全年事後熱度預測當次活動。

目前保留較簡單的核心 XGBoost 作對照，優先補一般日的可得性證據、逐日事前熱度及不同日期／類型的活動樣本，再以新的未見活動驗證。這一輪未變更部署。

此處核心 {value('core'):.2f} 是 V11 的 2:1 調參版；舊 V8 核心用 MAE 選參數的結果為 504.20。兩個數字對應不同調參規則，本輪以同規則控制做比較。

## 1. 一樣的資料與比較規則

- 21 場 × 四站 × 18–23 時，共 504 列。外層仍為 2024–2026 年九場北門、54 列，活動與年份不拆散。
- 核心輸入是車站、時段、車站×時段、週末、年內日序、年度趨勢、已公告結束後 0–60／60–120 分鐘×車站。
- 11 組輸入全部保留；前九組研究有事前來源的新增欄位，後兩組只作發布未核實的敏感度。沒有用外層最小 MAE 自動選變數或更換上線模型。
- 每組分「固定 V11 核心參數」與「重新選參數」。重選沿用 V11 的 Ridge 25、GAM 60、XGBoost 180 組；每年只看先前年份的北門時間留出預測，依散場低估 2 倍、高估 1 倍選擇。訓練損失仍是平方誤差與原懲罰。
- 這九場已反覆檢視，屬探索性回顧。歷史 OD 標籤發布時間與日曆版本仍有缺口，不能宣稱全流程已完成 D−1 上線驗證。

## 2. 重跑結果

單位：人次／小時，越小越好。下表是各組重新選參數的 MAE。

{table(display)}

![XGBoost 新增資訊比較](figures/feature_comparison.png)

![天氣單項比較](figures/weather_ablation.png)

所有組別的 RMSE、正向低估、尖峰絕對誤差、時點誤差、散場 2:1 誤差，見 [總表](tables/summary.csv)、[逐年](tables/annual.csv)、[逐場](tables/events.csv)。[每年度選定參數](selections.json)保留全部內層候選分數。

## 3. 壓軸和特別場：有公告才編碼

- 新增 `planned_finale`：官方完整安排中的系列最終／壓軸煙火場。不是把資料中每年的最後一筆標 1，也不是把歌手壓軸當活動壓軸。
- 另加 `planned_enhanced_nonfinale`：非壓軸，但公告另加碼／升級的煙火場。2026/7/25 是開幕特別展演，與 8/15 壓軸分開；2023/8/16、2025/8/20 是公告併入取消場次的加碼場。
- 21 場中，壓軸旗標有 {typ.planned_finale.notna().sum()} 場可編碼（其中 {int(typ.planned_finale.sum())} 場為 1）；非壓軸加碼旗標有 {typ.planned_enhanced_nonfinale.notna().sum()} 場可編碼（其中 {int(typ.planned_enhanced_nonfinale.sum())} 場為 1）。舊年度未核實者保留未知，不能當普通場。
- 2023/8/9 的加碼公告目前版本於前一天 18:04 更新，晚於 18:00 截止，因此加碼旗標保持未知；壓軸定位可用更早完整排程判定。
- [來源帳本](../../docs/d1_event_type_sources.json)保存逐場網址、保守可得時間、理由與來源文字雜湊；[編碼表](tables/event_types.csv)可查每場。當前網頁發布／更新欄不是當年網頁封存；不能聲稱已驗證歷史版本完全一致。
- 類型僅有少數正例，且與週末、時長、年份相依。它提供活動定位，不是新增大量獨立樣本，也不能從係數推論壓軸的因果效果。
- 檢查 2024／2025／2026 外層訓練前的已核實場次，分別為 7／10／13 場，壓軸旗標都與週末完全相同。2026/7/25 才在外層出現「週末但非壓軸」的反例。因此過去資料幾乎無法分辨週末與壓軸的獨立作用；缺值補值後的矩陣即使滿秩，也不代表這個實質混淆已解決。

## 4. 2025/8/30 21 時：實際 10,251 人次

{table(case_table)}

這張表同時保留固定參數結果，避免把「加入風速」的影響和「改選另一套參數」混在一起。各單項變化不具可加性，三項天氣一起加入的結果不等於三個單項誤差加總。小樣本、特徵相關與 0.25° 預報網格都可能影響泛化；這不是天氣對人流的因果估計。

![北門九場逐時預測](figures/beimen_profiles.png)

## 5. 你已有的 Google 資料可以怎麼分析

- 已讀取原 CSV：CP950、117 個月份，2017/1–2026/9，欄位是月份及 0–100 相對熱度；不是絕對搜尋次數。
- 檔案本身缺關鍵字、地區、查詢類別、查詢區間、下載時間與歷史版本，不能只因放在同一個 CSV 就視為已驗證同尺度。
- `<1` 保留區間 [0,1)，描述圖暫以 0.5 表示。下表「低／高值界」是這個截尾記號的取值界，不是信賴區間。零指數也不等於完全沒有人搜尋。
- 將每年 12 個月指數作算術平均，再對照該年已納入活動的北門六小時人次平均；不是把一年熱度複製成每一列獨立樣本。2026 只有九個月，另列且排除全年相關比較。
- 2017–2025 共 {google['n_years']} 個完整年度，描述性 Pearson r={google['pearson']:.3f}、Spearman ρ={google['spearman']:.3f}。沒有把小樣本相關當預測驗證或因果證據；年度活動構成也不同，且查詢尺度尚待核實。

{table(year_table)}

![Google 月序列與年度比較](figures/google_descriptive.png)

預測敏感度只使用「前一個完整曆月」指數：例如 2025 年八月三場都使用七月的 15，不能用八月的 100。這避免使用活動後月份內容，**但不能修復未知的查詢尺度／歷史版本**，所以仍不進主要事前輸入。需要逐場熱度時，應另保存截止前的逐日／七日查詢及尺度校準；目前的月份資料無法區分同月三場熱門程度。

本次 21 場的前月值均不是 `<1`，因此 0.5 的繪圖約定不影響前月熱度模型；它只影響年度描述平均。全年平均也不是 Google 官方另外提供的「年度指數」，而是本次對原月資料的描述性彙整。

## 6. 一般日背景已重建，但發布時間仍待補

- 由原始已彙整的人流 Parquet 及候選日帳本重新計算，與模型快照中的 504 個逐時基準全部一致。
- 同站、同時段，往前 56 天，匹配星期及既有日曆分類、排除活動日，取最近最多四天且至少兩天。用各來源小時的人次平均，不用活動當天真值，也不把六小時總量平均拆成每小時。
- 21 場均有數值；選中的站日配對共 280 組，其中 {int(baseline.same_month_station_day_pairs.sum())} 組屬活動同月。來源是按月公布的公開 OD；觀測已發生不代表該月 CSV 當時已發布。目前沒有逐檔歷史發布證據。
- 因此已做額外敏感度試驗，但仍標記 `retrospective_sensitivity_only`；不能靠改欄位名讓它通過 D−1。正式使用需保存發布版本，或取得捷運內部截止前可見的歷史 AFC。

![未驗證輸入敏感度](figures/unverified_sensitivity.png)

## 7. 重現、檢核與來源

```powershell
.\.venv\Scripts\python.exe -X utf8 src/run_analysis_v12.py
.\.venv\Scripts\python.exe -X utf8 src/verify_analysis_v12.py
.\.venv\Scripts\python.exe -X utf8 src/report_analysis_v12.py
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_analysis_v12.py -v
```

如需候選快取，使用 `run_analysis_v12.py --reuse`，來源／程式／設定雜湊必須一致。私人逐筆數據與預測在 `.local/analysis_v12/`；公開圖與總表在根目錄 `result/analysis_v12/`。[獨立數值檢核](independent_validation.json)用不同公式重算選參數與各表格，不等同補齊資料發布證據。

- [Google Trends 官方資料說明](https://support.google.com/trends/answer/4365533?hl=zh-Hant)：指數標準化與樣本限制；不能解讀為搜尋人數。
- [Hyndman & Athanasopoulos，FPP3 §5.10](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation)：時間向前驗證；本案年度切分與 2:1 偏好為研究設計。
- [Santanam 等，Public Transit for Special Events](https://arxiv.org/html/2106.05359)：特殊活動需求與列車模擬的研究路線；不保證本案任何新欄位必然提升精度。
- [官方 OD 資料說明](https://www.metro.taipei/cp.aspx?n=BDEB860F2BE3E249)與[政府開放資料目錄](https://data.nat.gov.tw/dataset/62261)：逐時 OD 不等於分鐘抵達月台。
- [預報時效證據與變數定義](../../docs/d1_weather_audit.md)：這次仍只用截止前 GFS 預報；不混入 CODiS 當天實測。
"""
    (output/"report.md").write_text(text, "utf-8")
    print(output/"report.md")
    print(json.dumps(google,ensure_ascii=False))


if __name__=="__main__":
    build()
