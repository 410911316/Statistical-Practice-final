# 大稻埕煙火人流分析與捷運班次模擬

**[公開網站：稻埕觀測室](https://shu-ming0808.github.io/Statistical-Practice-final/)** · [詳細預測資訊](https://shu-ming0808.github.io/Statistical-Practice-final/#/research)

目前人流預測先告一段落，**保留四站訓練、北門散場加權 XGBoost**，接續做候車情境與列車調度。Ridge、GAM 保留作比較及解釋，未加權 XGBoost 作加權效果對照；融合未採用。逐版實驗入口與重複圖表已移除，歷程仍可由 Git 查閱。

## 資料與分析流程

- **單位：**活動 × 車站 × 來源時段；21 場、北門／大橋頭／雙連／民權西路、18–23 時，共 504 列。504 列不是 504 個獨立活動。
- **目標：**北門各來源時段進站人次，主要評分為 2024–2026 同九場、54 列。
- **保留輸入：**車站、時段、車站×時段、週末、年內日期、年度趨勢、過去 56 日同星期一般日基準、事前預定展演結束後兩小時的車站重疊訊號。
- **未放入最終模型：**當日先前人流、實測天氣、事後網路聲量、年度經費、未核實的熱度與活動規模。
- **時間界線：**依活動前一天規劃輸入，但歷史 OD／日曆的實際發布時點尚未完全核實；目前是回顧比較，不能稱已驗證嚴格 D−1 可部署。九場已反覆用於開發，仍需新活動驗證。

~~~mermaid
flowchart TD
    A["活動 × 車站 × 時段<br/>來源、缺值與事前可得性檢查"] --> B["只用過去年份訓練<br/>訓練內補值、標準化與編碼"]
    B --> C["Ridge：線性與收縮"]
    B --> D["GAM：平滑非線性"]
    B --> E["XGBoost：門檻與交互作用"]
    C --> F["同九場北門比較 MAE、RMSE、尖峰"]
    D --> F
    E --> F
    F --> G["保留四站散場加權 XGBoost"]
    G --> H["分鐘到達情境 → 分方向候車 → 班次方案"]
~~~

## 如何調參，最後保留什麼？

1. **依年份向前驗證，不隨機打散。**預測 2024／2025／2026 時，分別只用過去 12／15／18 場訓練；更早年份的留出預測用來選參數。
2. **搜尋各模型的候選設定。**Ridge 比較 25 個 λ；GAM 比較 60 組平滑／收縮／基底；XGBoost 比較 180 組預先定義的組合，另試學習率×樹數與 early stopping。不是所有列出的數值做完整笛卡兒積。
3. **基礎設定曾用散場低估罰 2 倍選擇。**之後加入一般日基準時保留原設定；又用過去北門六時段 MAE 比較局部樹設定、早停及訓練加權。
4. **最後保留原樹設定＋散場訓練加權。**early stopping 不是目前模型的一部分。4 倍權重是提高特定訓練列的重要性，和「低估罰 2 倍」不同。保留 4 倍的決定參考了這九場比較，不能包裝成完全未見資料上的選模。
5. **目前固定模型，不再進行新搜尋。**下方年度設定與完整候選紀錄已保存，[重跑入口](src/run_forecast.py)只重現凍結結果。

| 模型 | 調整內容 | 在本專案的直觀角色 |
|---|---|---|
| Ridge | λ 共 25 值，約 10⁻⁴–10²；程式 alpha=nλ | 線性關係與車站×時段效果，收縮不穩定係數 |
| GAM | 平滑 λ：0.1／1／10／100／1000；係數懲罰：0.001／0.01／0.1／1；基底：4／5／7 | 允許有資料支持的連續變數彎曲，同時限制失穩 |
| XGBoost | 樹深、樹數、學習率、葉節點門檻、L1/L2、列／欄抽樣；另比較早停 | 學條件門檻與組合；是否較準以驗證結果決定 |

### 最後保留的年度設定

| 評估年度 | Ridge λ | GAM：平滑λ／係數懲罰／基底數 | XGBoost：樹深／樹數／學習率 |
|---|---:|---|---|
| 2024 | 0.000177828 | 0.1／0.001／5 | 3／250／0.10 |
| 2025 | 0.177827941 | 0.1／0.001／5 | 4／250／0.05 |
| 2026 | 0.177827941 | 0.1／0.001／7 | 4／250／0.05 |

XGBoost 其餘為 min_child_weight=1、reg_lambda=1、reg_alpha=0、subsample=1、colsample_bytree=1、平方誤差目標、hist。北門事前散場重疊比例為 o 時，原始權重 **1+3o**，其他列為 1，再令每場平均權重為 1。未知結束時間不加權。

[完整候選格點與調參紀錄](result/forecast/tuning_record.json)／[凍結設定](configs/forecast.json)／[擬合與 GAM 支持檢查](result/forecast/fits.json)

## 最終結果與解讀

同九場北門 54 個來源時段，單位為人次／來源時段，越低越好。

| 模型 | MAE | RMSE | 實際尖峰時段平均絕對誤差 |
|---|---:|---:|---:|
| Ridge | 845.44 | 1,547.49 | 2,681.26 |
| GAM | 830.71 | 1,315.75 | 2,305.11 |
| XGBoost 未加權 | 473.91 | 766.90 | 749.02 |
| **XGBoost 散場加權（保留）** | **440.28** | **690.91** | **747.14** |

加權相對未加權的 MAE 改善約 **7.1%**，尖峰誤差只略降。2024/8/10 最高峰實際 11,101、預測約 8,359，仍少估 **2,742 人次（24.7%）**。不能以整體平均誤差替代尖峰風險評估。

### 各場平均與誤差直方圖

![每場平均實際與模型預測](result/forecast/figures/event_means.png)

每組是同場六時段平均；這是分組長條圖，平均接近不代表每個時段都準。

![最終模型誤差直方圖與實際對照](result/forecast/figures/prediction_diagnostics.png)

直方圖負值＝低估，正值＝高估，每格 500 人次；右圖越接近虛線越準。[九場逐時圖](result/forecast/figures/hourly_profiles.png)可看尖峰時點與低估。

### 輸入變數相關熱力圖

![輸入相關熱力圖](result/forecast/figures/feature_correlations.png)

只用 **2026 前北門的 18 場訓練資料**、七項代表性輸入，以兩兩非缺值 Spearman 計算。這是變數相關，不是樹深／學習率等超參數的相關；同場六時段不獨立，不能直接作因果或顯著性解讀，也不據此自動刪變數。

## Ridge 最終公式與其他模型細節

Ridge 使用訓練期標準化後的向量 z（含車站、時段、交互作用及缺值旗標）：

$$
(\hat\beta_0,\hat\beta)=\arg\min_{\beta_0,\beta}
\left[\frac1n\sum_{i=1}^{n}(y_i-\beta_0-z_i^\top\beta)^2+\lambda\|\beta\|_2^2\right],
\qquad \hat y=\max(0,\hat\beta_0+z^\top\hat\beta).
$$

截距不懲罰；2026 評估折 n=432、λ=0.177827941，故 sklearn alpha=76.8216705。這一折截距為 **1589.8432**；部分係數如下，其餘不可省略，完整可計算公式見連結。

| 項目 | 係數 | 解讀尺度 |
|---|---:|---|
| 週末 | +278.50 | 0→1，其他設計欄位固定 |
| 一般日基準 | +532.87 | 增加一個訓練標準差（約 709.71 人次） |
| 北門散場第一小時重疊 | +452.73 | 標準化重疊欄增加 1 |
| 北門散場第二小時重疊 | +623.29 | 標準化重疊欄增加 1 |

這些是受懲罰的條件預測係數，不是活動造成的人數增量。參考車站為北門、時段為 18。[完整係數表](result/forecast/tables/ridge_coefficients_2026.csv)／[公式、補值及標準化常數](result/forecast/ridge_equation_2026.json)。

- **GAM：**Gaussian identity，加法結構；連續欄至少六場已知資料，指定可平滑欄另需至少八個不同值。平滑使用 cubic P-spline，搭配二階係數差分與全係數 L2；其餘為受懲罰線性項。一般日基準在此設定是線性項，不能宣稱所有連續變數都用曲線。[中心化曲線資料](result/forecast/tables/gam_components_2026.csv)。
- **XGBoost：**逐棵樹修正前面模型的誤差，可呈現門檻與交互作用；本案採樣本權重的平方誤差訓練。輸出低於 0 時截為 0。[特徵分裂增益](result/forecast/tables/tree_importance_2026.csv)不代表因果影響。
- **融合：**等權及過去留出預測學權重都曾比較，未採用。融合不是一定比單模型準；最終入口不再跑已淘汰的融合／選站／lag 實驗。

[完整結果](result/forecast/report.md)／[年度指標](result/forecast/tables/annual_metrics.csv)／[實作與完整公式](process.md)／[重現檢核](result/forecast/validation.json)

## 候車與網站

北門為高運量六節編組。保留 1,134 組「均勻／寬峰／集中峰 × 剩餘容量等」候車情境與四站官方班表。最大容量 2,220 人不等於本站可上車人數；進站前背景載量仍是假設，離站才加入上車人數。同方向有滯留時必已用滿可用容量。

首頁提供列車圖示及綠→黃→紅載客率；北門雙向候車按可調門檻變色，另外三站灰色待建模。現行一般班表搭配歷史需求是情境比較，不是當年實測候車或已完成全線班表最佳化。

[候車情境結果](result/queue_dispatch_v1/report.md)／[分鐘到達圖](result/queue_dispatch_v1/figures/arrival_queue_comparison.png)／[營運資料與缺口](docs/metro_operations.md)／[網站與發布](app/README.md)

## 快速開始

~~~powershell
uv sync --inexact --group analysis
uv run --group analysis python -X utf8 src/run_forecast.py
uv run --group analysis python -X utf8 -m unittest discover -s tests -p "test_forecast*.py"
~~~

隊員需私下取得 .local/forecast/analysis_input.csv 與 reference_predictions.csv；以雜湊確認凍結資料。**不必先跑十九個版本，也不用連 MySQL。**逐列資料與模型保持私有；只看 README／PNG 不用安裝環境。命令重現既定模型，不會以新資料自動調參或覆蓋網站快照。

<details>
<summary>原始資料前處理、資料庫與文字雲</summary>

~~~powershell
.\.venv\Scripts\python.exe -X utf8 src/main.py --reuse-clean --mysql
.\.venv\Scripts\python.exe -X utf8 src/import_weather.py
.\.venv\Scripts\python.exe -X utf8 src/threads_web_collect.py
.\.venv\Scripts\python.exe -X utf8 src/ptt_web_collect.py
~~~

這些流程保留；公開網站使用快照，不需 Tailscale。SQL、原始資料及帳密不推送。Threads／PTT 文字雲是公開搜尋樣本，不代表全站聲量。
</details>

## 專案結構

~~~text
統計實務/
├── README.md                  # 最終方法、結果與參考資料
├── process.md                 # 實作、公式、候車及發布細節
├── configs/forecast.json      # 凍結資料資格與年度參數
├── src/
│   ├── run_forecast.py        # 單一人流重現入口
│   ├── forecasting/           # 編碼、Ridge、GAM、XGBoost、指標及圖表
│   ├── queueing.py            # 列車批次服務
│   ├── run_queue_scenarios.py # 分鐘到達與候車情境
│   └── main.py                # 原始捷運資料前處理
├── result/
│   ├── forecast/              # 最終報告、參數、係數、表格及 PNG
│   └── queue_dispatch_v1/     # 候車與官方營運資料
├── app/                       # 前端、FastAPI、公開快照
├── data/                      # 本機原始／整理資料，不公開
├── .local/forecast/           # 私有輸入、逐列預測與模型
├── tests/                     # 模型、資料、候車與網站測試
├── docs/                      # 來源、規格與研究資料
└── results/                   # 文字雲及其他研究成果
~~~

## 詳細說明與參考資料

- [資料前處理](docs/data_preprocessing.md)／[資料庫結構](docs/database_schema.dbml)
- [Threads](docs/threads_collection.md)／[PTT](docs/ptt_collection.md)／[Google Trends](mysql/GOOGLE_TRENDS_README.md)
- [事前天氣證據](docs/d1_weather_audit.md)／[研究文獻](docs/references.md)

| 文獻 | 本專案用途與界線 |
|---|---|
| [Hoerl & Kennard（1970）](https://doi.org/10.1080/00401706.1970.10488634)／[Ridge 實作](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) | L2 收縮；本案採平均平方損失，所以 alpha=nλ |
| [Hastie & Tibshirani（1986）](https://doi.org/10.1214/ss/1177013604) | 加法平滑；不保證比線性或樹模型準 |
| [Eilers & Marx（1996）](https://doi.org/10.1214/ss/1038425655) | P-spline 與係數差分懲罰；基底、支持門檻為本案設定 |
| [Chen & Guestrin（2016）](https://arxiv.org/abs/1603.02754) | 正則化加法樹；格點、散場權重與選模規則為自行設定 |
| [Breiman（1996）](https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf) | 用留出預測學融合；未主張融合必勝 |
| [Cawley & Talbot（2010）](https://jmlr.org/papers/v11/cawley10a.html) | 選模與評估偏差；保留九場反覆使用的限制 |
| [Hyndman & Athanasopoulos，§5.10](https://otexts.com/fpp3/tscv.html) | 用過去預測未來的向前驗證 |
| [Santanam 等（2021）](https://arxiv.org/abs/2106.05359) | 需求預測銜接列車調度；未聲稱已重現最佳化 |

資料來源：

- [北捷分時進出站運量](https://data.taipei/dataset/detail?id=63f31c7e-7fc3-418b-bd82-b95158755b4d)
- [固定／變動成本](https://www.metro.taipei/cp.aspx?n=20CB8DE3F41B4111)：參考 114 年審定決算書。
- [變動／固定公里數](https://whhr.gov.taipei/News_Content.aspx?n=0121E4C78246FC0D&s=3E147DD6A74BE990)
- [天氣實測](https://codis.cwa.gov.tw/StationData)：氣象署 CODiS，只作事後探索。
- [GFS 歷史操作型預報](https://gdex.ucar.edu/datasets/d084001/)／[NOAA AWS 公開時間證據](https://registry.opendata.aws/noaa-gfs-bdp-pds/)：曾評估的事前天氣來源；最終模型未採用天氣欄位。
- [特殊節假日](https://data.gov.tw/dataset/14718)：政府行政機關辦公日曆表。
- [大稻埕煙火排程參考](https://data.gov.tw/dataset/7778)：觀光資訊活動資料庫。

候車與列車調度新增官方來源（2026-10-15 核對）：

- 四站時刻表：[北門](https://web.metro.taipei/pages2026/WebStation/105/7)／[大橋頭](https://web.metro.taipei/pages2026/WebStation/128/7)／[雙連](https://web.metro.taipei/pages2026/WebStation/054/7)／[民權西路](https://web.metro.taipei/pages2026/WebStation/055/7)。八份自 2026/8/30 生效，橘線往南勢角兩份自 10/1 生效；都是一般離站表。每份 PDF 的直連與 SHA-256 見 [來源清單](result/queue_dispatch_v1/metro_source_snapshot.csv)。
- [相鄰兩站行駛與停靠時間](https://data.taipei/dataset/detail?id=513e97fe-6a98-4a0d-b7dc-11122c8638d4)／[官方 CSV](https://data.taipei/api/frontstage/tpeod/dataset/resource.download?rid=c32998fa-c9b2-4324-9016-e19fbe0815f1)：推算同線候選發車站→四站的行車部分，不含未知整備／出庫／插車時間。
- [高運量六節編組最大載客量](https://www.dorts.gov.taipei/News_Content.aspx?n=2A66A485FACB0D5B&s=F32662CFF6FE4D80)：370 人／節、2,220 人／列；不是剩餘容量。
- [北捷 API 申請與資料目錄](https://www.metro.taipei/cp.aspx?n=BDEB860F2BE3E249)／[API 規格第 1 頁項次 6](https://www-ws.gov.taipei/001/Upload/405/relfile/18288/116939/26d47117-1927-48fb-9293-3f7879ec8092.pdf)：車次、方向及四級擁擠度，30 秒更新；尚未申請接入，不能直接當精確人數。
- [北捷載重與票證人流模型](https://www.metro.taipei/News_Content.aspx?n=30CCEFD2A45592BF&s=3C06D5F73D9377D0&sms=72544237BBE4C5F6)：車上背景載客估計的官方實務依據。
- [新莊機廠與迴龍進出軌](https://www.dorts.gov.taipei/cp.aspx?n=A7D84926581F5817&s=2091F0361804122D)／[新店機廠與小碧潭支線限制](https://www.dorts.gov.taipei/News_Content.aspx?n=2A66A485FACB0D5B&s=89085E307909460C&sms=87415A8B9CE81B16)：機廠位置與路徑線索，沒有公布活動當日可用備車與就緒時間。
- [114 年審定決算書 PDF 第 51 頁](https://www-ws.gov.taipei/001/Upload/405/relfile/18288/7602/4627504c-4034-4a4e-b403-c692be626487.pdf#page=51)：合併勞務成本明細表的年度固定／變動運輸費用；不是單班增量成本，參見 [成本口徑](docs/metro_operations.md#4-營運成本有找到仍不能直接當每加一班的成本)。
