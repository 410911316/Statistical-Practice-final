# GitHub 方法參考：先做好活動比較，再接預測與模擬

查閱日期：**2026-10-05**。本報告實際查閱 GitHub 原始倉庫、官方文件、GitHub API 與 PyPI 發布資料；沒有下載執行第三方倉庫程式，也沒有安裝或更換專案套件。以下「官方功能」是來源事實，「本專案建議」是依現有欄位與研究目的作的判斷，尚非本專案的實驗結果。

## 最推薦先採用的三項

1. **statsmodels：北門每場一列的簡單回歸與模型檢查。** 先比較歷史背景量、少量日曆／活動因子，不直接把許多小時當獨立場次。F 檢定須符合模型與誤差假設；計數回歸另比較 Poisson 與負二項。[官方 GLM 範例](https://www.statsmodels.org/stable/examples/notebooks/generated/glm.html)、[F 檢定限制](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.compare_f_test.html)。
2. **scikit-learn：把驗證切分做好。** 以整場活動為單位，主要回測採「以前的活動訓練、後面的活動測試」。GroupKFold 可防同場洩漏，但本身不保證時間順序；不能拿它直接代表未來預測能力。[分組切分](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)、[時間切分](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)。
3. **jieba ＋ word_cloud：沿用現有文字雲，增加可核對的主題表。** 專案已有每篇只算一次的詞頻；下一步比換模型更有用的是抽查斷詞、固定交通／天氣／觀賞體驗等類別字典，列出不同貼文占比，再人工核對。文字雲大小不等於效果大小或重要因果因子。[jieba 自訂字典及關鍵詞](https://github.com/fxsjy/jieba)、[word_cloud 中文範例](https://github.com/amueller/word_cloud/blob/main/examples/wordcloud_cn.py)。

**現有資料的界線：** 每月 OD 資料很多，不代表活動的獨立樣本很多。`origin_count` 是來源時段按起站加總；`destination_count_same_source_bin` 不能直接當該小時出閘量。步行路線、分鐘到達、閘門服務率、候車時間與列車剩餘容量均不能由這兩欄還原。Threads／PTT 是公開搜尋樣本；Trends 未校準尺度不能直接跨場當搜尋總量。氣象三表已匯入，但尚未自動併入回歸表。依據：[本專案資料規格](../data_preprocessing.md)、[研究規劃](../research_plan.md)、[README](../../README.md)。

## 八個參考項目

| 項目 | 可借用的方法 | 現在是否適合 | 預定接入位置 |
| --- | --- | --- | --- |
| [statsmodels/statsmodels](https://github.com/statsmodels/statsmodels) | OLS、F／Wald 檢定、Poisson／負二項 | 優先；先小模型與診斷 | 北門活動總量／逐時分析 |
| [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | 分組驗證、Pipeline、NMF 主題探索 | 優先用驗證方法 | 模型共用回測層 |
| [Nixtla/statsforecast](https://github.com/Nixtla/statsforecast) | SeasonalNaive、滾動回測 | 補足連續背景資料後 | 一般日流量基準 |
| [SimPy](https://simpy.readthedocs.io/en/stable/) | 有限資源與離散事件模擬 | 可做假設情境，未能校準實際排隊 | 未來模擬 API |
| [gboeing/osmnx](https://github.com/gboeing/osmnx) | 步行路網、最短／候選路徑 | 可改善地圖幾何；仍是候選路線 | 地圖離線 GeoJSON |
| [AequilibraE/aequilibrae](https://github.com/AequilibraE/aequilibrae) | OD 路徑分配、大眾運輸分配 | 缺路網與班距等，列後續 | 站間 OD → 軌道區間估計 |
| [fxsjy/jieba](https://github.com/fxsjy/jieba) | 自訂詞典、TF-IDF、TextRank | 現有程式已採用斷詞 | 貼文清理與主題詞 |
| [amueller/word_cloud](https://github.com/amueller/word_cloud) | 頻率字典、遮罩、中文字型 | 現有程式已採用 | 文字雲與詞頻下載 |

SimPy 是刻意保留的例外：GitHub 搜尋會找到 [simpx/simpy](https://github.com/simpx/simpy)，但頁面明示它是 fork，正式上游在 [team-simpy/simpy（GitLab）](https://gitlab.com/team-simpy/simpy)。本報告不把該舊 fork 當作當前官方程式。

### 1. statsmodels：可解釋的小模型與統計檢定

**官方功能／可借用：** [GLM notebook](https://www.statsmodels.org/stable/examples/notebooks/generated/glm.html) 提供模型建立與結果檢查；[離散型 NegativeBinomial](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.NegativeBinomial.html) 提供 NB1／NB2 及最大概似估計。注意 [GLM 的 NegativeBinomial family](https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.NegativeBinomial.html) 把 `alpha` 視為固定參數，不可未檢查就宣稱它自動估得過度分散。

**本專案建議：** 使用 `regression_event_station`，固定北門與 18–24 時，篩選完整視窗。先以 `origin_count_window` 為目標、`baseline_origin_window` 為比較基準；每次只加少數有事前可用性的變數。若對 `origin_excess_vs_baseline` 建模，它可能為負，不能套 Poisson／負二項。多站逐時模型要處理同一活動內的相關性，不讓重複列虛增自由度。

**F test 怎麼放：** 同一批列、同一個目標、巢狀 OLS 模型才作殘差平方和 F 比較；例如背景量模型對「再加一個活動因子」模型。精確小樣本 F 推論另以正確的線性均值模型、給定解釋變數後均值為零且**常態、獨立、同方差**的誤差為工作假設，並需設計矩陣滿秩、剩餘自由度為正。官方提醒未處理異質變異或相關性時比較 F 無效。[NIST 線性迴歸說明](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/line_fit.htm)、[compare_f_test 文件](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.compare_f_test.html)。

**本專案的相依敏感度：** 即使每場只留一列，同年度活動仍可能共受疫情、行銷或營運因素影響；不同活動也可能共用背景日，使含背景估計的比較不能逕自視為獨立。建議核對背景日重疊，作整年度留出或限制基準不重疊的敏感度比較，呈現係數與結論是否改變；這些檢查本身不保證滿足精確 F 假設。穩健 Wald 也有近似與小樣本限制，不能修正所有研究設計問題；沒有隨機化／可識別設計就不解釋成因果。此段是依資料規格提出的檢查建議，尚未執行。

**輸出建議：** 每模型可用場數、缺值剔除原因、係數與區間、殘差圖、假設診斷、逐場留出誤差；不要只交 p 值。授權：[BSD-3-Clause](https://github.com/statsmodels/statsmodels/blob/main/LICENSE.txt)。

### 2. scikit-learn：防資料洩漏的回測與可選文本探索

**官方功能／可借用：** [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) 讓同群組不跨訓練／測試；[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) 依時間向前切分，但文件要求等間隔樣本才有可比的測試時長；[Pipeline 防洩漏說明](https://scikit-learn.org/stable/common_pitfalls.html) 規定轉換只在訓練資料估計。

**本專案建議：** 活動日期不規則，且同場有多站／多小時，主回測應先排活動日期，再自行建立活動或年度邊界的 expanding-window folds，將整場所有列一起留出；不是直接對長表逐列套 TimeSeriesSplit。GroupKFold 可作「未見活動」的次要檢查，但訓練可能包含更晚日期。基準與所有候選模型共用完全相同 folds，回報逐場 MAE 與總體 WAPE，保留未能回測的早期場次。缺值、字典／特徵選擇及調參都只用訓練期。

**文本延伸：** 官方 [NMF／LDA 範例](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html) 可參考「TF-IDF→NMF」流程；中文需先斷詞，改掉英文停用詞。現有搜尋樣本可能偏少且偏向最相關，建議人工類別優先，NMF 僅探索並逐篇核對，不直接稱人口意見分布。授權：[BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)。

### 3. StatsForecast：一般日背景流量的時間序列基準

**官方功能／可借用：** [SeasonalNaive](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#seasonalnaive) 重複上一季節對應值；[cross_validation 範例](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/crossvalidation.html) 展示滾動截止點、預測視窗及誤差比較。輸入長表使用 `unique_id`、`ds`、`y`。

**本專案建議：** 不把相隔數週／跨年的活動總量當連續日資料。若先做每日同一時段流量，可用 `unique_id=station+source_hour`、`ds=service_date`、`y=origin_count`，考慮七日季節基準；只有補足連續日序列並處理真實缺值後才適用。不要把 02–04 時缺資料補零，再宣稱完整 24 小時週期。現有前 7／14／21／28 天平均已是實用基準，先比較它與上一週同時段，再決定是否需新套件。

**適用範圍：** 一般日基準／時間回測，不會自動學會煙火突增；取消、假日、疫情及其他活動仍需定義。輸出每截止點預測與實測、資料覆蓋和 MAE；不要把官方其他資料集的速度／準確率數字套到本專案。授權：[Apache-2.0](https://github.com/Nixtla/statsforecast/blob/main/LICENSE)。

### 4. SimPy：候車與閘門的情境模擬

**官方功能／可借用：** [Bank Renege](https://simpy.readthedocs.io/en/stable/examples/bank_renege.html) 示範到達、資源等待與離隊；[Resource API](https://simpy.readthedocs.io/en/stable/api_reference/simpy.resources.html) 支援有限同時服務數；[Monitoring](https://simpy.readthedocs.io/en/stable/topical_guides/monitoring.html) 示範保存隊列與資源使用變化。

**本專案建議：** 閘門可參考 `Resource(capacity=c)`，但須額外取得閘門數、個人服務時間與到達資料。列車依班表到站、一次帶走多名旅客，建議另建分方向的「列車批次載運事件」，不要把列車數直接當 M/M/c 的服務台數。需要方向比例、班距、到站剩餘容量、上下車量與站內移動延遲；目前小時 OD 不能辨識這些值。

**現階段可交付：** 固定每小時總量，在多種明確標示的到達形狀、班距、容量假設下，做多次隨機重複，輸出平均／高分位等待、最大隊列與末班殘留人數。分鐘序列是模擬；沒有實測校準就不稱真實候車時間或已驗證最適班次。這是針對資料限制提出的建模建議，不是套件自帶捷運模型。上游與授權：[官方概覽／MIT](https://simpy.readthedocs.io/en/stable/)。

### 5. OSMnx：將手繪線改善為可追溯的步行候選路線

**官方功能／可借用：** [路網 API](https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.graph.graph_from_point) 可下載 walk 網路；[官方 routing notebook](https://github.com/gboeing/osmnx-examples/blob/main/notebooks/02-routing-speed-time.ipynb) 及 [routing 原始碼](https://github.com/gboeing/osmnx/blob/main/osmnx/routing.py) 提供最短與 k 條候選路徑。

**本專案建議：** 活動區出口到北門等站的實際入口座標，先離線產生路線 GeoJSON、距離和來源日期，再載入 MapLibre。必須人工檢查水門、河堤出入口、無法步行路段和活動交通管制；道路車速推估不能當人群步速。若以站點 OD 量加粗候選線，圖例仍須寫「目的車站流量」，不叫該路段實際人數。

**不適用：** 不會觀測個人軌跡，也不會從捷運 OD 判定活動場地出發占比。當前路網不自動代表 2017 年路網。套件為 [MIT](https://github.com/gboeing/osmnx/blob/main/LICENSE.txt)；OSM 底層資料另遵循 [OpenStreetMap 授權與署名](https://www.openstreetmap.org/copyright)。**OSMnx 2.1 起不支援 Python 3.10**；若採最新版本，應另建離線路網環境，輸出檔案再接主站，不直接升級現有專案。[官方 Changelog](https://github.com/gboeing/osmnx/blob/main/CHANGELOG.md)。

### 6. AequilibraE：OD 到路網的分配，後續才需要

**官方功能／可借用：** [Public transport assignment with Optimal Strategies](https://www.aequilibrae.com/latest/python/_auto_examples/public_transport/plot_public_transit_assignment.html) 展示頻率型運輸網路分配；[Traffic Assignment without an AequilibraE Model](https://www.aequilibrae.com/latest/python/_auto_examples/traffic_assignment/plot_assignment_without_model.html) 展示建圖、OD 與分配流程。

**本專案建議：** 等拿到正式站碼、線路連接、方向、班距、旅行／轉乘時間與需求矩陣後，才評估把 `flow_edges_hourly` 的站間需求分配到軌道路段。需先建立原始站名到實體車站／路網節點的映射，檢查轉乘與同站 OD；結果是模型估計，應用區間載客觀測作外部檢查。

**目前檔案可用性：** 2026-10-05 實際檢查時，`data/processed/flow_edges_hourly.parquet` **不存在**；`src/aggregate_flows.py` 仍有建表及輸出邏輯，因此這是可重建的產物，不是本次已可直接讀取的成品。確定需要路徑分配時，再由既有彙整流程重建，或對既有乾淨 OD 作等價唯讀查詢；本輪沒有重建或修改資料。

**不適用：** 道路 BPR 容量函數不能直接搬成月台擁擠公式；靜態／頻率式分配也不能代替散場後逐班候車模擬。現有四條步行線展示暫時不需要整套引入。授權是 **MIT 加附加條款**，要求使用該軟體的工作文件作引用，不應誤標成純 MIT；以 [LICENSE.TXT 原文](https://github.com/AequilibraE/aequilibrae/blob/develop/LICENSE.TXT) 為準。此處僅記錄來源內容，不作法律適用判定。

### 7. jieba：中文詞典、主題候選詞與可重現清理

**官方功能／可借用：** [README](https://github.com/fxsjy/jieba/blob/master/README.md) 展示 `load_userdict`、`add_word`、自訂 IDF／停用詞和 TF-IDF、TextRank；可直接讀 [extract_tags.py 範例](https://github.com/fxsjy/jieba/blob/master/test/extract_tags.py) 與 [TextRank 原始碼](https://github.com/fxsjy/jieba/blob/master/jieba/analyse/textrank.py)。

**本專案建議：** 保留「大稻埕、北門站、大橋頭、延平河濱、交通管制」等詞，並人工核對臺灣用字、站名與「不塞／很塞」語境。維持現有每篇一次的 `document_frequency`，另做每平台、每活動前後時段的 `document_share`。TF-IDF／TextRank 是候選關鍵詞，不是自動完成有驗證的主題分類；預測用字典和 IDF 不能偷看未來測試貼文。

**現有實作：** `src/threads_web_report.py` 已用獨立 tokenizer、每篇詞集合、可重現排序，無須為找 GitHub 方法而重寫。繁體字典與停用詞修改要保存版本及抽查例子。授權：[MIT](https://github.com/fxsjy/jieba/blob/master/LICENSE)。發布較久，見下表；不能把「仍可使用」說成「近期活躍維護」。

### 8. word_cloud：把詞頻清楚呈現，不取代統計

**官方功能／可借用：** [中文範例](https://github.com/amueller/word_cloud/blob/main/examples/wordcloud_cn.py) 展示中文斷詞及字型；[WordCloud API](https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html) 支援 `generate_from_frequencies`、mask、random_state。該 API 文件標題仍是 1.8.1，參考當前行為時另核對 [main 原始碼](https://github.com/amueller/word_cloud/blob/main/wordcloud/wordcloud.py)，不能把文件版號當最新發布版。

**Windows 範例限制：** 上述 `wordcloud_cn.py` 呼叫 `jieba.enable_parallel(4)`，範例註解明示 Windows 不支援。本專案只借用它的字型、斷詞、遮罩概念，沿用現有獨立 `jieba.Tokenizer`，**不要原樣執行範例或照抄 `enable_parallel`**；現有文字雲程式不需因此改寫。[範例原始碼](https://github.com/amueller/word_cloud/blob/main/examples/wordcloud_cn.py)。

**本專案建議：** 圓形文字雲可繼續保留，旁邊增加前 20 詞的篇數／占比表及抽樣說明。比較不同期間時固定詞典、清理規則、字型和配色；字的像素面積不是精準的量尺，大小受排版與縮放參數影響。輸入已計算詞頻時，先在上游移除停用詞；不要以為 `generate_from_frequencies` 會再替字典套一次文字清理。

**輸入／輸出：** 已去重貼文→每篇一次的詞頻字典→PNG 加 CSV。不能把高頻「人潮」推論為現場人數，也不能把兩平台詞雲大小直接當平台聲量比例。授權：[MIT](https://github.com/amueller/word_cloud)；倉庫內附字型有獨立授權，中文使用本機字型時仍需確認使用方式。

## 維護與版本：本次實際查到的狀態

下表為 GitHub API 的 `pushed_at`、`releases/latest` 與 PyPI 的發布中繼資料，均於 2026-10-05 讀取。`pushed_at` 是倉庫推送時間，**不等於最後功能更新或維護品質保證**。查到的八個 GitHub 倉庫皆 `archived=false`；未封存也不表示仍活躍。日期以 UTC 日顯示。

| 項目 | 查得版本／發布日期 | GitHub 最近推送 | 該版 Python 宣告 | 來源 |
| --- | --- | --- | --- | --- |
| statsmodels | 0.15.0／2026-08-27 | 2026-10-04 | ≥3.10 | [Release](https://github.com/statsmodels/statsmodels/releases/tag/v0.15.0)、[API](https://api.github.com/repos/statsmodels/statsmodels)、[PyPI](https://pypi.org/pypi/statsmodels/0.15.0/json) |
| scikit-learn | 1.9.1／2026-09-11（GitHub release） | 2026-10-03 | ≥3.11 | [Release](https://github.com/scikit-learn/scikit-learn/releases/tag/1.9.1)、[API](https://api.github.com/repos/scikit-learn/scikit-learn)、[PyPI](https://pypi.org/pypi/scikit-learn/1.9.1/json) |
| StatsForecast | 2.1.1／2026-07-16 | 2026-10-01 | ≥3.10 | [Release](https://github.com/Nixtla/statsforecast/releases/tag/v2.1.1)、[API](https://api.github.com/repos/Nixtla/statsforecast)、[PyPI](https://pypi.org/pypi/statsforecast/2.1.1/json) |
| SimPy | 4.1.2／2026-05-24（PyPI） | 舊 fork：2021-02-03；上游推送日期本次未核實 | ≥3.8 | [官方](https://simpy.readthedocs.io/en/stable/)、[fork API](https://api.github.com/repos/simpx/simpy)、[PyPI](https://pypi.org/pypi/simpy/4.1.2/json) |
| OSMnx | 2.1.1／2026-07-21（Changelog） | 2026-07-31 | ≥3.11 | [Changelog](https://github.com/gboeing/osmnx/blob/main/CHANGELOG.md)、[API](https://api.github.com/repos/gboeing/osmnx)、[PyPI](https://pypi.org/pypi/osmnx/2.1.1/json) |
| AequilibraE | 1.7.0／2026-08-03 | 2026-10-02 | ≥3.11 | [Release](https://github.com/AequilibraE/aequilibrae/releases/tag/v1.7.0)、[API](https://api.github.com/repos/AequilibraE/aequilibrae)、[PyPI](https://pypi.org/pypi/aequilibrae/1.7.0/json) |
| jieba | 0.42.1／2020-01-20 | 2024-08-21 | 中繼資料未填，不能據此判相容 | [Release](https://github.com/fxsjy/jieba/releases/tag/v0.42.1)、[API](https://api.github.com/repos/fxsjy/jieba)、[PyPI](https://pypi.org/pypi/jieba/0.42.1/json) |
| word_cloud | 1.9.6／2026-01-22 | 2026-01-22 | ≥3.9 | [Release](https://github.com/amueller/word_cloud/releases/tag/1.9.6)、[API](https://api.github.com/repos/amueller/word_cloud)、[PyPI](https://pypi.org/pypi/wordcloud/1.9.6/json) |

專案固定 Python 3.10.11，因此 **scikit-learn 1.9.1、OSMnx 2.1.1、AequilibraE 1.7.0 不能直接裝進現有環境**。採用方法不等於必須升到最新版；後續應選經測試的相容版，或另建研究環境後用 Parquet／CSV／GeoJSON 與網站交換結果。此輪未驗證套件組合的安裝相容性。

## 對本專案的執行順序（建議，尚未實作）

1. **先把觀測層完成：** 固定資料版本與研究視窗，列活動／一般日覆蓋、基準天數、缺值與取消狀態。分析實際採用場數由統計報告依檔案核算，不沿用聊天中的舊場數。
2. **完成最低基準比較：** 北門每場視窗總量、逐時尖峰、相對一般日的差值；搭配同一批列的小模型及按時間整場留出。`origin_excess_vs_baseline` 只能作描述／另一目標，不能回填為原人次預測輸入。
3. **文字雲轉成可審核特徵：** 先人工主題類別與例句審核；事前預測只採截止時間以前的樣本。2026 年一個月的社群樣本不能直接補成 2017–2025 年的特徵。
4. **再做地圖與模擬：** OSM 步行候選路線改善展示；有班距與容量後才建 SimPy 分方向、批次上車情境。路徑分配與排隊模擬結果另存 `scenario_id`、假設、隨機種子與版本，與真實觀測明確分開。

這份文件提供選法與來源，不宣稱任何外部 repo 已經在本專案上通過重現、提高準確率或完成排隊模型驗證。
