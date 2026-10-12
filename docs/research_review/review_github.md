# GitHub 方法獨立審查

審查日期：2026-10-05。審查者：`review_github_methods`，與原作者 `research_github_methods` 分開執行。對象為 [方法報告](github_methods.md) 及 `results/research_review/2026-10-05/github_methods.json`。**最終結論：通過。GH-01、GH-02、GH-03 均已由原作者修正，審查者逐項核對 MD 與 JSON 後結案，沒有未解決項目。** 未執行外部倉庫程式、未安裝套件、未變更資料庫。

## 問題與修正狀態

| 編號 | 優先度 | 問題與證據 | 請原作者修改 | 狀態 |
| --- | --- | --- | --- | --- |
| GH-01 | P2 | 被推薦的 [wordcloud_cn.py](https://github.com/amueller/word_cloud/blob/main/examples/wordcloud_cn.py) 一開始呼叫 `jieba.enable_parallel(4)`，程式註解明確說明 Windows 不支援。報告已特別檢查 Python 相容性，卻未提醒這個範例本身不能原樣在本專案 Windows 執行。 | 在 MD 與 JSON 補註只借用字型、斷詞、遮罩概念；沿用現有獨立 `Tokenizer`，不要照抄 `enable_parallel`。不需改現有文字雲程式。 | Closed：MD 與 JSON 已核對 |
| GH-02 | P3 | AequilibraE 段落提到 `flow_edges_hourly`，但本次列出 `data/processed` 檔案未見該 Parquet。`src/aggregate_flows.py` 第 150 行確有建表邏輯、第 179–181 行可輸出；這是可重建的產物，不能讓讀者誤以為已在目前成品資料夾。 | 註明該 OD 邊表可從既有彙整流程重建，當前檔案不在 `data/processed`；需要時才重跑或用等價唯讀查詢取得。 | Closed：MD 與 JSON 已核對 |
| GH-03 | P3 | F 檢定文字只概括寫「適當誤差假設」。少量活動的精確 F 推論應明說線性均值模型、常態且同方差獨立誤差的工作假設；每場一列也不能排除同年度或重疊背景日造成的相依。 | 建議補一句精確 F 的條件，並說明按年度／基準重疊的敏感度；不用新增檢定或模型。原本不是錯誤公式，屬可讀性補強。 | Closed：條件與相依說明已補 |

已嘗試直接傳送上述具體修改給原作者；協作工具當時回覆 `agent thread limit reached`，因此已向主協作者回報，改由主協作者在空出執行名額後喚醒原作者。此紀錄不把未送達的訊息算成已交付。後續主協作者已轉交並喚醒原作者；本次複審確認修改完成。

## 本次獨立做了哪些查核

### 1. 全部八項版本、Python 條件與來源身分

審查者自己以 HTTPS 讀取八個 GitHub repo API、各版 PyPI JSON，另讀取 `releases/latest`；沒有重用原作者的 API 回傳作證據。以下值與原稿一致。日期為來源的 UTC 日期，不是相容性安裝測試。

| 項目 | 核對版號 | PyPI Python 條件 | 核對發布依據 | 結果 |
| --- | --- | --- | --- | --- |
| statsmodels | 0.15.0 | ≥3.10 | GitHub 2026-08-27 | 相符 |
| scikit-learn | 1.9.1 | ≥3.11 | GitHub 2026-09-11；PyPI 2026-09-10 | 原稿已區分兩者，相符 |
| StatsForecast | 2.1.1 | ≥3.10 | GitHub 2026-07-16 | 相符 |
| SimPy | 4.1.2 | ≥3.8 | PyPI 2026-05-24 | 相符；舊 GitHub 副本無 latest release |
| OSMnx | 2.1.1 | ≥3.11 | Changelog／PyPI 2026-07-21 | 相符；latest release API 回覆 404，不據此推斷沒有發布 |
| AequilibraE | 1.7.0 | ≥3.11 | GitHub 2026-08-03 | 相符 |
| jieba | 0.42.1 | 空字串 | GitHub／PyPI 2020-01-20 | 原稿正確保留相容性未知 |
| word_cloud | 1.9.6 | ≥3.9 | GitHub／PyPI 2026-01-22 | 相符 |

八項 `pushed_at`、`archived=false` 均與 JSON 逐項相符。專案 `pyproject.toml` 確為 `requires-python = "==3.10.11"`；因此原稿提醒三個最新版不能直接裝進目前環境是正確的。中繼資料只證明宣告條件，不證明套件組合可安裝。

API 來源依各專案分別為：

- [statsmodels repo](https://api.github.com/repos/statsmodels/statsmodels)、[PyPI 0.15.0](https://pypi.org/pypi/statsmodels/0.15.0/json)
- [scikit-learn repo](https://api.github.com/repos/scikit-learn/scikit-learn)、[PyPI 1.9.1](https://pypi.org/pypi/scikit-learn/1.9.1/json)
- [StatsForecast repo](https://api.github.com/repos/Nixtla/statsforecast)、[PyPI 2.1.1](https://pypi.org/pypi/statsforecast/2.1.1/json)
- [SimPy 舊副本 repo](https://api.github.com/repos/simpx/simpy)、[PyPI 4.1.2](https://pypi.org/pypi/simpy/4.1.2/json)
- [OSMnx repo](https://api.github.com/repos/gboeing/osmnx)、[PyPI 2.1.1](https://pypi.org/pypi/osmnx/2.1.1/json)
- [AequilibraE repo](https://api.github.com/repos/AequilibraE/aequilibrae)、[PyPI 1.7.0](https://pypi.org/pypi/aequilibrae/1.7.0/json)
- [jieba repo](https://api.github.com/repos/fxsjy/jieba)、[PyPI 0.42.1](https://pypi.org/pypi/jieba/0.42.1/json)
- [word_cloud repo](https://api.github.com/repos/amueller/word_cloud)、[PyPI 1.9.6](https://pypi.org/pypi/wordcloud/1.9.6/json)

### 2. 授權、範例及方法限制

- 逐一核對 GitHub API 的授權欄位，並另外閱讀 [AequilibraE LICENSE.TXT](https://github.com/AequilibraE/aequilibrae/blob/develop/LICENSE.TXT)：確有 MIT 以外的文件引用附加條款，原稿未誤標成純 MIT。
- [SimPy GitHub 頁面](https://github.com/simpx/simpy) 的 About 確實指向 GitLab 上游；雖 GitHub API 的 `fork` 布林值為 false，不能因此認定它是官方上游。原稿的「舊 fork」源於頁面作者自述，處理正確。[官方 SimPy](https://simpy.readthedocs.io/en/stable/) 與 [Resource API](https://simpy.readthedocs.io/en/stable/api_reference/simpy.resources.html) 支持資源／離散事件功能；列車批次上車是本專案的建模建議，原稿有清楚區分。
- [statsmodels compare_f_test](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.compare_f_test.html) 明示巢狀模型與異質變異／相關性限制；[NegativeBinomial family](https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.NegativeBinomial.html) 的 alpha 為固定參數。原稿對 OLS F 與 GLM 分散度的區分正確。
- [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)、[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) 與 [StatsForecast 回測範例](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/crossvalidation.html) 均實際開啟。原稿自行按活動日期建立整場時間切分，而未聲稱 GroupKFold 保證未來預測，判斷合理。
- [OSMnx Changelog](https://github.com/gboeing/osmnx/blob/main/CHANGELOG.md) 2.1.0 確實移除 Python 3.9／3.10；[官方路徑 notebook](https://github.com/gboeing/osmnx-examples/blob/main/notebooks/02-routing-speed-time.ipynb)、[AequilibraE 大眾運輸分配範例](https://www.aequilibrae.com/latest/python/_auto_examples/public_transport/plot_public_transit_assignment.html)、[不使用 model 的分配範例](https://www.aequilibrae.com/latest/python/_auto_examples/traffic_assignment/plot_assignment_without_model.html) 均可開啟。原稿未將模型候選路線冒充真實個人軌跡，亦未把靜態分配當逐班排隊。
- [jieba extract_tags.py](https://github.com/fxsjy/jieba/blob/master/test/extract_tags.py)、[wordcloud 中文範例](https://github.com/amueller/word_cloud/blob/main/examples/wordcloud_cn.py) 及 [GLM 範例](https://www.statsmodels.org/stable/examples/notebooks/generated/glm.html) 路徑有效。中文範例的 Windows 限制列為 GH-01。

### 3. 與本專案是否接得起來

已閱讀 `src/aggregate_flows.py`、`src/threads_web_report.py` 的相關欄位與邏輯，並核對 `docs/data_preprocessing.md`。`baseline_origin_mean_28d` 確實是前 7／14／21／28 日的可用非活動日均值；`destination_count_same_source_bin` 不是已核實的出閘時鐘；文字處理確實使用獨立 tokenizer 及每篇一次計數。原稿的資料界線和三個優先工作流程與程式一致。

推薦主要是成熟泛用庫，但每項皆提供具體範例／API、輸入欄位及適用限制，足以作為下一步的方法索引；不需為增加 repo 數量再納入資料來源不明的捷運專案。報告沒有宣稱已在本資料上重現性能或完成模型驗證。本輪審查也不代表完成這些外部程式的重現。

## 最終驗收（2026-10-05）

- GH-01：方法報告第 94 行與 JSON 的 jieba／word_cloud `limitations` 已說明 Windows 不支援 `enable_parallel(4)`，限定借用範例概念並沿用現有 tokenizer。
- GH-02：方法報告第 78 行與 JSON 的 `local_input_availability` 已清楚標明 OD 邊表成品目前不存在、可由既有流程重建，也沒有聲稱本輪已重建。
- GH-03：方法報告第 34–36 行與 statsmodels JSON 已補上精確 F 的條件、同年度／共用背景日相依及敏感度檢查，並明說敏感度本身不保證精確 F 假設成立。

JSON 可正常解析；作者將其標為等待獨立驗收，該作者狀態不取代本審查紀錄。本次最終驗收只確認三項修正，未擴大研究、執行模型或變更作者檔案。