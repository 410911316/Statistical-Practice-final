# 大稻埕人流網站

React＋shadcn 黑灰介面、MapLibre 地圖。提供「人流展示」「資料分析」「模型與候車」三頁；公開版使用結果快照，本機版可保留 FastAPI／MySQL。

## 公開成果網站（GitHub Pages）

網址：[稻埕觀測室](https://shu-ming0808.github.io/Statistical-Practice-final/)；[模型與候車](https://shu-ming0808.github.io/Statistical-Practice-final/#/research)。不需要 Tailscale，電腦關機後仍可看。

- 人流展示／資料分析：26 個活動日期、四站 18–23 時共 624 筆彙總觀測；缺值不補成 0。CSV 下載依當前所選車站。
- 預測結果：九場、54 小時，實際／預測比較及 MAE、RMSE。沿用 V14，未重訓。
- 候車情境：切換場次、方向、剩餘空間、方向比例與峰位置，比較三種到達形狀。北門高運量六節編組；每班 555／1,110／1,665 人是剩餘空間假設。
- 班表與調度：四站分線、方向、日型與終點；附官方 PDF、生效日及來源。候選路徑只計發車後行車時間，完整調度提前量仍缺整備、出庫與插車等待。

公開版只讀 `frontend/public/data/` 的彙總快照，頁面標示匯出時間；不會查詢 MySQL 或自動重算模型。模型九場驗證已反覆使用，候車為現行一般班表的情境估計，不是歷史實測。MySQL、原始 CSV、SQL dump、帳密與 `.local/` 都不發布。

### 更新公開快照

在本機更新資料與分析結果後，從專案根目錄執行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m app.export_snapshot
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_public_snapshot.py
npm.cmd --prefix app/frontend test
```

檢查 `app/frontend/public/data/` 的變更，再 commit／push。匯出程式核對封存模型 SHA256，模型有改動時會拒絕繼續，需先更新分析來源及快照契約。GitHub Actions 只建置與發布已審查快照，不需要資料庫憑證。

### 本機預覽公開版

```powershell
$env:VITE_DATA_MODE='snapshot'
$env:VITE_BASE_PATH='/Statistical-Practice-final/'
npm.cmd --prefix app/frontend run build -- --outDir dist-pages
npm.cmd --prefix app/frontend run preview -- --host 127.0.0.1 --port 4173 --outDir dist-pages
```

網址 `http://127.0.0.1:4173/Statistical-Practice-final/`。預覽完清除這兩個環境變數，才能建置下方即時資料庫版：

```powershell
Remove-Item Env:VITE_DATA_MODE, Env:VITE_BASE_PATH -ErrorAction SilentlyContinue
```

Pages 採 `HashRouter`（`#/research`），重新整理與分享子頁不會 404；建置會附上專案路徑前綴。工作流程見 [pages.yml](../.github/workflows/pages.yml)，部署只上傳前端 `dist/`，不會上傳整個專案。

依據：[GitHub Pages 自訂工作流程](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)、[Vite GitHub Pages 部署](https://vite.dev/guide/static-deploy.html#github-pages)。

## 啟動本機 MySQL 版

主機需有 Node.js 22.12 以上的相容 LTS、uv、本機 MySQL，以及既有的 `codex-local` MySQL login path。在專案根目錄執行：

```powershell
# 首次安裝；--inexact 保留原有的 notebook、社群分析等套件
uv sync --inexact --extra app
npm.cmd --prefix app/frontend ci
npm.cmd --prefix app/frontend run build

# 背景啟動／停止網站
.\app\start.ps1
.\app\start.ps1 -Stop
```

啟動後開啟 [http://127.0.0.1:8100](http://127.0.0.1:8100)。後續改完前端可在停止網站後執行 `.\app\start.ps1 -Build`，重新安裝鎖定套件、建置並啟動。一般啟動不下載套件；記錄放在 `app/.runtime/`。如果 PowerShell 阻擋本機腳本，可用 `powershell.exe -ExecutionPolicy Bypass -File .\app\start.ps1`，只作用於這次執行。

資料連線使用本機 MySQL 工具與加密登入設定，網站不接收或保存資料庫密碼，僅執行固定的唯讀查詢。需要其他登入名稱或工具路徑時，在啟動前設定 `APP_MYSQL_LOGIN_PATH`、`APP_MYSQL_BIN`。資料庫異常時會顯示錯誤，不以假資料代替。

## 隊員透過 Tailscale 使用

三人各自使用帳號加入並獲准存取同一個 tailnet；使用網站的人不需安裝 Python 或 MySQL。主機必須持續開機，MySQL、網站及 Tailscale 都要運作。

目前主機的私人網址：[開啟稻埕觀測室](https://desktop-4rv84m1.tail7191d4.ts.net:8443/)。已設定 8443 轉送到本機 8100，無須每次啟動重新設定。

先檢查既有轉送，保留其他服務：

```powershell
tailscale serve status
tailscale funnel status
# 確認 8443 未被使用後，新增私人入口
tailscale serve --bg --https=8443 http://127.0.0.1:8100
```

使用 Tailscale 回傳的 `https://主機名稱.tailnet名稱.ts.net:8443` 網址連線。若 tailnet 存取規則有限制，需允許這三位成員連線到主機的 TCP 8443。**此專案不使用 Funnel，也不修改既有 443／8000 的轉送。** 私人入口需由主機管理者完成首次 HTTPS 授權；`start.ps1` 不會自動更動 Tailscale。

## 資料與示意範圍

- 地圖上的會場到車站路線、粒子、人數分配與分鐘變化是示意設定，不是民眾真實軌跡；固定站點／入口位置不能證明實際行走路徑。
- MySQL 顯示的是活動日原始 OD 時段的站點彙總。目的站人數是「同一來源時段」加總，不能直接當成真實出站時間；缺值不補為零。
- 不同隊員各自操作日期、參數與播放進度，不會修改 MySQL 或影響其他人。回歸、M/M/c、候車時間與正式疏運模型尚未實作。

```text
app/
├── frontend/       # React 畫面、地圖與建置後的 dist/
├── backend/        # FastAPI、MySQL 唯讀查詢與 API 測試
├── start.ps1       # Windows 啟動／停止
└── README.md
```

既有 `web/meta-review/` 是公開隱私與刪除說明網站，與本網站分開保留。
