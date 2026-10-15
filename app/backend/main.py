"""Run from the repository root: uvicorn app.backend.main:app --host 127.0.0.1."""
from __future__ import annotations

import csv
from datetime import date
import io
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from .database import DATABASE_MESSAGE, DatabaseUnavailable, Repository


VERSION = "0.1.0"
TIME_SEMANTICS = "Asia/Taipei；原始 OD 時段。目的站人數是同一來源時段的目的站彙總，尚不能視為實際出站刷卡時刻。"
FLOW_FIELDS = [
    "event_date", "hour_start", "station", "origin_count",
    "destination_count_same_source_bin", "origin_complete", "destination_complete",
    "baseline_origin_mean_28d", "baseline_n_days",
]


def _csv_cell(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    if isinstance(value, bool):
        return int(value)
    return value


def create_app(repository: Repository | None = None, frontend_dist: Path | None = None) -> FastAPI:
    api = FastAPI(title="大稻埕人流資料 API", version=VERSION, docs_url=None, redoc_url=None)
    api.state.repository = repository or Repository()
    dist = (frontend_dist or Path(__file__).resolve().parents[1] / "frontend" / "dist").resolve()

    @api.middleware("http")
    async def response_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @api.exception_handler(DatabaseUnavailable)
    async def database_error(request: Request, exc: DatabaseUnavailable):
        return JSONResponse(status_code=503, content={"detail": str(exc) or DATABASE_MESSAGE})

    @api.get("/api/health")
    def health():
        try:
            api.state.repository.health()
        except DatabaseUnavailable:
            return JSONResponse(status_code=503, content={
                "status": "error", "database": "unavailable", "message": DATABASE_MESSAGE, "version": VERSION,
            })
        return {"status": "ok", "database": "connected", "version": VERSION}

    @api.get("/api/events")
    def events():
        return {"events": api.state.repository.events()}

    @api.get("/api/stations")
    def stations():
        return {"stations": api.state.repository.stations()}

    def selected_rows(event_date: date, stations: str, start_hour: int, end_hour: int):
        if end_hour <= start_hour:
            raise HTTPException(422, "結束小時必須大於開始小時；結束時刻不包含在查詢內。")
        selected = list(dict.fromkeys(s.strip() for s in stations.split(",") if s.strip()))
        if not selected or len(selected) > 20 or any(len(station) > 64 for station in selected):
            raise HTTPException(422, "請選擇 1 至 20 個車站。")
        repo = api.state.repository
        if event_date.isoformat() not in {event["event_date"] for event in repo.events()}:
            raise HTTPException(404, "查無這個活動日期。")
        known = {station["station"] for station in repo.stations()}
        if not set(selected).issubset(known):
            raise HTTPException(422, "車站名稱不在資料庫清單中，請重新選擇。")
        rows = repo.flows(event_date, selected, start_hour, end_hour)
        metadata = {
            "source": "MySQL", "timezone": "Asia/Taipei", "time_semantics": TIME_SEMANTICS,
            "event_date": event_date.isoformat(), "stations": selected,
            "start_hour": start_hour, "end_hour_exclusive": end_hour,
            "row_count": len(rows), "expected_rows": len(selected) * (end_hour - start_hour),
            "station_semantics": "原始站名標籤；未自動合併轉乘站標籤。",
            "missing_values": "null 代表缺資料或不完整，不能當成 0。",
            "simulation_note": "會場到車站的步行路線、人數分配及分鐘變化屬示意假設，非實測軌跡。",
            "cache_seconds": 30,
        }
        return {"rows": rows, "metadata": metadata}

    @api.get("/api/flows")
    def flows(event_date: date, stations: str = Query(default="北門", min_length=1, max_length=1300),
              start_hour: int = Query(default=18, ge=0, le=23),
              end_hour: int = Query(default=24, ge=1, le=24)):
        return selected_rows(event_date, stations, start_hour, end_hour)

    @api.get("/api/flows.csv")
    def flows_csv(event_date: date, stations: str = Query(default="北門", min_length=1, max_length=1300),
                  start_hour: int = Query(default=18, ge=0, le=23),
                  end_hour: int = Query(default=24, ge=1, le=24)):
        payload = selected_rows(event_date, stations, start_hour, end_hour)
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(FLOW_FIELDS)
        writer.writerows([_csv_cell(row.get(field)) for field in FLOW_FIELDS] for row in payload["rows"])
        return Response(buffer.getvalue().encode("utf-8-sig"), media_type="text/csv; charset=utf-8", headers={
            "Content-Disposition": f'attachment; filename="dadaocheng_{event_date}_{start_hour}-{end_hour}.csv"',
        })

    def index():
        page = dist / "index.html"
        if not page.is_file():
            return HTMLResponse("<html lang='zh-Hant'><meta charset='utf-8'><title>網站尚未建置</title>"
                                "<p>網站前端尚未建置，請先依 app/README.md 完成建置，再重新開啟。</p></html>",
                                status_code=503)
        return FileResponse(page, headers={"Cache-Control": "no-cache"})

    api.add_api_route("/", index, methods=["GET"], include_in_schema=False)
    api.add_api_route("/analysis", index, methods=["GET"], include_in_schema=False)
    api.add_api_route("/research", index, methods=["GET"], include_in_schema=False)

    @api.get("/data/{asset_path:path}", include_in_schema=False)
    def public_data(asset_path: str):
        base = (dist / "data").resolve()
        file = (base / asset_path).resolve()
        if not file.is_relative_to(base) or not file.is_file():
            raise HTTPException(404, "找不到公開快照。")
        return FileResponse(file, headers={"Cache-Control": "no-cache"})

    @api.get("/assets/{asset_path:path}", include_in_schema=False)
    def asset(asset_path: str):
        base = (dist / "assets").resolve()
        file = (base / asset_path).resolve()
        if not file.is_relative_to(base) or not file.is_file():
            raise HTTPException(404, "找不到檔案。")
        return FileResponse(file)

    return api


app = create_app()
