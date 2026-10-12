"""Download one official monthly OD CSV and load it into MySQL's raw landing table.

Usage: python mysql/import_original_data.py 202607
Requires an existing MySQL login path named codex-local (or --login-path override).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import pathlib
import re
import shutil
import subprocess
import sys
from urllib.parse import urlparse

import requests


ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "臺北捷運每日分時各站OD流量統計資料.csv"
MYSQL = pathlib.Path(r"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe")
EXPECTED_HEADER = ["日期", "時段", "進站", "出站", "人次"]


def mysql(sql: str, login_path: str) -> str:
    result = subprocess.run(
        [str(MYSQL), f"--login-path={login_path}", "--default-character-set=utf8mb4", "--batch", "--skip-column-names"],
        input=sql.encode("utf-8"), capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout.decode("utf-8", errors="replace").strip()


def official_url(month: str) -> str:
    with CATALOG.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if f"{int(row['西元年']):04d}{int(row['月']):02d}" == month:
                url = row["URL"].strip().replace("http://", "https://", 1)
                if urlparse(url).netloc != "tcgmetro.blob.core.windows.net":
                    raise ValueError("Unexpected source host in catalog")
                return url
    raise ValueError(f"Month {month} is not in the local official download catalog")


def download(url: str, destination: pathlib.Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    if not destination.exists():
        head = requests.head(url, timeout=20)
        head.raise_for_status()
        expected_size = int(head.headers["Content-Length"])
        chunk_size = 8 * 1024 * 1024

        def fetch(start: int) -> tuple[int, bytes]:
            end = min(start + chunk_size, expected_size) - 1
            last_error = None
            for _ in range(3):
                try:
                    response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 60))
                    response.raise_for_status()
                    if response.status_code != 206 or len(response.content) != end - start + 1:
                        raise IOError(f"Bad byte range {start}-{end}: HTTP {response.status_code}")
                    return start, response.content
                except (requests.RequestException, IOError) as exc:
                    last_error = exc
            raise IOError(f"Failed byte range {start}-{end}: {last_error}")

        starts = range(0, expected_size, chunk_size)
        with partial.open("wb") as handle:
            handle.truncate(expected_size)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(fetch, start) for start in starts]
                for number, future in enumerate(as_completed(futures), 1):
                    start, block = future.result()
                    handle.seek(start)
                    handle.write(block)
                    if number % 10 == 0 or number == len(futures):
                        print(f"  downloaded {number}/{len(futures)} chunks", flush=True)
        partial.replace(destination)
    digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_csv(path: pathlib.Path, month: str) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != EXPECTED_HEADER:
            raise ValueError(f"Unexpected CSV header: {header!r}")
        first = next(reader)
        if len(first) != 5 or first[0][:7].replace("-", "") != month:
            raise ValueError(f"Unexpected first row: {first!r}")
    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            count += block.count(b"\n")
    return count - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("month", help="YYYYMM, e.g. 202607")
    parser.add_argument("--login-path", default="codex-local")
    parser.add_argument("--download-only", action="store_true",
                        help="Download and validate CSV for the Parquet pipeline, without loading raw MySQL rows")
    args = parser.parse_args()
    if not re.fullmatch(r"20\d{4}", args.month):
        parser.error("month must be YYYYMM")

    url = official_url(args.month)
    local = ROOT / "data" / "original_data" / f"{args.month}.csv"
    print(f"Downloading/checking {args.month} ...", flush=True)
    sha256 = download(url, local)
    expected_rows = inspect_csv(local, args.month)
    print(f"Source: {local} | bytes={local.stat().st_size} | rows={expected_rows} | sha256={sha256}", flush=True)
    if args.download_only:
        return

    secure_dir = mysql("SHOW VARIABLES LIKE 'secure_file_priv';", args.login_path).split("\t", 1)[1]
    staged = pathlib.Path(secure_dir) / f"metro_od_{args.month}.csv"
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, staged)
    try:
        staged_sql_path = staged.as_posix().replace("'", "''")
        sql = f"""
        SET @source_row := 0;
        LOAD DATA INFILE '{staged_sql_path}' IGNORE INTO TABLE taipei_metro_analysis.original_data
        CHARACTER SET utf8mb4
        FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
        LINES TERMINATED BY '\\r\\n'
        IGNORE 1 LINES
        (@date_raw, @hour_raw, @entry_raw, @exit_raw, @count_raw)
        SET date_raw = @date_raw,
            hour_raw = @hour_raw,
            entry_station_raw = @entry_raw,
            exit_station_raw = @exit_raw,
            passenger_count_raw = @count_raw,
            source_month = '{args.month}',
            source_row = (@source_row := @source_row + 1);
        SELECT COUNT(*), MIN(date_raw), MAX(date_raw)
        FROM taipei_metro_analysis.original_data WHERE source_month = '{args.month}';
        """
        print("Importing into taipei_metro_analysis.original_data ...", flush=True)
        result = mysql(sql, args.login_path)
    finally:
        staged.unlink(missing_ok=True)
    print(f"MySQL count / date range: {result}")
    actual_rows = int(result.split("\t", 1)[0])
    if actual_rows != expected_rows:
        raise RuntimeError(f"Row count mismatch: source={expected_rows}, database={actual_rows}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
