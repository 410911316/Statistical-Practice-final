from datetime import date
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.backend.database import DATABASE_MESSAGE, DatabaseUnavailable, Repository, _read_json
from app.backend.main import _csv_cell, create_app


class SampleRepository:
    def __init__(self):
        self.queries = []

    def health(self):
        return True

    def events(self):
        return [{"event_date": "2026-08-15", "event_name": "大稻埕夏日節", "event_status": "held"}]

    def stations(self):
        return [{"station": "北門"}, {"station": "大橋頭站"}]

    def flows(self, event_date, stations, start, end):
        self.queries.append((event_date, stations, start, end))
        return [{"event_date": "2026-08-15", "hour_start": "2026-08-15T18:00:00",
                 "station": "北門", "origin_count": 100, "destination_count_same_source_bin": None,
                 "origin_complete": True, "destination_complete": False,
                 "baseline_origin_mean_28d": None, "baseline_n_days": 0}]


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.repo = SampleRepository()
        self.tmp = tempfile.TemporaryDirectory()
        self.dist = Path(self.tmp.name) / "dist"
        self.client = TestClient(create_app(self.repo, self.dist))

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()

    def test_research_route_and_only_public_dist_data_are_served(self):
        (self.dist / 'data').mkdir(parents=True)
        (self.dist / 'index.html').write_text('<h1>app</h1>')
        (self.dist / 'data' / 'research.json').write_text('{"public":true}')
        self.assertEqual(self.client.get('/research').status_code, 200)
        self.assertEqual(self.client.get('/data/research.json').json(), {'public': True})
        self.assertEqual(self.client.get('/data/missing.json').status_code, 404)
        self.assertEqual(self.client.get('/data/..%2Findex.html').status_code, 404)

    def test_valid_flow_keeps_missing_values_and_semantics(self):
        response = self.client.get("/api/flows", params={"event_date": "2026-08-15", "stations": "北門,北門,大橋頭站"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsNone(payload["rows"][0]["destination_count_same_source_bin"])
        self.assertFalse(payload["rows"][0]["destination_complete"])
        self.assertEqual(payload["metadata"]["expected_rows"], 12)
        self.assertIn("尚不能", payload["metadata"]["time_semantics"])
        self.assertEqual(self.repo.queries, [(date(2026, 8, 15), ["北門", "大橋頭站"], 18, 24)])
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_invalid_filters_never_reach_flow_query(self):
        for params in [
            {"event_date": "2026-02-30"},
            {"event_date": "2026-08-15", "start_hour": 24},
            {"event_date": "2026-08-15", "end_hour": 25},
            {"event_date": "2026-08-15", "start_hour": 20, "end_hour": 19},
            {"event_date": "2026-08-15", "stations": "北門');DROP TABLE x;--"},
            {"event_date": "2026-08-15", "stations": " , "},
            {"event_date": "2026-08-15", "stations": ",".join(str(i) for i in range(21))},
        ]:
            with self.subTest(params=params):
                self.assertEqual(self.client.get("/api/flows", params=params).status_code, 422)
        self.assertEqual(self.client.get("/api/flows?event_date=2026-08-16").status_code, 404)
        self.assertEqual(self.repo.queries, [])

    def test_csv_uses_same_filter_and_utf8_bom_and_blank_null(self):
        response = self.client.get("/api/flows.csv?event_date=2026-08-15&start_hour=0&end_hour=1")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content.startswith(b"\xef\xbb\xbf"))
        text = response.content.decode("utf-8-sig")
        self.assertIn("北門,100,,1,0,,0", text)
        self.assertEqual(self.repo.queries[0][2:], (0, 1))
        for value in ["=1+1", "+SUM(A1)", "-cmd", "@foo", "\t=2"]:
            self.assertEqual(_csv_cell(value), "'" + value)
        self.assertEqual(_csv_cell(20), 20)

    def test_safe_database_outage(self):
        unavailable = Repository(lambda sql: (_ for _ in ()).throw(DatabaseUnavailable(DATABASE_MESSAGE)))
        self.client.app.state.repository = unavailable
        for endpoint in ["health", "events", "stations", "flows?event_date=2026-08-15"]:
            response = self.client.get("/api/" + endpoint)
            self.assertEqual(response.status_code, 503)
            self.assertIn("目前無法讀取", response.text)
        self.assertEqual(self.client.get("/api/health").json()["database"], "unavailable")

    def test_only_known_spa_routes_and_assets_are_served(self):
        self.assertEqual(self.client.get("/").status_code, 503)
        self.dist.mkdir()
        (self.dist / "index.html").write_text("<html>test site</html>", encoding="utf-8")
        (self.dist / "assets").mkdir()
        (self.dist / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")
        (self.dist / "secret.txt").write_text("not an asset", encoding="utf-8")
        self.assertEqual(self.client.get("/analysis").status_code, 200)
        self.assertEqual(self.client.get("/assets/app.js").status_code, 200)
        for url in ["/api/nonexistent", "/secret.txt", "/assets/%2e%2e%2fsecret.txt", "/other"]:
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 404)


class ReaderTests(unittest.TestCase):
    def test_reader_is_read_only_and_bounded(self):
        result = subprocess.CompletedProcess([], 0, b'{"ok":1}\n', b"")
        with patch("app.backend.database.Path.is_file", return_value=True), \
             patch("app.backend.database.subprocess.run", return_value=result) as run, \
             patch.dict("os.environ", {"MYSQL_PWD": "not-for-subprocess"}):
            self.assertEqual(_read_json("SELECT JSON_OBJECT('ok',1);"), [{"ok": 1}])
            options = run.call_args.kwargs
            query = options["input"].decode("utf-8")
            self.assertIn("START TRANSACTION READ ONLY", query)
            self.assertIn("MAX_EXECUTION_TIME=8000", query)
            self.assertTrue(query.endswith("ROLLBACK;\n"))
            self.assertEqual(options["timeout"], 15)
            self.assertNotIn("MYSQL_PWD", options["env"])

    def test_reader_redacts_mysql_errors_and_handles_timeout(self):
        with patch("app.backend.database.Path.is_file", return_value=True):
            for failure in [subprocess.CompletedProcess([], 1, b"", b"password:secret"),
                            subprocess.TimeoutExpired("mysql", 15)]:
                kwargs = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
                with patch("app.backend.database.subprocess.run", **kwargs):
                    with self.assertRaises(DatabaseUnavailable) as caught:
                        _read_json("SELECT 1;")
                    self.assertEqual(str(caught.exception), DATABASE_MESSAGE)

    def test_cache_and_query_boundaries(self):
        calls = []
        def reader(sql):
            calls.append(sql)
            return [{"station": "北門", "origin_complete": 1, "destination_complete": 0}]
        repo = Repository(reader)
        first = repo.flows(date(2026, 8, 15), ["北門"], 18, 24)
        first[0]["station"] = "mutated"
        second = repo.flows(date(2026, 8, 15), ["北門"], 18, 24)
        self.assertEqual(second[0]["station"], "北門")
        self.assertEqual(len(calls), 1)
        self.assertIn("hour_start < TIMESTAMPADD(HOUR, 24", calls[0])
        self.assertIn("LIMIT 481", calls[0])
        self.assertNotIn("'北門'", calls[0])

    def test_unexpected_row_volume_fails_instead_of_silent_truncation(self):
        repo = Repository(lambda sql: [{} for _ in range(481)])
        with self.assertRaises(DatabaseUnavailable):
            repo.flows(date(2026, 8, 15), ["北門"], 18, 24)


if __name__ == "__main__":
    unittest.main()
