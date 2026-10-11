"""Archived D-1 weather must retain publication, units and interpolation evidence."""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("collect_d1_gfs", ROOT / "scripts/collect_d1_gfs.py")
gfs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gfs)


def listing(stamp):
    entries = "".join(
        f"<Contents><Key>gfs.20240809/00/atmos/gfs.t00z.pgrb2.0p25.f{lead:03d}</Key>"
        f"<LastModified>{stamp}</LastModified></Contents>" for lead in (33, 36, 39)
    )
    return ('<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            + entries + "</ListBucketResult>").encode()


class ArchivedWeatherTest(unittest.TestCase):
    def setUp(self):
        self.meta = {"sha256": "mock", "checked_at": "2026-10-11T00:00:00+00:00"}

    def test_publication_after_cutoff_or_before_initialisation_fails_before_values(self):
        for stamp in ("2024-08-09T10:00:01Z", "2024-08-08T23:59:59Z"):
            with self.subTest(stamp=stamp), \
                    patch.object(gfs, "fetch", return_value=(listing(stamp), self.meta)), \
                    patch.object(gfs, "point_values") as points:
                with self.assertRaisesRegex(ValueError, "outside D-1"):
                    gfs.collect_event("2024-08-10", Path("."))
                points.assert_not_called()

    def test_missing_publication_returns_unknown_instead_of_observation_or_zero(self):
        empty = b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"/>'
        with patch.object(gfs, "fetch", return_value=(empty, self.meta)), \
                patch.object(gfs, "point_values") as points:
            rows, summary = gfs.collect_event("2017-08-26", Path("."))
        self.assertEqual(summary["status"], "unknown")
        self.assertEqual(len(rows), 18)
        self.assertEqual({r["hour"] for r in rows}, set(range(18, 24)))
        self.assertTrue(all(r["value"] is None and r["published_at"] is None for r in rows))
        points.assert_not_called()

    def test_point_csv_requires_expected_units(self):
        content = ('time,station,latitude[unit="degrees_north"],longitude[unit="degrees_east"],'
                   'Temperature_height_above_ground[unit="degC"]\n'
                   '2024-08-10T12:00:00Z,point,25.000,121.500,28.5\n').encode()
        with patch.object(gfs, "fetch", return_value=(content, self.meta)):
            with self.assertRaisesRegex(ValueError, "Unexpected units"):
                gfs.point_values("20240809", 36, "temperature", Path("."))

    def test_point_csv_requires_exact_valid_time_and_nearest_grid(self):
        template = ('time,station,latitude[unit="degrees_north"],longitude[unit="degrees_east"],'
                    'Temperature_height_above_ground[unit="K"]\n{stamp},point,{lat},121.500,301.65\n')
        for stamp, lat, error in (("2024-08-10T11:00:00Z", 25, "valid time"),
                                  ("2024-08-10T12:00:00Z", 25.25, "Grid changed")):
            content = template.format(stamp=stamp, lat=lat).encode()
            with self.subTest(stamp=stamp, lat=lat), \
                    patch.object(gfs, "fetch", return_value=(content, self.meta)):
                with self.assertRaisesRegex(ValueError, error):
                    gfs.point_values("20240809", 36, "temperature", Path("."))

    def test_cache_content_hash_is_checked_before_reuse(self):
        url = "https://example.test/forecast.csv"
        key = hashlib.sha256(url.encode()).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            (cache / f"{key}.body").write_bytes(b"modified")
            (cache / f"{key}.json").write_text(json.dumps({"url": url, "sha256": "wrong"}))
            with patch.object(gfs.requests, "get") as request:
                with self.assertRaisesRegex(ValueError, "Cache identity/hash mismatch"):
                    gfs.fetch(url, cache)
                request.assert_not_called()

    def test_public_artifact_has_valid_cutoffs_and_unique_event_hour_feature(self):
        payload = json.loads((ROOT / "docs/d1_weather_sources.json").read_text(encoding="utf-8"))
        rows = payload["records"]
        self.assertEqual(len(rows), len({(r["event_date"], r["hour"], r["feature"]) for r in rows}))
        self.assertTrue(any(r["availability_status"] == "verified_pre_cutoff" for r in rows))
        for row in rows:
            with self.subTest(key=(row["event_date"], row["hour"], row["feature"])):
                if row["availability_status"] == "unknown":
                    self.assertIsNone(row["value"])
                    self.assertIsNone(row["published_at"])
                    continue
                self.assertEqual(row["availability_status"], "verified_pre_cutoff")
                cutoff = datetime.fromisoformat(row["event_date"] + "T18:00:00+08:00") - timedelta(days=1)
                published = datetime.fromisoformat(row["published_at"])
                initialised = datetime.fromisoformat(row["forecast_issued_at"])
                valid = datetime.fromisoformat(row["forecast_valid_at"])
                self.assertLessEqual(initialised, published)
                self.assertLessEqual(published, cutoff)
                self.assertLess(cutoff, valid)
                self.assertEqual(valid.hour, row["hour"])
                self.assertEqual(valid.date().isoformat(), row["event_date"])
                self.assertEqual(published, max(gfs.timestamp(e["last_modified"])
                                               for e in row["publication_evidence"]))
                self.assertEqual((row["grid_latitude"], row["grid_longitude"]), (25.0, 121.5))

    def test_public_values_reconstruct_from_forecast_endpoints_and_wind_vectors(self):
        payload = json.loads((ROOT / "docs/d1_weather_sources.json").read_text(encoding="utf-8"))
        checked = 0
        for row in payload["records"]:
            if row["availability_status"] != "verified_pre_cutoff":
                continue
            evidence, weight = row["subset_evidence"], row["interpolation_weight"]
            self.assertEqual(len(evidence), len(row["lead_hours_used"]))
            target_lead = row["hour"] + 16
            leads = row["lead_hours_used"]
            if len(leads) == 1:
                self.assertEqual(leads[0], target_lead)
                self.assertEqual(weight, 0)
            else:
                self.assertEqual(weight, (target_lead - leads[0]) / (leads[-1] - leads[0]))
            def interpolate(name):
                a = evidence[0]["values"][name]
                return a + weight * (evidence[-1]["values"][name] - a)
            if row["feature"] == "forecast_temperature_c":
                expected = interpolate("Temperature_height_above_ground") - 273.15
            elif row["feature"] == "forecast_wind_speed_m_s":
                expected = math.hypot(interpolate("u-component_of_wind_height_above_ground"),
                                      interpolate("v-component_of_wind_height_above_ground"))
            elif row["feature"] == "forecast_cloud_cover_pct":
                expected = interpolate("Total_cloud_cover_entire_atmosphere")
            else:
                self.fail(f"Unexpected feature without defined conversion: {row['feature']}")
            self.assertAlmostEqual(row["value"], expected, delta=5.01e-7)
            checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
