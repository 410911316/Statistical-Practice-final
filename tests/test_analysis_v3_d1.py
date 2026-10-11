"""D1 source-time gates must reject future or outcome-based feature leakage."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis_v1.d1_features import attach_d1_features


class D1EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frame = pd.DataFrame({"event_date": pd.to_datetime(["2024-08-10"]*2),
                                   "hour": [18, 19], "station": ["北門"]*2})
        self.info = {"manifest": [], "audit": {"d1_deployable": False, "model_feature_whitelist": []},
                     "feature_eligibility": {}, "missing_summary": {}}
        self.record = {"event_date": "2024-08-10", "feature": "planned_drone", "value": 1,
                       "published_at": "2024-08-09T17:59:00+08:00",
                       "source_url": "https://example.test/announcement", "evidence_text": "Planned show",
                       "availability_status": "verified_pre_cutoff"}

    def run_records(self, records, features=("planned_drone",)):
        (self.root / "sources.json").write_text(json.dumps({"records": records}), encoding="utf-8")
        return attach_d1_features(self.frame, self.info, self.root, ["sources.json"], list(features))

    def test_cutoff_uses_latest_known_version_and_never_future_revision(self):
        earlier = {**self.record, "value": 0, "published_at": "2024-08-08T15:00:00+08:00"}
        future = {**self.record, "value": 0, "published_at": "2024-08-09T18:00:01+08:00"}
        frame, info, audit = self.run_records([earlier, self.record, future])
        self.assertEqual(frame.planned_drone.tolist(), [1, 1])
        self.assertFalse(info["audit"]["d1_deployable"])
        self.assertTrue(audit.eligible.all())
        self.assertEqual(self.info["manifest"], [])

    def test_unverified_or_post_cutoff_updated_page_is_unknown_not_absent(self):
        for override in ({"availability_status": "unknown"},
                         {"updated_at": "2024-08-11T12:00:00+08:00"}):
            with self.subTest(override=override):
                frame, _, audit = self.run_records([{**self.record, **override}])
                self.assertTrue(frame.planned_drone.isna().all())
                self.assertFalse(audit.eligible.any())

    def test_naive_or_missing_evidence_timestamp_is_rejected(self):
        for stamp in (None, "2024-08-09 12:00"):
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                self.run_records([{**self.record, "published_at": stamp}])

    def test_conflicting_same_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.run_records([self.record, {**self.record, "value": 0}])

    def test_invalid_boolean_and_observed_weather_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "planned_drone"):
            self.run_records([{**self.record, "value": .5}])
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            self.run_records([{**self.record, "feature": "observed_temperature"}])

    def test_weather_requires_correct_vintage_and_valid_hour(self):
        record = {**self.record, "feature": "forecast_temperature_c", "value": 29,
                  "hour": 18, "forecast_issued_at": "2024-08-09T00:00:00Z",
                  "forecast_valid_at": "2024-08-10T18:00:00+08:00"}
        frame, _, _ = self.run_records([record], ("forecast_temperature_c",))
        self.assertEqual(frame.forecast_temperature_c.iloc[0], 29)
        self.assertTrue(np.isnan(frame.forecast_temperature_c.iloc[1]))
        for override in ({"forecast_valid_at": "2024-08-10T19:00:00+08:00"},
                         {"forecast_issued_at": "2024-08-10T00:00:00Z"}):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "inconsistent"):
                self.run_records([{**record, **override}], ("forecast_temperature_c",))


if __name__ == "__main__":
    unittest.main()
