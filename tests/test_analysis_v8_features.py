"""Source-version and timing regression checks for planned-show-end inputs."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis_v8.features import (END_FEATURES, STATION_CODES, attach_end_features,
                                  record_digest)
from analysis_v5.models import Encoder as SharedEncoder
from analysis_v7.models import Encoder as GAMEncoder
from analysis_v1.models import PLANNED_END_CANDIDATES


def frame_for(dates):
    return pd.DataFrame([{"event_date": pd.Timestamp(day), "station": station,
                          "hour": hour, "y": 999999.0, "nominal_end_hour": 3.0}
                         for day in dates for station in STATION_CODES.values()
                         for hour in range(18, 24)])


class ShowEndFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "docs").mkdir()
        for name in ("d1_program_sources.json", "d1_show_end_sources.json"):
            shutil.copyfile(ROOT / "docs" / name, self.root / "docs" / name)
        self.payload_path = self.root / "docs/d1_show_end_sources.json"
        self.parent_path = self.root / "docs/d1_program_sources.json"
        self.metadata = {"audit": {"d1_deployable": False,
                                   "model_feature_whitelist": ["hour"]},
                         "manifest": []}

    def tearDown(self):
        self.temp.cleanup()

    def payload(self):
        return json.loads(self.payload_path.read_text(encoding="utf-8"))

    def save(self, payload):
        self.payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def attach(self, dates, **kwargs):
        return attach_end_features(frame_for(dates), self.metadata, self.root, **kwargs)

    def test_all_21_events_keep_known_and_unknown_scope(self):
        dates = [r["event_date"] for r in self.payload()["records"]]
        frame, metadata, audit = self.attach(dates)
        self.assertEqual(len(frame), 504)
        self.assertEqual(int(audit.eligible.sum()), 18)
        self.assertEqual(audit.loc[audit.eligible, "scope"].value_counts().to_dict(),
                         {"fireworks_only": 16, "combined_sky_program": 2})
        self.assertEqual(int(frame[END_FEATURES[0]].notna().sum()), 432)
        self.assertFalse(metadata["audit"]["d1_deployable"])
        self.assertEqual(metadata["audit"]["planned_show_end_summary"]["unknown_events"], 3)

    def test_overlap_20_18_is_split_across_hour_bins(self):
        frame, _, audit = self.attach(["2022-07-30"])
        station = frame[frame.station.eq("北門")].set_index("hour")
        self.assertAlmostEqual(audit.iloc[0].planned_end_hour, 20.3)
        np.testing.assert_allclose(station.planned_end_0_60_beimen, [0, 0, .7, .3, 0, 0], atol=1e-12)
        np.testing.assert_allclose(station.planned_end_60_120_beimen, [0, 0, 0, .7, .3, 0], atol=1e-12)
        for window in ("0_60", "60_120"):
            self.assertAlmostEqual(station[f"planned_end_{window}_beimen"].sum(), 1.0)

    def test_each_station_only_activates_its_own_two_columns(self):
        frame, _, _ = self.attach(["2022-07-30"])
        for code, station in STATION_CODES.items():
            chosen = frame.loc[frame.station.eq(station), list(END_FEATURES)]
            own = [f"planned_end_{window}_{code}" for window in ("0_60", "60_120")]
            self.assertTrue((chosen.drop(columns=own) == 0).all().all())
            self.assertAlmostEqual(chosen[own].to_numpy().sum(), 2.0)

    def test_unknown_all_eight_missing_even_other_station_columns(self):
        frame, _, audit = self.attach(["2018-08-18", "2023-07-05", "2023-08-09"])
        self.assertTrue(frame[list(END_FEATURES)].isna().all().all())
        self.assertFalse(audit.eligible.any())
        reasons = audit.set_index("event_date").reason
        self.assertEqual(reasons["2018-08-18"], "published_after_cutoff")
        self.assertEqual(reasons["2023-08-09"], "published_after_cutoff")

    def test_combined_sky_start_keeps_fireworks_start_unknown(self):
        original = frame_for(["2026-07-25", "2026-08-15"])
        original["planned_fireworks_start_hour"] = np.nan
        frame, _, audit = attach_end_features(original, self.metadata, self.root)
        self.assertTrue(frame.planned_fireworks_start_hour.isna().all())
        self.assertTrue(audit.scope.eq("combined_sky_program").all())
        np.testing.assert_allclose(audit.planned_end_hour, 20 + 20 / 60)
        self.assertTrue(frame[list(END_FEATURES)].notna().all().all())

    def test_source_manifest_has_parent_and_derivative_hashes(self):
        _, metadata, _ = self.attach(["2022-07-30"])
        self.assertEqual({r["path"] for r in metadata["manifest"]},
                         {"docs/d1_program_sources.json", "docs/d1_show_end_sources.json"})
        for item in metadata["manifest"]:
            self.assertEqual(item["sha256"], hashlib.sha256((self.root / item["path"]).read_bytes()).hexdigest())
        self.assertTrue(set(END_FEATURES) <= set(metadata["audit"]["model_feature_whitelist"]))

    def test_changing_labels_or_nominal_actual_end_cannot_change_features(self):
        original = frame_for(["2022-07-30"])
        result, _, _ = attach_end_features(original, self.metadata, self.root)
        changed = original.copy()
        changed["y"] = -100000000
        changed["nominal_end_hour"] = 23
        compared, _, _ = attach_end_features(changed, self.metadata, self.root)
        pd.testing.assert_frame_equal(result[list(END_FEATURES)], compared[list(END_FEATURES)])
        pd.testing.assert_frame_equal(result[original.columns], original)
        self.assertEqual(self.metadata, {"audit": {"d1_deployable": False,
                                                "model_feature_whitelist": ["hour"]}, "manifest": []})

    def test_tampered_derived_value_rejected(self):
        payload = self.payload()
        payload["records"][0]["planned_end_hour"] = 21.9
        self.save(payload)
        with self.assertRaisesRegex(ValueError, "differs from validated parent"):
            self.attach(["2017-08-26"])

    def test_unknown_cannot_be_promoted_by_derived_json(self):
        payload = self.payload()
        record = next(r for r in payload["records"] if r["event_date"] == "2018-08-18")
        record.update(availability_status="verified_pre_cutoff", scope="fireworks_only",
                      planned_start_hour=20.0, planned_duration_minutes=5.0, planned_end_hour=20.0833333333)
        self.save(payload)
        with self.assertRaisesRegex(ValueError, "cannot be promoted"):
            self.attach(["2018-08-18"])

    def test_rewriting_parent_requires_reauditing_derivative(self):
        self.parent_path.write_text(self.parent_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "parent source changed"):
            self.attach(["2022-07-30"])

    def test_parent_record_digest_checked_even_if_whole_file_hash_updated(self):
        parent = json.loads(self.parent_path.read_text(encoding="utf-8"))
        parent["records"][1]["value"] = 11
        self.parent_path.write_text(json.dumps(parent, ensure_ascii=False), encoding="utf-8")
        payload = self.payload()
        payload["parent_source_sha256"] = hashlib.sha256(self.parent_path.read_bytes()).hexdigest()
        self.save(payload)
        with self.assertRaisesRegex(ValueError, "parent reference differs"):
            self.attach(["2017-08-26"])

    def test_naive_derived_timestamp_rejected(self):
        payload = self.payload()
        payload["records"][0]["published_at"] = "2017-08-21T23:59:59"
        self.save(payload)
        with self.assertRaisesRegex(ValueError, "explicit timezone"):
            self.attach(["2017-08-26"])

    def test_timezone_equivalent_timestamp_is_accepted(self):
        payload = self.payload()
        record = payload["records"][0]
        record["published_at"] = pd.Timestamp(record["published_at"]).tz_convert("UTC").isoformat()
        self.save(payload)
        _, _, audit = self.attach(["2017-08-26"])
        self.assertTrue(audit.iloc[0].eligible)

    def test_source_cutoff_cannot_be_delayed_to_event_day(self):
        payload = self.payload()
        payload["records"][0]["cutoff_at"] = "2017-08-26T18:00:00+08:00"
        self.save(payload)
        with self.assertRaisesRegex(ValueError, "D-1 18:00"):
            self.attach(["2017-08-26"])

    def test_unknown_event_outside_source_stays_missing(self):
        frame, _, audit = self.attach(["2027-08-01"])
        self.assertTrue(frame[list(END_FEATURES)].isna().all().all())
        self.assertEqual(audit.iloc[0].reason, "no_verified_record")

    def test_combined_program_start_cannot_be_changed(self):
        payload = self.payload()
        record = next(r for r in payload["records"] if r["event_date"] == "2026-07-25")
        record["structured_start_evidence"]["value"] = 20.2
        self.save(payload)
        with self.assertRaisesRegex(ValueError, "combined program start"):
            self.attach(["2026-07-25"])

    def test_invalid_hour_station_and_timezone_date_rejected(self):
        for field, value in (("hour", 20.5), ("station", "不存在"),
                             ("event_date", pd.Timestamp("2022-07-30", tz="Asia/Taipei"))):
            with self.subTest(field=field):
                original = frame_for(["2022-07-30"])
                original[field] = value
                with self.assertRaises(ValueError):
                    attach_end_features(original, self.metadata, self.root)

    def test_actual_shared_and_gam_encoder_accept_audited_eight_features(self):
        dates = ["2017-08-26", "2018-08-18", "2019-08-03", "2020-08-29",
                 "2021-10-16", "2022-07-30", "2023-07-12"]
        frame, _, _ = self.attach(dates)
        self.assertEqual(set(END_FEATURES), PLANNED_END_CANDIDATES)
        for encoder_type in (SharedEncoder, GAMEncoder):
            with self.subTest(encoder=encoder_type.__name__):
                encoder = encoder_type().fit(frame, END_FEATURES)
                self.assertTrue(np.isfinite(encoder.transform(frame)).all())
                self.assertEqual(set(encoder.imputation_values), set(END_FEATURES))
                if isinstance(encoder, GAMEncoder):
                    self.assertTrue(all(item["known_training_events"] == 6
                                        for item in encoder.continuous_audit))

    def test_encoder_imputation_is_from_train_never_holdout(self):
        dates = ["2017-08-26", "2018-08-18", "2019-08-03", "2020-08-29",
                 "2021-10-16", "2022-07-30", "2023-07-12"]
        train, _, _ = self.attach(dates)
        holdout, _, _ = self.attach(["2026-08-15"])
        feature = END_FEATURES[0]
        for encoder_type in (SharedEncoder, GAMEncoder):
            encoder = encoder_type().fit(train, END_FEATURES)
            before = copy.deepcopy(encoder.imputation_values)
            self.assertEqual(before[feature], float(train[feature].median()))
            self.assertEqual(before[feature], 0.0)
            holdout[feature] = .99
            holdout.loc[holdout.index[0], feature] = np.nan
            self.assertEqual(encoder._numeric_values(holdout).iloc[0][feature], before[feature])
            self.assertTrue(np.isfinite(encoder.transform(holdout)).all())
            self.assertEqual(encoder.imputation_values, before)

    def test_encoder_rejects_nonfinite_out_of_range_and_arbitrary_features(self):
        dates = ["2017-08-26", "2019-08-03", "2020-08-29", "2021-10-16",
                 "2022-07-30", "2023-07-12"]
        train, _, _ = self.attach(dates)
        for encoder_type in (SharedEncoder, GAMEncoder):
            for invalid in (-.01, 1.01, np.inf):
                with self.subTest(encoder=encoder_type.__name__, invalid=invalid):
                    changed = train.copy()
                    changed[END_FEATURES[0]] = invalid
                    with self.assertRaises(ValueError):
                        encoder_type().fit(changed, END_FEATURES)
            changed = train.copy()
            changed["actual_end_or_unapproved_signal"] = 20.0
            with self.assertRaisesRegex(ValueError, "Unapproved"):
                encoder_type().fit(changed, ("actual_end_or_unapproved_signal",))
            fitted = encoder_type().fit(train, END_FEATURES)
            changed = train.copy()
            changed[END_FEATURES[0]] = 1.01
            with self.assertRaises(ValueError):
                fitted.transform(changed)


if __name__ == "__main__":
    unittest.main()
