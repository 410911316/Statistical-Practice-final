"""Behavioral checks for timing, unknowns, primary eligibility and compatibility."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis_v12.data import event_types, read_google, check_variant
from analysis_v12.models import Encoder, GAMEncoder, fit_one
from analysis_v11.models import fit_one as old_fit


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads((ROOT / "docs/d1_event_type_sources.json").read_text("utf-8"))

    def read_mutation(self, doc):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "source.json"
            p.write_text(json.dumps(doc), "utf-8")
            return event_types(p)

    def test_unknown_not_zero(self):
        d = event_types(ROOT / "docs/d1_event_type_sources.json").set_index("event_date")
        self.assertTrue(pd.isna(d.loc["2018-08-18", "planned_finale"]))
        self.assertEqual(d.loc["2026-07-25", "planned_finale"], 0)
        self.assertEqual(d.loc["2026-07-25", "planned_enhanced_nonfinale"], 1)
        self.assertEqual(d.loc["2026-08-15", "planned_finale"], 1)

    def test_late_announcement_rejected(self):
        r = self.doc["records"][0]
        r["published_at"] = "2017-08-25T18:00:01+08:00"
        with self.assertRaisesRegex(ValueError, "after cutoff"):
            self.read_mutation(self.doc)

    def test_late_enhancement_rejected(self):
        r = next(r for r in self.doc["records"] if r["event_date"] == "2025-08-20")
        r["enhancement_published_at"] = "2025-08-20T00:00:00+08:00"
        with self.assertRaisesRegex(ValueError, "after cutoff"):
            self.read_mutation(self.doc)

    def test_bad_binary_rejected(self):
        self.doc["records"][0]["planned_finale"] = 2
        with self.assertRaises(ValueError):
            self.read_mutation(self.doc)

    def test_duplicate_rejected(self):
        self.doc["records"].append(copy.deepcopy(self.doc["records"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.read_mutation(self.doc)

    def test_google_censor_and_partial_year(self):
        g = read_google(ROOT / "data/original_data/google_網路聲量.csv")
        self.assertEqual(len(g), 117)
        self.assertEqual((g.month.dt.year == 2026).sum(), 9)
        c = g.loc[g.censored]
        np.testing.assert_array_equal(c.midpoint, .5)
        np.testing.assert_array_equal(c.lower, 0)
        np.testing.assert_array_equal(c.upper, 1)
        self.assertEqual(g.loc[g.month.eq(pd.Period("2025-08")), "midpoint"].iloc[0], 100)

    def test_unsafe_primary_rejected(self):
        cfg = json.loads((ROOT / "configs/analysis_v12.json").read_text("utf-8"))
        for name in ("retrospective_baseline_56d", "unverified_google_prev_month"):
            with self.assertRaisesRegex(ValueError, "unverified"):
                check_variant("core", [name], cfg)
        check_variant("sensitivity_baseline", ["retrospective_baseline_56d"], cfg)

    def test_outcomes_rejected(self):
        with self.assertRaisesRegex(ValueError, "Post-event"):
            check_variant("core", ["y"], {"sensitivity_only": []})


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(ROOT / ".local/analysis_v8/both/analysis_input.csv", parse_dates=["event_date"])
        typ = event_types(ROOT / "docs/d1_event_type_sources.json")
        cls.frame = cls.frame.merge(typ, on="event_date", validate="many_to_one")
        cls.config = json.loads((ROOT / "configs/analysis_v8_core.json").read_text("utf-8"))
        cls.groups = tuple(cls.config["candidate_groups"])

    def test_unknown_feature_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            Encoder().fit(self.frame, ("observed_has_drone_show",))

    def test_baseline_shortcut_rejected(self):
        with self.assertRaisesRegex(ValueError, "sensitivity"):
            Encoder().fit(self.frame, self.groups, include_historical_baseline=True)

    def test_fit_no_holdout_imputation(self):
        train = self.frame[self.frame.year < 2025]
        groups = self.groups + ("planned_finale", "forecast_temperature_c")
        encoder = Encoder().fit(train, groups)
        before = copy.deepcopy(encoder.imputation_values)
        holdout = self.frame[self.frame.year == 2025].copy()
        holdout["forecast_temperature_c"] = 1e6
        encoder.transform(holdout)
        self.assertEqual(before, encoder.imputation_values)
        self.assertEqual(before["forecast_temperature_c"], train.forecast_temperature_c.median())

    def test_three_models_core_unchanged(self):
        train, valid = self.frame[self.frame.year < 2024], self.frame[self.frame.year == 2024]
        parameters = {"ridge": {"lambda": .01},
                      "gam": {"lambda": 1., "ridge_alpha": .01, "n_splines": 5},
                      "xgboost": {"max_depth": 3, "n_estimators": 50}}
        for model, p in parameters.items():
            with self.subTest(model=model):
                fitted = fit_one(train, model, self.groups, p, self.config)
                old = old_fit(train, model, self.groups, p, self.config)
                np.testing.assert_allclose(fitted.predict(valid), old.predict(valid), rtol=0, atol=1e-8)

    def test_gam_does_not_learn_unsupported_event_type(self):
        train = self.frame[self.frame.year < 2023]
        enc = GAMEncoder().fit(train, self.groups + ("planned_finale",), self.config)
        self.assertNotIn("planned_finale", enc.names)
        self.assertNotIn("planned_finale:missing", enc.names)


if __name__ == "__main__":
    unittest.main()
