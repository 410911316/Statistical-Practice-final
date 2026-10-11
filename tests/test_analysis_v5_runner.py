"""Output boundaries and prespecified v5 design must survive configuration edits."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_analysis_v5 import validate_config


class ConfigurationBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/analysis_v5_both.json").read_text(encoding="utf-8-sig"))

    def test_prespecified_design_and_private_predictions(self):
        public, private = validate_config(self.config)
        self.assertTrue(public.is_relative_to(ROOT / "result/analysis_v5"))
        self.assertTrue(private.is_relative_to(ROOT / ".local/analysis_v5"))
        for field, bad in (("include_historical_baseline", True),
                           ("station_hour_interaction", False),
                           ("d1_cutoff_hour", 19), ("fusion_top_k", 3),
                           ("linear_design", "forward"),
                           ("output_dir", "result/analysis_v4/both"),
                           ("private_output_dir", "result/analysis_v5/predictions")):
            changed = copy.deepcopy(self.config)
            changed[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_config(changed)


if __name__ == "__main__":
    unittest.main()
