"""Prevent a rerun from leaking private predictions or overwriting prior runs."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_analysis_v8 import validate_config


class OutputBoundaryTest(unittest.TestCase):
    def test_ablation_cannot_silently_change_model_grid_or_temporal_controls(self):
        config = json.loads((ROOT / "configs/analysis_v8_both.json").read_text(encoding="utf-8-sig"))
        changes = {
            "gam_min_known_events": 5,
            "end_signal_policy": "choose_peak_from_test_labels",
            "candidate_groups": config["candidate_groups"] + ["actual_show_end"],
            "end_source_file": "docs/event_timing_sources.json",
        }
        for key, value in changes.items():
            changed = copy.deepcopy(config)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(changed)

    def test_private_outputs_cannot_escape_and_old_results_cannot_be_overwritten(self):
        config = json.loads((ROOT / "configs/analysis_v8_both.json").read_text(encoding="utf-8-sig"))
        public, private = validate_config(config)
        self.assertTrue(public.is_relative_to(ROOT / "result/analysis_v8"))
        self.assertTrue(private.is_relative_to(ROOT / ".local/analysis_v8"))
        for field, path in (("output_dir", "result/analysis_v7/both"),
                            ("output_dir", "result/analysis_v8/../analysis_v7/both"),
                            ("private_output_dir", "result/analysis_v8/private"),
                            ("private_output_dir", ".local/analysis_v8/../../result/private")):
            changed = copy.deepcopy(config)
            changed[field] = path
            with self.subTest(field=field, path=path), self.assertRaises(ValueError):
                validate_config(changed)


if __name__ == "__main__":
    unittest.main()
