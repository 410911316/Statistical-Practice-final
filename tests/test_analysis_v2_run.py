"""Keep version comparisons reproducible and protect their original run."""
from argparse import Namespace
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_analysis import ROOT, resolve_run_paths, verify_comparison_inputs


class VersionedRunTest(unittest.TestCase):
    def test_new_config_preserves_v1_destinations_and_candidate_controls(self):
        config = json.loads((ROOT / "configs/analysis_v2.json").read_text(encoding="utf-8"))
        old = json.loads((ROOT / "configs/analysis_v1.json").read_text(encoding="utf-8"))
        output, private, reference, manifest = resolve_run_paths(Namespace(output=None, private_output=None), config)
        self.assertEqual(output, ROOT / "result/analysis_v2")
        self.assertEqual(private, ROOT / ".local/analysis_v2")
        self.assertNotEqual(reference.parent, private)
        self.assertNotEqual(manifest.parent, output)
        self.assertEqual(len(config["xgb_grid"]), 36)
        for key in ("outer_years", "ridge_lambdas", "candidate_groups", "seed"):
            self.assertEqual(config[key], old[key])
        for candidate in old["xgb_grid"]:
            effective = {**old["xgb_fixed"], **candidate}
            self.assertIn(effective, [{**config["xgb_fixed"], **c} for c in config["xgb_grid"]])

    def test_overwrite_reference_is_rejected(self):
        config = json.loads((ROOT / "configs/analysis_v2.json").read_text(encoding="utf-8"))
        for args in (Namespace(output=str(ROOT / "result/analysis_v1"), private_output=None),
                     Namespace(output=None, private_output=str(ROOT / ".local/analysis_v1"))):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "overwrite"):
                resolve_run_paths(args, config)

    def test_input_revision_cannot_be_mistaken_for_a_model_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "pred.csv"
            predictions.write_text("dummy", encoding="utf-8")
            metadata = root / "manifest.json"
            data = {"manifest": [{"path": "source.csv", "sha256": "original_hash"}]}
            metadata.write_text(json.dumps({"data": data, "analysis_version": "old"}), encoding="utf-8")
            result = verify_comparison_inputs(predictions, metadata, data)
            self.assertTrue(result["same_input_hashes"])
            self.assertEqual(len(result["reference_predictions_sha256"]), 64)
            with self.assertRaisesRegex(ValueError, "source hashes differ"):
                verify_comparison_inputs(predictions, metadata, {"manifest": [{"path": "source.csv", "sha256": "revised_hash"}]})


if __name__ == "__main__":
    unittest.main()
