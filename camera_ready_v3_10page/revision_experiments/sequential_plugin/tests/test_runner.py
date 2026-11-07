from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def test_platform_replay_help_does_not_load_data_or_fit_models(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "investigate_platform_variation.py"), "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--prepared", result.stdout)

    def test_preparation_help_does_not_read_raw_data(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "prepare_accepted.py"), "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--source", result.stdout)
        self.assertIn("--check-only", result.stdout)

    def test_preparation_missing_source_has_actionable_preflight_error(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "prepare_accepted.py"), "--source",
             str(ROOT / "missing.csv.gz"), "--check-only"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Accepted source missing", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_missing_input_is_reported_without_starting_fits_or_creating_output(self):
        self.assertTrue((ROOT / "run.py").exists(), "isolated runner is not implemented")
        result = subprocess.run(
            [sys.executable, str(ROOT / "run.py"), "--prepared", str(ROOT / "missing.parquet"), "--check-only"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("prepared cohort missing", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_output_cannot_escape_isolated_results_directory(self):
        self.assertTrue((ROOT / "run.py").exists(), "isolated runner is not implemented")
        result = subprocess.run(
            [sys.executable, str(ROOT / "run.py"), "--prepared", "missing.parquet", "--output", str(ROOT.parent)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Output must be inside", result.stderr)


if __name__ == "__main__":
    unittest.main()
