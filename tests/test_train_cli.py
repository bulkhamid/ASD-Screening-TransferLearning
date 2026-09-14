import os
import subprocess
import sys
import unittest


PYTHON = sys.executable


class TrainCliTests(unittest.TestCase):
    def run_train(self, *args):
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [PYTHON, "train.py", *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=60,
        )

    def test_help_succeeds(self):
        result = self.run_train("--help")

        self.assertEqual(result.returncode, 0)
        self.assertIn("--num_workers", result.stdout)
        self.assertIn("--wandb", result.stdout)

    def test_cached_cv_fails_fast(self):
        result = self.run_train("--model", "r3d", "--cached", "--cv")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--cv is disabled", result.stderr)

    def test_legacy_training_requires_acknowledgement(self):
        result = self.run_train("--model", "simple2dcnn", "--cached")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--legacy_holdout_ack", result.stderr)

    def test_subject_benchmark_help_succeeds(self):
        result = subprocess.run(
            [PYTHON, "-m", "experiments.run_subject_benchmark", "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--validate_only", result.stdout)

    def test_subject_benchmark_fails_cleanly_without_private_cache(self):
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "experiments.run_subject_benchmark",
                "--cache_dir",
                "definitely_missing_grouped_cache",
                "--validate_only",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("cache_config.json", result.stderr)


if __name__ == "__main__":
    unittest.main()
