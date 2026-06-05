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
        self.assertIn("--cached --cv is disabled", result.stderr)


if __name__ == "__main__":
    unittest.main()
