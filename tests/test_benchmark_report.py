import csv
import json
import unittest
import uuid
from pathlib import Path

from tools.summarize_subject_benchmark import load_subject_predictions, summarize_permutations


class BenchmarkReportTests(unittest.TestCase):
    def write_predictions(self, ids):
        root = Path("tmp_tests_workspace") / f"report_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        path = root / "predictions.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["level", "subject_id", "label", "logit"])
            writer.writeheader()
            for subject in ids:
                writer.writerow(dict(level="subject", subject_id=subject, label=1, logit=2.0))
        return path

    def test_duplicate_predictions_are_rejected_even_with_correct_set(self):
        path = self.write_predictions(["a", "b", "a"])
        with self.assertRaisesRegex(ValueError, "coverage"):
            load_subject_predictions(path, {"a", "b"})

    def test_predictions_from_the_wrong_fold_are_rejected(self):
        path = self.write_predictions(["a", "b"])
        with self.assertRaisesRegex(ValueError, "held-out"):
            load_subject_predictions(path, {"a", "c"})

    def test_incomplete_permutation_suite_is_not_reported_as_complete(self):
        root = Path("tmp_tests_workspace") / f"permutations_{uuid.uuid4().hex}"
        primary, controls, output = (root / name for name in ("primary", "controls", "output"))
        primary.mkdir(parents=True)
        controls.mkdir()
        (primary / "configuration.json").write_text(json.dumps({
            "cache_manifest_sha256": "test", "protocol_config": {},
        }))
        (primary / "fold_assignments.csv").write_text("fold,subject_id,label,role\n")
        with self.assertRaisesRegex(ValueError, "incomplete: 0/7"):
            summarize_permutations(primary, controls, output)
        self.assertFalse((output / "permutation_controls.json").exists())


if __name__ == "__main__":
    unittest.main()
