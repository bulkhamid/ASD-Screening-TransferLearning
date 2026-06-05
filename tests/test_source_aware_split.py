import json
import unittest
from argparse import Namespace
from pathlib import Path

from data.save_numpy_dataset import capture_month, split_source_aware, write_split_diagnostics


class SourceAwareSplitTests(unittest.TestCase):
    def sample_rows(self):
        rows = []
        for idx in range(20):
            label = idx % 2
            month = ["2024-07", "2024-08", "2024-09", "2024-10"][idx % 4]
            ext = ".mp4" if idx % 5 == 0 else ".webm"
            rows.append(
                {
                    "path": f"C:/dataset/{'positive' if label else 'negative'}/{month}_{idx}{ext}",
                    "label": label,
                    "source_month": month,
                    "source_extension": ext,
                    "total_frames": 5000 + idx,
                    "failed_reads": 0,
                }
            )
        return rows

    def test_capture_month_from_supported_filenames(self):
        self.assertEqual(capture_month(Path("20240829054410.webm")), "2024-08")
        self.assertEqual(capture_month(Path("video_2024-10-20_03-10-55.mp4")), "2024-10")
        self.assertEqual(capture_month(Path("unknown.webm")), "unknown")

    def test_source_aware_split_is_stratified_and_deterministic(self):
        rows = self.sample_rows()
        labels = [row["label"] for row in rows]
        args = Namespace(seed=7, val_frac=0.10, test_frac=0.20, source_aware_candidates=20)

        first, first_score = split_source_aware(rows, labels, args)
        second, second_score = split_source_aware(rows, labels, args)

        self.assertEqual(first_score, second_score)
        self.assertEqual(
            [[row["path"] for row in split] for split in first],
            [[row["path"] for row in split] for split in second],
        )
        for split in first:
            self.assertEqual({row["label"] for row in split}, {0, 1})

    def test_write_split_diagnostics_reports_confounded_sources(self):
        rows = self.sample_rows()
        for row in rows:
            if row["source_extension"] == ".mp4":
                row["label"] = 1
        output = Path("tmp_tests_workspace") / "source_diag"
        output.mkdir(parents=True, exist_ok=True)

        write_split_diagnostics(
            output,
            "source_aware",
            rows,
            [("train", rows[:14]), ("val", rows[14:16]), ("test", rows[16:])],
            1.23,
        )

        diagnostics = json.loads((output / "split_diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(diagnostics["split_strategy"], "source_aware")
        self.assertTrue(any("class-confounded" in warning for warning in diagnostics["warnings"]))
        self.assertIn("extension_counts", diagnostics["overall"])


if __name__ == "__main__":
    unittest.main()
