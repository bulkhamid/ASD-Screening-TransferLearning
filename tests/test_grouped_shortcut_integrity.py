import csv
import unittest
import uuid
from pathlib import Path

import numpy as np

from data.save_numpy_dataset import array_hash, file_hash
from experiments.grouped_shortcut_baselines import capture_group, read_legacy_data


class GroupedShortcutIntegrityTests(unittest.TestCase):
    def test_missing_date_cannot_become_a_unique_video_group(self):
        with self.assertRaisesRegex(ValueError, "Missing capture date"):
            capture_group("unknown.mp4", "date")

    def test_hash_check_detects_reordered_videos_with_identical_labels(self):
        root = Path("tmp_tests_workspace") / f"shortcut_integrity_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        rows = []
        for split in ("train", "val", "test"):
            raw = root / f"{split}.mp4"
            raw.write_bytes(split.encode())
            videos = np.array([np.full((2, 4, 4, 3), value, dtype=np.uint8) for value in (10, 20)])
            np.savez(root / f"{split}.npz", X=videos, y=np.array([0, 0]))
            for video in videos:
                rows.append(dict(split=split, path=str(raw), label=0,
                                 raw_hash=file_hash(raw), decoded_hash=array_hash(video)))
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        X, labels, _ = read_legacy_data(root, manifest)
        self.assertEqual(len(X), 6)
        self.assertEqual(len(labels), 6)
        with np.load(root / "train.npz") as data:
            reordered = data["X"][::-1].copy()
        np.savez(root / "train.npz", X=reordered, y=np.array([0, 0]))
        with self.assertRaisesRegex(ValueError, "decoded hash"):
            read_legacy_data(root, manifest)


if __name__ == "__main__":
    unittest.main()
