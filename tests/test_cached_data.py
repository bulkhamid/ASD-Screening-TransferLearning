import hashlib
import unittest
from pathlib import Path

import numpy as np


DATA_DIR = Path("data")


def split_path(split):
    return DATA_DIR / f"{split}.npz"


@unittest.skipUnless(
    all(split_path(split).exists() for split in ("train", "val", "test")),
    "cached npz files are not present",
)
class CachedDataIntegrityTests(unittest.TestCase):
    def test_cached_splits_have_expected_shape_and_labels(self):
        for split in ("train", "val", "test"):
            with self.subTest(split=split):
                npz = np.load(split_path(split))
                X = npz["X"]
                y = npz["y"]

                self.assertEqual(X.ndim, 5)
                self.assertEqual(X.shape[1:], (60, 112, 112, 3))
                self.assertEqual(X.dtype, np.uint8)
                self.assertEqual(y.shape, (X.shape[0],))
                self.assertTrue(set(np.unique(y)).issubset({0, 1}))
                self.assertGreater(int(y.sum()), 0)
                self.assertGreater(len(y) - int(y.sum()), 0)

    def test_no_all_zero_cached_videos(self):
        for split in ("train", "val", "test"):
            with self.subTest(split=split):
                X = np.load(split_path(split))["X"]
                all_zero = sum(np.all(X[i] == 0) for i in range(len(X)))

                self.assertEqual(all_zero, 0)

    def test_no_exact_decoded_duplicates_across_splits(self):
        hashes = {}
        for split in ("train", "val", "test"):
            X = np.load(split_path(split))["X"]
            hashes[split] = {
                hashlib.sha1(np.ascontiguousarray(X[i]).tobytes()).hexdigest()
                for i in range(len(X))
            }

        self.assertFalse(hashes["train"] & hashes["val"])
        self.assertFalse(hashes["train"] & hashes["test"])
        self.assertFalse(hashes["val"] & hashes["test"])

    def test_split_manifest_matches_cached_count_when_present(self):
        manifest = DATA_DIR / "split_manifest.csv"
        if not manifest.exists():
            self.skipTest("split manifest is not present")

        import csv

        counts = {"train": 0, "val": 0, "test": 0}
        with manifest.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                counts[row["split"]] += 1

        for split in ("train", "val", "test"):
            with self.subTest(split=split):
                X = np.load(split_path(split))["X"]
                self.assertEqual(counts[split], X.shape[0])


if __name__ == "__main__":
    unittest.main()
