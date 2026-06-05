import unittest
import uuid
from pathlib import Path

import numpy as np
import torch

from src.utils.data_cached import _jitter_padding, load_split


class DataCachedAugmentationTests(unittest.TestCase):
    def test_jitter_padding_changes_only_padding_pixels(self):
        video = torch.zeros(3, 2, 4, 4)
        video[:, :, :, :] = 0.5
        video[:, :, 1:3, 1:3] = 1.0

        jittered = _jitter_padding(video, pad_value=128, jitter=16)

        self.assertTrue(torch.all(jittered[:, :, 1:3, 1:3] == 1.0))
        self.assertFalse(torch.equal(jittered[:, :, 0, 0], video[:, :, 0, 0]))

    def test_load_split_respects_data_dir(self):
        tmp_path = Path("tmp_tests_workspace") / f"load_split_{uuid.uuid4().hex}"
        tmp_path.mkdir(parents=True, exist_ok=True)
        X = np.zeros((2, 3, 4, 5, 3), dtype=np.uint8)
        X[1] = 255
        y = np.array([0, 1], dtype=np.uint8)
        np.savez_compressed(tmp_path / "train.npz", X=X, y=y)

        ds = load_split("train", data_dir=tmp_path)

        self.assertEqual(len(ds), 2)
        video, label = ds[1]
        self.assertEqual(tuple(video.shape), (3, 3, 4, 5))
        self.assertTrue(torch.allclose(video, torch.ones_like(video)))
        self.assertEqual(float(label), 1.0)


if __name__ == "__main__":
    unittest.main()
