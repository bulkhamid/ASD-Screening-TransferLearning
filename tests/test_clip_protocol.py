import unittest
from unittest.mock import Mock

import numpy as np
import torch

from data.clip_cache import deterministic_clip_indices, _read_frame
from src.utils.subject_data import ClipTransform, KINETICS_MEAN, KINETICS_STD


class ClipProtocolTests(unittest.TestCase):
    def test_substituted_frame_records_its_actual_previous_index(self):
        capture = Mock()
        capture.read.return_value = (False, None)
        previous = np.zeros((4, 4, 3), dtype=np.uint8)
        frame, actual, substituted = _read_frame(capture, 100, (previous, 98))
        self.assertTrue(substituted)
        self.assertEqual(actual, 98)
        np.testing.assert_array_equal(frame, previous)

    def test_clip_indices_are_deterministic_uniform_and_15_fps(self):
        first = deterministic_clip_indices(2700, 30.0)
        second = deterministic_clip_indices(2700, 30.0)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertTrue(all(len(clip) == 16 for clip in first))
        self.assertTrue(all(np.diff(clip).tolist() == [2] * 15 for clip in first))
        self.assertEqual(first[0][0], 0)
        self.assertEqual(first[-1][-1], 2699)

    def test_fractional_native_rate_is_resampled_near_15_fps(self):
        indices = deterministic_clip_indices(2500, 25.0, clips_per_video=1)[0]

        self.assertEqual(indices[:6], [0, 2, 3, 5, 7, 8])
        self.assertEqual(indices[-1], 25)

    def test_training_augmentation_is_clip_consistent(self):
        frame = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        clip = np.repeat(frame[None], 16, axis=0)
        mask = np.ones((16, 32, 32), dtype=bool)
        transform = ClipTransform((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), training=True)

        output = transform(clip, mask, sample_seed=19)

        self.assertTrue(torch.equal(output[:, 0], output[:, 15]))
        self.assertTrue(torch.equal(output, transform(clip, mask, sample_seed=19)))

    def test_padding_is_zero_after_normalization(self):
        clip = np.full((16, 16, 16, 3), 128, dtype=np.uint8)
        mask = np.zeros((16, 16, 16), dtype=bool)
        mask[:, 4:12, 2:14] = True
        transform = ClipTransform(KINETICS_MEAN, KINETICS_STD, training=False)

        output = transform(clip, mask)
        temporal_first = output.permute(1, 0, 2, 3)
        expanded_mask = torch.from_numpy(mask).unsqueeze(1).expand_as(temporal_first)
        output_padding = temporal_first[~expanded_mask]

        self.assertTrue(torch.all(output_padding == 0))

    def test_r3d_content_uses_kinetics_normalization(self):
        pixel = np.asarray([110, 100, 90], dtype=np.uint8)
        clip = np.broadcast_to(pixel, (16, 4, 4, 3)).copy()
        mask = np.ones((16, 4, 4), dtype=bool)

        output = ClipTransform(KINETICS_MEAN, KINETICS_STD, training=False)(clip, mask)
        expected = (pixel.astype(float) / 255.0 - np.asarray(KINETICS_MEAN)) / np.asarray(KINETICS_STD)

        np.testing.assert_allclose(output[:, 0, 0, 0].numpy(), expected, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
