import unittest

import numpy as np

from experiments.review_clip_overlap import overlap_evidence


class ClipOverlapTests(unittest.TestCase):
    def test_near_identical_encoded_copy_is_detected(self):
        left = np.arange(5 * 16 * 4 * 4 * 3, dtype=np.uint8).reshape(5, 16, 4, 4, 3)
        right = left.copy()
        right[4, :5, 0, 0, 0] ^= 1
        result = overlap_evidence(left, right)
        self.assertTrue(result["verified_overlap"])
        self.assertEqual(result["identical_frames"], 75)

    def test_shared_intro_alone_does_not_trigger_exclusion(self):
        left = np.zeros((5, 16, 4, 4, 3), dtype=np.uint8)
        right = left.copy()
        right[1:] = 100
        self.assertFalse(overlap_evidence(left, right)["verified_overlap"])


if __name__ == "__main__":
    unittest.main()
