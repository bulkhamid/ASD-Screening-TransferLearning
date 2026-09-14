import unittest

import numpy as np

from experiments.perceptual_duplicate_audit import nearest_cross_subject_pairs


class PerceptualDuplicateAuditTests(unittest.TestCase):
    def test_nearest_pairs_exclude_same_subject(self):
        embeddings = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.99, 0.01]], dtype=np.float32
        )
        metadata = [
            {"item_index": 0, "subject_id": "same"},
            {"item_index": 1, "subject_id": "same"},
            {"item_index": 2, "subject_id": "other"},
        ]

        pairs = nearest_cross_subject_pairs(embeddings, metadata, top_k=3)

        self.assertTrue(pairs)
        self.assertTrue(
            all(pair["left"]["subject_id"] != pair["right"]["subject_id"] for pair in pairs)
        )


if __name__ == "__main__":
    unittest.main()
