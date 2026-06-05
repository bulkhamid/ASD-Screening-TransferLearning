import unittest

import numpy as np

from experiments.feature_distribution_audit import cohen_d, month_token, summarize_feature_matrix


class FeatureDistributionAuditTests(unittest.TestCase):
    def test_cohen_d_positive_minus_negative(self):
        self.assertGreater(cohen_d([3, 4, 5], [0, 1, 2]), 1.0)
        self.assertLess(cohen_d([0, 1, 2], [3, 4, 5]), -1.0)

    def test_month_token_handles_supported_names(self):
        self.assertEqual(month_token(r"C:\dataset\positive\20240829054410.webm"), "2024-08")
        self.assertEqual(month_token(r"C:\dataset\positive\video_2024-10-20_03-10-55.mp4"), "2024-10")
        self.assertEqual(month_token(r"C:\dataset\positive\unknown.webm"), "unknown")

    def test_summarize_feature_matrix_sorts_by_effect_size(self):
        X = np.array(
            [
                [10.0, 1.0],
                [11.0, 2.0],
                [0.0, 1.5],
                [1.0, 1.6],
            ]
        )
        y = np.array([1, 1, 0, 0])
        rows = summarize_feature_matrix(X, y, ["large_gap", "small_gap"])
        self.assertEqual(rows[0]["feature"], "large_gap")


if __name__ == "__main__":
    unittest.main()
