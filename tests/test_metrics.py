import unittest

import numpy as np

from src.utils.metrics import compute_metrics


class ComputeMetricsTests(unittest.TestCase):
    def test_threshold_changes_thresholded_metrics_not_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.10, 0.40, 0.60, 0.90])

        low_threshold = compute_metrics(y_true, y_prob, threshold=0.05)
        default_threshold = compute_metrics(y_true, y_prob, threshold=0.5)

        self.assertEqual(low_threshold["auc"], default_threshold["auc"])
        self.assertEqual(default_threshold["accuracy"], 1.0)
        self.assertLess(low_threshold["accuracy"], default_threshold["accuracy"])
        self.assertLess(low_threshold["precision"], default_threshold["precision"])

    def test_single_class_auc_falls_back_to_chance(self):
        y_true = np.array([1, 1, 1])
        y_prob = np.array([0.20, 0.80, 0.90])

        metrics = compute_metrics(y_true, y_prob)

        self.assertEqual(metrics["auc"], 0.5)


if __name__ == "__main__":
    unittest.main()
