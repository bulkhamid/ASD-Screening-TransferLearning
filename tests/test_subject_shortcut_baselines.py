import unittest

import numpy as np

from experiments.subject_shortcut_baselines import evaluate_family
from src.utils.subject_data import make_subject_folds


class SubjectShortcutBaselineTests(unittest.TestCase):
    def test_predictions_are_out_of_fold_at_subject_level(self):
        rows = []
        features = []
        for label in (0, 1):
            for index in range(3):
                subject = f"subject-{label}-{index}"
                rows.append(
                    {
                        "video_id": f"video-{subject}",
                        "session_id": f"session-{subject}",
                        "subject_id": subject,
                        "label": str(label),
                    }
                )
                features.append([float(label * 2 - 1), float(index)])
        folds = make_subject_folds(rows, max_folds=3, split_seed=42)

        subject_rows, metrics = evaluate_family(
            np.asarray(features), rows, folds, seed=42
        )

        self.assertEqual(len(subject_rows), 6)
        self.assertEqual(metrics["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
