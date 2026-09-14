import unittest

from src.utils.subject_metrics import compute_subject_metrics, hierarchical_aggregate


class SubjectMetricsTests(unittest.TestCase):
    def test_single_class_auc_is_undefined(self):
        metrics = compute_subject_metrics([{"subject_id": "s", "label": 1, "logit": 1.0}])
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["pr_auc"])

    def test_empty_and_nonfinite_predictions_are_rejected(self):
        for rows in ([], [{"subject_id": "s", "label": 0, "logit": float("nan")}],
                     [{"subject_id": "s", "label": 1, "logit": float("inf")}],
                     [{"subject_id": "s", "label": 2, "logit": 1.0}]):
            with self.assertRaises(ValueError):
                compute_subject_metrics(rows)

    def test_conflicting_video_identity_is_not_silently_overwritten(self):
        row = {"video_id": "v", "session_id": "session", "subject_id": "a", "label": 1, "logit": 1.0}
        with self.assertRaisesRegex(ValueError, "conflicting"):
            hierarchical_aggregate([row, {**row, "subject_id": "b"}])

    def test_hierarchical_aggregation_weights_sessions_equally(self):
        clips = []
        for video_index in range(3):
            clips.append(
                {
                    "video_id": f"v{video_index}",
                    "session_id": "many-videos",
                    "subject_id": "subject",
                    "label": 1,
                    "logit": 3.0,
                }
            )
        clips.append(
            {
                "video_id": "v3",
                "session_id": "one-video",
                "subject_id": "subject",
                "label": 1,
                "logit": -1.0,
            }
        )

        subjects = hierarchical_aggregate(clips)

        self.assertEqual(len(subjects), 1)
        self.assertAlmostEqual(subjects[0]["logit"], 1.0)
        self.assertEqual(subjects[0]["sessions"], 2)

    def test_auc_uses_logits_even_when_probabilities_would_saturate(self):
        rows = [
            {"subject_id": "negative", "label": 0, "logit": 1000.0},
            {"subject_id": "positive", "label": 1, "logit": 1001.0},
        ]

        metrics = compute_subject_metrics(rows)

        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
