import csv
import random
import unittest
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

from src.utils.subject_data import (
    ClipTransform,
    SubjectBalancedClipDataset,
    assert_group_isolation,
    audit_clip_cache,
    make_subject_folds,
    shuffled_training_labels,
    split_inner_subjects,
)


def cache_dir(name):
    path = Path("tmp_tests_workspace") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def write_cached_video(root, video_id, value, subject, label, session=None, shape=(5, 16, 8, 8, 3)):
    clips = np.full(shape, value, dtype=np.uint8)
    masks = np.ones(shape[:-1], dtype=bool)
    clip_path = Path("clips") / f"{video_id}.npy"
    mask_path = Path("masks") / f"{video_id}.npy"
    (root / clip_path).parent.mkdir(exist_ok=True)
    (root / mask_path).parent.mkdir(exist_ok=True)
    np.save(root / clip_path, clips)
    np.save(root / mask_path, np.packbits(masks, axis=-1))
    return {
        "video_id": video_id,
        "label": str(label),
        "subject_id": subject,
        "session_id": session or f"session-{subject}",
        "cache_path": clip_path.as_posix(),
        "mask_path": mask_path.as_posix(),
    }


class SubjectDataTests(unittest.TestCase):
    def test_inner_split_retains_both_classes_under_extreme_imbalance(self):
        for counts in ((2, 20), (20, 2), (2, 2), (3, 80)):
            rows = [
                {"subject_id": f"{label}-{index}", "label": str(label)}
                for label, count in enumerate(counts) for index in range(count)
            ]
            subjects = {row["subject_id"] for row in rows}
            for seed in range(10):
                train, val = split_inner_subjects(rows, subjects, seed=seed)
                self.assertFalse(train & val)
                self.assertEqual(train | val, subjects)
                for partition in (train, val):
                    self.assertEqual({int(row["label"]) for row in rows if row["subject_id"] in partition}, {0, 1})
                self.assertEqual((train, val), split_inner_subjects(rows, subjects, seed=seed))

    def test_folds_are_subject_isolated_and_stratified(self):
        rows = []
        for label in (0, 1):
            for index in range(3):
                rows.append(
                    {
                        "subject_id": f"s{label}-{index}",
                        "session_id": f"session-{label}-{index}",
                        "label": str(label),
                    }
                )

        folds = make_subject_folds(rows, max_folds=5, split_seed=42)

        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertFalse(fold["outer_train"] & fold["outer_test"])
            test_labels = [int(row["label"]) for row in rows if row["subject_id"] in fold["outer_test"]]
            self.assertEqual(Counter(test_labels), Counter({0: 1, 1: 1}))

    def test_session_overlap_is_rejected(self):
        rows = [
            {"subject_id": "a", "session_id": "same", "label": "0"},
            {"subject_id": "b", "session_id": "same", "label": "1"},
        ]
        with self.assertRaisesRegex(ValueError, "Session overlap"):
            assert_group_isolation(rows, {"a"}, set(), {"b"})

    def test_subject_balancing_draws_equal_samples(self):
        root = cache_dir("balanced")
        rows = [
            write_cached_video(root, "v0", 20, "s0", 0),
            write_cached_video(root, "v1", 220, "s1", 1),
        ]
        transform = ClipTransform((0.5,) * 3, (0.25,) * 3, training=False)
        dataset = SubjectBalancedClipDataset(rows, root, transform, samples_per_subject=4)

        labels = [int(dataset[index][1]) for index in range(len(dataset))]

        self.assertEqual(Counter(labels), Counter({0: 4, 1: 4}))

    def test_session_is_selected_before_video(self):
        root = cache_dir("session_balance")
        rows = [write_cached_video(root, "a", 10, "subject", 0, session="short")]
        rows.extend(
            write_cached_video(root, f"b{index}", 20 + index, "subject", 0, session="long")
            for index in range(4)
        )
        dataset = SubjectBalancedClipDataset(
            rows,
            root,
            ClipTransform((0.5,) * 3, (0.25,) * 3, training=False),
        )

        selected_sessions = Counter(
            dataset.sample_row("subject", random.Random(seed))["session_id"]
            for seed in range(1000)
        )

        self.assertLess(abs(selected_sessions["short"] - selected_sessions["long"]), 100)

    def test_shuffle_changes_training_subjects_only(self):
        rows = [
            {"subject_id": "a", "label": "0"},
            {"subject_id": "b", "label": "0"},
            {"subject_id": "c", "label": "1"},
            {"subject_id": "d", "label": "1"},
            {"subject_id": "held-out", "label": "1"},
        ]

        shuffled = shuffled_training_labels(rows, {"a", "b", "c", "d"}, seed=3)

        self.assertEqual(set(shuffled), {"a", "b", "c", "d"})
        self.assertEqual(Counter(shuffled.values()), Counter({0: 2, 1: 2}))
        self.assertNotIn("held-out", shuffled)

    def test_integrity_rejects_exact_cross_subject_duplicates(self):
        root = cache_dir("duplicates")
        shape = (5, 16, 112, 112, 3)
        rows = [
            write_cached_video(root, "v0", 40, "s0", 0, shape=shape),
            write_cached_video(root, "v1", 40, "s1", 1, shape=shape),
        ]

        with self.assertRaisesRegex(ValueError, "cross subject"):
            audit_clip_cache(rows, root)


if __name__ == "__main__":
    unittest.main()
