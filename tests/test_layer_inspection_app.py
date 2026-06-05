import unittest

from tools.layer_inspection_app import frame_indices, select_rows


class LayerInspectionAppTests(unittest.TestCase):
    def test_select_rows_keeps_split_order_and_limit(self):
        rows = [
            {"split": "test", "path": "test_1"},
            {"split": "train", "path": "train_1"},
            {"split": "val", "path": "val_1"},
            {"split": "train", "path": "train_2"},
        ]

        selected = select_rows(rows, ["train", "val"], videos_per_split=1)

        self.assertEqual([row["path"] for row in selected], ["train_1", "val_1"])

    def test_select_rows_supports_offset(self):
        rows = [
            {"split": "train", "path": "train_1"},
            {"split": "train", "path": "train_2"},
            {"split": "train", "path": "train_3"},
        ]

        selected = select_rows(rows, ["train"], videos_per_split=1, start_index=1)

        self.assertEqual([row["path"] for row in selected], ["train_2"])

    def test_frame_indices_are_reproducible(self):
        first = frame_indices(max_frames=60, frames_per_video=3, seed=42, row_index=7)
        second = frame_indices(max_frames=60, frames_per_video=3, seed=42, row_index=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(0 <= idx < 60 for idx in first))

    def test_single_frame_defaults_to_middle(self):
        self.assertEqual(frame_indices(60, 1, 42, 0), [30])


if __name__ == "__main__":
    unittest.main()
