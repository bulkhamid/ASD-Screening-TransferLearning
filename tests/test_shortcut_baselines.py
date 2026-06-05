import unittest

from experiments.shortcut_baselines import path_text_features


class ShortcutBaselineTests(unittest.TestCase):
    def test_path_text_features_exclude_label_folder(self):
        rows = {
            "train": [
                {
                    "path": r"C:\dataset\positive\20240101010101.webm",
                    "split": "train",
                }
            ],
            "val": [
                {
                    "path": r"C:\dataset\negative\20240202020202.mp4",
                    "split": "val",
                }
            ],
            "test": [
                {
                    "path": r"C:\dataset\positive\child_video_20240303.avi",
                    "split": "test",
                }
            ],
        }
        texts = path_text_features(rows)

        all_text = " ".join(texts["train"] + texts["val"] + texts["test"])
        self.assertNotIn("positive", all_text)
        self.assertNotIn("negative", all_text)
        self.assertIn("webm", texts["train"][0])
        self.assertIn("20240101010101", texts["train"][0])


if __name__ == "__main__":
    unittest.main()
