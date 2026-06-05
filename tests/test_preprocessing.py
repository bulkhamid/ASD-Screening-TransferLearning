import unittest
import uuid
from pathlib import Path

import numpy as np

from data.save_numpy_dataset import array_hash, crawl_videos, file_hash, load_video
from data.video_utils import resize_with_letterbox
from tools.gradcam_dashboard import overlay_cam

def workspace_case_dir(name):
    path = Path("tmp_tests_workspace") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


class PreprocessingHelperTests(unittest.TestCase):
    def test_file_hash_reads_full_file(self):
        tmpdir = workspace_case_dir("file_hash")
        path = tmpdir / "sample.bin"
        path.write_bytes(b"abc" + b"x" * (1024 * 1024 + 7))

        digest1 = file_hash(path)
        digest2 = file_hash(path)

        self.assertEqual(digest1, digest2)
        self.assertEqual(len(digest1), 40)

    def test_array_hash_depends_on_array_content(self):
        first = np.zeros((2, 2), dtype=np.uint8)
        second = np.ones((2, 2), dtype=np.uint8)

        self.assertNotEqual(array_hash(first), array_hash(second))

    def test_unreadable_video_is_excluded_by_default(self):
        tmpdir = workspace_case_dir("unreadable_exclude")
        path = tmpdir / "bad.webm"
        path.write_bytes(b"not a real video")

        video, meta = load_video(
            path,
            max_frames=3,
            target_size=(8, 8),
            unreadable_policy="exclude",
            pad_value=0,
        )

        self.assertIsNone(video)
        self.assertEqual(meta["status"], "excluded_unreadable")
        self.assertEqual(meta["decoded_frames"], 0)

    def test_unreadable_video_can_be_zero_filled_explicitly(self):
        tmpdir = workspace_case_dir("unreadable_zeros")
        path = tmpdir / "bad.webm"
        path.write_bytes(b"not a real video")

        video, meta = load_video(
            path,
            max_frames=3,
            target_size=(8, 8),
            unreadable_policy="zeros",
            pad_value=0,
        )

        self.assertEqual(meta["status"], "unreadable_zero_filled")
        self.assertEqual(video.shape, (3, 8, 8, 3))
        self.assertTrue(np.all(video == 0))

    def test_crawl_videos_requires_positive_and_negative_folders(self):
        root = workspace_case_dir("crawl_missing_negative")
        (root / "positive").mkdir()

        with self.assertRaises(FileNotFoundError):
            crawl_videos(root)

    def test_resize_with_letterbox_preserves_aspect_ratio(self):
        frame = np.full((72, 128, 3), 255, dtype=np.uint8)

        resized = resize_with_letterbox(frame, (112, 112), pad_value=128)

        self.assertEqual(resized.shape, (112, 112, 3))
        content_rows = np.where(resized[:, :, 0].max(axis=1) > 128)[0]
        content_cols = np.where(resized[:, :, 0].max(axis=0) > 128)[0]
        self.assertEqual(len(content_cols), 112)
        self.assertEqual(len(content_rows), 63)

    def test_overlay_cam_does_not_blend_padding_when_masked(self):
        frame = np.full((8, 8, 3), 128, dtype=np.uint8)
        frame[2:6, 2:6] = 255
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        cam = np.ones((8, 8), dtype=np.float32)

        overlay = overlay_cam(frame, cam, content_mask=mask)

        self.assertTrue(np.all(overlay[~mask] == 128))
        self.assertFalse(np.array_equal(overlay[mask], frame[mask]))


if __name__ == "__main__":
    unittest.main()
