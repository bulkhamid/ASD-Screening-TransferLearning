import unittest

from tools.model_inspection_app import choose_samples


class ModelInspectionAppTests(unittest.TestCase):
    def test_choose_samples_is_reproducible(self):
        rows = [{"path": f"video_{i}.mp4", "label": str(i % 2)} for i in range(10)]

        first = choose_samples(rows, videos=4, frames_per_video=3, max_frames=60, seed=123)
        second = choose_samples(rows, videos=4, frames_per_video=3, max_frames=60, seed=123)

        self.assertEqual(first, second)

    def test_choose_samples_limits_video_and_frame_counts(self):
        rows = [{"path": f"video_{i}.mp4", "label": str(i % 2)} for i in range(3)]

        samples = choose_samples(rows, videos=10, frames_per_video=5, max_frames=4, seed=7)

        self.assertEqual(len(samples), 3)
        self.assertTrue(all(len(sample["frames"]) == 4 for sample in samples))
        self.assertTrue(all(0 <= frame < 4 for sample in samples for frame in sample["frames"]))


if __name__ == "__main__":
    unittest.main()
