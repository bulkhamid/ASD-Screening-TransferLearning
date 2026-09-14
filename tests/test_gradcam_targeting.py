import unittest

import numpy as np
import torch
import torch.nn as nn

from tools.gradcam_dashboard import GradCAM, tensor_from_frames


class TinyVideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(3, 1, kernel_size=1, bias=False)
        nn.init.constant_(self.conv.weight, 1.0)

    def forward(self, x):
        return self.conv(x).mean(dim=(1, 2, 3, 4))


class GradCamTargetingTests(unittest.TestCase):
    def test_positive_and_negative_targets_use_opposite_evidence(self):
        model = TinyVideoModel()
        runner = GradCAM(model, model.conv)
        frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        frames[:, 2:6, 2:6] = 255
        x = tensor_from_frames(frames, mean=(0, 0, 0), std=(1, 1, 1))

        _positive_logit, positive_cam = runner(x, "positive")
        _negative_logit, negative_cam = runner(x, "negative")
        runner.close()

        self.assertGreater(float(positive_cam.sum()), 0.0)
        self.assertEqual(float(negative_cam.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
