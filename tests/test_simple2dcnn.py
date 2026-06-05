import unittest

import torch

from src.models.simple2dcnn import Simple2DCNN


class Simple2DCNNTests(unittest.TestCase):
    def test_forward_returns_one_logit_per_video(self):
        model = Simple2DCNN(p_drop=0.0)
        x = torch.randn(2, 3, 4, 112, 112)

        y = model(x)

        self.assertEqual(tuple(y.shape), (2,))


if __name__ == "__main__":
    unittest.main()
