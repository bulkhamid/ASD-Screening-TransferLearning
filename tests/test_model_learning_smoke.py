import unittest

import torch
import torch.nn as nn

from src.models.pretrained_r3d import PretrainedR3D
from src.models.simple2dcnn import Simple2DCNN
from src.models.simple3dlstm_v2 import Simple3DLSTMV2


class ModelLearningSmokeTests(unittest.TestCase):
    def train_one_batch(self, model):
        model.train()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        before = [parameter.detach().clone() for parameter in trainable]
        optimizer = torch.optim.SGD(trainable, lr=1e-3)
        inputs = torch.randn(2, 3, 4, 32, 32)
        labels = torch.tensor([0.0, 1.0])
        loss = nn.BCEWithLogitsLoss()(model(inputs), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, trainable)))

    def test_simple2dcnn_learns_one_batch(self):
        self.train_one_batch(Simple2DCNN(p_drop=0.0))

    def test_simple3dlstm_v2_learns_one_batch(self):
        self.train_one_batch(Simple3DLSTMV2(p_drop=0.0))

    def test_r3d_head_learns_one_batch(self):
        model = PretrainedR3D(pretrained=False)
        model.configure_stage("head")
        self.train_one_batch(model)


if __name__ == "__main__":
    unittest.main()
