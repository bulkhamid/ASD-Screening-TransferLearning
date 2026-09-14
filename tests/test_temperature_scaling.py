import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.temperature_scaling import TemperatureScaler


class IdentityLogitModel(torch.nn.Module):
    def forward(self, inputs):
        return inputs[:, 0]


class TemperatureScalingTests(unittest.TestCase):
    def test_temperature_is_bounded_and_does_not_worsen_calibration_nll(self):
        logits = torch.tensor([[8.0], [5.0], [-4.0], [-7.0]])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)

        scaler = TemperatureScaler(IdentityLogitModel()).set_temperature(loader, "cpu")

        self.assertGreaterEqual(float(scaler.temperature), 0.05)
        self.assertLessEqual(float(scaler.temperature), 10.0)
        self.assertLessEqual(scaler.nll_after, scaler.nll_before + 1e-8)


if __name__ == "__main__":
    unittest.main()
