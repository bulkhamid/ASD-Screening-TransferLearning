"""Bounded, experimental temperature scaling for raw-logit classifiers."""

import math

import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """Learn a bounded scalar T using an independent calibration loader."""

    def __init__(self, model, minimum=0.05, maximum=10.0):
        super().__init__()
        if not 0 < minimum < 1 < maximum:
            raise ValueError("Temperature bounds must satisfy 0 < minimum < 1 < maximum")
        self.model = model
        self.net = model
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.log_T = nn.Parameter(torch.zeros(1))
        self.nll_before = None
        self.nll_after = None
        self.at_boundary = False

    @property
    def temperature(self):
        return self.log_T.exp().clamp(self.minimum, self.maximum)

    def forward(self, x):
        return self.model(x) / self.temperature

    def set_temperature(self, loader, device="cuda"):
        """Fit T with BCEWithLogitsLoss; use independent calibration subjects."""
        device = torch.device(device)
        self.to(device)
        self.eval()
        criterion = nn.BCEWithLogitsLoss()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                logits_list.append(self.net(inputs).float().view(-1))
                labels_list.append(labels.to(device).float().view(-1))
        if not logits_list:
            raise ValueError("Calibration loader is empty")
        logits = torch.cat(logits_list).detach().cpu()
        labels = torch.cat(labels_list).detach().cpu()

        self.log_T.data = torch.zeros_like(self.log_T.data, device="cpu")
        self.nll_before = float(criterion(logits, labels))
        optimizer = optim.LBFGS(
            [self.log_T], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            temperature = self.log_T.exp().clamp(self.minimum, self.maximum)
            loss = criterion(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.log_T.clamp_(math.log(self.minimum), math.log(self.maximum))
            calibrated = criterion(logits / self.temperature.detach().cpu(), labels)
            self.nll_after = float(calibrated)
            if self.nll_after > self.nll_before + 1e-8:
                self.log_T.zero_()
                self.nll_after = self.nll_before
            value = float(self.temperature.detach().cpu())
            self.at_boundary = (
                abs(value - self.minimum) < 1e-6 or abs(value - self.maximum) < 1e-6
            )
        self.log_T.data = self.log_T.data.to(device)
        print(
            f">> experimental calibration T={value:.4f} "
            f"NLL {self.nll_before:.4f}->{self.nll_after:.4f} boundary={self.at_boundary}"
        )
        return self
