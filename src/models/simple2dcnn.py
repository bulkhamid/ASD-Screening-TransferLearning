import torch
import torch.nn as nn


class Simple2DCNN(nn.Module):
    """Frame-level 2D CNN with temporal average pooling.

    Input shape is (B, C, T, H, W). Each frame is processed independently by
    Conv2D layers, then frame features are averaged across time.
    """

    def __init__(self, p_drop: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p_drop),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p_drop),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        batch, channels, frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        features = self.cnn(x).flatten(1)
        features = features.reshape(batch, frames, -1).mean(dim=1)
        return self.head(features).view(-1)
