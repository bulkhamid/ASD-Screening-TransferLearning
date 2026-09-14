"""Lower-capacity 3D-CNN/LSTM for small subject-grouped datasets."""

import torch.nn as nn


class Simple3DLSTMV2(nn.Module):
    def __init__(self, p_drop: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout3d(p_drop),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout3d(p_drop),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout3d(p_drop),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        features = self.cnn(x).mean(dim=(-1, -2))  # B,C,T
        features = features.permute(0, 2, 1)
        sequence, _state = self.lstm(features)
        return self.head(sequence[:, -1]).view(-1)
