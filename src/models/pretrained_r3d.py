# src/models/pretrained_r3d.py
import torch.nn as nn
import torchvision.models as models
import torch
from torchvision.models.video import R3D_18_Weights

class PretrainedR3D(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.video.r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        for p in backbone.parameters():
            p.requires_grad = False
        # final head now outputs *1 logit* instead of a probability
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.net = backbone

    def forward(self, x):
        # returns raw logits
        return self.net(x).view(-1)

