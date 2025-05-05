# src/models/pretrained_r3d.py
import torch.nn as nn
import torchvision.models as models
import torch

class PretrainedR3D(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.video.r3d_18(weights=True)
        for p in backbone.parameters():
            p.requires_grad = False
        # final head now outputs *1 logit* instead of a probability
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.net = backbone

    def forward(self, x):
        # returns raw logits
        return self.net(x).view(-1)

