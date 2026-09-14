# src/models/pretrained_r3d.py
import torch.nn as nn
import torchvision.models as models
from torchvision.models.video import R3D_18_Weights

class PretrainedR3D(nn.Module):
    def __init__(self, pretrained=True, freeze_batchnorm=True):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        backbone = models.video.r3d_18(weights=weights)
        for p in backbone.parameters():
            p.requires_grad = False
        # final head now outputs *1 logit* instead of a probability
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.net = backbone
        self.freeze_batchnorm = freeze_batchnorm

    def configure_stage(self, stage):
        """Select trainable parameters while keeping BatchNorm frozen."""
        if stage not in {"head", "layer4", "full"}:
            raise ValueError(f"Unknown R3D stage: {stage}")
        for parameter in self.net.parameters():
            parameter.requires_grad_(stage == "full")
        if stage == "head":
            for parameter in self.net.fc.parameters():
                parameter.requires_grad_(True)
        elif stage == "layer4":
            for name, parameter in self.net.named_parameters():
                parameter.requires_grad_(name.startswith("layer4") or name.startswith("fc"))
        if self.freeze_batchnorm:
            for module in self.net.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_batchnorm:
            for module in self.net.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, x):
        # returns raw logits
        return self.net(x).view(-1)

