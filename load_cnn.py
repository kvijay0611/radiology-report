import torch
import torch.nn as nn
from torchvision import models


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        effnet = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.DEFAULT
        )

        self.backbone = effnet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.project = nn.Linear(1792, 512)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.project(x)
        return x


def load_cnn(device):
    model = CNNEncoder().to(device)

    state_dict = torch.load("cnn_encoder_only.pt", map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model
