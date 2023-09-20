import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvBlock()
        self.classifier = nn.Linear(56448, 32)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )

    def forward(self, x):
        return self.encoder(x)
