import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        self.encoder = ConvBlock()
        self.classifier = nn.Linear()

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.MaxPool2d()
        )

    def forward(self, x):
        return self.encoder(x)
