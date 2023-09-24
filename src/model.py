import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(input_channels=3, output_channels=32),
            ConvBlock(input_channels=32, output_channels=64),
        )
        self.classifier = nn.Linear(10816, 43)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, input_channels=3, output_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(output_channels),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

    def forward(self, x):
        return self.encoder(x)
