from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from custom_dataset import TrafficSignDatset

class CNNModule(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate=1e-3):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss = self.loss(output, target)
        self.log("train loss", loss)
        _, pred = torch.max(output, dim=1)
        accuracy = torch.sum(pred==target)/len(target)
        self.log("train accuracy", accuracy.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss = self.loss(output, target)
        self.log("test loss", loss)
        _, pred = torch.max(output, dim=1)
        accuracy = torch.sum(pred==target)/len(target)
        self.log("test accuracy", accuracy.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


class TrafficSignDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage):
        data_paths = self.get_train_paths(self.data_dir)
        traffic_train_paths, traffic_val_paths = random_split(data_paths, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
        self.traffic_train = TrafficSignDatset(traffic_train_paths)
        self.traffic_val = TrafficSignDatset(traffic_val_paths)

    def train_dataloader(self):
        return DataLoader(self.traffic_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.traffic_val, batch_size=self.batch_size)

    def get_train_paths(self, data_path: Path):
        data_paths = []
        for class_dir in data_path.iterdir():
            class_id = int(class_dir.parts[-1])
            for img in class_dir.iterdir():
                data_paths.append({"label": class_id, "image": img})

        return data_paths
