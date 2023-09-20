from pathlib import Path

import lightning.pytorch as pl

from lightning_modules import CNNModule, TrafficSignDataModule
from model import CNN

def main():
    data_path = Path("/home/lukasz/traffic_sign_recognition/data/preprocessed")

    data_module = TrafficSignDataModule(data_dir=data_path)
    model = CNN()
    cnn_module = CNNModule(model)

    trainer = pl.Trainer(max_epochs=10)

    trainer.fit(cnn_module, datamodule=data_module)

if __name__ == "__main__":
    main()
