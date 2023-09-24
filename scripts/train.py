from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning_modules import CNNModule, TrafficSignDataModule
from model import CNN


def main():
    data_path = Path("/home/lukasz/traffic_sign_recognition/data/preprocessed")
   
    model = CNN()
    cnn_module = CNNModule(model)

    wandb_logger = WandbLogger(project='traffic-sign-recognition', name=None)
    checkpoint_callback = ModelCheckpoint(dirpath=Path.cwd() / "models", filename="best-model.ckpt", save_top_k=1, monitor="val_accuracy")
    data_module = TrafficSignDataModule(data_dir=data_path)

    trainer = pl.Trainer(max_epochs=1, logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(cnn_module, datamodule=data_module)


if __name__ == "__main__":
    main()
