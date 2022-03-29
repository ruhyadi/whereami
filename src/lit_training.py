"""
Copyright Didi Ruhyadi
"""
import argparse

from model import LitModel
from dataset import LitWhereAmIDataset
from torchvision import models

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# logger
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.loggers import WandbLogger
import wandb

from utils import WandbLog


def train(
    dataset_path='data',
    last_model='weights',
    fpm=20,
    epochs=10, 
    batch_size=32,
    fast_dev_run=False,
    comet_api='xxx',
    wandb_api='xxx'
    ):

    # model
    resnet18 = models.resnet18(pretrained=True)
    model = LitModel(resnet18)

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val/acc', # monitor log see model.py
        dirpath='weights',
        filename='model_{epoch:02d}_{val_loss:.2f}',
        mode='max', # save model with min val_loss
        save_top_k=2, # save two best model based on val_loss
    )
    
    # comet logger
    comet_logger = CometLogger(
        api_key=comet_api,
        save_dir="logs",  # Optional
        project_name="whereami",  # Optional
    )
    # wandb logger
    wandb.login(key=wandb_api)
    wandblog = WandbLog()
    wandb_logger = WandbLogger(project="whereami", log_model="all")

    # trainer
    trainer = Trainer(
        auto_select_gpus=True,
        auto_lr_find=True, # auto learning rate finder
        benchmark=True, # speedup if input size same
        check_val_every_n_epoch=5, # check val every n epoch
        callbacks=[checkpoint_callback, wandblog],
        fast_dev_run=fast_dev_run,
        gpus=-1, # select available GPU
        logger=[comet_logger, wandb_logger],
        max_epochs=epochs,
        )

    # tune lr and batch_size
    # trainer.tune(model)

    # dataset
    dataset = LitWhereAmIDataset(
        data_dir=dataset_path,
        fpm=fpm,
        train_val_test_split=[0.8, 0.1, 0.1],
        batch_size=batch_size,
    )

    # check last model

    # fit trainer
    trainer.fit(
        model=model,
        datamodule=dataset
    )

def main():
    parser = argparse.ArgumentParser(description='Training model')
    parser.add_argument('--dataset_path', type=str, default='data', help='dataset path')
    parser.add_argument('--last_model', type=str, default='weights', help='last model path')
    parser.add_argument('--fpm', type=int, default=20, help='frame per minute')
    parser.add_argument('--epochs', type=int, default=10, help='num of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='num of batch size')    
    parser.add_argument('--fast_dev_run', type=bool, default=False, help='debugging run')
    parser.add_argument('--comet_api', type=str, default='xxx', help='comet ml api')
    parser.add_argument('--wandb_api', type=str, default='xxx', help='wandb api')

    args = parser.parse_args()

    train(**vars(args))

if __name__ == "__main__":
    main()