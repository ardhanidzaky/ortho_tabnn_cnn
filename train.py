import os
import argparse
import yaml

import pytorch_lightning as pl
import pandas as pd

from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.dataset import FKGDataset
from utils.utils import PrepareDataset, data_transform
from utils.lightning import FKGLightningModule

def main(config_file):
    def load_config(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    config = load_config(config_file)
    device = config['device']

    # Dataset
    df = pd.read_csv('data/front_v1.csv', delimiter=';')
    task = PrepareDataset(
        image_path=config['data']['image_path']
        , subtask=config['data']['task']
        , data=df
    )
    train, val = task.get_train_test_split()   

    print("Preparing the training dataset.")
    td = FKGDataset(
        dataframe=train
        , transform=data_transform(config['data']['resolutions'])
        , target_width=config['data']['resolutions'][1]
        , target_height=config['data']['resolutions'][0]
    )
    tloader = DataLoader(td, batch_size=2, shuffle=True, num_workers=0)
    print()

    print("Preparing the validation dataset.")
    vd = FKGDataset(
        dataframe=val
        , transform=data_transform(config['data']['resolutions'])
        , target_width=config['data']['resolutions'][1]
        , target_height=config['data']['resolutions'][0]
    )
    vloader = DataLoader(vd, batch_size=1, shuffle=False, num_workers=0)
    print()

    # Model
    model = FKGLightningModule(**config['model']).to(config['device'])
    weight_path, log_path = create_path(model.initiated_at)
    
    logger = TensorBoardLogger(
        save_dir=log_path,
        version='events'
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    ckpt1 = ModelCheckpoint(
        monitor="val/loss",
        dirpath=weight_path,
        filename="{epoch:02d}",
        save_top_k=5,
        mode="min",
        save_weights_only=True
    )
    ckpt2 = ModelCheckpoint(
        save_last=True,
        dirpath=weight_path,
        filename="last",
        save_weights_only=False
    )
    callbacks = [lr_monitor, ckpt1, ckpt2]
    
    if config['device'] != 'cpu':
        trainer = pl.Trainer(
            accelerator='gpu'
            , devices='auto'
            , deterministic='warn'
            , logger=logger
            , callbacks=callbacks
            , gradient_clip_val=0.5
            , max_epochs=config['epoch']
        )
    else:
        trainer = pl.Trainer(
            accelerator='cpu'
            , devices='auto'
            , deterministic='warn'
            , logger=logger
            , callbacks=callbacks
            , max_epochs=config['epoch']
        )

    print('Starting training')
    print()
    start_time = datetime.now()
    trainer.fit(model=model, train_dataloaders=tloader, val_dataloaders=vloader)
    
    # Result section
    create_csv(model.initiated_at, model.result_file)
    print()
    end_time = datetime.now()
    time_taken = end_time - start_time

    print(f'Training done, result can be viewed in logs/{model.initiated_at}/{model.result_file}')
    print(f'Time taken: {time_taken}')

def create_path(version):
    weight_path = Path.cwd() / "weights" / Path(version)
    log_path = Path.cwd() / "logs" / Path(version)

    weight_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    
    return weight_path, log_path
    
def create_csv(version, key):
    df_l = pd.read_csv(f'logs/{version}/train_{key}')
    df_r = pd.read_csv( f'logs/{version}/val_{key}').iloc[1:, :].reset_index(drop=True)
    res = pd.concat([df_l, df_r], axis=1)
    res.columns = [
        'train_loss', 'train_acc', 'train_recall', 'train_fpr'
        , 'val_loss', 'val_acc', 'val_recall', 'val_fpr'
    ]

    res.to_csv(f'logs/{version}/{key}')
    os.remove(f'logs/{version}/train_{key}')
    os.remove(f'logs/{version}/val_{key}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ardhani <> Alfina Training Script')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the YAML configuration file')
    args = parser.parse_args()

    main(args.config)