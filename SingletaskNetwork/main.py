from dis import pretty_flags
import numpy as np
import os
from time import gmtime, strftime

from rising.loading import DataLoader, default_transform_call
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50, SEResNet50
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torchsummary import summary

import pandas as pd
from sklearn.model_selection import StratifiedKFold,  train_test_split
from pathlib import Path
from torch.optim import SGD, ASGD, Adamax

import parser
from data import Scans3dDM
from transforms import get_train_transforms, get_val_transforms
from model import LitBrainMRI, create_pretrained_medical_resnet, FineTuneCB

argparser = parser.create_parser()
args = argparser.parse_args()


PATH_MODELS = '/cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/SingletaskNetwork/pretrained_weights/'

data_dir = Path(args.data_directory)
df = pd.read_csv(args.data_csv)
experiment_name = f'{args.experiment}_{args.model_name}_{str(args.model_size)}'

print(experiment_name)
print(df['Label'].value_counts())

skf = StratifiedKFold(n_splits=args.num_splits, shuffle=True, random_state=42)
skf.get_n_splits(df, df.Study)

fold=0
for train_index, test_index in skf.split(df, df.Study):

    fold += 1
    train_index, val_index = train_test_split(
        train_index,
        test_size=args.val_split_pct, 
        shuffle=True,
        random_state=42
    )
    train_df = df.iloc[train_index].reset_index(drop=True)
    val_df = df.iloc[val_index].reset_index(drop=True)
    test_cv_df = df.iloc[test_index].reset_index(drop=True)


    data_module = Scans3dDM(
        data_dir=data_dir,
        train_df=train_df,
        validation_df=val_df,
        test_df=test_cv_df,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_processes,
        train_transforms=get_train_transforms(),
        valid_transforms=get_val_transforms()
    )
    break


data_module.setup()

model = LitBrainMRI(
    args=args,
    pretrained_path=PATH_MODELS,
    train_transforms=get_train_transforms(),
    val_transforms=get_val_transforms()
)

fine = FineTuneCB(unfreeze_epoch=1)
swa = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.6)

monitor = {
    'Regression':'valid/corr',
    'Classification':'valid/auroc'
}[args.experiment_type]

if args.leave_site_out == True:
    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath = f'ModelCheckpoints_LSO/{args.lso_main_task}/{experiment_name}/{(strftime("%Y_%m_%d_%H_%M_%S", gmtime()))}/',
        monitor=monitor,
        save_top_k=5,
        save_last=True,
        filename=f'checkpoint/{{epoch:02d}}-{{{monitor}:.4f}}',
        mode='max',
    )
else:
    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath = f'ModelCheckpoints/{experiment_name}/{(strftime("%Y_%m_%d_%H_%M_%S", gmtime()))}/',
        monitor=monitor,
        save_top_k=5,
        save_last=True,
        filename=f'checkpoint/{{epoch:02d}}-{{{monitor}:.4f}}',
        mode='max',
    )

# logger = TensorBoardLogger("tb_logs", name=experiment_name)
wandb_logger = WandbLogger(save_dir = 'wandb',name=experiment_name, project='SingletaskNetwork_LSO')

trainer = pl.Trainer(
    fast_dev_run=False,
    gpus=[0],
    callbacks=[ckpt, fine, swa],
    max_epochs=6,
    precision=16,
    benchmark=True,
    accumulate_grad_batches=2,
    val_check_interval=0.25,
    progress_bar_refresh_rate=10,
    log_every_n_steps=20,
    weights_summary='top',
    auto_lr_find=False,
    logger=[wandb_logger]
)

wandb_logger.watch(model, log_freq=500)

# trainer.tune(
#     model, 
#     datamodule=data_module,       
#     lr_find_kwargs=dict(min_lr=2e-6, max_lr=3e-2, num_training=15),
# )
print(f"Batch size: {data_module.batch_size}")
print(f"Learning Rate: {model.learning_rate}")

trainer.fit(model=model, datamodule=data_module)
