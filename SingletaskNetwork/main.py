# from model import 
import numpy as np
from data import Scans3dDM
from model import LitBrainMRI, create_pretrained_medical_resnet, FineTuneCB
import argparse
import os

from rising.loading import DataLoader, default_transform_call
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50, SEResNet50
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torchsummary import summary

import pandas as pd
from sklearn.model_selection import StratifiedKFold,  train_test_split
from pathlib import Path
from torch.optim import SGD, ASGD, Adamax

parser = argparse.ArgumentParser(description='Process Training Parameters')
parser.add_argument(
    '--experiment',
    type=str,
    help='Name of the experiment')
parser.add_argument(
    '--data_csv', 
    type=str,
    help='Path to the csv file containing the data')
parser.add_argument(
    '--model_size', 
    type=int,
    help='Size of the model - 10, 18, 34, 50')
parser.add_argument(
    '--batch_size', 
    type=int,
    help='Size of the batch')
args = parser.parse_args()

PATH_MODELS = '/cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/SingletaskNetwork/pretrained_weights/'
PATH_PRETRAINED_WEIGHTS = os.path.join(PATH_MODELS, f"resnet_{str(args.model_size)}.pth")

data_dir = Path('/cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/DataPrep/Preprocessing/BrainAligned')
df = pd.read_csv(args.data_csv)
experiment_name = args.experiment + '_ResNet' + str(args.model_size)

print(experiment_name)
print(df['Label'].value_counts())

skf = StratifiedKFold(n_splits=4)
skf.get_n_splits(df, df.Study)

fold=0
for train_index, test_index in skf.split(df, df.Study):

    fold += 1
    train_index, val_index = train_test_split(train_index, test_size=0.3, )
    train_df = df.iloc[train_index].reset_index(drop=True)
    val_df = df.iloc[val_index].reset_index(drop=True)
    test_cv_df = df.iloc[test_index].reset_index(drop=True)


    data_module = Scans3dDM(
        data_dir=data_dir,
        train_df=train_df,
        validation_df=val_df,
        test_df=test_cv_df

    )
    break


data_module.setup()
data_module.batch_size = args.batch_size

if args.model_size == 10:
    net, pretraineds_layers = create_pretrained_medical_resnet(
        PATH_PRETRAINED_WEIGHTS,
        model_constructor=resnet10)
elif args.model_size == 18:
    net, pretraineds_layers = create_pretrained_medical_resnet(
        PATH_PRETRAINED_WEIGHTS,
        model_constructor=resnet18)
elif args.model_size == 34:
    net, pretraineds_layers = create_pretrained_medical_resnet(
        PATH_PRETRAINED_WEIGHTS,
        model_constructor=resnet34)
elif args.model_size == 50:
    net, pretraineds_layers = create_pretrained_medical_resnet(
        PATH_PRETRAINED_WEIGHTS,
        model_constructor=resnet50)

model = LitBrainMRI(net = net, pretrained_params=None, lr=0.000608, optimizer=Adamax)
fine = FineTuneCB(unfreeze_epoch=1)
swa = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.6)


ckpt = pl.callbacks.ModelCheckpoint(
    dirpath = 'ModelCheckpoints',
    monitor='valid/mae',
    save_top_k=5,
    save_last=True,
    filename='checkpoint/{epoch:02d}-{valid/mae:.4f}',
    mode='max',
)

logger = TensorBoardLogger("tb_logs", name=experiment_name)
wandb_logger = WandbLogger(save_dir = 'wandb',name=experiment_name)

trainer = pl.Trainer(
    fast_dev_run=False,
    gpus=[0],
    callbacks=[ckpt, fine, swa],
    max_epochs=10,
    precision=16,
    benchmark=False,
    accumulate_grad_batches=2,
    val_check_interval=0.25,
    progress_bar_refresh_rate=10,
    log_every_n_steps=5,
    weights_summary='top',
    auto_lr_find=False,
    logger=[logger, wandb_logger]
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
