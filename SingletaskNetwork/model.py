import logging
import os
from typing import Any, Optional, Sequence, Tuple, Type, Union
from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from monai.networks.nets import EfficientNetBN, ResNet
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50, SEResNet50
from pytorch_lightning import Callback, LightningModule
from torch import nn, Tensor
from torch.optim import AdamW, Adagrad, RMSprop, Adamax, Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics import (
    Accuracy,
    AUROC,
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
)
from tqdm.auto import tqdm


class LitBrainMRI(LightningModule):
    def __init__(
        self,
        args: ArgumentParser,
        train_transforms: Any,
        val_transforms: Any,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.experiment_type = args.experiment_type
        self.name = args.model_name
        self.prediction_endpoint = args.prediction_endpoint
        self.resnet_size = args.model_size
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        self.pretrained_weights = args.pretrained_weights
        self.num_splits = args.num_splits
        self.dataloader_num_processes = (args.dataloader_num_processes,)
        self.accumulate_grad_batches = (args.accumulate_grad_batches,)
        self.swa_epoch_start = (args.swa_epoch_start,)
        self.mixed_precision = (args.mixed_precision,)
        self.pretrained_path = pretrained_path
        self.optimizer = AdamW
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        self.loss_fn = {
            "Classification": F.binary_cross_entropy_with_logits,
            "Regression": F.mse_loss,
        }[self.experiment_type]

        self.net = create_pretrained_medical_resnet(
            pretrained_path=self.pretrained_path, model_size=self.resnet_size
        )
        if self.experiment_type == "Classification":
            self.train_auroc = AUROC(num_classes=1, compute_on_step=False)
            self.train_acc_step = Accuracy(num_classes=1)
            self.train_acc_final = Accuracy(num_classes=1, compute_on_step=False)

            # self.train_f1_score = F1()
            self.val_auroc = AUROC(num_classes=1, compute_on_step=False)
            self.val_acc = Accuracy(num_classes=1, compute_on_step=False)

            # self.val_f1_score = F1()
        elif self.experiment_type == "Regression":
            self.train_mae_step = MeanAbsoluteError()
            self.train_mae_final = MeanAbsoluteError(compute_on_step=False)
            self.train_corr = PearsonCorrCoef(compute_on_step=False)

            self.val_mae = MeanAbsoluteError(compute_on_step=False)
            self.val_corr = PearsonCorrCoef(compute_on_step=False)

        self.save_hyperparameters(ignore=["net"])

    def forward(self, x: Tensor) -> Tensor:
        if self.experiment_type == "Classification":
            return torch.sigmoid(self.net(x)[:, 0])
        elif self.experiment_type == "Regression":
            return self.net(x)[:, 0]

    @staticmethod
    def compute_loss(
        y_hat: Tensor, y: Tensor, loss_fn: Optional[Type[nn.Module]] = None
    ) -> Tensor:
        return loss_fn(y_hat, y.to(y_hat.dtype))

    def training_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]

        # # Applying Monai transforms to the data on GPU one image at a time
        # # TODO: These transforms should be applied to the entire batch at once
        # scans = torch.Tensor().cuda()
        # for i in range(img.shape[0]):
        #     scans = torch.cat((scans, torch.unsqueeze(self.train_transforms(img[i]),0)), dim=0)
        # img = scans

        y_hat = self(img)
        loss = self.compute_loss(y_hat, y, self.loss_fn)

        if self.experiment_type == "Regression":
            self.log("train/loss", loss, prog_bar=False)
            self.train_corr(y_hat, y)
            self.train_mae_final(y_hat, y)
            self.log("train/mae_step", self.train_mae_step(y_hat, y), prog_bar=False)
            self.log(
                "train/mae_final", self.train_mae_final, on_step=False, on_epoch=True
            )
            self.log("train/corr", self.train_corr, on_step=False, on_epoch=True)
        if self.experiment_type == "Classification":
            y = y.int()
            self.log("train/loss", loss, prog_bar=False)
            self.train_auroc(y_hat, y)
            self.train_acc_final(y_hat, y)
            self.log("train/acc_step", self.train_acc_step(y_hat, y), prog_bar=False)
            self.log(
                "train/acc_final", self.train_acc_final, on_step=False, on_epoch=True
            )
            self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]
        y_hat = self(img)
        loss = self.compute_loss(y_hat, y, self.loss_fn)

        if self.experiment_type == "Regression":
            self.log("valid/loss", loss, prog_bar=False)
            self.val_corr(y_hat, y)
            self.val_mae(y_hat, y)
            self.log("valid/mae", self.val_mae, on_step=False, on_epoch=True)
            self.log("valid/corr", self.val_corr, on_step=False, on_epoch=True)
        if self.experiment_type == "Classification":
            y = y.int()
            self.log("valid/loss", loss, prog_bar=False)
            self.val_auroc(y_hat, y)
            self.val_acc(y_hat, y)
            self.log("valid/acc", self.val_acc, on_step=False, on_epoch=True)
            self.log("valid/auroc", self.val_auroc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.net.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        return [optimizer], [scheduler]


def create_pretrained_medical_resnet(
    pretrained_path: str,
    model_size: int,
    spatial_dims: int = 3,
    n_input_channels: int = 1,
    num_classes: int = 1,
    **kwargs_monai_resnet: Any,
) -> Tuple[ResNet, Sequence[str]]:

    if model_size == 10:
        model_constructor = resnet10
    elif model_size == 18:
        model_constructor = resnet18
    elif model_size == 34:
        model_constructor = resnet34
    elif model_size == 50:
        model_constructor = resnet50
    pretrained_path = os.path.join(pretrained_path, f"resnet_{str(model_size)}.pth")

    net = model_constructor(
        pretrained=False,
        spatial_dims=spatial_dims,
        n_input_channels=n_input_channels,
        num_classes=num_classes,
        **kwargs_monai_resnet,
    )
    net_dict = net.state_dict()
    pretrain = torch.load(pretrained_path)
    pretrain["state_dict"] = {
        k.replace("module.", ""): v for k, v in pretrain["state_dict"].items()
    }
    missing = tuple({k for k in net_dict.keys() if k not in pretrain["state_dict"]})
    print(f"missing in pretrained: {len(missing)}")
    inside = tuple({k for k in pretrain["state_dict"] if k in net_dict.keys()})
    print(f"inside pretrained: {len(inside)}")
    unused = tuple({k for k in pretrain["state_dict"] if k not in net_dict.keys()})
    print(f"unused pretrained: {len(unused)}")
    assert len(inside) > len(missing)
    assert len(inside) > len(unused)

    pretrain["state_dict"] = {
        k: v for k, v in pretrain["state_dict"].items() if k in net_dict.keys()
    }
    net.load_state_dict(pretrain["state_dict"], strict=False)
    return net


class FineTuneCB(Callback):
    # add callback to freeze/unfreeze trained layers
    def __init__(self, unfreeze_epoch: int) -> None:
        self.unfreeze_epoch = unfreeze_epoch

    def on_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.current_epoch != self.unfreeze_epoch:
            return
        for n, param in pl_module.net.named_parameters():
            param.requires_grad = True
        optimizers, _ = pl_module.configure_optimizers()
        trainer.optimizers = optimizers
