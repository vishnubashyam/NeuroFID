import logging
import os
from typing import Any, Optional, Sequence, Tuple, Type, Union, List, Dict

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from monai.networks.nets import EfficientNetBN, ResNet, resnet18
from pytorch_lightning import Callback, LightningModule
from torch import nn, Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, AUROC, MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef
import torchmetrics
from tqdm.auto import tqdm


def create_pretrained_medical_resnet(
    pretrained_path: str,
    model_constructor: callable = resnet18,
    spatial_dims: int = 3,
    n_input_channels: int = 1,
    num_classes: int = 1,
    **kwargs_monai_resnet: Any
) -> Tuple[ResNet, Sequence[str]]:
    """This si specific constructor for MONAI ResNet module loading MedicalNEt weights.
    See:
    - https://github.com/Project-MONAI/MONAI
    - https://github.com/Borda/MedicalNet
    """
    net = model_constructor(
        pretrained=False,
        spatial_dims=spatial_dims,
        n_input_channels=n_input_channels,
        num_classes=num_classes,
        **kwargs_monai_resnet
    )
    net_dict = net.state_dict()
    pretrain = torch.load(pretrained_path)
    pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
    missing = tuple({k for k in net_dict.keys() if k not in pretrain['state_dict']})
    print(f"missing in pretrained: {len(missing)}")
    inside = tuple({k for k in pretrain['state_dict'] if k in net_dict.keys()})
    print(f"inside pretrained: {len(inside)}")
    unused = tuple({k for k in pretrain['state_dict'] if k not in net_dict.keys()})
    print(f"unused pretrained: {len(unused)}")
    assert len(inside) > len(missing)
    assert len(inside) > len(unused)

    pretrain['state_dict'] = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    net.load_state_dict(pretrain['state_dict'], strict=False)
    return net, inside


class FineTuneCB(Callback):
    # add callback to freeze/unfreeze trained layers
    def __init__(self, unfreeze_epoch: int) -> None:
        self.unfreeze_epoch = unfreeze_epoch

    def on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch != self.unfreeze_epoch:
            return
        for n, param in pl_module.net.named_parameters():
            param.requires_grad = True
        optimizers, _ = pl_module.configure_optimizers()
        trainer.optimizers = optimizers


class LitBrainMRI(LightningModule):

    def __init__(
        self,
        net: Union[nn.Module, str],
        targets: List,
        criterion: Dict,        
        pretrained_params: Optional[Sequence[str]] = None,
        lr: float = 1e-3,
        optimizer: Optional[Type[Optimizer]] = None,
    ):
        super().__init__()
        self.name = net.__class__.__name__
        self.net = net
        self.targets = targets
        self.criterion = criterion
        self.learning_rate = lr
        self.optimizer = optimizer or AdamW
        # self.save_hyperparameters()

        self.train_metrics = nn.ModuleList()
        self.val_metrics = nn.ModuleList()

        self.train_test = torchmetrics.MetricCollection([
            AUROC(num_classes=1, compute_on_step=False),
            Accuracy(num_classes=1, compute_on_step=False)])


        for i, target in enumerate(self.targets):
            key = (list(target.keys())[0]).replace(' ', '_')
            if (list(target.values())[0]['Type']) == 'Categorical':
                self.train_metrics.append(nn.ModuleDict({
                    f'train/{key}_acc': Accuracy(num_classes=1), 
                    f'train/{key}_auc': AUROC(num_classes=1, compute_on_step=False)
                }))
                self.val_metrics.append(nn.ModuleDict({
                    f'valid/{key}_acc': Accuracy(num_classes=1), 
                    f'valid/{key}_auc': AUROC(num_classes=1, compute_on_step=False)
                }))
            if (list(target.values())[0]['Type']) == 'Numerical':
                self.train_metrics.append(nn.ModuleDict({
                    f'train/{key}_mae': MeanAbsoluteError(),
                    f'train/{key}_corr': PearsonCorrCoef(compute_on_step=False) 

                }))                    
                self.val_metrics.append(nn.ModuleDict({
                    f'valid/{key}_mae': MeanAbsoluteError(),
                    f'valid/{key}_corr': PearsonCorrCoef(compute_on_step=False) 
                }))

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        outputs = []
        for i, target in enumerate(self.targets):
            if (list(target.values())[0]['Type']) == 'Categorical':
                x[:,i] = (torch.sigmoid(x[:,i]))
        return x

    @staticmethod
    def compute_loss(
        pred: Tensor,
        labels: Tensor, 
        masks: Tensor, 
        targets: List, 
        criterion: Dict
    ) -> Tensor:
        loss = torch.Tensor([0]).to(pred.device) 
        for i, target in enumerate(targets):
            if (list(target.values())[0]['Type']) == 'Categorical':
                loss_tmp =  criterion['Categorical'](pred[:,i], labels[:,i], reduction='none')
                loss_tmp = torch.mean(loss_tmp[masks[i]])
                if not loss_tmp.isnan():
                    loss += loss_tmp
            if (list(target.values())[0]['Type']) == 'Numerical':
                loss_tmp =  criterion['Numerical'](pred[:,i], labels[:,i], reduction='none')
                loss_tmp = torch.mean(loss_tmp[masks[i]])
                if not loss_tmp.isnan():
                    loss += loss_tmp
        return loss

    def training_step(self, batch, batch_idx):
        img, label, mask = batch["data"], batch["label"], batch["masks"]
        pred = self(img)
        loss = self.compute_loss(pred, label, mask, self.targets, self.criterion)
        self.log("train/loss", loss, prog_bar=False)

        for i, metric in enumerate(self.train_metrics):
            for key, metric in metric.items():
                label_tmp = label[:,i]
                if not mask[i].any():
                    continue 
                if ('auc' in key or 'acc' in key):
                    label_tmp = label_tmp.int()
                if ('acc' in key or 'mae' in key):
                    metric(pred[:,i][mask[i]], label_tmp[mask[i]])
                    try:
                        self.log(key, metric, prog_bar=False)
                    except:
                        print(pred[:,i], label_tmp, mask[i])
                else:
                    metric(pred[:,i][mask[i]], label_tmp[mask[i]])
                    try:
                        self.log(key, metric, prog_bar=False, on_step=False, on_epoch=True)
                    except:
                        print(pred[:,i], label_tmp, mask[i])
        return loss

    def validation_step(self, batch, batch_idx):
        img, label, mask = batch["data"], batch["label"], batch["masks"]
        pred = self(img)
        loss = self.compute_loss(pred, label, mask, self.targets, self.criterion)
        self.log("valid/loss", loss, prog_bar=False)

        for i, metric in enumerate(self.val_metrics):
            for key, metric in metric.items():
                label_tmp = label[:,i]
                if not mask[i].any():
                    continue
                if ('auc' in key or 'acc' in key):
                    label_tmp = label_tmp.int()
                if ('acc' in key or 'mae' in key):
                    try:
                        metric(pred[:,i][mask[i]], label_tmp[mask[i]])
                        self.log(key, metric, prog_bar=False)
                    except:
                        print("Exception ##### "+key)
                else:
                    metric(pred[:,i][mask[i]], label_tmp[mask[i]])
                    try:
                        self.log(key, metric, prog_bar=False, on_step=False, on_epoch=True)
                    except:
                        print("Exception ##### "+key)


    def configure_optimizers(self):
        optimizer = self.optimizer(self.net.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        return [optimizer], [scheduler]
