from typing import Dict, Any
import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

import numpy as np
import numpy.typing as npt

from conduit.data import TernarySample
from efficientnet_pytorch import EfficientNet

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class EfficientNetModule(pl.LightningModule):

    """
    This is an implementation of EfficentNet MLP, it can be run on the dataset
    """
    def __init__(
            self,
            classifier: torch.nn.Module,
            state_dict: Dict[str, torch.Tensor]=None,
            version: str='efficientnet-b0',
            in_channels: int=1,
            num_classes: int=1280,
            frozen: bool=False,
            optimizer: torch.optim.Optimizer=None,
            scheduler: torch.optim.lr_scheduler=None,
            ) -> None:

        super().__init__()

        self.save_hyperparameters()

        self.efficientnet = EfficientNet.from_name(model_name=version,
                                               in_channels=in_channels,
                                               num_classes=num_classes)


        self.classifier = classifier
        self.embedder = 'EfficientNetMLP'

        if state_dict:
            print('Loading EfficientNet Encoder with pretrained weights')
            self.load_state_dict(state_dict)
            print('Freezing EfficientNet weights')
            if frozen:
                # freeze teacher parameters
                for param in self.parameters():
                   param.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        x_flat = x.view(-1, 1, x.size(3), x.size(4)) # [batch_size*60, 1, 96, 64]
        z_flat = self.efficientnet(x_flat)
        # calculate each observations mean over frames [batch_size, 128]
        z = z_flat.view(x.size(0), -1, z_flat.size(1)).mean(axis=1)
        return  self.classifier(z)

    def model_step(self, x: TernarySample, y: torch.Tensor) -> Dict[str, torch.Tensor]:

        x = x.squeeze(1) # [batch_size, 60, 1, 96, 64]
        # Flatten each observation's frames into same dimensions
        x_flat = x.view(-1, 1, x.size(3), x.size(4)) # [batch_size*60, 1, 96, 64]
        z_flat = self.efficientnet(x_flat)
        # calculate each observations mean over frames [batch_size, 128]
        z = z_flat.view(x.size(0), -1, z_flat.size(1)).mean(axis=1)
        y_preds = self.classifier(z)

        # pass through mlp
        losses = self.classifier.loss_func(y_preds, y)

        return {**losses, **y_preds, 'z': z.detach()}

    def training_step(self,
            batch: TernarySample,
            batch_idx: int) -> Dict[str, torch.Tensor]:

        x, y, s = batch

        outputs = self.model_step(x, y)

        return {**outputs,
                'y': y.detach(),
                's': s.detach()}

    def on_train_batch_end(self,
            outputs: Dict[str, torch.Tensor],
            batch: TernarySample,
            bathc_idx: int) -> None:

        self.log_dict({"train/loss": outputs["loss"].item(),
                       "train/abundence": outputs["abundenceLoss"].item(),
                       "train/richness": outputs["richnessLoss"].item(),
                       "train/habitat": outputs["habitatLoss"].item()})

    def on_train_epoch_end(self):
        pass


    def validation_step(self,
            batch: TernarySample,
            batch_idx: int) -> Dict[str, torch.Tensor]:

        x, y, s = batch

        outputs = self.model_step(x, y)
        # calculate each observations latent space (mean over frames) [batch_size, 128]

        return {**outputs,
                'y': y.detach(),
                's': s.detach()}

    def on_validation_batch_end(self,
            outputs: Dict[str, torch.Tensor],
            batch: TernarySample,
            bathc_idx: int) -> None:

        self.log_dict({"val/loss": outputs["loss"].item(),
                       "val/abundence": outputs["abundenceLoss"].item(),
                       "val/richness": outputs["richnessLoss"].item(),
                       "val/habitat": outputs["habitatLoss"].item()})

    def on_validation_epoch_end(self):
        pass


    def test_step(self, batch: TernarySample, batch_idx: int):

        x, y, s = batch

        outputs = self.model_step(x, y)

        return {**outputs,
                'y': y.detach(),
                's': s.detach()}

    def on_test_batch_end(self,
            outputs: Dict[str, torch.Tensor],
            batch: TernarySample,
            bathc_idx: int) -> None:

        # update and log metrics
        self.log_dict({"test/loss": outputs["loss"].item(),
                       "test/abundence": outputs["abundenceLoss"].item(),
                       "test/richness": outputs["richnessLoss"].item(),
                       "test/habitat": outputs["habitatLoss"].item()})


    def predict_step(self,
                 batch: Dict[Any, TernarySample],
                 batch_idx: int,
                 dataloader_idx: int) -> Dict[Any, Dict[str, npt.NDArray]]:

        split = 'train' if dataloader_idx == 0 else 'val'

        x, y, s = batch
        # encode latent representation z
        output = self.model_step(x, y)
        # return logits and targets
        outputs = {**output,
                   "y": y.detach(),
                   "s": s.detach()}

        return {split: outputs}


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
