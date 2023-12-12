from typing import Dict, Any
import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

import numpy as np

from conduit.data import TernarySample

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.VAEModule import VAEModule
from src.models.components.VAE import VAE
from src.models.components.encoder import Encoder
from src.models.components.decoder import Decoder
from src.models.components.VGGish import VGGish
from src.models.components.efficientNet import EncoderEfficientNet


class MultiTaskMLPModule(pl.LightningModule):

    """
        MLP latent classifier, an embedder is required except for VGGish
    """
    def __init__(
            self,
            classifier: torch.nn.Module,
            optimizer: torch.optim.Optimizer=None,
            scheduler: torch.optim.lr_scheduler=None,
            embedder_ckpt: str=None,
            embedder: str='Eff'
            ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=['encoder'])


        self.classifier = classifier

        self.embedder = embedder

        # load appropriate encoder
        if self.embedder in 'VAE' and embedder_ckpt:
            print('Using VAE')
            checkpoint=torch.load(embedder_ckpt)
            self.encoder = VAEModule(model=VAE(Encoder(), Decoder()),
                                     state_dict=checkpoint['state_dict'],
                                     frozen=True)
        elif self.embedder in 'Eff' and embedder_ckpt:
            print('Using EfficientNet VAE')
            checkpoint=torch.load(embedder_ckpt)
            state_dict = {k.replace('student.',''): v for k,v in checkpoint['state_dict'].items() if 'student' in k}
            self.encoder = EncoderEfficientNet(state_dict=state_dict,
                                               frozen=True)
        elif self.embedder in 'VGGish' and not embedder_ckpt:
            print('Using VGGish')
            self.encoder = VGGish()

        self.encoder.eval()

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return  self.classifier(z)

    def encode(self, x: torch.Tensor) -> None:
        # enocde into latent space if VAE else use VGGish
        if self.embedder == 'VAE' or self.embedder == 'Eff':
            _, _, z_flat = self.encoder(x)
        elif self.embedder == 'VGGish':
            z_flat = self.encoder(x)
        return z_flat


    def model_step(self, x: TernarySample, y: torch.Tensor) -> Dict[str, torch.Tensor]:

        x = x.squeeze(1) # [batch_size, 60, 1, 96, 64]
        # Flatten each observation's frames into same dimensions
        x_flat = x.view(-1, 1, x.size(3), x.size(4)) # [batch_size*60, 1, 96, 64]
        z_flat = self.encode(x_flat)
        # calculate each observations latent space (mean over frames) [batch_size, 128]
        z = z_flat.view(x.size(0), -1, z_flat.size(1)).mean(axis=1)

        # pass through mlp
        outputs = self.classifier(z)
        losses = self.classifier.loss_func(outputs, y)

        return {**losses,
                **outputs,
                'z': z.detach(),
                }

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
                 dataloader_idx: int) -> Dict[Any, Dict[str, np.ndarray]]:

        split = 'train' if dataloader_idx == 0 else 'val'

        x, y, s = batch
        output = self.model_step(x, y)
        outputs = {**output,
                   "y": y.detach(),
                   "s": s.detach()}

        return {split: outputs}


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        optimizer = self.hparams.optimizer(params=self.classifier.parameters())
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
