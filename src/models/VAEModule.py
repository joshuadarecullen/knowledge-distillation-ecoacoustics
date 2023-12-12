from typing import Any, Dict, Union

import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

import numpy as np

from conduit.data import TernarySample

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class VAEModule(pl.LightningModule):

    """
        LightningModule for Ecoacoustic VAE, the model has to be passed in only with optimiser and scheduler.
    """
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer=None,
            scheduler: torch.optim.lr_scheduler=None,
            kl_scheduler: torch.optim.lr_scheduler=None,
            state_dict: dict=None,
            frozen: bool=False,
            train_sigma: bool=True,
            ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        self.model = model
        self.train_sigma = train_sigma
        self.kl_scheduler = kl_scheduler
        self.scheduler = scheduler

        if state_dict:
            print("Using pretrained weights")
            self.load_state_dict(state_dict)
            if frozen:
                print("Freezing weights")
                # freeze teacher parameters
                for param in self.parameters():
                   param.requires_grad_(False)


    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
            inference for the latent space
        '''
        return self.model.encode(x)

    def model_step(self, x: TernarySample) -> Dict[str, torch.Tensor]:

        # x has shape [batch_size, 1, 60, 1, 96, 64]

        x = x.squeeze(1) # [batch_size, 60, 1, 96, 64]

        # Flatten each observation's frames into same dimensions
        x_flat = x.view(-1, 1, x.size(3), x.size(4)) # [batch_size*60, 1, 96, 64]

        # pass through the model outputs shape [batch_size*60, 128]
        mean_flat, logvar_flat, z_flat, recon = self.model(x_flat)

        # calculate each observations latent space (mean over frames) [batch_size, 128]
        z = z_flat.view(x.size(0), -1, z_flat.size(1)).mean(axis=1)

        # if training reconstruction sigma
        if self.train_sigma:
            # extract spectrograms and their variance, both [batch_size*60, 1, 96, 64]
            recon_flat, x_logvar_flat = recon.chunk(2, dim=1)
            # reshape to extract observations from frames [batch_size, 60, 1, 96, 64]
            x_hat = recon_flat.view(x.shape)
        else:
            x_logvar_flat = torch.zeros_like(recon)
            x_hat = recon.view(x.shape)

        x_log_var = x_logvar_flat.view(x.shape)

        # reshape mean and log var of the latent space [batch_size, 60, 128]
        mean = mean_flat.view(x.size(0), -1, mean_flat.size(1))
        logvar = logvar_flat.view(x.size(0), -1, logvar_flat.size(1))

        if self.kl_scheduler:
            # Compute beta for cyclic annealing
            beta = self.kl_scheduler.scale
        else:
            beta = 1.0

        # compute the reconstruction loss
        loss = self.model.loss_func(
                x,
                x_hat,
                mean,
                logvar,
                x_log_var,
                beta
                )

        return {**loss,
                'beta-kl': beta*loss['kl_divergence'].item(),
                'x_hat': x_hat.detach(),
                'x': x.detach(),
                'z': z.detach(),
                'xlogvar': x_log_var.detach()}


    def training_step(self,
            batch: TernarySample,
            batch_idx: int) -> Dict[str, torch.Tensor]:

        x, y, s = batch

        outputs = self.model_step(x)


        return {**outputs,
                'y': y.detach(),
                's': s.detach()}

    def on_train_batch_end(self,
            outputs: Dict[str, torch.Tensor],
            batch: TernarySample,
            bathc_idx: int) -> None:
        self.kl_scheduler()

        # update and log metrics
        self.log_dict({"train/loss": outputs["loss"].item(),
                       "train/real_loss": outputs["real_loss"].item(),
                       "train/log_likelihood_of_data": -outputs["log_likelihood"].item(),
                       "train/kl_divergence": outputs["kl_divergence"].item(),
                       "train/beta_kl_divergence": outputs["beta-kl"]})


    def validation_step(self,
            batch: TernarySample,
            batch_idx: int) -> Dict[str, torch.Tensor]:

        x, y, s = batch

        outputs = self.model_step(x)

        return {**outputs,
                'y': y.detach(),
                's': s.detach()}

    def on_validation_batch_end(self,
            outputs: Dict[str, torch.Tensor],
            batch: TernarySample,
            bathc_idx: int) -> None:

        # update and log metrics
        self.log_dict({"val/loss": outputs["loss"].item(),
                       "val/real_loss": outputs["real_loss"].item(),
                       "val/log_likelihood_of_data": -outputs["log_likelihood"].item(),
                       "val/kl_divergence": outputs["kl_divergence"].item()})


    def test_step(self, batch: TernarySample, batch_idx: int):

        x, y, s = batch

        outputs = self.model_step(x)

        return {**outputs,
                'y': y.detach(),
                's': s.detach()}

    def on_test_batch_end(self,
            outputs: Dict[str, torch.Tensor],
            batch: TernarySample,
            bathc_idx: int) -> None:

        # update and log metrics
        self.log_dict({"test/loss": outputs["loss"].item(),
                       "test/real_loss": outputs["real_loss"].item(),
                       "test/log_likelihood_of_data": -outputs["log_likelihood"].item(),
                       "test/kl_divergence": outputs["kl_divergence"].item() })

    def predict_step(self,
                     batch: Dict[Any, TernarySample],
                     batch_idx: int,
                     dataloader_idx: int) -> Dict[Any, Dict[str, np.ndarray]]:

        split = 'train' if dataloader_idx == 0 else 'val'

        x, y, s = batch
        # pass through the model
        output = self.model_step(x)
        outputs = {**output,
                   "y": y.detach(),
                   "s": s.detach()}

        return {split: outputs}


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        optimizer = self.hparams.optimizer(params=self.model.parameters())
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
