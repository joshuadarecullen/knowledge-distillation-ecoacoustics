from typing import Any, Dict, Union

import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

import numpy as np

from conduit.data import TernarySample

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.components.VAE import VAE
from src.models.VAEModule import VAEModule
from src.models.components.encoder import Encoder
from src.models.components.decoder import Decoder
from src.models.components.efficientNet import EncoderEfficientNet


class DistillModule(pl.LightningModule):
    def __init__(self,
                 teacher_ckpt: str,
                 student: EncoderEfficientNet,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler=None,
                 apply_recon: bool= True
                ) -> None:

        super().__init__()
        self.save_hyperparameters(ignore=['teacher'])

        # load the teachers checkpoint
        if teacher_ckpt:
            checkpoint=torch.load(teacher_ckpt)
            self.teacher = VAEModule(model=VAE(Encoder(), Decoder()),
                                     state_dict=checkpoint['state_dict'],
                                     frozen=True)
            self.teacher.eval() # set model to evaluation mode
        self.student = student
        self.apply_recon = apply_recon # if we want to apply recon loss

    def kl_divergence(self, mu1, logvar1, mu2, logvar2):
        # finding the differnce between two probility distributions
        sigma1 = torch.exp(0.5 * logvar1)
        sigma2 = torch.exp(0.5 * logvar2)
        kl_divergence = torch.log(sigma2/sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5
        kl_divergence = kl_divergence.view(kl_divergence.size(0), -1).sum(axis=1)
        return kl_divergence
    
    def loss_func(self,
                  x,
                  student_mu,
                  student_logvar,
                  student_recon,
                  student_x_log_var,
                  teacher_mu,
                  teacher_logvar,
                  beta: float = 1) -> Dict[str, torch.Tensor]:

        # calcuate the kl between teacher and student
        kl_loss = self.kl_divergence(student_mu,
                                     student_logvar,
                                     teacher_mu,
                                     teacher_logvar).mean(axis=0)

        # recreate spectrogram using teachers decoder
        recon_loss = self.teacher.model.likelihood(x,
                                                   student_recon,
                                                   student_x_log_var).mean(axis=0)

        # if recon loss to be applied, then add it
        loss =  kl_loss + recon_loss*beta if self.apply_recon else kl_loss

        # calcuate teachers for metrics
        teacher_loss = self.teacher.model.kl_divergence(teacher_mu,
                teacher_logvar).mean(axis=0)

        return {'loss': loss,
                'teacher_loss': teacher_loss,
                'log_likelihood': recon_loss,
                'kl_divergence': kl_loss}

    def forward(self, x):
        return self.student(x)

    def model_step(self, x: TernarySample) -> Dict[str, torch.Tensor]:
        # x has shape [batch_size, 1, 60, 1, 96, 64]

        x = x.squeeze(1) # [batch_size, 60, 1, 96, 64]

        # Flatten each observation's frames into same dimensions
        x_flat = x.view(-1, 1, x.size(3), x.size(4)) # [batch_size*60, 1, 96, 64]

        teacher_mu_flat, teacher_logvar_flat, _= self.teacher(x_flat)
        student_mu_flat, student_logvar_flat, student_z_flat = self.student(x_flat)

        # calculate each observations latent space (mean over frames) [batch_size, 128]
        student_z = student_z_flat.view(x.size(0), -1, student_z_flat.size(1)).mean(axis=1)

        # decoder students z_flat predictions
        student_recon = self.teacher.model.decode(student_z_flat)

        # extract spectrograms and their variance, both [batch_size*60, 1, 96, 64]
        student_recon_flat, student_x_logvar_flat = student_recon.chunk(2, dim=1)

        # reshape to extract observations from frames [batch_size, 60, 1, 96, 64]
        student_x_hat = student_recon_flat.view(x.shape)
        student_x_log_var = student_x_logvar_flat.view(x.shape)

        # reshape mean and log var of the latent space [batch_size, 60, 128]
        student_mu = student_mu_flat.view(x.size(0), -1, student_mu_flat.size(1))
        student_logvar = student_logvar_flat.view(x.size(0), -1, student_logvar_flat.size(1))
        teacher_mu = teacher_mu_flat.view(x.size(0), -1, teacher_mu_flat.size(1))
        teacher_logvar = teacher_logvar_flat.view(x.size(0), -1, teacher_logvar_flat.size(1))
        beta=1.0

        losses = self.loss_func(x,
                                student_mu,
                                student_logvar,
                                student_x_hat,
                                student_x_log_var,
                                teacher_mu,
                                teacher_logvar,
                                beta)
        return {**losses,
                'z': student_z.detach(),
                'x': x.detach(),
                'xlogvar': student_x_log_var.detach(),
                'x_hat': student_x_hat.detach()}

    # all the below functions are the modular parts of pytorch lighting depending on wat trainer function you call
    # train, test, predict
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

        self.log_dict({"train/student_loss": outputs["loss"].item(),
                       "train/teacher_loss": outputs["teacher_loss"].item(),
                       "train/student_log_likelihood_of_data": -outputs["log_likelihood"].item(),
                       "train/student_kl_divergence": outputs["kl_divergence"].item() })


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

        self.log_dict({"val/student_loss": outputs["loss"].item(),
                       "train/teacher_loss": outputs["teacher_loss"].item(),
                       "val/student_log_likelihood_of_data": -outputs["log_likelihood"].item(),
                       "val/student_kl_divergence": outputs["kl_divergence"].item() })

    def on_validation_epoch_end(self):
        pass


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

        self.log_dict({"test/loss": outputs["loss"].item(),
                       "test/log_likelihood_of_data": -outputs["log_likelihood"].item(),
                       "test/kl_divergence": outputs["kl_divergence"].item() })


    def predict_step(self,
                 batch: TernarySample,
                 batch_idc: int,
                 dataloader_idx: int) -> Dict[Any, Dict[str, np.ndarray]]:

        split = 'train' if dataloader_idx == 0 else 'val'

        x, y, s = batch
        output = self.model_step(x)
        outputs = {**output,
                   "y": y.detach(),
                   "s": s.detach()}

        return {split: outputs}


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        optimizer = self.hparams.optimizer(params=self.student.parameters())
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
