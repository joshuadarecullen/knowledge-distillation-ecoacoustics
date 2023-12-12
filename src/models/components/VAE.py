from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class VAE(nn.Module):

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 ) -> None:

        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        # model predicts mu and var of the learned distriution to apply
        # reparameterisation trick to sample z from learned distribution

        mean_flat, log_var_flat = self.encoder(x)
        z_flat = self.reparameterise(mean_flat, log_var_flat)
        return mean_flat, log_var_flat, z_flat

    def decode(self, z):
        # input is  variational latent dimension vector sampled
        # from learned distribution and outputs reconstruction and uncertainty.
        return self.decoder(z)

    def forward(self, x):
        mean_flat, log_var_flat, z_flat = self.encode(x)
        return mean_flat, log_var_flat, z_flat, self.decode(z_flat)

    def loss_func(self,
            x: torch.Tensor,
            x_hat: torch.Tensor,
            mean: torch.Tensor,
            logvar: torch.Tensor,
            x_log_var: torch.Tensor,
            beta: float) -> Dict:


        # sum the loss components, taking batchwise mean
        log_likelihood = self.likelihood(x, x_hat, x_log_var).mean(axis=0)
        kl_divergence = self.kl_divergence(mean, logvar).mean(axis=0)
        real_loss = log_likelihood + kl_divergence

        loss = log_likelihood + kl_divergence * beta

        # return a dict of loss tensors
        return {"loss": loss,
                "real_loss": real_loss,
                "log_likelihood": log_likelihood,
                "kl_divergence": kl_divergence}

    def likelihood(self,
                     x: torch.Tensor,
                     x_hat: torch.Tensor,
                     x_log_var: torch.Tensor) -> torch.Tensor:
        # treat input x and output x_hat as samples from gaussian distributions
        # with variance x_var, calculate gaussian log probability of the data
        # calculate the loss element-wise
        log_likelihood = F.gaussian_nll_loss(x_hat, x, x_log_var.exp(), reduction="none")

        # reduce along all axes except the first 
        return log_likelihood.view(log_likelihood.size(0), -1).sum(axis=1)

    def kl_divergence(self,
                        mean: torch.Tensor,
                        log_variance: torch.Tensor,
                        ) -> torch.Tensor:
        # sum over all values in each spectrogram
        # calculate KL divergence between prior and surrogate posterior and take the batchwise mean
        # calculate the KL divergence element-wise
        kl_divergence = -1/2 * (1 + log_variance - mean.pow(2) - log_variance.exp())
        # reduce along axes except the first
        kl_divergence = kl_divergence.view(kl_divergence.size(0), -1).sum(axis=1)

        return kl_divergence

    def reparameterise(self,
                       mean: torch.Tensor,
                       log_variance: torch.Tensor) -> torch.Tensor:
        # given our prior on z is a standard normal distribution, use the
        # reparameterisation trick to sample from the approximate posterior
        # distribution to derive a variational latent vector z

        # calculate standard deviation from log variance
        std_dev = (1/2 * log_variance).exp()
        # generate random samples from prior on z, a normal distribution with mean 0 and variance 1
        # update prior, return samples z from shifted normal distributions learned from data
        return Normal(mean, std_dev).rsample()
