from typing import Optional, List
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from efficientnet_pytorch import EfficientNet

class EncoderEfficientNet(nn.Module):
    def __init__(self,
                 version: str='efficientnet-b0',
                 frozen: bool=False,
                 state_dict: dict=None,
                 in_channels: int=1,
                 in_features: int= 1280,
                 out_features: int= 128,
                 num_outputs: int=2) -> None:
        '''
        Custom implementation of efficientNet to take 1 channel inputs
        and two final layers that produce mean and log_variance.

        Args:
            version: the version of efficient net to load; default b0
            frozen: load with frozen weights
            state_dict: pretrained weights
            in_channels: amount of input channels in first conv layer
            in_features: amount of faetures in the final layer of efficient net
            out_features: latent space size
            num_outputs: num of outputs for the final layer
        '''
        super().__init__()

        self.features = EfficientNet.from_name(model_name=version,
                                               in_channels=in_channels,
                                               num_classes=in_features)


        self.outputs = nn.ModuleList([nn.Linear(in_features=in_features, out_features=out_features)
                                      for _ in range(num_outputs)])

        if state_dict:
            print('Loading EfficientNet Encoder with pretrained weights')
            self.load_state_dict(state_dict)
            print('Freezing EfficientNet weights')
            if frozen:
                # freeze teacher parameters
                for param in self.parameters():
                   param.requires_grad_(False)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.features(x)
        return [output(x) for output in self.outputs]

    def forward(self, x):
        mean_flat, log_var_flat = self.encode(x)
        z_flat = self.reparameterise(mean_flat, log_var_flat)
        return mean_flat, log_var_flat, z_flat

    def reparameterise(self,
                       mean: torch.Tensor,
                       log_variance: torch.Tensor) -> torch.Tensor:
        # calculate standard deviation from log variance
        std_dev = (1/2 * log_variance).exp()
        # generate random samples from prior on z, a normal distribution with mean 0 and variance 1
        # update prior, return samples (z) derived from shifted normal distributions learned from data
        return Normal(mean, std_dev).rsample()

if __name__ == "__main__":
    model = EncoderEfficientNet()

