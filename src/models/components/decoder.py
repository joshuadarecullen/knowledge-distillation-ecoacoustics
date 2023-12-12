from typing import Tuple
import torch
from torch import nn

class Decoder(nn.Module):
    #decoder maps latent space sample z back into input data space
    #mirrored architecture to encoder, see VGGish encoder
    def __init__(self,
                 in_features: int = 128,
                 out_channels: int = 2,
                 negative_slope: float = 0.03):
        super().__init__()
        # decoder fully connected network using single hidden layer
        # output (n, 512, 6, 4)
        self.embedding = nn.Sequential(nn.Linear(in_features=in_features, out_features=4096),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(in_features=4096, out_features=4096),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(in_features=4096, out_features=512 * 4 * 6),
                                       nn.ReLU(inplace=True)) 
        # convolutions applied in reverse utlising learned upsampling
        # shape = (n, 512, 6, 4)
        self.features = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
                                      nn.GroupNorm(num_groups=1, num_channels=512),
                                      nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                                      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                      nn.GroupNorm(num_groups=1, num_channels=512),
                                      nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                                      nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                      nn.GroupNorm(num_groups=1, num_channels=256),
                                      nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                                      nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                                      nn.GroupNorm(num_groups=1, num_channels=256),
                                      nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                                      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                      nn.GroupNorm(num_groups=1, num_channels=256),
                                      nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                                      nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                                      nn.GroupNorm(num_groups=1, num_channels=128),
                                      nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                                      nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                                      nn.GroupNorm(num_groups=1, num_channels=128),
                                      nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                                      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                                      nn.GroupNorm(num_groups=1, num_channels=64),
                                      nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                                      nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
                                      nn.GroupNorm(num_groups=1, num_channels=64),
                                      nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                                      nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1))
        # final shape = (n, m, 96, 64)

    def unflatten(self,
                  x: torch.Tensor) -> torch.Tensor:
        #unflatten embedding to apply convolution input has shape (N, 12288)
        #returns of shape (N, 512, 6, 4)
        return x.view(x.size(0), 6, 4, 512).permute(0, 3, 1, 2)

    def forward(self,
                x: torch.Tensor) -> Tuple[torch.Tensor,
                                          torch.Tensor]:
        # foward pass of the decoder, latent layer all the way to the 2 output, uncertainty and mean
        x = self.embedding(x)
        x = self.unflatten(x)
        x = self.features(x)
        return x

