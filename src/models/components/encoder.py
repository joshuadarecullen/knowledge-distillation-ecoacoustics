from typing import List
import torch
from torch import nn
from .VGGish import VGGish


class Encoder(nn.Module):
    def __init__(self,
                 vggish_pretrain: bool = False,
                 vggish_frozen: bool = False,
                 out_features: int = 128,
                 negative_slope: float = 0.03,
                 num_outputs: int = 2):
        super().__init__()
        # encoder maps input x log melfrequency spectrograms
        # to  normal distribution in the latent space. The final layer
        # removed and with two output layers added to output mean and log variance

        # import vggish with pre-loaded weights
        self.vggish = VGGish(pretrain=vggish_pretrain,
                             frozen=vggish_frozen)
        # remove the final layer
        self.vggish.embeddings = nn.Sequential(*[self.vggish.embeddings[i] for i in range(4)])

        # create n output layers depedning if sigma is being trainied
        # shape = (n, 4096, 1, 1)
        self.outputs = nn.ModuleList([nn.Linear(in_features=4096, out_features=out_features) for _ in range(num_outputs)])

    def forward(self,
                x: torch.Tensor) -> List[torch.Tensor]:
        #perform a forward pass to encoder a latent feature embedding
        # run VGGish CNN for feature detection and embeddings
        x = self.vggish(x)
        # compute outputs
        return [output(x) for output in self.outputs]
