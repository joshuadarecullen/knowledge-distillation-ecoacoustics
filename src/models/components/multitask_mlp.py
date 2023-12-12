import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

'''
a multitask multilayer perceptron to classify habitat, species abundence and richness
'''
class MultiTaskMLP(nn.Module):
    def __init__(self,
                 shared_layer: bool= True,
                 input_dim: int=128,
                 hidden_dim: int=1024,
                 num_classes: int=6) -> None:
        super(MultiTaskMLP, self).__init__()

        self.shared_layer = shared_layer

        if self.shared_layer:
            self.shared_layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            )
            self.richness = nn.Linear(hidden_dim, 1)  # Regression output for Task 1: Species count
            self.abundence = nn.Linear(hidden_dim, 1)  # Regression output for Task 2: Abundence
            self.habitat = nn.Linear(hidden_dim, num_classes)  # Multi-class classification for Task 3: habitat
        else:
            self.richness = nn.Linear(input_dim, 1)  # Regression output for Task 1: Species count
            self.abundence = nn.Linear(input_dim, 1)  # Regression output for Task 2: Abundence
            self.habitat = nn.Linear(input_dim, num_classes)  # Multi-class classification for Task 3: habitat

        self.richnessLoss = nn.MSELoss()
        self.abundenceLoss = nn.MSELoss()
        self.habitatLoss = nn.CrossEntropyLoss()

    def forward(self, x) -> tuple:
        outputs = {}
        # if we have decided to have a shared layer then compute
        if self.shared_layer:
            shared = self.shared_layers(x)
            outputs['shared'] = shared
            abundence = F.softplus(self.abundence(shared))
            richness = F.softplus(self.richness(shared))
            habitat = self.habitat(shared)
        else:
            richness = F.softplus(self.richness(x))
            abundence = F.softplus(self.abundence(x))
            habitat = self.habitat(x)

        return {**outputs,
                'NN': richness,
                'N0': abundence,
                'habitat': habitat}

    def loss_func(self,
                  outputs: torch.Tensor,
                  targets: torch.Tensor) -> Dict[str, torch.Tensor]:

        # calculate loss for all targets
        richL = self.richnessLoss(outputs['NN'].flatten(), targets[:,0].float())
        abunL = self.abundenceLoss(outputs['N0'].flatten(), targets[:,1].float())
        habL = self.habitatLoss(outputs['habitat'], targets[:,2])
        loss = abunL + richL + habL # sum the losses
        losses = {
                'loss': loss,
                'abundenceLoss': abunL,
                'richnessLoss': richL,
                'habitatLoss': habL
                }

        return losses
