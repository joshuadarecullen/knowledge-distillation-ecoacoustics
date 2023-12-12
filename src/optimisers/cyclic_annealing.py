import torch.nn as nn
import math

class CyclicAnnealing(nn.Module):
    def __init__(self,
                 L: int,
                 M: int, 
                 warmup_epochs: int = 5,
                 alpha1: float = 0.0,
                 alpha2: float = 1.0) -> None:
        """
        Initialize the CyclicAnnealing module.

        Args:
        - L: the period of each cycle, e.g., number of epochs per cycle.
        - M: number of cycles.
        - alpha1: starting annealing factor, default is 0.0.
        - alpha2: ending annealing factor, default is 1.0.
        """
        super().__init__()
        self.update_type = 'epoch'
        self.L = L
        self.M = M
        self.warmup_epochs = warmup_epochs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.current_epoch = 0

    def forward(self, current_epoch: int) -> float:
        """
        Compute the annealing factor for the given epoch.

        Args:
        - current_epoch: the current epoch (0-indexed).

        Returns:
        - float: the annealing factor.
        """
        if current_epoch < self.warmup_epochs:
            self.current_epoch += 1
            return self.alpha1
        else:
            epoch_adj = current_epoch - self.warmup_epochs
            self.current_epoch += 1
            
            # Sinusoidal annealing after warmup
            annealing_factor = self.alpha1 + 0.5 * (self.alpha2 - self.alpha1) * (1 + math.cos(math.pi * (epoch_adj % self.L) / self.L))
            return annealing_factor
            return 

class CyclicBatchAnnealer(nn.Module):
    def __init__(self,
                 cycle_length: int = 20*151,
                 max_value: float = 1.0, 
                 warmup_batches: int = 4*152) -> None:
        """
        Implements batch-wise cyclic annealing.

        Args:
        - cycle_length: Number of batches for a complete cycle (from 0 to max_value and back to 0).
        - max_value: Maximum value of the annealing factor, default is 1.
        """
        super(CyclicBatchAnnealer, self).__init__()
        self.update_type = 'batch'
        self.cycle_length = cycle_length
        self.max_value = max_value
        self.warmup_batches = warmup_batches
        self.current_step = 0
        self.scale = 0.0

    def forward(self) -> float:

        if self.current_step < self.warmup_batches:
            self.current_step += 1
        else:
            # Calculate the position within the current cycle
            cycle_pos = (self.current_step-self.warmup_batches) % self.cycle_length

            # Calculate the scale using a triangle sinusoidal function
            self.scale = self.max_value * (1 - abs(cycle_pos / (self.cycle_length / 2) - 1))

            self.current_step += 1
