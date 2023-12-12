import torch

class LinearAnnealing(torch.nn.Module):
    def __init__(self, warmup_epochs: int, total_epochs: int):
        """
        Initialize the LinearAnnealing module.

        Args:
        - total_epochs: Total number of epochs over which annealing should reach 1.
        - warmup_epocs: the amount of epochs before kl is introduced
        """
        super().__init__()
        self.name = 'epoch'
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

    def forward(self, current_epoch: int) -> float:
        """
        Compute the annealing factor for the given epoch.

        Args:
        - current_epoch: the current epoch (0-indexed).

        Returns:
        - float: the annealing factor.
        """
        if current_epoch < self.warmup_epochs:
            return 0.0
        else:
            epoch_adj = current_epoch - self.warmup_epochs
            return min(epoch_adj / self.total_epochs, 1.0)


class BatchLinearAnnealer(torch.nn.Module):
    def __init__(self,
                 max_value: float=1.0,
                 total_batches: int=12500,
                 warmup_batches: int=608):

        super().__init__()
        self.name = 'batch'

        self.max_value = max_value
        self.total_batches = total_batches
        self.warmup_batches = warmup_batches
        self.current_step = 0
        self.scale = 0.0

    def forward(self):
        if self.current_step < self.warmup_batches:
            self.current_step += 1
        else:
            batch_idx_adj = self.current_step - self.warmup_batches
            """Return the annealing factor for the current batch index."""
            self.scale = min(self.max_value, (batch_idx_adj / self.total_batches))
            self.current_step += 1
