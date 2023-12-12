from typing import Callable, List, Optional

import numpy as np
from ranzen import parsable
import torch
from torch import Tensor, nn
import torchaudio.transforms as T

from conduit.data.datasets.utils import AudioTform


class LogMelSpectrogram(T.MelSpectrogram):
    @parsable
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 125.0,
        f_max: Optional[float] = 7500.00,
        n_mels: int = 64,
        log_offset: float = 0.0,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
        )
        self.log_offset = log_offset

    def forward(self, waveform: Tensor) -> Tensor:
        x = super().forward(waveform)
        return torch.log(x + self.log_offset)


class Framing(nn.Module):
    def __init__(self,
                window_secs: int = 96,
                n_mels: int = 64
                ):

        self.window_secs = window_secs
        self.n_mels = n_mels
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # frequecy onto vertical axis
        specgram = torch.transpose(data[:,:,:5760], -2, -1)
        # reshape into 96x64 frames
        specgram = torch.reshape(specgram, (60, 1, self.window_secs, self.n_mels))
        return specgram.unsqueeze(0)

    # inverse of foward method
    def backward(self, data: torch.Tensor) -> torch.Tensor:
        specgram = torch.reshape(data, (1, -1, 64))
        # specgram = torch.reshape(data, (num_points, 1, -1, 64))
        specgram = torch.transpose(specgram, -2, -1)
        return specgram
