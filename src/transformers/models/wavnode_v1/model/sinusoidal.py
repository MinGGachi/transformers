#! /usr/bin/env python3
# written by Min Jun Choi
# Music & Audio Research Group, Seoul National University
# Last modified: 25.07.24

"""
this module generates sinusoidal embeddings for time-varying modules.

based on the following github repository:
- qkvflow: https://github.com/SDML-KU/qkvflow (time embedding)
"""

from typing import Tuple, Dict, Optional, Union
from torch import Tensor

import math
import numpy as np
import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Fourier features for time-varying modules."""
    def __init__(self,
                 time_dim: int = 128,
                 max_period: int = 10000,
                 scale: int = 1000,
                 requires_grad: bool = True,
                 ):
        """
        Args:
            sinusoidal_dim (int): Dimension of sinusoidal features.
            max_period (int): Maximum period of sinusoidal features.
            scale (int): Scaling factor for time.
        """
        super().__init__()
        self.time_dim = time_dim
        self.scale = scale
        self.requires_grad = requires_grad

        freq = torch.exp(
            -torch.log(torch.tensor(max_period))
            * torch.arange(self.time_dim)
            / self.time_dim
        )
        self.register_buffer("freq", freq)

    def forward(self, time: Tensor) -> Tensor:
        """
        Args:
            time (torch.Tensor): Tensor of timestep or sequence of timesteps with length L.

        Returns:
            emb (torch.Tensor): Fourier features with length 2 * sinusoidal_dim + 1.
        """
        # Check if time requires grad
        requires_grad = self.requires_grad or time.requires_grad

        assert time.dim() in [0, 1], "time must be a 0D or 1D tensor"
        
        if time.dim() == 0:
            time = time.reshape(-1)
        
        scaled_time = time * self.scale
        emb = scaled_time.reshape(-1, 1) * self.freq.reshape(1, -1)
        emb = torch.cat([time.reshape(-1, 1), torch.sin(emb), torch.cos(emb)], dim=1)

        return emb.squeeze(0).requires_grad_(requires_grad)