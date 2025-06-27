#! /usr/bin/env python3
# written by Min Jun Choi
# Music & Audio Research Group, Seoul National University
# Last modified: 25.06.06

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


class SinusoidalEmbedding():
    def __init__(self,
                 time_dim: int = 128,
                 max_period: int = 10000,
                 scale: int = 1000,
                 ):
        """
        Args:
            sinusoidal_dim (int): Dimension of sinusoidal features.
            max_period (int): Maximum period of sinusoidal features.
            scale (int): Scaling factor for time.
        """
        self.time_dim = time_dim
        self.max_period = max_period
        self.scale = scale

    def __call__(self,
                 time: Tensor,
                 ) -> Tensor:
        """
        For generating time-varying weights for encoder layer.

        Args:
            time (torch.Tensor): Tensor of timesteps with length L.

        Returns:
            emb (torch.Tensor): Fourier features with length 2 * sinusoidal_dim + 1.
        """
        assert len(time.shape) in [0, 1], "time must be a 0D or 1D tensor"
        if len(time.shape) == 0:
            time = time.reshape(-1)

        max_period = torch.tensor(self.max_period, dtype=time.dtype)
        scaled_time = time * self.scale
        freq = torch.exp(
            -torch.log(max_period)
            * torch.arange(self.time_dim, dtype=time.dtype)
            / self.time_dim
            )
        emb = scaled_time.reshape(-1, 1) * freq.reshape(1, -1)
        emb = torch.cat([time.reshape(-1, 1), torch.sin(emb), torch.cos(emb)], dim=1)

        return emb.squeeze(0)
    

class LearnableSinusoidalEmbedding(nn.Module):
    # TODO: Think about this class's efficiency
    # does it better than SinusoidalEmbedding?
    pass