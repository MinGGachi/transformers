#! /usr/bin/env python3
# written by Min Jun Choi
# Music & Audio Research Group, Seoul National University
# Last modified: 25.06.18

"""
this module generates time-dynamic LoRA style weights and biases for nn.Linear, nn.Conv1d, nn.LayerNorm.

based on the following github repositories:
- LoRA: https://github.com/microsoft/LoRA (LoRA style weight generation)
- qkvflow: https://github.com/SDML-KU/qkvflow (weight generating network)
"""

from typing import Tuple, Union
from torch import Tensor

import math
import torch
import torch.nn as nn


# TODO: check Conv1d's LoRA style weight generation
class WeightGenerator(nn.Module):
    """
    This neural network generates dynamic weights and biases based on time embeddings,
    allowing for time-dependent linear and convolutional transformations.
    You can choose to generate full weights and biases or low-rank weights and biases.
    """
    def __init__(self,
                 module: Union[nn.Linear, nn.Conv1d, nn.LayerNorm],
                 rank: int,
                 time_dim: int,
                 hidden_dim: int,
                 activation: nn.Module,
                 ):
        """
        Args:
            module (nn.Module): target module to generate weights and biases. 
                                this module can generate weights and biases only for nn.Linear, nn.Conv1d, nn.LayerNorm.
            rank (int): rank of low-rank weights and biases.
                        but rank doesn't affect the dimension of weights and biases for nn.LayerNorm.
            time_dim (int): dimension of time embedding. sinusoidal embedding's output dimension is 2 * time_dim + 1.
            hidden_dim (int): dimension of hidden layer for MLP.
            activation (nn.Module): activation function to use for MLP.
        """
        super().__init__()

        # in and out dimension of weight generating network
        input_dim = 2 * time_dim + 1
        target_dim = self.target_dim(module, rank)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
        )
        self.projection = nn.Linear(hidden_dim, target_dim)

    def target_dim(self,
                   module: Union[nn.Linear, nn.Conv1d, nn.LayerNorm],
                   rank: int,
                   ) -> int:
        # sanity check for rank
        assert rank > 0 and isinstance(rank, int), "rank must be greater than 0"
        
        # check if module has bias
        bias = module.bias is not None

        if isinstance(module, nn.Linear):
            return rank * (module.in_features + module.out_features) \
                + int(bias) * module.out_features
        elif isinstance(module, nn.Conv1d):
            return rank * module.kernel_size[0] \
                * (module.in_channels // module.groups * module.kernel_size[0] + module.out_channels) \
                + int(bias) * module.out_channels
        elif isinstance(module, nn.LayerNorm):
            return math.prod(module.normalized_shape) * (1 + int(bias))

    def forward(self,
                sinusoidal_emb: Tensor,
                ) -> Tensor:
        """
        Generate time-dynamic unshaped weights and biases for the target module.

        Args:
            sinusoidal_emb (torch.Tensor): Sinusoidal embedding of time.

        Returns:
            weight (torch.Tensor): Time-dynamic unshaped weights for the target module.
        """
        # Proj(MLP(Sinusoidal Embedding(time)))

        return self.projection(self.mlp(sinusoidal_emb))