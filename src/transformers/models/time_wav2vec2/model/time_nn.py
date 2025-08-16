#! /usr/bin/env python3
# written by Min Jun Choi
# Music & Audio Research Group, Seoul National University
# Last modified: 25.07.24

from typing import Tuple, Optional, Union
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .weight_generator import WeightGenerator


class TimeModule(nn.Module):
    """
    This module CAN NOT be used as a standalone module.
    You should use it as a base class for other modules.
    """
    def __init__(self,
                 *,
                 module: nn.Module,
                 rank: int,
                 time_dim: int,
                 hidden_dim: int,
                 activation: nn.Module = nn.SiLU(),
                 scale_init: float = 0.0,
                 **layer_kwargs,
                 ):
        """
        Args:
            module (nn.Module): target module to generate time-dynamic weights and biases
            rank (int): rank of the generated weights
            time_dim (int): dimension of the time embedding
            hidden_dim (int): dimension of the hidden layer
            activation (nn.Module): activation function
        """
        super().__init__(**layer_kwargs)
        self.rank = rank
        self.scale_init = scale_init
        self.weight_generator = WeightGenerator(module=module,
                                                rank=rank,
                                                time_dim=time_dim,
                                                hidden_dim=hidden_dim,
                                                activation=activation)
        
    def reshape_weight(self,
                       unshaped_weight: Tensor,
                       ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Reshape the weight to the shape of the target module.

        Args:
            unshaped_weight (Tensor): generated unshaped weight

        Returns:
            Tuple[Tensor, Optional[Tensor]]: reshaped weight and bias(optional)
        """
        #### self._reshape_impl MUST be implemented in the child class ####
        return self._reshape_impl(unshaped_weight)
    
    def _weight_sum_with_bias(self,
                              unshaped_weight: Tensor,
                              raw_scaling_factor: Tensor,
                              )-> Tuple[Tensor, Tensor]:
        # autocast?
        weight, bias = self.reshape_weight(unshaped_weight)
        scaling_factor = F.softplus(raw_scaling_factor + self.scale_init)

        weight_ = self.weight + scaling_factor * weight
        bias_ = self.bias + scaling_factor * bias

        return weight_, bias_
    
    def _weight_sum_without_bias(self,
                                 unshaped_weight: Tensor,
                                 raw_scaling_factor: Tensor,
                                 )-> Tuple[Tensor, None]:
        # autocast?
        weight, _ = self.reshape_weight(unshaped_weight)
        scaling_factor = F.softplus(raw_scaling_factor + self.scale_init)

        weight_ = self.weight + scaling_factor * weight

        return weight_, None


class Linear(TimeModule, nn.Linear):
    """
    A module that encapsulates nn.Linear and generates LoRA style time-dynamic weights and biases.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 rank: int,
                 time_dim: int,
                 hidden_dim: int,
                 activation: nn.Module,
                 **kwargs,
                 ):
        """
        Args:
            in_features (int): input feature dimension
            out_features (int): output feature dimension
            kwargs (dict): keyword arguments for nn.Linear
        """
        super().__init__(in_features=in_features, 
                         out_features=out_features,
                         module=self,
                         rank=rank,
                         time_dim=time_dim,
                         hidden_dim=hidden_dim,
                         activation=activation,
                         **kwargs)
        
        if self.bias is not None:
            self._weight_sum = self._weight_sum_with_bias
            self._reshape_impl = self._reshape_with_bias
        else:
            self._weight_sum = self._weight_sum_without_bias
            self._reshape_impl = self._reshape_without_bias
        
    def _reshape_with_bias(self,
                           unshaped_weight: Tensor
                           )-> Tuple[Tensor, Tensor]:
        weight_A, weight_B, bias = unshaped_weight.split([self.in_features * self.rank, 
                                                    self.out_features * self.rank, 
                                                    self.out_features], 
                                                    dim=-1)
            
        weight_A = weight_A.view(self.rank, self.in_features)
        weight_B = weight_B.view(self.out_features, self.rank)
        bias = bias.view(self.out_features)

        weight = weight_B @ weight_A

        return weight.view(self.weight.shape), bias.view(self.bias.shape)
    
    def _reshape_without_bias(self,
                              unshaped_weight: Tensor
                              )-> Tuple[Tensor, None]:
        weight_A, weight_B = unshaped_weight.split([self.in_features * self.rank, 
                                                    self.out_features * self.rank], 
                                                    dim=-1)
        
        weight_A = weight_A.view(self.rank, self.in_features)
        weight_B = weight_B.view(self.out_features, self.rank)

        weight = weight_B @ weight_A

        return weight.view(self.weight.shape), None

    def forward(self, 
                x: Tensor, 
                sinusoidal_emb: Tensor,
                ):
        """
        Args:
            x (Tensor): input tensor
            sinusoidal_emb (Tensor): time embedding
        """
        unshaped_weight, scaling_factor = self.weight_generator(sinusoidal_emb)
        weight, bias = self._weight_sum(unshaped_weight, scaling_factor)

        return F.linear(x, weight=weight, bias=bias)


class Conv1d(TimeModule, nn.Conv1d):
    """
    A module that encapsulates nn.Conv1d and generates LoRA style time-dynamic weights and biases.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 rank: int,
                 time_dim: int,
                 hidden_dim: int,
                 activation: nn.Module,
                 **kwargs,
                 ):
        """
        Args:
            in_channels (int): input channel dimension
            out_channels (int): output channel dimension
            kernel_size (Union[int, Tuple[int]]): kernel size
            kwargs (dict): keyword arguments for nn.Conv1d
        """
        super().__init__(in_channels=in_channels, 
                         out_channels=out_channels, 
                         kernel_size=kernel_size,
                         module=self,
                         rank=rank,
                         time_dim=time_dim,
                         hidden_dim=hidden_dim,
                         activation=activation,
                         **kwargs)
        
        if self.bias is not None:
            self._weight_sum = self._weight_sum_with_bias
            self._reshape_impl = self._reshape_with_bias
        else:
            self._weight_sum = self._weight_sum_without_bias
            self._reshape_impl = self._reshape_without_bias
    
    def _reshape_with_bias(self,
                           unshaped_weight: Tensor
                           )-> Tuple[Tensor, Tensor]:
        kernel_size = self.kernel_size[0]
        weight_A, weight_B, bias = unshaped_weight.split([self.in_channels // self.groups * (kernel_size ** 2) * self.rank,
                                                          self.out_channels * kernel_size * self.rank,
                                                          self.out_channels], 
                                                          dim=-1)
        
        weight_A = weight_A.view(self.rank * kernel_size, self.in_channels // self.groups * kernel_size)
        weight_B = weight_B.view(self.out_channels, kernel_size * self.rank)

        weight = weight_B @ weight_A

        return weight.view(self.weight.shape), bias.view(self.bias.shape)

    def _reshape_without_bias(self,
                              unshaped_weight: Tensor
                              )-> Tuple[Tensor, None]:
        kernel_size = self.kernel_size[0]
        weight_A, weight_B = unshaped_weight.split([self.in_channels // self.groups * (kernel_size ** 2) * self.rank,
                                                    self.out_channels * kernel_size * self.rank], 
                                                    dim=-1)
        
        weight_A = weight_A.view(self.rank * kernel_size, self.in_channels // self.groups * kernel_size)
        weight_B = weight_B.view(self.out_channels, kernel_size * self.rank)

        weight = weight_B @ weight_A

        return weight.view(self.weight.shape), None

    def forward(self,
                x: Tensor,
                sinusoidal_emb: Tensor,
                ):
        """
        Args:
            x (Tensor): input tensor
            sinusoidal_emb (Tensor): time embedding
        """
        unshaped_weight, scaling_factor = self.weight_generator(sinusoidal_emb)
        weight, bias = self._weight_sum(unshaped_weight, scaling_factor)

        return self._conv_forward(x, weight=weight, bias=bias)


class LayerNorm(TimeModule, nn.LayerNorm):
    """
    A module that encapsulates nn.LayerNorm and generates time-dynamic weights.
    """
    def __init__(self,
                 normalized_shape: int,
                 rank: int,
                 time_dim: int,
                 hidden_dim: int,
                 activation: nn.Module,
                 **kwargs,
                 ):
        """
        Args:
            normalized_shape (int): dimension of the normalized shape
            kwargs (dict): keyword arguments for nn.LayerNorm
        """
        if rank > 1 or rank is None:
            print("in LayerNorm, rank doesn't affect the dimension of weights and biases")
            rank = 1
        super().__init__(normalized_shape=normalized_shape,
                         module=self,
                         rank=rank,
                         time_dim=time_dim,
                         hidden_dim=hidden_dim,
                         activation=activation,
                         **kwargs)
        
        with torch.no_grad():
            self.weight.data.fill_(1.0)
            self.bias.data.fill_(0.0)
        self.weight.requires_grad = False
        self.bias.requires_grad = False

        self._reshape_impl = self._reshape_with_bias
        self._weight_sum = self._weight_sum_with_bias
    
    def _reshape_with_bias(self,
                           unshaped_weight: Tensor
                           )-> Tuple[Tensor, Tensor]:
        
        weight, bias = unshaped_weight.split([self.normalized_shape[0], 
                                              self.normalized_shape[0]], 
                                              dim=-1)

        return weight.view(self.weight.shape), bias.view(self.bias.shape)

    def forward(self, 
                x: Tensor,
                sinusoidal_emb: Tensor,
                ):
        """
        Args:
            x (Tensor): input tensor
            sinusoidal_emb (Tensor): time embedding
        """
        unshaped_weight, scaling_factor = self.weight_generator(sinusoidal_emb)
        weight, bias = self._weight_sum(unshaped_weight, scaling_factor)

        return F.layer_norm(x, normalized_shape=self.normalized_shape, weight=weight, bias=bias)
    

__all__ = ["Linear", 
           "Conv1d", 
           "LayerNorm"]