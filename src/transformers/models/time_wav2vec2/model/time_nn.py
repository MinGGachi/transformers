#! /usr/bin/env python3
# written by Min Jun Choi
# Music & Audio Research Group, Seoul National University
# Last modified: 25.06.20

from typing import Tuple, Optional, Union
from torch import Tensor

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .weight_generator import WeightGenerator


class TimeModule():
    """
    This module CAN NOT be used as a standalone module.
    You should use it as a base class for other modules.
    """
    def __init__(self,
                 module: nn.Module,
                 rank: int,
                 time_dim: int,
                 hidden_dim: int,
                 activation: nn.Module = nn.SiLU(),
                 ):
        """
        Args:
            module (nn.Module): target module to generate time-dynamic weights and biases
            rank (int): rank of the generated weights
            time_dim (int): dimension of the time embedding
            hidden_dim (int): dimension of the hidden layer
            activation (nn.Module): activation function
        """
        self.rank = rank
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
        #### self._reshape_impl must be implemented in the child class ####
        return self._reshape_impl(unshaped_weight)
    
    def _weight_sum_with_bias(self,
                              sinusoidal_emb: Tensor,
                              )-> Tuple[Tensor, Tensor]:
        unshaped_weight = self.weight_generator(sinusoidal_emb)
        weight, bias = self.reshape_weight(unshaped_weight)

        weight_ = self.weight + weight
        bias_ = self.bias + bias

        return weight_, bias_
    
    def _weight_sum_without_bias(self,
                                 sinusoidal_emb: Tensor,
                                 )-> Tuple[Tensor, None]:
        unshaped_weight = self.weight_generator(sinusoidal_emb)
        weight, _ = self.reshape_weight(unshaped_weight)
        
        weight_ = self.weight + weight

        return weight_, None
        
        
class Linear(nn.Linear, TimeModule):
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
        nn.Linear.__init__(self, 
                           in_features=in_features, 
                           out_features=out_features, 
                           **kwargs)
        TimeModule.__init__(self, 
                            module=self, 
                            rank=rank, 
                            time_dim=time_dim, 
                            hidden_dim=hidden_dim, 
                            activation=activation)
        
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
        weight, bias = self._weight_sum(sinusoidal_emb)

        return F.linear(x, weight=weight, bias=bias)


class Conv1d(nn.Conv1d, TimeModule):
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
        nn.Conv1d.__init__(self, 
                           in_channels=in_channels, 
                           out_channels=out_channels, 
                           kernel_size=kernel_size,
                           **kwargs)
        TimeModule.__init__(self, 
                            module=self, 
                            rank=rank, 
                            time_dim=time_dim, 
                            hidden_dim=hidden_dim, 
                            activation=activation)
        
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
        weight, bias = self._weight_sum(sinusoidal_emb)

        return self._conv_forward(x, weight=weight, bias=bias)


class LayerNorm(nn.LayerNorm, TimeModule):
    """
    A module that encapsulates nn.LayerNorm and generates time-dynamic weights and biases.
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
        nn.LayerNorm.__init__(self, 
                              normalized_shape=normalized_shape,
                              **kwargs)
        TimeModule.__init__(self, 
                            module=self, 
                            rank=rank, 
                            time_dim=time_dim, 
                            hidden_dim=hidden_dim, 
                            activation=activation)
        
        if self.bias is not None:
            self._reshape_impl = self._reshape_with_bias
        else:
            self._reshape_impl = self._reshape_without_bias

    def _reshape_with_bias(self,
                           unshaped_weight: Tensor
                           )-> Tuple[Tensor, Tensor]:
        weight, bias = unshaped_weight.split([math.prod(self.normalized_shape), 
                                              math.prod(self.normalized_shape)], 
                                              dim=-1)

        return weight.view(self.weight.shape), bias.view(self.bias.shape)
    
    def _reshape_without_bias(self,
                              unshaped_weight: Tensor
                              )-> Tuple[Tensor, None]:
        weight = unshaped_weight.view(self.normalized_shape)

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
        weight, bias = self._weight_sum(sinusoidal_emb)

        return F.layer_norm(x, normalized_shape=self.normalized_shape, weight=weight, bias=bias)
    

__all__ = ["Linear", 
           "Conv1d", 
           "LayerNorm"]