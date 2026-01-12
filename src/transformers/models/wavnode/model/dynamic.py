#! /usr/bin/env python3
# written by Min Jun Choi
# Music & Audio Research Group, Seoul National University
# Last modified: 25.09.25

# TODO: add "evaluate_at" method
# the method that "compile" the temporal dynamic weight to static weight
# check https://github.com/SDML-KU/qkvflow/blob/f9419fb7d1816a133721aab050d85b4e77e72abc/qkvflow/nn/dynamic.py#L112

import numbers
from torch import Tensor, Size
from typing import Union, List, Tuple, Optional

_shape_t = Union[int, List[int], Size]

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .orthonormalize import _orthonormalize, PowerIterState


class LinearHead(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 rank: int,
                 tau: float = 0.2,
                 beta: float = 0.8,
                 ):
        super().__init__()
        self.head = nn.Linear(hidden_dim, rank)
        self.sig = nn.Softplus(beta=beta)
        self.tau = tau

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.head.bias, -0.05)

    def forward(self, t_emb: Tensor) -> Tensor:
        """
            Forward pass to generate time-dependent singular values.
        """
        raw = self.head(t_emb)

        return self.sig(raw / self.tau)
        

class LayerNormHead(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 latent_dim: int,
                 bias: bool = True,
                 ):
        super().__init__()
        target_dim = 2 * latent_dim if bias else latent_dim
        self.head = nn.Linear(hidden_dim, target_dim)

    def forward(self, t_emb: Tensor) -> Tensor:
        """
            Forward pass to generate time-dependent LayerNorm weights and biases.
        """
        return self.head(t_emb)


class WeightGenerator(nn.Module):
    def __init__(self,
                 module: str,
                 time_dim: int,
                 hidden_dim: int,
                 activation: nn.Module,
                 **kwargs
                 ):
        """
        """
        super().__init__()
        input_dim = 2 * time_dim + 1
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
        )
        if module == "linear":
            assert "rank" in kwargs, "rank must be specified for linear module"
            self.head = LinearHead(hidden_dim=hidden_dim,
                                   rank=kwargs.get("rank"),
                                   tau=kwargs.get("tau", 0.2),
                                   beta=kwargs.get("beta", 0.8))
        elif module == "layernorm":
            assert "latent_dim" in kwargs, "latent_dim must be specified for layernorm module"
            self.head = LayerNormHead(hidden_dim=hidden_dim,
                                      latent_dim=kwargs.get("latent_dim"),
                                      bias=kwargs.get("bias", True))
        
    def forward(self,
                t_emb: Tensor,) -> Tensor:
        
        t_emb = self.mlp(t_emb)
        
        return self.head(t_emb)


class Linear(nn.Module):
    """
        Time-Dynamic Linear Layer using basis and rank decomposition.
    """
    __constants__ = ['in_features', 'out_features', 'rank']
    in_features: int
    out_features: int
    rank: int

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 rank: int = None,
                 time_dim: int = 128,
                 hidden_dim: int = 128,
                 activation: nn.Module = nn.SiLU(),
                 device=None,
                 dtype=None,
                 ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        if rank is None:
            rank = min(in_features, out_features) // 2
        self.rank = rank

        self.V = nn.Parameter(torch.randn((rank, in_features), **factory_kwargs))
        self.S = WeightGenerator(module="linear",
                                 time_dim=time_dim,
                                 hidden_dim=hidden_dim,
                                 activation=activation,
                                 rank=rank,
                                 tau=0.2,
                                 beta=0.8,)
        self.U = nn.Parameter(torch.randn((out_features, rank), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter('bias', None)

        if in_features >= out_features:
            self.forward_impl = self.forward_shrink
        else:
            self.forward_impl = self.forward_expand

        self._sigma_state_U: Optional[PowerIterState] = None
        self._sigma_state_V: Optional[PowerIterState] = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.orthogonal_(self.V)
        init.orthogonal_(self.U)

    def forward_shrink(self,
                       t: Tensor,
                       x: Tensor,) -> Tensor:
        S = self.S(t)

        x = F.linear(x, weight=self.V)
        x = F.linear(x, weight=self.U * S[None, :], bias=self.bias)

        return x

    def forward_expand(self,
                       t: Tensor,
                       x: Tensor,) -> Tensor:
        S = self.S(t)

        x = F.linear(x, weight=S[:, None] * self.V)
        x = F.linear(x, weight=self.U, bias=self.bias)

        return x

    def forward(self,
                t: Tensor,
                x: Tensor,) -> Tensor:
        return self.forward_impl(t, x)
    
    def _orthonormalize(self):
        self._sigma_state_V = _orthonormalize(self.V, sigma_state=self._sigma_state_V)
        self._sigma_state_U = _orthonormalize(self.U, sigma_state=self._sigma_state_U)


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'eps']
    normalized_shape: Tuple[int, ...]
    eps: float

    def __init__(self,
                 normalized_shape: _shape_t,
                 time_dim: int,
                 hidden_dim: int,
                 activation: nn.Module,
                 eps: float = 1e-5,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        self.register_buffer('weight', torch.ones(self.normalized_shape, **factory_kwargs))
        self.use_bias = bias

        self.weight_generator = WeightGenerator(module='layernorm',
                                                time_dim=time_dim,
                                                hidden_dim=hidden_dim,
                                                activation=activation,
                                                latent_dim=self.normalized_shape[0],
                                                bias=bias,
                                                )

    def forward(self, 
                t: Tensor,
                x: Tensor,
                ):
        """
        Args:
            t (Tensor): time embedding
            x (Tensor): input tensor
        """
        weight_bias = self.weight_generator(t)
        
        if self.use_bias:
            delta_w, delta_b = weight_bias.split([self.normalized_shape[0], 
                                                  self.normalized_shape[0]], dim=-1)
            bias = delta_b.view(self.normalized_shape)
        else:
            delta_w = weight_bias
            bias = None

        delta_w = delta_w.view(self.normalized_shape)
        weight = self.weight + delta_w

        return F.layer_norm(x, normalized_shape=self.normalized_shape, weight=weight, bias=bias, eps=self.eps)
    

__all__ = ["Linear", 
           "LayerNorm"]