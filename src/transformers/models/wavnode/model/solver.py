#! /usr/bin/env python3
####
# written by Min Jun Choi
# Ph.D. Student, Music & Audio Research Group, Seoul Nat'l Univ.
# Last modified: 25.10.01
####

from typing import Callable, Dict, List, Tuple, Union, Any
from torch import Tensor

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

# =========================================================
# Step Registry: register / get / list
# ---------------------------------------------------------
# step_fn signature:
#   step_fn(y, t, dt, f_module, **kwargs) -> y_next
# =========================================================
_STEP_REGISTRY: Dict[str, Callable] = {}

def register_step(name: str, fn: Callable):
    if not callable(fn):
        raise TypeError("fn must be callable")
    _STEP_REGISTRY[name.lower()] = fn

def get_step(name: str) -> Callable:
    key = name.lower()
    if key not in _STEP_REGISTRY:
        raise KeyError(f"Unknown step method '{name}'. Registered: {list(_STEP_REGISTRY)}")
    return _STEP_REGISTRY[key]

def list_steps() -> List[str]:
    return sorted(_STEP_REGISTRY.keys())

# =========================================================
# Basic fixed-step solvers forward-pass step implementations
# - Euler, Midpoint, RK4
# TODO: add more step methods
# =========================================================
def _euler_step(f: nn.Module, 
                x: Tensor, 
                t: Tensor, 
                dt: Tensor, 
                **kwargs) -> Tensor:
    attn_weights = () if kwargs.get("output_attentions", False) else None

    k1, aux_outputs = f(hidden_states=x, time=t, **kwargs)
    if kwargs.get("position_bias") is None:
        kwargs['position_bias'] = aux_outputs['position_bias']
    if kwargs.get("output_attentions", False):
        attn_weights = attn_weights + (aux_outputs['attn_weights'],)

    x_dx = x + k1 * dt

    return x_dx, kwargs, attn_weights

def _midpoint_step(f: nn.Module, 
                   x: Tensor, 
                   t: Tensor, 
                   dt: Tensor, 
                   **kwargs) -> Tensor:
    # time variable
    hdt = dt / 2
    t_hdt = t + hdt

    attn_weights = () if kwargs.get("output_attentions", False) else None

    k1, aux_outputs = f(hidden_states=x, time=t, **kwargs)
    if kwargs.get("position_bias") is None:
        kwargs['position_bias'] = aux_outputs['position_bias']
    if kwargs.get("output_attentions", False):
        attn_weights = attn_weights + (aux_outputs['attn_weights'],)

    k2, aux_outputs = f(hidden_states=x + k1 * hdt, time=t_hdt, **kwargs)
    if kwargs.get("output_attentions", False):
        attn_weights = attn_weights + (aux_outputs['attn_weights'],)

    x_dx = x + k2 * dt

    return x_dx, kwargs, attn_weights

def _rk4_step(f: nn.Module, 
              x: Tensor, 
              t: Tensor, 
              dt: Tensor, 
              **kwargs) -> Tensor:
    # time variables
    hdt = dt / 2
    t_hdt = t + hdt
    t_dt = t + dt

    attn_weights = () if kwargs.get("output_attentions", False) else None

    k1, aux_outputs = f(hidden_states=x, time=t, **kwargs)
    if kwargs.get("position_bias") is None:
        kwargs['position_bias'] = aux_outputs['position_bias']
    if kwargs.get("output_attentions", False):
        attn_weights = attn_weights + (aux_outputs['attn_weights'],)

    k2, aux_outputs = f(hidden_states=x + k1 * hdt, time=t_hdt, **kwargs)
    if kwargs.get("output_attentions", False):
        attn_weights = attn_weights + (aux_outputs['attn_weights'],)

    k3, aux_outputs = f(hidden_states=x + k2 * hdt, time=t_hdt, **kwargs)
    if kwargs.get("output_attentions", False):
        attn_weights = attn_weights + (aux_outputs['attn_weights'],)

    k4, aux_outputs = f(hidden_states=x + k3 * dt,  time=t_dt,  **kwargs)
    if kwargs.get("output_attentions", False):
        attn_weights = attn_weights + (aux_outputs['attn_weights'],)

    x_dx = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

    return x_dx, kwargs, attn_weights

# IF YOU WANT TO ADD MORE STEP METHODS, YOU CAN ADD THEM HERE
register_step("euler", _euler_step)
register_step("midpoint", _midpoint_step)
register_step("rk4", _rk4_step)


class Solver(nn.Module):
    def __init__(self,
                 f: nn.Module,
                 step_method: str = "euler",
                 use_checkpoint: bool = True,):
        super().__init__()
        self.f = f
        self.training = f.training
        self.step_method = step_method.lower()
        self.step_fn = get_step(self.step_method)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: Tensor, ts: Tensor, return_trajectory: bool = True, **kwargs) -> Tensor:
        attn_weights = {} if kwargs.get("output_attentions", False) else None
        T = ts.shape[0]

        if return_trajectory:
            # pre-allocate memory for trajectory
            trajectory = torch.empty((T, *x.shape), dtype=x.dtype, device=x.device)
            trajectory[0] = x

        for i in range(T - 1):
            t = ts[i]
            dt = ts[i + 1] - t
            x, kwargs, attn_weights_i = self._forward_step(x, t, dt, **kwargs)
            if kwargs.get("output_attentions", False):
                attn_weights[t] = attn_weights_i
            
            if return_trajectory:
                trajectory[i + 1] = x

        trajectory = trajectory if return_trajectory else x

        return trajectory, attn_weights

    def _forward_step(self, x: Tensor, t: Tensor, dt: Tensor, **kwargs) -> Tensor:
        def run_step(x: Tensor, t: Tensor, dt: Tensor, **kwargs) -> Tensor:
            if not self.f.training:
                with torch.no_grad():
                    return self.step_fn(self.f, x, t, dt, **kwargs)
            else:
                return self.step_fn(self.f, x, t, dt, **kwargs)
        
        if self.use_checkpoint and self.f.training:
            return cp.checkpoint(run_step, x, t, dt, **kwargs, use_reentrant=False)
        else:
            return run_step(x, t, dt, **kwargs)


__all__ = [
    "Solver",
]