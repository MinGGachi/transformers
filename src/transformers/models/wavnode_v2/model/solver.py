#! /usr/bin/env python3
####
# written by Min Jun Choi
# Ph.D. Student, Music & Audio Research Group, Seoul Nat'l Univ.
# Last modified: 26.01.08
####

from typing import Any, Callable, Dict, List, Tuple
from torch import Tensor

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import functools

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

def _run_f(f: nn.Module, t: Tensor, x: Tensor, **kwargs) -> Tuple[Tensor, Tuple[Any, ...]]:
    """
    Helper to run the ODE function f.
    Handles variable return values (tensor vs tuple).
    Assumption: f returns (dx, *aux_outputs) or just dx.
    If f is WavNodeEncoderLayer, it expects (time, hidden_states, ...).
    """
    outputs = f(time=t, hidden_states=x, **kwargs)
    
    if isinstance(outputs, tuple):
        # outputs = (dx, aux1, aux2, ...)
        return outputs[0], outputs[1:]
    # outputs = dx
    return outputs, ()

def _euler_step(f: nn.Module, 
                t: Tensor, 
                x: Tensor, 
                dt: Tensor, 
                **aux_inputs) -> Tuple[Tensor, Tuple[Any, ...]]:
    k1, aux_outputs = _run_f(f, t, x, **aux_inputs)
    x_next = x + k1 * dt
    return x_next, aux_outputs

def _midpoint_step(f: nn.Module, 
                   t: Tensor, 
                   x: Tensor, 
                   dt: Tensor, 
                   **aux_inputs) -> Tuple[Tensor, Tuple[Any, ...]]:
    hdt = dt / 2
    t_hdt = t + hdt

    k1, _ = _run_f(f, t, x, **aux_inputs)
    k2, aux_outputs = _run_f(f, t_hdt, x + k1 * hdt, **aux_inputs)

    x_next = x + k2 * dt
    return x_next, aux_outputs

def _rk4_step(f: nn.Module, 
              t: Tensor, 
              x: Tensor, 
              dt: Tensor, 
              **aux_inputs) -> Tuple[Tensor, Tuple[Any, ...]]:
    hdt = dt / 2
    t_hdt = t + hdt
    t_dt = t + dt
    
    k1, _ = _run_f(f, t, x, **aux_inputs)
    k2, _ = _run_f(f, t_hdt, x + k1 * hdt, **aux_inputs)
    k3, _ = _run_f(f, t_hdt, x + k2 * hdt, **aux_inputs)
    k4, aux_outputs = _run_f(f, t_dt, x + k3 * dt, **aux_inputs)

    x_next = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

    return x_next, aux_outputs


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
        self.step_method = step_method.lower()
        self.step_fn = get_step(self.step_method)
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        ts: Tensor,
        x: Tensor,
        return_trajectory: bool = True,
        **kwargs: Any,
    ) -> Tuple[Tensor, List[Any]]:
        # aux outputs (e.g., attention weights) per step are gathered when provided by f
        aux_outputs: List[Any] = []
        T = ts.shape[0]

        if return_trajectory:
            # pre-allocate memory for trajectory
            trajectory = torch.empty((T, *x.shape), dtype=x.dtype, device=x.device)
            trajectory[0] = x

        for i in range(T - 1):
            t = ts[i]
            dt = ts[i + 1] - t
            x, aux_out = self._forward_step(t, x, dt, **kwargs)
            aux_outputs.append(aux_out)
            
            if return_trajectory:
                trajectory[i + 1] = x

        trajectory = trajectory if return_trajectory else x

        return trajectory, aux_outputs

    def _forward_step(self, t: Tensor, x: Tensor, dt: Tensor, **kwargs) -> Tuple[Tensor, Any]:
        # Wrap step_fn to handle kwargs via closure for checkpointing
        def run_step(t: Tensor, x: Tensor, dt: Tensor) -> Tuple[Tensor, Any]:
            if not self.f.training:
                # Inference mode
                with torch.no_grad():
                    return self.step_fn(self.f, t, x, dt, **kwargs)
            # Training mode
            return self.step_fn(self.f, t, x, dt, **kwargs)

        # Apply checkpointing if required and enabled
        if self.use_checkpoint and self.f.training:
            # Checkpointing with use_reentrant=False requires at least one input with requires_grad=True.
            # If the input state 'x' is frozen (e.g., first layer input from frozen features), 
            # we must pass a dummy tensor with grad enabled to trigger the checkpointing machinery correctly for parameters.
            if not x.requires_grad:
                dummy_grad = torch.tensor([], requires_grad=True, device=x.device)
                
                # Wrapper to accept dummy
                def run_step_wrapper(dummy, t, x, dt):
                    return run_step(t, x, dt)
                    
                return cp.checkpoint(run_step_wrapper, dummy_grad, t, x, dt, use_reentrant=False)

            # use_reentrant=False is generally recommended for newer PyTorch versions
            # It supports passing non-tensor args via partial, but here we capture in closure run_step
            return cp.checkpoint(run_step, t, x, dt, use_reentrant=False)

        return run_step(t, x, dt)


__all__ = [
    "Solver",
]