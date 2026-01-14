from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import Tensor


@dataclass
class PowerIterState:
    mode: str          # "u" or "v"
    vec: Tensor        # shape [m,1] if mode="u", else [n,1]
    itrs: int = 0      # optional bookkeeping


@torch.no_grad()
def power_iter_sigma_max(
    W: Tensor,
    itrs: int = 2,
    state: Optional[PowerIterState] = None,
    eps: float = 1e-12,
    safe_mult: float = 1.1,
    compute_dtype: Optional[torch.dtype] = None,
    prefer_small_side: bool = True,
    return_state: bool = True,
) -> Union[Tensor, Tuple[Tensor, PowerIterState]]:
    assert W.ndim == 2, "W must be 2D"
    m, n = W.shape

    if compute_dtype is None:
        compute_dtype = torch.float32 if W.dtype in (torch.float16, torch.bfloat16) else W.dtype

    Wc = W.to(dtype=compute_dtype)

    if prefer_small_side:
        mode = "u" if m <= n else "v"
    else:
        mode = "v"

    def _init_vec(mode_: str) -> Tensor:
        dim = m if mode_ == "u" else n
        v = torch.randn(dim, 1, device=W.device, dtype=compute_dtype)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(v, src=0)
            
        v = v / (v.norm() + eps)
        return v

    if state is None or state.mode not in ("u", "v") or state.mode != mode:
        vec = _init_vec(mode)
        state = PowerIterState(mode=mode, vec=vec, itrs=0)
    else:
        vec = state.vec
        if vec.ndim == 1:
            vec = vec[:, None]
        expected = (m, 1) if mode == "u" else (n, 1)
        if vec.shape != expected or vec.device != W.device or vec.dtype != compute_dtype:
            vec = _init_vec(mode)
            state.vec = vec
            state.itrs = 0
        else:
            vec = vec / (vec.norm() + eps)

    if itrs > 0:
        if mode == "v":
            v = vec
            for _ in range(itrs):
                v = Wc.transpose(0, 1) @ (Wc @ v)   # [n,1]
                v = v / (v.norm() + eps)
            vec = v
        else:
            u = vec
            for _ in range(itrs):
                u = Wc @ (Wc.transpose(0, 1) @ u)   # [m,1]
                u = u / (u.norm() + eps)
            vec = u

    if mode == "v":
        sigma_hat = (Wc @ vec).norm().clamp_min(eps)
    else:
        sigma_hat = (Wc.transpose(0, 1) @ vec).norm().clamp_min(eps)

    sigma_safe = (sigma_hat * safe_mult).to(dtype=W.dtype)

    state.vec = vec.detach()
    state.itrs += itrs

    if return_state:
        return sigma_safe, state
    return sigma_safe


@torch.no_grad()
def _orthonormalize(
    W: Tensor,
    itrs: int = 5,
    beta: float = 0.5,
    eps: float = 1e-12,
    # warm-start for sigma_max scaling (optional)
    sigma_state: Optional[PowerIterState] = None,
    sigma_itrs: int = 2,
    safe_mult: float = 1.1,
    compute_dtype: Optional[torch.dtype] = None,
) -> Optional[PowerIterState]:
    """
    In-place Björck semi-orthonormalization.

    If rows >= cols: makes columns orthonormal  => W^T W ≈ I_cols
    If rows <  cols: makes rows orthonormal     => W W^T ≈ I_rows

    Returns updated sigma_state (for warm-start) if provided/desired.
    """
    assert W.ndim == 2, "Input must be a 2D tensor."
    rows, cols = W.shape

    if compute_dtype is None:
        compute_dtype = torch.float32 if W.dtype in (torch.float16, torch.bfloat16) else W.dtype

    # 1) Scale so that ||W||2 <= 1 (needed for Björck convergence)
    sigma, sigma_state = power_iter_sigma_max(
        W, itrs=sigma_itrs, state=sigma_state, eps=eps,
        safe_mult=safe_mult, compute_dtype=compute_dtype,
        prefer_small_side=True, return_state=True
    )
    # operate in compute_dtype for stability
    Wc = W.to(dtype=compute_dtype) / (sigma.to(dtype=compute_dtype) + eps)

    # 2) Björck updates
    if rows >= cols:
        # column-orthonormalize: W <- W ((1+β)I - β W^T W)
        I = torch.eye(cols, device=W.device, dtype=compute_dtype)
        for _ in range(itrs):
            G = Wc.transpose(0, 1) @ Wc          # [cols, cols]
            Wc = Wc @ ((1.0 + beta) * I - beta * G)
    else:
        # row-orthonormalize: W <- ((1+β)I - β W W^T) W
        I = torch.eye(rows, device=W.device, dtype=compute_dtype)
        for _ in range(itrs):
            G = Wc @ Wc.transpose(0, 1)          # [rows, rows]
            Wc = ((1.0 + beta) * I - beta * G) @ Wc

    # 3) copy back
    W.copy_(Wc.to(dtype=W.dtype))
    
    return sigma_state