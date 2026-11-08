"""Soft demodulation against registered complex constellation points."""
from __future__ import annotations
import math
import operator
import torch

@torch.no_grad()
def soft_demodulate(symbols: torch.Tensor, modulation: object, noise_variance: float) -> torch.Tensor:
    """Compute max-log logits, log(P(bit=1)/P(bit=0)), in MSB-first order.

    symbols: [L] or [W,L] complex tensor. Output: [L*m] or [W,L*m].
    noise_variance is E[|n|^2] at the normalized symbol input. The caller
    must provide an estimate; its accuracy affects the decoder metrics.
    """
    if not isinstance(symbols, torch.Tensor):
        raise TypeError("symbols 必须为 torch.Tensor")
    if symbols.ndim not in (1, 2) or not symbols.is_complex():
        raise ValueError("symbols 必须是一维或二维复数张量")
    if not bool(torch.isfinite(symbols).all().item()):
        raise ValueError("symbols 不能包含非有限值")
    if isinstance(noise_variance, bool):
        raise TypeError("noise_variance 必须为正实数")
    try:
        variance = float(noise_variance)
    except (ValueError, TypeError, OverflowError) as exc:
        raise TypeError("noise_variance 必须为正实数") from exc
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError("noise_variance 必须为有限正数")
    if isinstance(getattr(modulation, "bits_per_symbol", None), bool):
        raise TypeError("调制器必须提供整数 bits_per_symbol")
    try:
        m = operator.index(modulation.bits_per_symbol)
    except (TypeError, AttributeError) as exc:
        raise TypeError("调制器必须提供整数 bits_per_symbol") from exc
    if m <= 0:
        raise ValueError("调制器 bits_per_symbol 必须为正数")
    points_fn = getattr(modulation, "constellation_points", None)
    if not callable(points_fn):
        raise TypeError("调制器必须提供 constellation_points(device)")
    points = points_fn(symbols.device)
    if not isinstance(points, torch.Tensor) or points.ndim != 1 or not points.is_complex():
        raise TypeError("constellation_points 必须返回一维复数张量")
    if points.device != symbols.device:
        raise ValueError("星座与输入符号必须在同一设备")
    if points.numel() != 2**m:
        raise ValueError("星座点数与每符号位数不匹配")
    if not bool(torch.isfinite(points).all().item()):
        raise ValueError("星座点不能包含非有限值")
    points = points.to(dtype=symbols.dtype)
    flat = symbols.reshape(-1)
    indices = torch.arange(points.numel(), device=symbols.device, dtype=torch.long)
    masks = tuple(((indices >> bit) & 1).bool() for bit in range(m - 1, -1, -1))
    # Bound the temporary distance matrix for large constellations.
    chunk_size = max(1, 1_048_576 // points.numel())
    chunks: list[torch.Tensor] = []
    for start in range(0, flat.numel(), chunk_size):
        distances = (flat[start:start + chunk_size, None] - points[None, :]).abs().square()
        per_bit = [(distances[:, ~mask].amin(dim=-1) - distances[:, mask].amin(dim=-1))
                   / variance for mask in masks]
        chunks.append(torch.stack(per_bit, dim=-1))
    if chunks:
        logits = torch.cat(chunks, dim=0)
    else:
        logits = torch.empty((0, m), dtype=symbols.real.dtype, device=symbols.device)
    return logits.reshape(*symbols.shape[:-1], symbols.shape[-1] * m)
