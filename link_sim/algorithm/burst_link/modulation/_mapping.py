"""Complex constellation mapping for one-dimensional or word-oriented bits."""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def _gray_pam(bits: Sequence[int]) -> int:
    """根据 Gray 标签计算一维等间距幅度。"""
    if len(bits) == 1:
        return 1 - 2 * bits[0]
    return (1 - 2 * bits[0]) * (2 ** (len(bits) - 1) - _gray_pam(bits[1:]))


class ConstellationModulator:
    """Map MSB-first bit groups to a fixed, unit-average-energy constellation.

    ``points[label]`` is the symbol for the corresponding binary label.
    This class performs the mapping itself; individual modulation modules
    supply their own constellation geometry and labeling.
    """

    def __init__(self, points: Sequence[complex]) -> None:
        if len(points) < 2 or len(points) & (len(points) - 1):
            raise ValueError("constellation size must be a power of two >= 2")
        self.bits_per_symbol = int(math.log2(len(points)))
        constellation = torch.tensor(points, dtype=torch.complex64)
        if not torch.isfinite(constellation).all():
            raise ValueError("constellation points must be finite")
        energy = constellation.abs().square().mean()
        if energy <= 0:
            raise ValueError("constellation must have positive average energy")
        self._cpu_points = constellation / energy.sqrt()
        self._device_points: dict[torch.device, torch.Tensor] = {
            self._cpu_points.device: self._cpu_points
        }

    def _points_on(self, device: torch.device) -> torch.Tensor:
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device not in self._device_points:
            self._device_points[device] = self._cpu_points.to(device=device)
        return self._device_points[device]

    def constellation_points(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return normalized, label-ordered points without exposing the cache."""
        return self._points_on(torch.device(device)).clone()

    def modulate(self, bits: torch.Tensor) -> torch.Tensor:
        """Map [L] or [W,L] binary bits to complex symbols on the same device."""
        if not isinstance(bits, torch.Tensor):
            raise TypeError("bits must be a torch.Tensor")
        if bits.ndim not in (1, 2):
            raise ValueError("bits must have shape [L] or [W,L]")
        if bits.is_complex():
            raise ValueError("bits must be real binary values")
        m = self.bits_per_symbol
        if bits.shape[-1] % m:
            raise ValueError("last bit dimension must be divisible by bits_per_symbol")
        if torch.any((bits != 0) & (bits != 1)).item():
            raise ValueError("bits must contain only 0 or 1")
        num_symbols = bits.shape[-1] // m
        groups = bits.reshape(*bits.shape[:-1], num_symbols, m).to(torch.int64)
        weights = torch.arange(m - 1, -1, -1, dtype=torch.int64, device=bits.device)
        indices = torch.bitwise_left_shift(groups, weights).sum(dim=-1)
        return self._points_on(bits.device)[indices]
