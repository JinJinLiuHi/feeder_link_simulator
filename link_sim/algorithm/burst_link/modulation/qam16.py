"""16QAM 调制。"""
from __future__ import annotations

from ._mapping import ConstellationModulator, _gray_pam


def qam16() -> ConstellationModulator:
    """创建 I/Q 交错比特、Gray 映射的 16QAM 调制器。"""
    points = []
    for label in range(16):
        bits = [(label >> shift) & 1 for shift in range(3, -1, -1)]
        points.append(complex(_gray_pam(bits[0::2]), _gray_pam(bits[1::2])))
    modulator = ConstellationModulator(points)
    modulator.order = 16
    return modulator
