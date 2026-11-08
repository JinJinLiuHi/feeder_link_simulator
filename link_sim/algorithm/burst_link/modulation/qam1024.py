"""1024QAM 调制。"""
from __future__ import annotations

from ._mapping import ConstellationModulator, _gray_pam


def qam1024() -> ConstellationModulator:
    """创建 I/Q 交错比特、Gray 映射的 1024QAM 调制器。"""
    points = []
    for label in range(1024):
        bits = [(label >> shift) & 1 for shift in range(9, -1, -1)]
        points.append(complex(_gray_pam(bits[0::2]), _gray_pam(bits[1::2])))
    modulator = ConstellationModulator(points)
    modulator.order = 1024
    return modulator
