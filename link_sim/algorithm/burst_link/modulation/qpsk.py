"""QPSK 星座与调制器。"""
from __future__ import annotations

import cmath
import math

from ._mapping import ConstellationModulator


def qpsk() -> ConstellationModulator:
    """创建单位平均能量的 QPSK 调制器，使用 Gray 标号。"""
    order = 4
    phase = math.pi / 4
    width = order.bit_length() - 1
    points = [0j] * order
    for position in range(order):
        gray = position ^ (position >> 1)
        label = int(f"{gray:0{width}b}"[::-1], 2)
        points[label] = cmath.exp(1j * (phase + 2 * math.pi * position / order))
    modulator = ConstellationModulator(points)
    modulator.order = order
    return modulator
