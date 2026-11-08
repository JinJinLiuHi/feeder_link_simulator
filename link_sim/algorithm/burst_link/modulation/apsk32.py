"""Individual APSK constellation; bit labels are MSB first."""

import cmath
import math

from ._mapping import ConstellationModulator


def apsk32() -> ConstellationModulator:
    """Build 32APSK, constellation profile 3/4."""
    radii = (1.0, 2.84, 5.27)
    counts = (4, 12, 16)
    phases = (math.pi / 4, math.pi / 12, 0.0)
    labels = (17, 21, 23, 19, 16, 0, 1, 5, 4, 20, 22, 6, 7, 3, 2, 18, 24, 8, 25, 9, 13, 29, 12, 28, 30, 14, 31, 15, 11, 27, 10, 26)
    points = [0j] * 32
    offset = 0
    for radius, count, phase in zip(radii, counts, phases):
        for position in range(count):
            points[labels[offset + position]] = radius * cmath.exp(
                1j * (phase + 2 * math.pi * position / count)
            )
        offset += count
    modulator = ConstellationModulator(points)
    modulator.order = 32
    modulator.rate = '3/4'
    return modulator
