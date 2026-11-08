"""Individual APSK constellation; bit labels are MSB first."""

import cmath
import math

from ._mapping import ConstellationModulator


def apsk16() -> ConstellationModulator:
    """Build 16APSK, constellation profile 3/4."""
    radii = (1.0, 2.85)
    counts = (4, 12)
    phases = (math.pi / 4, math.pi / 12)
    labels = (12, 14, 15, 13, 4, 0, 8, 10, 2, 6, 7, 3, 11, 9, 1, 5)
    points = [0j] * 16
    offset = 0
    for radius, count, phase in zip(radii, counts, phases):
        for position in range(count):
            points[labels[offset + position]] = radius * cmath.exp(
                1j * (phase + 2 * math.pi * position / count)
            )
        offset += count
    modulator = ConstellationModulator(points)
    modulator.order = 16
    modulator.rate = '3/4'
    return modulator
