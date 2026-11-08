"""Individual APSK constellation; bit labels are MSB first."""

import cmath
import math

from ._mapping import ConstellationModulator


def apsk64() -> ConstellationModulator:
    """Build 64APSK, constellation profile 128/180."""
    radii = (1.0, 1.88, 2.72, 3.95)
    ring_by_prefix = {'00': 0, '01': 1, '10': 3, '11': 2}
    base_by_suffix = {'00': 1/16, '01': 3/16, '10': 7/16, '11': 5/16}
    points = [0j] * 64
    for prefix, ring_index in ring_by_prefix.items():
        for q in (0, 1):
            for p in (0, 1):
                for suffix, base in base_by_suffix.items():
                    if (p, q) == (0, 0):
                        angle = base
                    elif (p, q) == (0, 1):
                        angle = 2 - base
                    elif (p, q) == (1, 0):
                        angle = 1 - base
                    else:
                        angle = 1 + base
                    label = int(f"{prefix}{q}{p}{suffix}", 2)
                    points[label] = radii[ring_index] * cmath.exp(1j * math.pi * angle)
    modulator = ConstellationModulator(points)
    modulator.order = 64
    modulator.rate = '128/180'
    return modulator
