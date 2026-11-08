"""Individual APSK constellation; bit labels are MSB first."""

import cmath
import math

from ._mapping import ConstellationModulator


def apsk256() -> ConstellationModulator:
    """Build 256APSK, constellation profile 116/180."""
    radii = (1.0, 1.791, 2.405, 2.98, 3.569, 4.235, 5.078, 6.536)
    ring_by_prefix = {'000': 0, '001': 1, '010': 3, '011': 2, '100': 7, '101': 6, '110': 4, '111': 5}
    base_by_suffix = {'000': 1/32, '001': 3/32, '010': 7/32, '011': 5/32, '100': 15/32, '101': 13/32, '110': 9/32, '111': 11/32}
    points = [0j] * 256
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
    modulator.order = 256
    modulator.rate = '116/180'
    return modulator
