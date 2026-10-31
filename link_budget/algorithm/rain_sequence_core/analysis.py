from __future__ import annotations

import numpy as np


def acf_curve(values: np.ndarray, dt_s: float, max_seconds: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    n = centered.size
    spectrum = np.fft.rfft(centered, n=2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    acf /= np.maximum(np.arange(n, 0, -1), 1)
    if acf[0] > 0:
        acf /= acf[0]
    count = min(n, max(2, int(max_seconds / max(dt_s, 1e-9))))
    return np.arange(count) * dt_s, acf[:count]


def ccdf_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=float))
    exceedance = 1.0 - (np.arange(ordered.size) + 0.5) / ordered.size
    return ordered, exceedance


def downsample_indices(n: int, max_points: int = 5000) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points, dtype=int)


def minmax_lod_indices(
    values: np.ndarray,
    start: int = 0,
    stop: int | None = None,
    max_points: int = 12000,
) -> np.ndarray:
    """Return ordered min/max indices for a visible time-series window.

    Small windows are returned at full resolution.  Large windows retain both
    extrema of each time bucket so short peaks are not lost as they are with
    uniform point sampling.
    """
    array = np.asarray(values)
    size = array.size
    start = max(0, min(int(start), size))
    stop = size if stop is None else max(start, min(int(stop), size))
    count = stop - start
    if count <= max_points:
        return np.arange(start, stop, dtype=int)
    bucket_count = max(1, (max_points - 2) // 2)
    edges = np.linspace(start, stop, bucket_count + 1, dtype=int)
    selected: list[int] = [start]
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        segment = array[left:right]
        finite = np.isfinite(segment)
        if not np.any(finite):
            selected.extend((left, right - 1))
            continue
        finite_indices = np.flatnonzero(finite)
        finite_values = segment[finite]
        minimum = left + int(finite_indices[int(np.argmin(finite_values))])
        maximum = left + int(finite_indices[int(np.argmax(finite_values))])
        selected.extend((minimum, maximum) if minimum <= maximum else (maximum, minimum))
    selected.append(stop - 1)
    return np.unique(np.asarray(selected, dtype=int))
