from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


GRID_STEP_DEG = 1.5
RAIN_HEIGHT_OFFSET_KM = 0.36


def _default_grid_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "data" / "P.839-4 rain_height" / "h0.txt"
    )


@lru_cache(maxsize=1)
def _load_zero_isotherm_grid() -> np.ndarray:
    path = _default_grid_path()
    if not path.exists():
        raise FileNotFoundError(f"缺少ITU-R P.839数字地图：{path}")
    grid = np.loadtxt(path, dtype=float)
    if grid.shape != (121, 241):
        raise ValueError(f"ITU-R P.839数字地图尺寸异常：{grid.shape}，期望(121, 241)。")
    return grid


def zero_isotherm_height_km(latitude_deg: float, longitude_deg: float) -> float:
    """Return mean annual 0 °C isotherm height h0 (km AMSL), per ITU-R P.839-4.

    The official 1.5 degree grid is bilinearly interpolated. Longitude is wrapped
    into [0, 360), while latitude is constrained to the valid [-90, 90] range.
    """

    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("纬度必须位于-90°～90°。")
    longitude = longitude_deg % 360.0
    row = (90.0 - latitude_deg) / GRID_STEP_DEG
    col = longitude / GRID_STEP_DEG
    i0 = min(int(np.floor(row)), 119)
    j0 = min(int(np.floor(col)), 239)
    di = row - i0
    dj = col - j0
    grid = _load_zero_isotherm_grid()
    h00 = grid[i0, j0]
    h10 = grid[i0 + 1, j0]
    h01 = grid[i0, j0 + 1]
    h11 = grid[i0 + 1, j0 + 1]
    return float(
        h00 * (1.0 - di) * (1.0 - dj)
        + h10 * di * (1.0 - dj)
        + h01 * (1.0 - di) * dj
        + h11 * di * dj
    )


def rain_height_km(latitude_deg: float, longitude_deg: float) -> tuple[float, float]:
    """Return ``(h0, hR)`` in km above mean sea level.

    ITU-R P.839-4 defines the mean annual rain height as hR = h0 + 0.36 km.
    """

    h0 = zero_isotherm_height_km(latitude_deg, longitude_deg)
    return h0, h0 + RAIN_HEIGHT_OFFSET_KM
