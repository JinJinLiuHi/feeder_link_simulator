from __future__ import annotations

import numpy as np


def _gaussian_sum(
    log_frequency: float,
    a: tuple[float, ...],
    b: tuple[float, ...],
    c: tuple[float, ...],
    slope: float,
    intercept: float,
) -> float:
    terms = [ai * np.exp(-((log_frequency - bi) / ci) ** 2) for ai, bi, ci in zip(a, b, c)]
    return float(sum(terms) + slope * log_frequency + intercept)


def p838_coefficients(
    frequency_ghz: float,
    elevation_deg: np.ndarray | float,
    polarization_tilt_deg: float = 45.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ITU-R P.838-3 k and alpha for an Earth-space path.

    ``polarization_tilt_deg`` is measured from horizontal.  A tilt of 45 degrees
    also gives the circular-polarization/mean H-V coefficient.
    """
    frequency = float(np.clip(frequency_ghz, 1.0, 1000.0))
    log_frequency = float(np.log10(frequency))
    log_k_h = _gaussian_sum(
        log_frequency,
        (-5.33980, -0.35351, -0.23789, -0.94158),
        (-0.10008, 1.26970, 0.86036, 0.64552),
        (1.13098, 0.45400, 0.15354, 0.16817),
        -0.18961,
        0.71147,
    )
    log_k_v = _gaussian_sum(
        log_frequency,
        (-3.80595, -3.44965, -0.39902, 0.50167),
        (0.56934, -0.22911, 0.73042, 1.07319),
        (0.81061, 0.51059, 0.11899, 0.27195),
        -0.16398,
        0.63297,
    )
    alpha_h = _gaussian_sum(
        log_frequency,
        (-0.14318, 0.29591, 0.32177, -5.37610, 16.1721),
        (1.82442, 0.77564, 0.63773, -0.96230, -3.29980),
        (-0.55187, 0.19822, 0.13164, 1.47828, 3.43990),
        0.67849,
        -1.95537,
    )
    alpha_v = _gaussian_sum(
        log_frequency,
        (-0.07771, 0.56727, -0.20238, -48.2991, 48.5833),
        (2.33840, 0.95545, 1.14520, 0.791669, 0.791459),
        (-0.76284, 0.54039, 0.26809, 0.116226, 0.116479),
        -0.053739,
        0.83433,
    )
    k_h = 10.0**log_k_h
    k_v = 10.0**log_k_v
    elevation = np.asarray(elevation_deg, dtype=float)
    polarization_term = np.cos(np.deg2rad(elevation)) ** 2 * np.cos(
        2.0 * np.deg2rad(polarization_tilt_deg)
    )
    k = 0.5 * (k_h + k_v + (k_h - k_v) * polarization_term)
    alpha = (
        k_h * alpha_h
        + k_v * alpha_v
        + (k_h * alpha_h - k_v * alpha_v) * polarization_term
    ) / np.maximum(2.0 * k, 1e-12)
    return np.asarray(k, dtype=float), np.asarray(alpha, dtype=float)


def p838_specific_attenuation_db_km(
    rain_rate_mm_h: np.ndarray,
    frequency_ghz: float,
    elevation_deg: np.ndarray,
    polarization_tilt_deg: float = 45.0,
) -> np.ndarray:
    rain_rate = np.maximum(np.asarray(rain_rate_mm_h, dtype=float), 0.0)
    k, alpha = p838_coefficients(frequency_ghz, elevation_deg, polarization_tilt_deg)
    return k * rain_rate**alpha


def p618_effective_path_km(
    slant_rain_path_km: np.ndarray,
    specific_attenuation_db_km: np.ndarray,
    frequency_ghz: float,
    elevation_deg: np.ndarray,
    station_lat_deg: float,
) -> np.ndarray:
    """P.618-14 0.01%-path reduction used as a dynamic engineering mapping.

    P.618 defines this reduction for the R0.01 design point.  Here it is applied
    sample by sample to the instantaneous P.838 specific attenuation.  The path
    is capped by the geometric liquid path so the result cannot exceed a
    homogeneous full-path rain field.
    """
    slant_path = np.maximum(np.asarray(slant_rain_path_km, dtype=float), 0.0)
    gamma = np.maximum(np.asarray(specific_attenuation_db_km, dtype=float), 0.0)
    elevation = np.asarray(elevation_deg, dtype=float)
    elevation_rad = np.deg2rad(elevation)
    horizontal_path = slant_path * np.cos(elevation_rad)
    horizontal_reduction = 1.0 / np.maximum(
        1.0
        + 0.78 * np.sqrt(np.maximum(horizontal_path * gamma / max(frequency_ghz, 1e-9), 0.0))
        - 0.38 * (1.0 - np.exp(-2.0 * horizontal_path)),
        0.05,
    )
    reduced_slant_path = slant_path * horizontal_reduction
    chi = max(36.0 - abs(float(station_lat_deg)), 0.0)
    vertical_term = (
        31.0
        * (1.0 - np.exp(-elevation / (1.0 + chi)))
        * np.sqrt(np.maximum(reduced_slant_path * gamma, 0.0))
        / max(frequency_ghz**2, 1e-9)
        - 0.45
    )
    vertical_reduction = 1.0 / np.maximum(
        1.0 + np.sqrt(np.maximum(np.sin(elevation_rad), 0.0)) * vertical_term,
        0.1,
    )
    effective_path = reduced_slant_path * vertical_reduction
    effective_path = np.minimum(effective_path, slant_path)
    return np.where(gamma > 0.0, np.maximum(effective_path, 0.0), 0.0)
