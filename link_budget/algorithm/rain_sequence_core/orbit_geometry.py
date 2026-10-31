from __future__ import annotations

import math

import numpy as np


EARTH_RADIUS_KM = 6371.0
EARTH_MU_KM3_S2 = 398600.4418


def leo_altitude_km(orbit_or_altitude: str | float) -> float:
    """Resolve a generic altitude or a legacy built-in LEO scene name."""
    if isinstance(orbit_or_altitude, (int, float)):
        return float(orbit_or_altitude)
    if "500" in orbit_or_altitude:
        return 500.0
    if "1200" in orbit_or_altitude:
        return 1200.0
    raise ValueError(f"无法从“{orbit_or_altitude}”确定LEO轨道高度。")


def central_angle_from_elevation_deg(elevation_deg: float, altitude_km: float) -> float:
    """Earth-centred station/sub-satellite angle for a specified elevation."""
    elevation_rad = math.radians(elevation_deg)
    orbital_radius_km = EARTH_RADIUS_KM + altitude_km
    argument = (EARTH_RADIUS_KM / orbital_radius_km) * math.cos(elevation_rad)
    return math.acos(max(-1.0, min(1.0, argument))) - elevation_rad


def leo_visible_duration_s(
    orbit_or_altitude: str | float,
    elevation_min_deg: float,
    elevation_max_deg: float,
) -> float:
    """Maximum centred pass duration between the two minimum-elevation crossings.

    The built-in LEO scenes use a circular orbit without Earth rotation.  The
    maximum elevation defines the pass cross-track offset; therefore an
    elevation_max below 90 degrees still produces a shorter, off-centre pass.
    """
    altitude_km = leo_altitude_km(orbit_or_altitude)
    orbital_radius_km = EARTH_RADIUS_KM + altitude_km
    mean_motion_rad_s = math.sqrt(EARTH_MU_KM3_S2 / orbital_radius_km**3)
    edge_angle = central_angle_from_elevation_deg(elevation_min_deg, altitude_km)
    closest_angle = central_angle_from_elevation_deg(elevation_max_deg, altitude_km)
    ratio = math.cos(edge_angle) / max(math.cos(closest_angle), 1e-15)
    half_duration_s = math.acos(max(-1.0, min(1.0, ratio))) / mean_motion_rad_s
    return 2.0 * half_duration_s


def leo_elevation_profile_deg(
    time_s: np.ndarray,
    orbit_or_altitude: str | float,
    collection_duration_s: float,
    elevation_min_deg: float,
    elevation_max_deg: float,
) -> np.ndarray:
    """Calculate a physical, TCA-centred elevation profile for a visible window."""
    altitude_km = leo_altitude_km(orbit_or_altitude)
    orbital_radius_km = EARTH_RADIUS_KM + altitude_km
    mean_motion_rad_s = math.sqrt(EARTH_MU_KM3_S2 / orbital_radius_km**3)
    closest_angle = central_angle_from_elevation_deg(elevation_max_deg, altitude_km)
    relative_time_s = np.asarray(time_s, dtype=float) - collection_duration_s / 2.0
    cosine_angle = np.cos(closest_angle) * np.cos(mean_motion_rad_s * relative_time_s)
    central_angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    elevation_rad = np.arctan2(
        np.cos(central_angle) - EARTH_RADIUS_KM / orbital_radius_km,
        np.maximum(np.sin(central_angle), 1e-15),
    )
    return np.rad2deg(elevation_rad)


def slant_range_from_elevation_km(elevation_deg: np.ndarray, altitude_km: float) -> np.ndarray:
    """Station-to-satellite slant range for spherical Earth and circular orbit."""
    elevation_rad = np.deg2rad(np.asarray(elevation_deg, dtype=float))
    orbital_radius_km = EARTH_RADIUS_KM + altitude_km
    projected_km = EARTH_RADIUS_KM * np.cos(elevation_rad)
    return (
        np.sqrt(np.maximum(orbital_radius_km**2 - projected_km**2, 0.0))
        - EARTH_RADIUS_KM * np.sin(elevation_rad)
    )
