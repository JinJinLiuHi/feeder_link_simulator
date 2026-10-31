from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import signal, stats

from .config import HYDROMETEOR_PHASE_MODES, ModelConfig, OutputConfig, SimulationConfig, config_to_dict
from .orbit_geometry import leo_altitude_km, leo_elevation_profile_deg, slant_range_from_elevation_km
from .rain_physics import p618_effective_path_km, p838_specific_attenuation_db_km
from .rain_height import rain_height_km as calculate_itu_rain_height


@dataclass(slots=True)
class SimulationResult:
    time_s: np.ndarray
    data: dict[str, np.ndarray]
    metrics: dict[str, float]
    metadata: dict[str, Any]


def _param(cfg: ModelConfig, name: str, default: float) -> float:
    value = cfg.parameters.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _ar1_noise(n: int, dt_s: float, tau_s: float, rng: np.random.Generator) -> np.ndarray:
    tau_s = max(tau_s, dt_s / 10.0)
    a = float(np.exp(-dt_s / tau_s))
    b = float(np.sqrt(max(1.0 - a * a, 1e-12)))
    white = rng.standard_normal(n)
    return signal.lfilter([b], [1.0, -a], white)


def _rank_to_gamma(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
    ranks = stats.rankdata(x, method="average") / (x.size + 1.0)
    return stats.gamma.ppf(np.clip(ranks, 1e-6, 1 - 1e-6), a=max(shape, 0.2), scale=max(scale, 0.01))


def _event_envelope(
    n: int,
    dt_s: float,
    mean_duration_s: float,
    mean_gap_s: float,
    rng: np.random.Generator,
    rise_fraction: float = 0.45,
    shape_power: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    env = np.zeros(n, dtype=float)
    event_id = np.zeros(n, dtype=np.int32)
    first_window = max(1, min(int(mean_gap_s / dt_s), max(n - 1, 1), max(int(0.35 * n), 1)))
    cursor = int(rng.uniform(0, first_window))
    eid = 0
    while cursor < n:
        eid += 1
        duration = max(8, int(rng.lognormal(np.log(max(mean_duration_s, dt_s)), 0.42) / dt_s))
        full_end = cursor + duration
        end = min(n, full_end)
        idx = np.arange(cursor, end)
        rise_fraction = float(np.clip(rise_fraction, 0.12, 0.82))
        peak_phase = float(np.clip(rng.normal(rise_fraction, 0.035), 0.1, 0.9))
        phase = (idx - cursor) / max(duration - 1, 1)
        rise_phase = np.clip(phase / peak_phase, 0.0, 1.0)
        fall_phase = np.clip((1.0 - phase) / (1.0 - peak_phase), 0.0, 1.0)
        pulse = np.where(
            phase <= peak_phase,
            np.sin(0.5 * np.pi * rise_phase) ** max(shape_power, 0.5),
            np.sin(0.5 * np.pi * fall_phase) ** max(shape_power, 0.5),
        )
        env[idx] = np.maximum(env[idx], pulse)
        event_id[idx] = eid
        gap = max(1, int(rng.exponential(max(mean_gap_s, dt_s)) / dt_s))
        cursor = full_end + gap
    return env, event_id


def _rain_mb(n: int, dt_s: float, cfg: ModelConfig, rng: np.random.Generator) -> np.ndarray:
    beta = max(_param(cfg, "mb_beta_s_inv", 0.055), 1e-4)
    mu = _param(cfg, "mb_log_mean", 1.05)
    sigma = max(_param(cfg, "mb_log_sigma", 0.72), 0.05)
    floor = max(_param(cfg, "mb_floor_db", 0.15), 0.0)
    intermittency = float(np.clip(_param(cfg, "mb_intermittency", 0.2), 0.0, 0.95))
    latent = _ar1_noise(n, dt_s, 1.0 / beta, rng)
    attenuation = np.exp(mu + sigma * latent)
    gate_state = _ar1_noise(n, dt_s, max(4.0 / beta, dt_s), rng)
    gate = np.clip((gate_state + 1.1 - 2.0 * intermittency) / 1.4, 0.0, 1.0)
    return floor + attenuation * gate


def _rain_itu(n: int, dt_s: float, cfg: ModelConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    tau = _param(cfg, "itu_tau_s", 12.0)
    marginal_scale = max(
        _param(cfg, "itu_marginal_scale", _param(cfg, "itu_a001_db", 28.0) / 13.0),
        0.2,
    )
    shape = _param(cfg, "itu_gamma_shape", 1.8)
    event_mean = _param(cfg, "event_mean_s", 120.0)
    event_gap = _param(cfg, "event_gap_s", 240.0)
    rise = _param(cfg, "itu_rise_fraction", 0.34)
    fast_tau = max(_param(cfg, "itu_fast_tau_s", 1.8), dt_s)
    fast_sigma = max(_param(cfg, "itu_fast_sigma", 0.18), 0.0)
    envelope, labels = _event_envelope(n, dt_s, event_mean, event_gap, rng, rise_fraction=rise, shape_power=1.35)
    latent = _ar1_noise(n, dt_s, tau, rng)
    marginal = _rank_to_gamma(latent, shape, marginal_scale)
    fast = _ar1_noise(n, dt_s, fast_tau, rng)
    modulation = np.exp(fast_sigma * fast)
    return np.maximum(0.0, envelope * (1.2 + marginal) * modulation), labels


def _rain_onera(n: int, dt_s: float, cfg: ModelConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    fast_tau = _param(cfg, "fast_tau_s", cfg.fast_tau_s)
    slow_tau = _param(cfg, "slow_tau_s", cfg.slow_tau_s)
    fast_sigma = _param(cfg, "onera_fast_sigma", 0.28)
    slow_shape = _param(cfg, "onera_slow_shape", cfg.marginal_shape)
    peak_scale = _param(cfg, "onera_peak_scale_db", 4.2)
    event_mean = _param(cfg, "event_mean_s", cfg.event_mean_s)
    event_gap = _param(cfg, "event_gap_s", cfg.event_gap_s)
    rise = _param(cfg, "onera_rise_fraction", 0.42)
    envelope, labels = _event_envelope(n, dt_s, event_mean, event_gap, rng, rise_fraction=rise, shape_power=2.2)
    slow = _ar1_noise(n, dt_s, slow_tau, rng)
    fast = _ar1_noise(n, dt_s, fast_tau, rng)
    slow_power = _rank_to_gamma(slow, slow_shape, peak_scale)
    modulation = np.exp(fast_sigma * fast)
    rain = envelope * (0.8 + slow_power) * modulation
    return np.maximum(0.0, rain), labels


def _rain_sst(n: int, dt_s: float, cfg: ModelConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    wind_speed = _param(cfg, "sst_wind_speed_m_s", cfg.wind_speed_m_s)
    cell_length_km = max(_param(cfg, "sst_cell_length_km", 2.4), 0.05)
    spectral_slope = max(_param(cfg, "sst_spectral_slope", 1.67), 0.3)
    convective_ratio = np.clip(_param(cfg, "sst_convective_ratio", 0.68), 0.0, 1.0)
    melting_factor = max(_param(cfg, "sst_melting_factor", 1.08), 0.2)
    event_mean = _param(cfg, "event_mean_s", 150.0)
    event_gap = _param(cfg, "event_gap_s", 190.0)
    envelope, labels = _event_envelope(n, dt_s, event_mean, event_gap, rng, rise_fraction=0.28, shape_power=1.1)
    white = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n, d=dt_s)
    corner = max(wind_speed / (cell_length_km * 1000.0), 1e-5)
    shaping = 1.0 / np.sqrt(1.0 + (freqs / corner) ** spectral_slope)
    field = np.fft.irfft(np.fft.rfft(white) * shaping, n=n)
    field = (field - field.mean()) / max(field.std(), 1e-9)
    stratiform = 2.0 + 2.2 * np.maximum(_ar1_noise(n, dt_s, event_mean * 0.7, rng), -0.8)
    convective = 5.0 + 5.8 * np.maximum(field, -0.82)
    rain = melting_factor * envelope * ((1.0 - convective_ratio) * stratiform + convective_ratio * convective)
    return np.maximum(0.0, rain), labels


def _rain_excell(n: int, dt_s: float, cfg: ModelConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n) * dt_s
    rain = np.zeros(n)
    labels = np.zeros(n, dtype=np.int32)
    total_cells = max(1, int(round(_param(cfg, "excell_cell_count", cfg.cell_count))))
    radius_km = max(_param(cfg, "excell_radius_km", 1.1), 0.03)
    wind_speed = max(_param(cfg, "excell_wind_speed_m_s", cfg.wind_speed_m_s), 0.1)
    peak_shape = max(_param(cfg, "excell_peak_shape", cfg.marginal_shape), 0.2)
    peak_scale = max(_param(cfg, "excell_peak_scale_db", 4.0), 0.05)
    background = max(_param(cfg, "excell_background_db", 0.0), 0.0)
    for cell in range(1, total_cells + 1):
        center = rng.uniform(0, max(t[-1], dt_s))
        sigma = rng.lognormal(np.log(max(radius_km * 1000.0 / wind_speed, dt_s)), 0.34)
        peak = rng.gamma(peak_shape, peak_scale)
        radial_power = np.clip(_param(cfg, "excell_radial_power", 1.7), 0.5, 4.0)
        contribution = peak * np.exp(-0.5 * np.abs((t - center) / sigma) ** radial_power)
        rain += contribution
        labels[contribution > max(0.15 * peak, 0.1)] = cell
    return rain + background, labels


def _rain_gamma(n: int, dt_s: float, cfg: ModelConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    tau = _param(cfg, "gamma_tau_s", 45.0)
    shape = _param(cfg, "gamma_shape", 2.2)
    scale = _param(cfg, "gamma_scale_db", 3.8)
    dry_probability = np.clip(_param(cfg, "gamma_dry_probability", 0.45), 0.0, 0.98)
    colored = _ar1_noise(n, dt_s, tau, rng)
    gamma_process = _rank_to_gamma(colored, shape, scale)
    gate_process = _ar1_noise(n, dt_s, tau * 2.4, rng)
    gate_threshold = stats.norm.ppf(dry_probability)
    wet_state = gate_process > gate_threshold
    wet = 1.0 / (1.0 + np.exp(-np.clip((gate_process - gate_threshold) / 0.22, -30.0, 30.0)))
    labels = np.zeros(n, dtype=np.int32)
    transitions = np.flatnonzero(np.diff(np.r_[False, wet_state]) > 0)
    for event_number, start in enumerate(transitions, 1):
        end_candidates = np.flatnonzero(~wet_state[start:])
        end = start + end_candidates[0] if end_candidates.size else n
        labels[start:end] = event_number
    return gamma_process * wet, labels


def _generate_reference_rain(
    n: int, dt_s: float, cfg: ModelConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    name = cfg.rain_model
    if name == "Maseng–Bakken":
        rain = _rain_mb(n, dt_s, cfg, rng)
        labels = (rain > np.quantile(rain, 0.65)).astype(np.int32)
    elif name == "ITU-R P.1853":
        rain, labels = _rain_itu(n, dt_s, cfg, rng)
    elif name == "SST/E-SST":
        rain, labels = _rain_sst(n, dt_s, cfg, rng)
    elif name == "EXCELL/Multi-EXCELL":
        rain, labels = _rain_excell(n, dt_s, cfg, rng)
    elif name == "Gamma相关过程":
        rain, labels = _rain_gamma(n, dt_s, cfg, rng)
    elif name == "融合模型":
        onera, labels = _rain_onera(n, dt_s, cfg, rng)
        excell, labels2 = _rain_excell(n, dt_s, cfg, rng)
        w1 = np.clip(_param(cfg, "fusion_dynamic_weight", cfg.fusion_weight_dynamic), 0.0, 1.0)
        w2 = np.clip(_param(cfg, "fusion_event_weight", cfg.fusion_weight_event), 0.0, 1.0)
        tail_boost = max(_param(cfg, "fusion_tail_boost", 1.15), 0.1)
        total = max(w1 + w2, 1e-9)
        if cfg.fusion_mode == "事件门控融合":
            gate = signal.lfilter([0.08], [1.0, -0.92], (labels2 > 0).astype(float))
            gate = np.clip(gate, 0.0, 1.0)
            event_mix = (w1 * onera + w2 * excell) / total
            rain = (1.0 - gate) * onera + gate * event_mix
        elif cfg.fusion_mode == "验证误差加权":
            target_p99 = _param(cfg, "fusion_target_p99_db", 25.0)
            onera_error = abs(float(np.quantile(onera, 0.99)) - target_p99) + 1e-6
            excell_error = abs(float(np.quantile(excell, 0.99)) - target_p99) + 1e-6
            adaptive_onera = 1.0 / onera_error
            adaptive_excell = 1.0 / excell_error
            rain = (adaptive_onera * onera + adaptive_excell * excell) / (adaptive_onera + adaptive_excell)
        elif cfg.fusion_mode == "单模型":
            rain = onera
            labels2[:] = 0
        else:
            rain = (w1 * onera + w2 * excell) / total
        if cfg.fusion_mode != "单模型":
            threshold = np.quantile(rain, 0.9)
            rain = np.where(rain > threshold, threshold + (rain - threshold) * tail_boost, rain)
        labels = np.maximum(labels, labels2)
    else:
        rain, labels = _rain_onera(n, dt_s, cfg, rng)
    return rain, labels


def _geo_elevation_and_range(cfg: SimulationConfig) -> tuple[float, float]:
    earth_radius_km = 6371.0
    orbital_radius_km = earth_radius_km + cfg.geo_altitude_km
    latitude = np.deg2rad(cfg.station_lat_deg)
    longitude_delta = np.deg2rad(((cfg.geo_longitude_deg - cfg.station_lon_deg + 180.0) % 360.0) - 180.0)
    cos_central = float(np.clip(np.cos(latitude) * np.cos(longitude_delta), -1.0, 1.0))
    central = float(np.arccos(cos_central))
    elevation = np.rad2deg(np.arctan2(np.cos(central) - earth_radius_km / orbital_radius_km, max(np.sin(central), 1e-12)))
    slant_range = np.sqrt(earth_radius_km**2 + orbital_radius_km**2 - 2.0 * earth_radius_km * orbital_radius_km * cos_central)
    return float(elevation), float(slant_range)


def _elevation_profile(time_s: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    if cfg.orbit == "GEO":
        elevation, _distance = _geo_elevation_and_range(cfg)
        if elevation <= 0.0:
            raise ValueError("当前信关站不可见该GEO卫星，请调整GEO星下点经度。")
        return np.full_like(time_s, elevation, dtype=float)
    if cfg.orbit in ("LEO", "500 km LEO", "1200 km LEO"):
        altitude_km = (
            500.0 if cfg.orbit == "500 km LEO"
            else 1200.0 if cfg.orbit == "1200 km LEO"
            else cfg.leo_altitude_km
        )
        return leo_elevation_profile_deg(
            time_s,
            altitude_km,
            cfg.duration_s,
            cfg.elevation_min_deg,
            cfg.elevation_max_deg,
        )
    # External ephemerides are not yet imported into the generator.  Retain a
    # neutral preview profile rather than applying a built-in LEO altitude.
    return np.full_like(time_s, cfg.elevation_max_deg, dtype=float)


def _slant_range_km(elevation_deg: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    if cfg.orbit == "GEO":
        _elevation, distance = _geo_elevation_and_range(cfg)
        return np.full_like(elevation_deg, distance, dtype=float)
    elif cfg.orbit in ("LEO", "500 km LEO", "1200 km LEO"):
        altitude_km = (
            leo_altitude_km(cfg.orbit) if cfg.orbit != "LEO" else cfg.leo_altitude_km
        )
        return slant_range_from_elevation_km(elevation_deg, altitude_km)
    return np.full_like(elevation_deg, np.nan, dtype=float)


def _rain_path_length_km(elevation_deg: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    return _path_to_height_km(cfg.rain_height_km, elevation_deg, cfg)


def _path_to_height_km(top_height_km: float, elevation_deg: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    height_difference = max(top_height_km - cfg.station_alt_km, 0.0)
    theta = np.deg2rad(elevation_deg)
    effective_earth_radius_km = 8500.0
    high_elevation = height_difference / np.maximum(np.sin(theta), 1e-12)
    low_elevation = 2.0 * height_difference / (
        np.sqrt(np.sin(theta) ** 2 + 2.0 * height_difference / effective_earth_radius_km) + np.sin(theta)
    )
    return np.where(elevation_deg >= 5.0, high_elevation, low_elevation)


def _slant_layer_path_km(
    lower_height_km: float,
    upper_height_km: float,
    elevation_deg: np.ndarray,
    cfg: SimulationConfig,
) -> np.ndarray:
    if upper_height_km <= lower_height_km:
        return np.zeros_like(elevation_deg, dtype=float)
    upper_path = _path_to_height_km(upper_height_km, elevation_deg, cfg)
    lower_path = _path_to_height_km(max(lower_height_km, cfg.station_alt_km), elevation_deg, cfg)
    return np.maximum(upper_path - lower_path, 0.0)


def _circular_components(
    h_hh: np.ndarray, h_hv: np.ndarray, h_vh: np.ndarray, h_vv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    jones = np.empty((h_hh.size, 2, 2), dtype=complex)
    jones[:, 0, 0] = h_hh
    jones[:, 0, 1] = h_hv
    jones[:, 1, 0] = h_vh
    jones[:, 1, 1] = h_vv
    linear_from_circular = np.array([[1.0, 1.0], [-1j, 1j]], dtype=complex) / np.sqrt(2.0)
    circular = np.einsum("ab,nbc,cd->nad", linear_from_circular.conj().T, jones, linear_from_circular)
    return circular[:, 0, 0], circular[:, 0, 1], circular[:, 1, 0], circular[:, 1, 1]


def _polarization_tilt_deg(polarization: str) -> float:
    if polarization == "单极化 H":
        return 0.0
    if polarization == "单极化 V":
        return 90.0
    return 45.0


def _activity_aligned_occurrence(
    activity: np.ndarray,
    n: int,
    dt_s: float,
    probability: float,
    mean_duration_s: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive wet events from the model activity while matching time occupancy.

    This avoids multiplying a model event by a second, unrelated random event.
    The 0.5 crossing has the requested occupancy; compact smooth-step shoulders
    retain finite event rise and decay slopes.
    """
    probability = float(np.clip(probability, 0.0, 1.0))
    if probability <= 0.0:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=np.int32)
    if probability >= 1.0:
        return np.ones(n, dtype=float), np.ones(n, dtype=np.int32)
    values = np.maximum(np.asarray(activity, dtype=float), 0.0)
    scale = float(np.quantile(values[values > 0.0], 0.5)) if np.any(values > 0.0) else 1.0
    score = np.log1p(values / max(scale, 1e-9))
    tie_breaker = _ar1_noise(n, dt_s, max(mean_duration_s * 0.45, dt_s), rng)
    score += 0.035 * (tie_breaker - np.median(tie_breaker))
    gate_tau_s = max(0.35, min(mean_duration_s * 0.015, 1.5))
    gate_a = float(np.exp(-dt_s / gate_tau_s))
    initial_state = signal.lfilter_zi([1.0 - gate_a], [1.0, -gate_a]) * score[0]
    score, _ = signal.lfilter([1.0 - gate_a], [1.0, -gate_a], score, zi=initial_state)
    threshold = float(np.quantile(score, 1.0 - probability))
    spread = float(np.quantile(score, 0.75) - np.quantile(score, 0.25))
    half_width = max(0.18 * spread, 1e-5)
    normalized = np.clip((score - (threshold - half_width)) / (2.0 * half_width), 0.0, 1.0)
    envelope = normalized * normalized * (3.0 - 2.0 * normalized)
    wet_state = score >= threshold
    starts = wet_state & ~np.r_[False, wet_state[:-1]]
    labels = (np.cumsum(starts, dtype=np.int32) * wet_state).astype(np.int32)
    return envelope, labels


def _rain_rate_series(
    raw_activity: np.ndarray,
    occurrence: np.ndarray,
    target_rate_mm_h: float,
    dt_s: float,
    correlation_time_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Map dimensionless model activity to a physical rain-rate series.

    The typical rain rate is the median over wet samples.  A correlated
    background prevents event models from becoming identically zero between
    their internal rain-cell features when occurrence probability is one.
    """
    activity = np.maximum(np.asarray(raw_activity, dtype=float), 0.0)
    positive = activity[activity > 0.0]
    native_scale = float(np.median(positive)) if positive.size else 1.0
    native = activity / max(native_scale, 1e-9)
    native = np.clip(native, 0.0, 12.0)
    background = np.exp(0.20 * _ar1_noise(activity.size, dt_s, max(correlation_time_s, dt_s), rng))
    background /= max(float(np.median(background)), 1e-9)
    conditional_rate = 0.88 * native + 0.12 * background
    relative_rate = conditional_rate * np.asarray(occurrence, dtype=float)
    wet = np.asarray(occurrence, dtype=float) >= 0.5
    if not np.any(wet):
        return np.zeros_like(relative_rate)
    wet_median = float(np.median(relative_rate[wet]))
    rate_ratio = relative_rate / max(wet_median, 1e-9)
    maximum_ratio = 2.4
    rate_ratio = maximum_ratio * np.tanh(rate_ratio / maximum_ratio)
    rate_ratio /= max(float(np.median(rate_ratio[wet])), 1e-9)
    return max(float(target_rate_mm_h), 0.0) * rate_ratio


def _model_rain_cell_length_km(model: ModelConfig) -> float:
    name = model.rain_model
    if name == "SST/E-SST":
        return max(_param(model, "sst_cell_length_km", 2.4), 0.2)
    if name == "EXCELL/Multi-EXCELL":
        return max(2.0 * _param(model, "excell_radius_km", 1.1), 0.2)
    if name == "ONERA双过程":
        return max(_param(model, "onera_cell_length_km", 1.0), 0.2)
    if name == "ITU-R P.1853":
        return max(_param(model, "itu_cell_length_km", 1.0), 0.2)
    if name == "Maseng–Bakken":
        return max(_param(model, "mb_cell_length_km", 1.0), 0.2)
    if name == "Gamma相关过程":
        return max(_param(model, "gamma_cell_length_km", 1.0), 0.2)
    return max(_param(model, "fusion_cell_length_km", 1.2), 0.2)


def _limit_path_by_rain_cell(
    effective_path_km: np.ndarray,
    geometric_path_km: np.ndarray,
    elevation_deg: np.ndarray,
    cell_length_km: float,
) -> np.ndarray:
    horizontal_projection = np.maximum(np.cos(np.deg2rad(elevation_deg)), 0.2)
    cell_limited_slant = cell_length_km / horizontal_projection
    return np.minimum(np.asarray(effective_path_km), np.minimum(geometric_path_km, cell_limited_slant))


def _coherence_time(acf: np.ndarray, dt_s: float, threshold: float) -> float:
    below = np.flatnonzero(acf < threshold)
    return float(below[0] * dt_s) if below.size else float((acf.size - 1) * dt_s)


def _acf_fft(x: np.ndarray, max_lags: int) -> np.ndarray:
    centered = x - np.mean(x)
    n = centered.size
    spectrum = np.fft.rfft(centered, n=2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    acf /= np.maximum(np.arange(n, 0, -1), 1)
    if acf[0] > 0:
        acf /= acf[0]
    return acf[:max_lags]


def _resample(data: np.ndarray, time_in: np.ndarray, time_out: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(data):
        real = np.interp(time_out, time_in, data.real)
        imag = np.interp(time_out, time_in, data.imag)
        return real + 1j * imag
    if np.issubdtype(data.dtype, np.integer):
        indices = np.searchsorted(time_in, time_out, side="left").clip(0, len(time_in) - 1)
        return data[indices]
    return np.interp(time_out, time_in, data)


def simulate_channel(
    sim: SimulationConfig,
    model: ModelConfig,
    output: OutputConfig,
    sequence_index: int = 0,
) -> SimulationResult:
    if sim.rain_height_mode.startswith("ITU-R"):
        h0, h_r = calculate_itu_rain_height(sim.station_lat_deg, sim.station_lon_deg)
        sim = replace(sim, zero_isotherm_height_km=h0, rain_height_km=h_r)
    else:
        sim = replace(sim, zero_isotherm_height_km=max(sim.rain_height_km - 0.36, sim.station_alt_km))
    sim.validate(output.output_dt_ms)
    if sim.hydrometeor_phase_mode not in HYDROMETEOR_PHASE_MODES:
        raise ValueError(f"未知水凝物相态模式：{sim.hydrometeor_phase_mode}")
    dt_ms = min(sim.internal_dt_ms, output.output_dt_ms)
    dt_s = dt_ms / 1000.0
    n = int(np.floor(sim.duration_s / dt_s)) + 1
    time_internal = np.arange(n, dtype=float) * dt_s
    rng = np.random.default_rng(sim.seed + sequence_index * 104729)
    elevation = _elevation_profile(time_internal, sim)
    slant_range_km = _slant_range_km(elevation, sim)
    rain_path_length_km = _rain_path_length_km(elevation, sim)
    slant = 1.0 / np.maximum(np.sin(np.deg2rad(elevation)), np.sin(np.deg2rad(5.0)))
    slant /= np.median(slant)

    raw_activity, _native_event_id = _generate_reference_rain(n, dt_s, model, rng)
    event_mean_s = max(_param(model, "event_mean_s", model.event_mean_s), 8.0 * dt_s)
    occurrence, event_id = _activity_aligned_occurrence(
        raw_activity,
        n,
        dt_s,
        sim.rain_probability if sim.enable_rain else 0.0,
        event_mean_s,
        rng,
    )
    rain_rate_series = _rain_rate_series(
        raw_activity,
        occurrence,
        sim.rain_rate_mm_h,
        dt_s,
        max(_param(model, "itu_tau_s", model.slow_tau_s), dt_s),
        rng,
    )
    hydrometeor_ref = rain_rate_series
    if not sim.enable_rain or sim.rain_probability == 0.0:
        hydrometeor_ref[:] = 0.0
        rain_rate_series[:] = 0.0
        event_id[:] = 0

    melting_lower_km = max(sim.station_alt_km, sim.zero_isotherm_height_km - sim.melting_layer_thickness_km / 2.0)
    melting_upper_km = max(melting_lower_km, sim.zero_isotherm_height_km + sim.melting_layer_thickness_km / 2.0)
    rain_phase_path_km = _path_to_height_km(melting_lower_km, elevation, sim)
    melting_path_km = _slant_layer_path_km(melting_lower_km, melting_upper_km, elevation, sim)
    frozen_path_km = _slant_layer_path_km(
        melting_upper_km,
        melting_upper_km + sim.ice_layer_thickness_km,
        elevation,
        sim,
    )
    surface_snow_path_km = _slant_layer_path_km(
        sim.station_alt_km,
        sim.station_alt_km + sim.snow_layer_thickness_km,
        elevation,
        sim,
    )
    mode = sim.hydrometeor_phase_mode
    if mode in ("分层混合相（推荐）",):
        active_rain_path = rain_phase_path_km
    elif mode == "纯降雪":
        active_rain_path = np.zeros_like(rain_path_length_km)
    else:
        active_rain_path = rain_path_length_km
    polarization_tilt = _polarization_tilt_deg(sim.polarization)
    gamma_u = p838_specific_attenuation_db_km(
        rain_rate_series, sim.uplink_ghz, elevation, polarization_tilt
    )
    gamma_d = p838_specific_attenuation_db_km(
        rain_rate_series, sim.downlink_ghz, elevation, polarization_tilt
    )
    effective_rain_path_u = p618_effective_path_km(
        active_rain_path, gamma_u, sim.uplink_ghz, elevation, sim.station_lat_deg
    )
    effective_rain_path_d = p618_effective_path_km(
        active_rain_path, gamma_d, sim.downlink_ghz, elevation, sim.station_lat_deg
    )
    rain_cell_length_km = _model_rain_cell_length_km(model)
    effective_rain_path_u = _limit_path_by_rain_cell(
        effective_rain_path_u, active_rain_path, elevation, rain_cell_length_km
    )
    effective_rain_path_d = _limit_path_by_rain_cell(
        effective_rain_path_d, active_rain_path, elevation, rain_cell_length_km
    )
    rain_u = gamma_u * effective_rain_path_u
    rain_d = gamma_d * effective_rain_path_d
    melting_u = np.zeros_like(rain_u)
    melting_d = np.zeros_like(rain_d)
    if mode == "分层混合相（推荐）":
        effective_melting_path_u = p618_effective_path_km(
            melting_path_km, gamma_u, sim.uplink_ghz, elevation, sim.station_lat_deg
        )
        effective_melting_path_d = p618_effective_path_km(
            melting_path_km, gamma_d, sim.downlink_ghz, elevation, sim.station_lat_deg
        )
        effective_melting_path_u = _limit_path_by_rain_cell(
            effective_melting_path_u, melting_path_km, elevation, rain_cell_length_km
        )
        effective_melting_path_d = _limit_path_by_rain_cell(
            effective_melting_path_d, melting_path_km, elevation, rain_cell_length_km
        )
        melting_u = gamma_u * effective_melting_path_u * sim.melting_layer_enhancement
        melting_d = gamma_d * effective_melting_path_d * sim.melting_layer_enhancement

    gas_base = (0.23 + 0.0045 * (sim.downlink_ghz - 30.0)) * slant
    gas_u = gas_base * (sim.uplink_ghz / sim.downlink_ghz) ** 1.18
    gas_d = gas_base
    if not sim.enable_gas:
        gas_u[:] = 0.0
        gas_d[:] = 0.0

    cloud_latent = _ar1_noise(n, dt_s, model.cloud_tau_s, rng)
    cloud_d = np.maximum(0.0, 0.45 + 0.22 * cloud_latent + 0.006 * hydrometeor_ref) * slant
    cloud_u = cloud_d * (sim.uplink_ghz / sim.downlink_ghz) ** 1.32
    if not sim.enable_cloud:
        cloud_u[:] = 0.0
        cloud_d[:] = 0.0

    scint_base = _ar1_noise(n, dt_s, model.scint_tau_s, rng)
    scint_d = 0.18 * np.sqrt(slant) * scint_base
    scint_u = scint_d * (sim.uplink_ghz / sim.downlink_ghz) ** 0.72
    if not sim.enable_scintillation:
        scint_u[:] = 0.0
        scint_d[:] = 0.0

    positive_hydrometeors = hydrometeor_ref[hydrometeor_ref > 0.0]
    activity_scale = float(np.quantile(positive_hydrometeors, 0.9)) if positive_hydrometeors.size else 1.0
    phase_activity = np.clip(hydrometeor_ref / max(activity_scale, 1e-9), 0.0, 2.5)
    ice_variation = np.clip(0.72 + 0.18 * cloud_latent, 0.15, 1.5)
    ice_fraction = sim.frozen_ice_fraction if mode == "分层混合相（推荐）" else 1.0
    snow_fraction = 1.0 - ice_fraction if mode == "分层混合相（推荐）" else 1.0
    ice_path_km = (
        _slant_layer_path_km(
            sim.rain_height_km,
            sim.rain_height_km + sim.ice_layer_thickness_km,
            elevation,
            sim,
        )
        if mode == "降雨+高层冰晶"
        else frozen_path_km
    )
    snow_path_km = frozen_path_km if mode == "分层混合相（推荐）" else surface_snow_path_km
    ice_40 = sim.ice_specific_atten_db_km_40ghz * ice_path_km * ice_variation * phase_activity * ice_fraction
    snow_40 = sim.snow_specific_atten_db_km_40ghz * snow_path_km * phase_activity * snow_fraction
    if mode == "仅液态降雨":
        ice_40[:] = 0.0
        snow_40[:] = 0.0
    elif mode == "降雨+高层冰晶":
        snow_40[:] = 0.0
    elif mode == "纯降雪":
        rain_u[:] = 0.0
        rain_d[:] = 0.0
        melting_u[:] = 0.0
        melting_d[:] = 0.0
        ice_40[:] = 0.0
    elif mode == "独立叠加试验":
        ice_path_km = _slant_layer_path_km(
            sim.station_alt_km,
            sim.station_alt_km + sim.ice_layer_thickness_km,
            elevation,
            sim,
        )
        snow_path_km = surface_snow_path_km
        ice_40 = sim.ice_specific_atten_db_km_40ghz * ice_path_km * ice_variation
        snow_gate = signal.lfilter([0.06], [1.0, -0.94], (event_id > 0).astype(float))
        snow_gate = np.clip(snow_gate, 0.0, 1.0)
        snow_40 = sim.snow_specific_atten_db_km_40ghz * snow_path_km * snow_gate
        if not sim.enable_ice_crystal:
            ice_40[:] = 0.0
        if not sim.enable_snow:
            snow_40[:] = 0.0

    ice_d = ice_40 * (sim.downlink_ghz / 40.0) ** 1.35
    ice_u = ice_40 * (sim.uplink_ghz / 40.0) ** 1.35
    snow_d = snow_40 * (sim.downlink_ghz / 40.0) ** 1.25
    snow_u = snow_40 * (sim.uplink_ghz / 40.0) ** 1.25

    total_u = np.maximum(0.0, rain_u + melting_u + gas_u + cloud_u + scint_u + ice_u + snow_u)
    total_d = np.maximum(0.0, rain_d + melting_d + gas_d + cloud_d + scint_d + ice_d + snow_d)

    xpd_u = np.maximum(model.xpd_floor_db, model.terminal_isolation_db - 0.47 * rain_u - 0.55 * melting_u - 0.12 * cloud_u - 0.2 * ice_u - 0.25 * snow_u)
    xpd_d = np.maximum(model.xpd_floor_db, model.terminal_isolation_db - 0.47 * rain_d - 0.55 * melting_d - 0.12 * cloud_d - 0.2 * ice_d - 0.25 * snow_d)
    differential_attenuation_u = 0.055 * rain_u + 0.07 * melting_u + 0.025 * snow_u + 0.012 * cloud_u
    differential_attenuation_d = 0.055 * rain_d + 0.07 * melting_d + 0.025 * snow_d + 0.012 * cloud_d
    differential_phase_u = 2.1 * np.sqrt(np.maximum(rain_u + 0.8 * melting_u + 0.3 * snow_u, 0.0))
    differential_phase_d = 2.1 * np.sqrt(np.maximum(rain_d + 0.8 * melting_d + 0.3 * snow_d, 0.0))
    phase_common = np.cumsum(rng.normal(0.0, 0.003 * np.sqrt(dt_s), n))
    uplink_h_hh = 10 ** (-total_u / 20.0) * np.exp(1j * phase_common)
    uplink_cross_ratio = 10 ** (-xpd_u / 20.0)
    uplink_h_hv = uplink_h_hh * uplink_cross_ratio * np.exp(1j * np.deg2rad(differential_phase_u))
    uplink_h_vh = uplink_h_hh * uplink_cross_ratio * np.exp(-1j * np.deg2rad(differential_phase_u))
    uplink_h_vv = uplink_h_hh * 10 ** (-differential_attenuation_u / 20.0) * np.exp(1j * (phase_common + np.deg2rad(differential_phase_u)))
    downlink_h_hh = 10 ** (-total_d / 20.0) * np.exp(1j * phase_common)
    downlink_cross_ratio = 10 ** (-xpd_d / 20.0)
    downlink_h_hv = downlink_h_hh * downlink_cross_ratio * np.exp(1j * np.deg2rad(differential_phase_d))
    downlink_h_vh = downlink_h_hh * downlink_cross_ratio * np.exp(-1j * np.deg2rad(differential_phase_d))
    downlink_h_vv = downlink_h_hh * 10 ** (-differential_attenuation_d / 20.0) * np.exp(1j * (phase_common + np.deg2rad(differential_phase_d)))
    if not sim.enable_polarization:
        uplink_h_hv[:] = 0.0
        uplink_h_vh[:] = 0.0
        downlink_h_hv[:] = 0.0
        downlink_h_vh[:] = 0.0
        xpd_u[:] = np.inf
        xpd_d[:] = np.inf

    uplink_h_rr, uplink_h_rl, uplink_h_lr, uplink_h_ll = _circular_components(
        uplink_h_hh, uplink_h_hv, uplink_h_vh, uplink_h_vv
    )
    downlink_h_rr, downlink_h_rl, downlink_h_lr, downlink_h_ll = _circular_components(
        downlink_h_hh, downlink_h_hv, downlink_h_vh, downlink_h_vv
    )
    if sim.polarization == "单极化 V":
        uplink_selected_pol, downlink_selected_pol = uplink_h_vv, downlink_h_vv
    elif sim.polarization == "单极化 RHCP":
        uplink_selected_pol, downlink_selected_pol = uplink_h_rr, downlink_h_rr
    elif sim.polarization == "单极化 LHCP":
        uplink_selected_pol, downlink_selected_pol = uplink_h_ll, downlink_h_ll
    else:
        uplink_selected_pol, downlink_selected_pol = uplink_h_hh, downlink_h_hh

    output_dt_s = output.output_dt_ms / 1000.0
    time_out = np.arange(0.0, sim.duration_s + output_dt_s * 0.25, output_dt_s)
    fields = {
        "elevation_deg": elevation,
        "slant_range_km": slant_range_km,
        "zero_isotherm_height_km": np.full_like(elevation, sim.zero_isotherm_height_km),
        "rain_height_km": np.full_like(elevation, sim.rain_height_km),
        "rain_vertical_extent_km": np.full_like(elevation, sim.rain_height_km - sim.station_alt_km),
        "rain_path_length_km": rain_path_length_km,
        "rain_phase_path_km": active_rain_path,
        "rain_rate_mm_h": rain_rate_series,
        "rain_specific_atten_uplink_db_km": gamma_u,
        "rain_specific_atten_downlink_db_km": gamma_d,
        "rain_effective_path_uplink_km": effective_rain_path_u,
        "rain_effective_path_downlink_km": effective_rain_path_d,
        "rain_cell_length_km": np.full_like(elevation, rain_cell_length_km),
        "melting_layer_lower_km": np.full_like(elevation, melting_lower_km),
        "melting_layer_upper_km": np.full_like(elevation, melting_upper_km),
        "melting_path_km": melting_path_km,
        "ice_path_km": ice_path_km,
        "snow_path_km": snow_path_km,
        "rain_uplink_db": rain_u,
        "rain_downlink_db": rain_d,
        "melting_uplink_db": melting_u,
        "melting_downlink_db": melting_d,
        "gas_uplink_db": gas_u,
        "gas_downlink_db": gas_d,
        "cloud_uplink_db": cloud_u,
        "cloud_downlink_db": cloud_d,
        "scint_uplink_db": scint_u,
        "scint_downlink_db": scint_d,
        "ice_uplink_db": ice_u,
        "ice_downlink_db": ice_d,
        "snow_uplink_db": snow_u,
        "snow_downlink_db": snow_d,
        "ice_snow_uplink_db": melting_u + ice_u + snow_u,
        "ice_snow_downlink_db": melting_d + ice_d + snow_d,
        "uplink_total_db": total_u,
        "downlink_total_db": total_d,
        "xpd_db": xpd_u,
        "xpd_uplink_db": xpd_u,
        "xpd_downlink_db": xpd_d,
        "differential_attenuation_db": differential_attenuation_u,
        "differential_attenuation_uplink_db": differential_attenuation_u,
        "differential_attenuation_downlink_db": differential_attenuation_d,
        "differential_phase_deg": differential_phase_u,
        "differential_phase_uplink_deg": differential_phase_u,
        "differential_phase_downlink_deg": differential_phase_d,
        "h_hh": uplink_h_hh,
        "h_hv": uplink_h_hv,
        "h_vh": uplink_h_vh,
        "h_vv": uplink_h_vv,
        "uplink_h_hh": uplink_h_hh,
        "uplink_h_hv": uplink_h_hv,
        "uplink_h_vh": uplink_h_vh,
        "uplink_h_vv": uplink_h_vv,
        "downlink_h_hh": downlink_h_hh,
        "downlink_h_hv": downlink_h_hv,
        "downlink_h_vh": downlink_h_vh,
        "downlink_h_vv": downlink_h_vv,
        "uplink_h_rr": uplink_h_rr,
        "uplink_h_rl": uplink_h_rl,
        "uplink_h_lr": uplink_h_lr,
        "uplink_h_ll": uplink_h_ll,
        "downlink_h_rr": downlink_h_rr,
        "downlink_h_rl": downlink_h_rl,
        "downlink_h_lr": downlink_h_lr,
        "downlink_h_ll": downlink_h_ll,
        "uplink_selected_pol": uplink_selected_pol,
        "downlink_selected_pol": downlink_selected_pol,
        "event_id": event_id,
    }
    data = {name: _resample(values, time_internal, time_out) for name, values in fields.items()}

    finite_u = data["uplink_total_db"]
    event_wet = data["event_id"] > 0
    rain_wet = data["rain_uplink_db"][event_wet]
    increment_100ms_lag = max(1, int(round(100.0 / output.output_dt_ms)))
    rain_increment_100ms = (
        data["rain_uplink_db"][increment_100ms_lag:] - data["rain_uplink_db"][:-increment_100ms_lag]
        if output.output_dt_ms <= 100.0 and increment_100ms_lag < data["rain_uplink_db"].size
        else np.asarray([], dtype=float)
    )
    max_lags = min(finite_u.size, max(2, int(60.0 / output_dt_s)))
    acf = _acf_fft(finite_u, max_lags)
    metrics = {
        "uplink_max_db": float(np.max(data["uplink_total_db"])),
        "downlink_max_db": float(np.max(data["downlink_total_db"])),
        "uplink_p99_db": float(np.quantile(data["uplink_total_db"], 0.99)),
        "dynamic_range_db": float(np.quantile(finite_u, 0.999) - np.quantile(finite_u, 0.01)),
        "t95_s": _coherence_time(acf, output_dt_s, 0.95),
        "t99_s": _coherence_time(acf, output_dt_s, 0.99),
        "rain_path_min_km": float(np.min(data["rain_path_length_km"])),
        "rain_path_max_km": float(np.max(data["rain_path_length_km"])),
        "rain_rate_wet_median_mm_h": float(np.median(data["rain_rate_mm_h"][event_wet])) if np.any(event_wet) else 0.0,
        "rain_rate_max_mm_h": float(np.max(data["rain_rate_mm_h"])),
        "rain_time_fraction": float(np.mean(event_wet)),
        "rain_wet_std_db": float(np.std(rain_wet)) if rain_wet.size else 0.0,
        "rain_delta_100ms_std_db": float(np.std(rain_increment_100ms)) if rain_increment_100ms.size else float("nan"),
        "xpd_p01_db": float(np.quantile(data["xpd_uplink_db"][np.isfinite(data["xpd_uplink_db"])], 0.01)) if np.any(np.isfinite(data["xpd_uplink_db"])) else float("inf"),
    }
    for interval_ms in (2, 5, 10, 20):
        lag = max(1, int(round(interval_ms / output.output_dt_ms)))
        if output.output_dt_ms <= interval_ms and lag < finite_u.size:
            metrics[f"max_delta_{interval_ms}ms_db"] = float(np.max(np.abs(finite_u[lag:] - finite_u[:-lag])))
        else:
            metrics[f"max_delta_{interval_ms}ms_db"] = float("nan")

    metadata = config_to_dict(sim, model, output)
    metadata["effective_internal_dt_ms"] = dt_ms
    metadata["sequence_index"] = sequence_index
    metadata["rain_mapping"] = {
        "rain_rate_semantics": "wet-sample median rain rate",
        "occurrence_semantics": "target rain time occupancy; 1 means continuously wet",
        "specific_attenuation": "ITU-R P.838-3",
        "effective_path": "ITU-R P.618-14 0.01%-path reduction applied dynamically and capped by geometric path",
        "spatial_limit": "model-specific horizontal rain-cell length converted to slant length",
        "variance_control": "model fast-process correlation time and log standard deviation; reported by wet fade sigma and 100 ms increment sigma",
    }
    return SimulationResult(time_out, data, metrics, metadata)
