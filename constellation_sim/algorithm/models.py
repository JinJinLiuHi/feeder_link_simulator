"""Shared scenario data models."""

from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    source: str
    tle_path: str
    satellite_count: int
    planes: int
    walker_layout: str
    altitude_km: float
    inclination_deg: float
    eccentricity: float
    arg_perigee_deg: float
    duration_s: float
    step_s: float
    speed: float
    latitude_limit_deg: float
    min_elevation_deg: float
    max_gsl_km: float
    max_isl_km: float
    isl_pattern: str
    seam_enabled: bool
    seam_after_plane: int
    stations: list[dict]
    center_plane: int
    display_plane_count: int
    show_satellites: bool
    show_same_isl: bool
    show_cross_isl: bool
    show_gsl: bool
    show_motion: bool
    show_labels: bool
    traffic_enabled: bool = True
    traffic_grid_path: str = ""
    traffic_dense_percentile: float = 95.0
    traffic_dense_max_satellites: int = 3
    traffic_random_sigma: float = 0.0
    traffic_outside_china_percent: float = 0.0
    traffic_total_gbps: float = 1000.0
    traffic_cluster_radius_km: float = 180.0
    planning_enabled: bool = True
    planning_period_s: float = 300.0
    feeder_bandwidth_mhz: float = 500.0
    snr_margin_db: float = 1.0
    modcod_table_path: str = ""
    capacity_threshold_percent: float = 85.0
    switching_loss_percent: float = 2.0
    rain_enabled: bool = True
    rain_source: str = "geographic_auto"
    rain_sense_sigma: float = 0.20
    rain_forecast_hours: float = 2.0
    rain_use_p90: bool = True
    satellite_antenna_count: int = 4
    satellite_channel_count: int = 4
    satellite_channel_capacity_gbps: float = 10.0
    geo_interference_enabled: bool = True
    gateway_eirp_dbw: float = 65.0
    gateway_gt_dbk: float = 30.0
    gateway_array_loss_db: float = 1.0
    walker_mode: str = "single"
    walker_shells: list[dict] = field(default_factory=list)
    result_directory: str = ""
    scenario_directory: str = ""
    gateway_planning_enabled: bool = True
    gateway_min_sites: int = 3
    gateway_max_sites: int = 5
    gateway_redundancy: int = 2
    gateway_max_antennas: int = 20
    gateway_planning_mode: str = "automatic"
    planning_horizon_s: float = 1200.0
    planning_slot_s: float = 15.0
    fast_evaluation_period_s: float = 1.0
    fast_adjust_period_s: float = 5.0
    route_update_period_s: float = 15.0
    planning_freeze_window_s: float = 30.0
    traffic_landing_percent: float = 30.0
