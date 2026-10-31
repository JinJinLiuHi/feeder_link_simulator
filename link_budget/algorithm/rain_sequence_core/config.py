from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .orbit_geometry import leo_visible_duration_s


HYDROMETEOR_PHASE_MODES = (
    "仅液态降雨",
    "降雨+高层冰晶",
    "分层混合相（推荐）",
    "纯降雪",
    "独立叠加试验",
)


@dataclass(slots=True)
class SimulationConfig:
    orbit: str = "LEO"
    leo_altitude_km: float = 500.0
    station_name: str = "上海站"
    station_lat_deg: float = 31.23
    station_lon_deg: float = 121.47
    station_alt_km: float = 0.02
    rain_height_mode: str = "ITU-R P.839-4自动计算"
    zero_isotherm_height_km: float = 4.64
    rain_height_km: float = 5.0
    direction: str = "上下行同时输出"
    uplink_ghz: float = 50.0
    downlink_ghz: float = 40.0
    polarization: str = "双线极化 H/V"
    duration_s: float = 440.0
    internal_dt_ms: float = 100.0
    sequence_count: int = 1
    elevation_min_deg: float = 10.0
    elevation_max_deg: float = 70.0
    geo_longitude_deg: float = 110.5
    geo_altitude_km: float = 35786.0
    rain_rate_mm_h: float = 35.0
    rain_probability: float = 0.32
    seed: int = 20260821
    enable_rain: bool = True
    enable_gas: bool = True
    enable_cloud: bool = True
    enable_scintillation: bool = True
    enable_polarization: bool = True
    hydrometeor_phase_mode: str = "分层混合相（推荐）"
    enable_ice_crystal: bool = False
    enable_snow: bool = False
    melting_layer_thickness_km: float = 0.72
    melting_layer_enhancement: float = 1.35
    frozen_ice_fraction: float = 0.45
    ice_layer_thickness_km: float = 2.0
    ice_specific_atten_db_km_40ghz: float = 0.003
    snow_layer_thickness_km: float = 1.0
    snow_specific_atten_db_km_40ghz: float = 0.02

    def validate(self, output_dt_ms: float) -> None:
        if self.duration_s <= 0:
            raise ValueError("持续时间必须大于0 s。")
        if self.internal_dt_ms <= 0 or output_dt_ms <= 0:
            raise ValueError("时间步长必须大于0 ms。")
        if not 1 <= self.sequence_count <= 100:
            raise ValueError("序列条数应在1～100之间。")
        if not 1.0 <= self.elevation_min_deg < self.elevation_max_deg <= 90.0:
            raise ValueError("仰角范围设置不正确。")
        is_leo = self.orbit in ("LEO", "500 km LEO", "1200 km LEO")
        if is_leo:
            altitude_km = (
                500.0 if self.orbit == "500 km LEO"
                else 1200.0 if self.orbit == "1200 km LEO"
                else self.leo_altitude_km
            )
            if not 160.0 <= altitude_km <= 2000.0:
                raise ValueError("LEO轨道高度应位于160～2000 km。")
            visible_s = leo_visible_duration_s(
                altitude_km, self.elevation_min_deg, self.elevation_max_deg
            )
            if self.duration_s > visible_s + 1e-6:
                raise ValueError(
                    f"LEO {altitude_km:g} km在{self.elevation_min_deg:g}°～{self.elevation_max_deg:g}°仰角范围内的"
                    f"单次最大可见时长约为{visible_s:.2f} s；当前持续时间{self.duration_s:g} s超出可见窗口。"
                    "请缩短持续时间、点击“采用可见上限”，或改用GEO/外部星历。"
                )
        if self.uplink_ghz <= 0 or self.downlink_ghz <= 0:
            raise ValueError("上下行频率必须大于0 GHz。")
        if not 0.0 <= self.rain_probability <= 1.0:
            raise ValueError("降水出现概率必须位于0～1之间。")
        if self.rain_rate_mm_h < 0.0:
            raise ValueError("典型雨强不能为负数。")
        if self.rain_height_km <= self.station_alt_km:
            raise ValueError("雨层顶高度必须高于信关站海拔。")
        if self.orbit == "GEO":
            if not -180.0 <= self.geo_longitude_deg <= 180.0:
                raise ValueError("GEO星下点经度必须位于-180°～180°。")
            if not 30000.0 <= self.geo_altitude_km <= 45000.0:
                raise ValueError("GEO轨道高度建议位于30000～45000 km。")
        if self.melting_layer_thickness_km <= 0.0:
            raise ValueError("融化层厚度必须大于0 km。")
        if self.melting_layer_enhancement < 0.0:
            raise ValueError("融化层增强系数不能为负数。")
        if not 0.0 <= self.frozen_ice_fraction <= 1.0:
            raise ValueError("冻结层冰晶份额必须位于0～1之间。")
        if min(
            self.ice_layer_thickness_km,
            self.snow_layer_thickness_km,
            self.ice_specific_atten_db_km_40ghz,
            self.snow_specific_atten_db_km_40ghz,
        ) < 0.0:
            raise ValueError("冰雪层厚度和比衰减不能为负数。")
        points = self.duration_s * 1000.0 / min(self.internal_dt_ms, output_dt_ms)
        if points > 3_000_000:
            raise ValueError("单条序列内部采样点超过300万，请缩短持续时间或增大时间步长。")


@dataclass(slots=True)
class ModelConfig:
    rain_model: str = "ONERA双过程"
    fusion_mode: str = "分层物理融合"
    fast_tau_s: float = 2.5
    slow_tau_s: float = 75.0
    event_mean_s: float = 90.0
    event_gap_s: float = 180.0
    wind_speed_m_s: float = 12.0
    cell_count: int = 5
    marginal_shape: float = 2.2
    fusion_weight_dynamic: float = 0.65
    fusion_weight_event: float = 0.35
    cloud_tau_s: float = 180.0
    scint_tau_s: float = 0.15
    terminal_isolation_db: float = 32.0
    xpd_floor_db: float = 15.0
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class OutputConfig:
    output_dt_ms: float = 100.0
    output_format: str = "MAT"
    selected: dict[str, bool] = field(
        default_factory=lambda: {
            "uplink_total_db": True,
            "downlink_total_db": True,
            "rain_components_db": True,
            "gas_cloud_scint_db": True,
            "ice_snow_components_db": False,
            "polarization_matrix": True,
            "xpd_and_differential": True,
            "geometry": True,
            "event_labels": False,
        }
    )


def config_to_dict(*configs: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for config in configs:
        merged[type(config).__name__] = asdict(config)
    return merged
