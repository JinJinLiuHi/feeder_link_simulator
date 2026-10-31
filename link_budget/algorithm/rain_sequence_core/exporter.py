from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from scipy.io import savemat

from .engine import SimulationResult


OUTPUT_GROUPS = {
    "uplink_total_db": ("uplink_total_db",),
    "downlink_total_db": ("downlink_total_db",),
    "rain_components_db": (
        "rain_rate_mm_h",
        "rain_specific_atten_uplink_db_km", "rain_specific_atten_downlink_db_km",
        "rain_effective_path_uplink_km", "rain_effective_path_downlink_km",
        "rain_uplink_db", "rain_downlink_db",
    ),
    "gas_cloud_scint_db": (
        "gas_uplink_db",
        "gas_downlink_db",
        "cloud_uplink_db",
        "cloud_downlink_db",
        "scint_uplink_db",
        "scint_downlink_db",
    ),
    "ice_snow_components_db": (
        "melting_uplink_db", "melting_downlink_db",
        "ice_uplink_db", "ice_downlink_db", "snow_uplink_db", "snow_downlink_db",
    ),
    "polarization_matrix": (
        "uplink_h_hh", "uplink_h_hv", "uplink_h_vh", "uplink_h_vv",
        "downlink_h_hh", "downlink_h_hv", "downlink_h_vh", "downlink_h_vv",
        "uplink_h_rr", "uplink_h_rl", "uplink_h_lr", "uplink_h_ll",
        "downlink_h_rr", "downlink_h_rl", "downlink_h_lr", "downlink_h_ll",
        "uplink_selected_pol", "downlink_selected_pol",
    ),
    "xpd_and_differential": (
        "xpd_uplink_db", "xpd_downlink_db",
        "differential_attenuation_uplink_db", "differential_attenuation_downlink_db",
        "differential_phase_uplink_deg", "differential_phase_downlink_deg",
    ),
    "geometry": (
        "slant_range_km", "elevation_deg", "zero_isotherm_height_km", "rain_height_km",
        "rain_vertical_extent_km", "rain_path_length_km", "rain_phase_path_km",
        "rain_cell_length_km",
        "melting_layer_lower_km", "melting_layer_upper_km", "melting_path_km",
        "ice_path_km", "snow_path_km",
    ),
    "event_labels": ("event_id",),
}


def selected_names(result: SimulationResult, selected: dict[str, bool]) -> list[str]:
    names: list[str] = []
    for group, enabled in selected.items():
        if enabled:
            names.extend(name for name in OUTPUT_GROUPS.get(group, ()) if name in result.data)
    return list(dict.fromkeys(names))


def _metadata_json(result: SimulationResult) -> str:
    return json.dumps(result.metadata, ensure_ascii=False, indent=2, default=str)


def export_csv(path: str | Path, result: SimulationResult, names: Iterable[str]) -> Path:
    path = Path(path)
    columns: dict[str, np.ndarray] = {"time_s": result.time_s}
    for name in names:
        values = result.data[name]
        if np.iscomplexobj(values):
            columns[f"{name}_real"] = values.real
            columns[f"{name}_imag"] = values.imag
        else:
            columns[name] = values
    pd.DataFrame(columns).to_csv(path, index=False, encoding="utf-8-sig")
    path.with_suffix(path.suffix + ".metadata.json").write_text(_metadata_json(result), encoding="utf-8")
    return path


def export_mat(path: str | Path, result: SimulationResult, names: Iterable[str]) -> Path:
    path = Path(path)
    payload = {"time_s": result.time_s, **{name: result.data[name] for name in names}}
    payload["metrics_json"] = json.dumps(result.metrics, ensure_ascii=False)
    payload["metadata_json"] = _metadata_json(result)
    savemat(path, payload, do_compression=True)
    return path


def export_hdf5(path: str | Path, result: SimulationResult, names: Iterable[str]) -> Path:
    path = Path(path)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("time_s", data=result.time_s, compression="gzip")
        group = handle.create_group("channel")
        for name in names:
            group.create_dataset(name, data=result.data[name], compression="gzip")
        metrics_group = handle.create_group("metrics")
        for key, value in result.metrics.items():
            metrics_group.attrs[key] = value
        handle.attrs["metadata_json"] = _metadata_json(result)
    return path


def export_result(
    path: str | Path,
    output_format: str,
    result: SimulationResult,
    selected: dict[str, bool],
) -> Path:
    names = selected_names(result, selected)
    if not names:
        raise ValueError("至少选择一种输出数据。")
    if output_format == "CSV":
        return export_csv(path, result, names)
    if output_format == "HDF5":
        return export_hdf5(path, result, names)
    return export_mat(path, result, names)
