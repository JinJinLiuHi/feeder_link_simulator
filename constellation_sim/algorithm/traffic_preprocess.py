"""Offline WorldPop-to-H3 preprocessing for the traffic model.

The operation is intentionally separate from frame simulation. It may process a
large GeoTIFF once and write a compact CSV which is reused by every scenario.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .traffic_model import TrafficCell


def _h3_cell(h3_module, lat: float, lon: float, resolution: int) -> str:
    if hasattr(h3_module, "latlng_to_cell"):
        return h3_module.latlng_to_cell(lat, lon, resolution)
    return h3_module.geo_to_h3(lat, lon, resolution)


def _h3_center(h3_module, cell_id: str) -> tuple[float, float]:
    if hasattr(h3_module, "cell_to_latlng"):
        return tuple(h3_module.cell_to_latlng(cell_id))
    return tuple(h3_module.h3_to_geo(cell_id))


def aggregate_worldpop_to_h3(
    geotiff_path: str | Path,
    resolution: int = 4,
    minimum_population: float = 0.0,
) -> list[TrafficCell]:
    """Aggregate valid WorldPop pixels into H3 population weights.

    Requires ``rasterio`` and ``h3`` in the preprocessing environment. Windows
    are read block by block so the complete population raster is never loaded in
    memory.
    """
    try:
        import h3
        import numpy as np
        import rasterio
    except ImportError as exc:
        raise RuntimeError("离线处理WorldPop需要安装 rasterio 和 h3") from exc

    totals: dict[str, float] = defaultdict(float)
    with rasterio.open(str(geotiff_path)) as dataset:
        nodata = dataset.nodata
        for _, window in dataset.block_windows(1):
            values = dataset.read(1, window=window, masked=True)
            row0, col0 = int(window.row_off), int(window.col_off)
            valid_mask = ~np.ma.getmaskarray(values)
            for local_row, local_col in zip(*(valid_mask.nonzero())):
                value = float(values[local_row, local_col])
                if value <= 0.0 or (nodata is not None and value == nodata):
                    continue
                lon, lat = dataset.xy(row0 + int(local_row), col0 + int(local_col))
                totals[_h3_cell(h3, float(lat), float(lon), resolution)] += value

    cells = []
    for cell_id, population in totals.items():
        if population < minimum_population:
            continue
        lat, lon = _h3_center(h3, cell_id)
        cells.append(TrafficCell(cell_id, lat, lon, population, population))
    return cells


def save_cells_csv(cells: list[TrafficCell], output_path: str | Path) -> None:
    with Path(output_path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "grid_id", "latitude_deg", "longitude_deg", "population",
            "base_weight", "enabled",
        ])
        for cell in cells:
            writer.writerow([
                cell.cell_id, cell.latitude_deg, cell.longitude_deg,
                cell.population, cell.base_weight, int(cell.enabled),
            ])


def load_cells_csv(path: str | Path) -> list[TrafficCell]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        return [
            TrafficCell(
                cell_id=row["grid_id"],
                latitude_deg=float(row["latitude_deg"]),
                longitude_deg=float(row["longitude_deg"]),
                population=float(row.get("population", 0.0)),
                base_weight=float(row["base_weight"]),
                enabled=str(row.get("enabled", "1")).lower() not in {"0", "false", "no"},
            )
            for row in rows
        ]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="将WorldPop GeoTIFF聚合为业务模型使用的H3 CSV")
    parser.add_argument("worldpop_tif", help="WorldPop GeoTIFF路径")
    parser.add_argument("output_csv", help="输出CSV路径")
    parser.add_argument("--resolution", type=int, default=4, help="H3分辨率，默认4")
    parser.add_argument("--minimum-population", type=float, default=0.0, help="输出网格的最低人口")
    args = parser.parse_args()
    cells = aggregate_worldpop_to_h3(
        args.worldpop_tif, args.resolution, args.minimum_population)
    save_cells_csv(cells, args.output_csv)
    print(f"已生成 {len(cells):,} 个业务网格：{args.output_csv}")


if __name__ == "__main__":
    main()

