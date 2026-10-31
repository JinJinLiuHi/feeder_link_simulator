"""Multiprocess worker for building time-varying traffic cache frames."""

from __future__ import annotations

import marshal
import zlib
from datetime import datetime, timedelta

import numpy as np

from .traffic_model import calculate_visibility


_orbit = None
_planes = None
_model = None
_cells = None
_cell_by_id = None
_min_elevation_deg = 0.0
_epoch = None
_step_s = 1.0
_duration_s = 0.0


def initialize(orbit_path, planes, model, cells, min_elevation_deg,
               epoch_iso, step_s, duration_s):
    """Attach one spawned process to the read-only orbit memory map."""
    global _orbit, _planes, _model, _cells, _cell_by_id
    global _min_elevation_deg, _epoch, _step_s, _duration_s
    _orbit = np.load(orbit_path, mmap_mode="r")
    _planes = np.asarray(planes, dtype=np.int32)
    _model = model
    _cells = cells
    _cell_by_id = {cell.cell_id: cell for cell in cells}
    _min_elevation_deg = float(min_elevation_deg)
    _epoch = datetime.fromisoformat(epoch_iso)
    _step_s = float(step_s)
    _duration_s = float(duration_s)


def build_frame(frame_index):
    """Build and compress one traffic frame for the parent process."""
    cached = np.asarray(_orbit[frame_index])
    count = cached.shape[0]
    positions = np.empty((count, 5), dtype=np.float64)
    positions[:, 0] = np.arange(count)
    positions[:, 1] = _planes
    positions[:, 2:5] = cached[:, 0:3]
    visibility = calculate_visibility(_cells, positions, _min_elevation_deg)
    frame_time = _epoch + timedelta(seconds=min(frame_index * _step_s, _duration_s))
    result = _model.run_frame(frame_time, visibility)
    traffic_rows = [
        [item.cell_id, _cell_by_id[item.cell_id].latitude_deg,
         _cell_by_id[item.cell_id].longitude_deg, item.demand_weight,
         item.dense, item.unassigned_weight,
         [[sid, value] for sid, value in item.allocations.items()],
         item.local_hour,
         [[candidate.satellite_id, candidate.elevation_deg]
          for candidate in visibility.get(item.cell_id, ())],
         item.dense_cluster_id]
        for item in result.cells
    ]
    satellite_rows = [
        [sid, load.assigned_weight, load.served_cell_count]
        for sid, load in result.satellites.items()
        if load.assigned_weight > 0.0
    ]
    return frame_index, zlib.compress(marshal.dumps((traffic_rows, satellite_rows)), 1)
