from __future__ import annotations

import json
from pathlib import Path


BUILTIN_STATIONS: dict[str, dict[str, float]] = {
    "上海站": {"lat_deg": 31.23, "lon_deg": 121.47, "alt_km": 0.02},
    "北京站": {"lat_deg": 39.90, "lon_deg": 116.40, "alt_km": 0.05},
    "广州站": {"lat_deg": 23.13, "lon_deg": 113.26, "alt_km": 0.02},
}


class StationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.custom: dict[str, dict[str, float]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.custom = {}
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.custom = {
                str(name): {
                    "lat_deg": float(values["lat_deg"]),
                    "lon_deg": float(values["lon_deg"]),
                    "alt_km": float(values.get("alt_km", 0.0)),
                }
                for name, values in raw.get("stations", {}).items()
            }
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            self.custom = {}

    def names(self) -> tuple[str, ...]:
        return tuple(BUILTIN_STATIONS) + tuple(sorted(self.custom))

    def get(self, name: str) -> dict[str, float] | None:
        record = BUILTIN_STATIONS.get(name) or self.custom.get(name)
        return record.copy() if record else None

    def save(self, name: str, lat_deg: float, lon_deg: float, alt_km: float) -> None:
        name = name.strip()
        if not name:
            raise ValueError("站址名称不能为空。")
        if name in BUILTIN_STATIONS:
            raise ValueError("不能覆盖内置站址，请使用新的站址名称。")
        if not -90.0 <= lat_deg <= 90.0:
            raise ValueError("纬度必须位于-90°～90°。")
        if not -180.0 <= lon_deg <= 180.0:
            raise ValueError("经度必须位于-180°～180°。")
        if not -0.5 <= alt_km <= 10.0:
            raise ValueError("站高应位于-0.5～10 km。")
        self.custom[name] = {"lat_deg": float(lat_deg), "lon_deg": float(lon_deg), "alt_km": float(alt_km)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"stations": self.custom}, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

