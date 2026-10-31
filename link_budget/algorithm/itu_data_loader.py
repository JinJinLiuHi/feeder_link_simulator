"""
ITU-R 数据加载与插值模块
支持三文件格式（LAT.TXT / LON.TXT / DATA.TXT）

目录结构（放在项目根目录 data/ 下）：
    data/
      P.837_R001_Maps/
        LAT_R001.TXT      ← 每行一个纬度值 (°)
        LON_R001.TXT      ← 每行一个经度值 (°)
        R001.TXT          ← 每行一个 R_0.01 (mm/h)
      P.839-4 rain_height/
        Lat.txt           ← 每行一个纬度值 (°)
        Lon.txt           ← 每行一个经度值 (°)
        h0.txt            ← 每行一个雨顶高度 H_R (km)

三个文件的行数相同，第 i 行表示同一个格点的坐标和数值。
插值方式：scipy.interpolate.LinearNDInterpolator（散点线性插值）
          若格点是规则格网，自动切换为 RegularGridInterpolator（更快）。
"""

import os
import math
import numpy as np
from typing import Optional, Tuple

# ── scipy 插值（惰性导入，避免启动时拖慢）────────────────
_LinearNDInterpolator   = None
_RegularGridInterpolator = None
_griddata               = None

def _ensure_scipy():
    global _LinearNDInterpolator, _RegularGridInterpolator, _griddata
    if _LinearNDInterpolator is None:
        from scipy.interpolate import (
            LinearNDInterpolator, RegularGridInterpolator, griddata)
        _LinearNDInterpolator    = LinearNDInterpolator
        _RegularGridInterpolator = RegularGridInterpolator
        _griddata                = griddata


# ── 数据路径 ──────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE_DIR, "data")

_R001_DIR   = os.path.join(_DATA_DIR, "P.837_R001_Maps")
_HRAIN_DIR  = os.path.join(_DATA_DIR, "P.839-4 rain_height")

_R001_LAT   = os.path.join(_R001_DIR,  "LAT_R001.TXT")
_R001_LON   = os.path.join(_R001_DIR,  "LON_R001.TXT")
_R001_DATA  = os.path.join(_R001_DIR,  "R001.TXT")

_HRAIN_LAT  = os.path.join(_HRAIN_DIR, "Lat.txt")
_HRAIN_LON  = os.path.join(_HRAIN_DIR, "Lon.txt")
_HRAIN_DATA = os.path.join(_HRAIN_DIR, "h0.txt")


# ══════════════════════════════════════════════════════════
#  中国城市列表（名称, 纬度, 经度）
# ══════════════════════════════════════════════════════════

CHINA_CITIES = [
    ("北京",      39.90,  116.41),
    ("上海",      31.23,  121.47),
    ("天津",      39.08,  117.20),
    ("重庆",      29.56,  106.55),
    ("石家庄",    38.04,  114.51),
    ("太原",      37.87,  112.55),
    ("呼和浩特",  40.84,  111.75),
    ("沈阳",      41.80,  123.43),
    ("长春",      43.88,  125.32),
    ("哈尔滨",    45.75,  126.63),
    ("南京",      32.06,  118.78),
    ("杭州",      30.27,  120.15),
    ("合肥",      31.86,  117.28),
    ("福州",      26.07,  119.30),
    ("南昌",      28.68,  115.88),
    ("济南",      36.67,  117.00),
    ("武汉",      30.59,  114.31),
    ("长沙",      28.23,  112.94),
    ("郑州",      34.75,  113.65),
    ("广州",      23.13,  113.26),
    ("深圳",      22.54,  114.05),
    ("南宁",      22.82,  108.37),
    ("海口",      20.04,  110.34),
    ("成都",      30.66,  104.07),
    ("贵阳",      26.58,  106.71),
    ("昆明",      25.05,  102.71),
    ("拉萨",      29.65,   91.13),
    ("西安",      34.27,  108.95),
    ("兰州",      36.06,  103.79),
    ("西宁",      36.62,  101.78),
    ("银川",      38.47,  106.27),
    ("乌鲁木齐",  43.79,   87.60),
    ("香港",      22.32,  114.17),
    ("台北",      25.03,  121.56),
]

CITY_NAMES = [c[0] for c in CHINA_CITIES]

# 内置回退值 {城市: (H_R km, R001 mm/h)}
_FALLBACK = {
    "北京":     (4.0,  42.0), "上海":     (4.5,  62.0),
    "天津":     (4.0,  40.0), "重庆":     (4.8,  68.0),
    "石家庄":   (3.8,  38.0), "太原":     (3.6,  30.0),
    "呼和浩特": (3.2,  22.0), "沈阳":     (3.8,  45.0),
    "长春":     (3.5,  40.0), "哈尔滨":   (3.2,  38.0),
    "南京":     (4.6,  65.0), "杭州":     (4.7,  72.0),
    "合肥":     (4.5,  60.0), "福州":     (5.0,  85.0),
    "南昌":     (4.8,  75.0), "济南":     (4.2,  50.0),
    "武汉":     (4.7,  68.0), "长沙":     (4.8,  72.0),
    "郑州":     (4.0,  48.0), "广州":     (5.0,  98.0),
    "深圳":     (5.0, 105.0), "南宁":     (5.2,  92.0),
    "海口":     (5.5, 120.0), "成都":     (4.5,  58.0),
    "贵阳":     (4.6,  65.0), "昆明":     (4.2,  55.0),
    "拉萨":     (3.0,  18.0), "西安":     (3.8,  38.0),
    "兰州":     (3.2,  22.0), "西宁":     (3.0,  18.0),
    "银川":     (3.2,  22.0), "乌鲁木齐": (2.8,  12.0),
    "香港":     (5.2, 115.0), "台北":     (5.0, 100.0),
}


# ══════════════════════════════════════════════════════════
#  三文件格式加载器
# ══════════════════════════════════════════════════════════

class _TriFileDataset:
    """
    从 LAT/LON/DATA 三个文本文件加载 ITU 格点数据集，
    支持规则格网（快速）和散点（通用）两种插值方式。
    """

    def __init__(self, lat_file: str, lon_file: str, data_file: str,
                 name: str = ""):
        self.name       = name
        self._interp    = None   # 插值器（惰性构建）
        self._loaded    = False
        self._lat_file  = lat_file
        self._lon_file  = lon_file
        self._data_file = data_file
        self._ok        = False  # 文件全部存在且成功加载

    def _load(self):
        if self._loaded:
            return
        self._loaded = True

        for f in (self._lat_file, self._lon_file, self._data_file):
            if not os.path.isfile(f):
                return   # 文件缺失，_ok 保持 False

        try:
            lat_arr  = np.loadtxt(self._lat_file).ravel()
            lon_arr  = np.loadtxt(self._lon_file).ravel()
            data_arr = np.loadtxt(self._data_file).ravel()
        except Exception:
            return

        if not (len(lat_arr) == len(lon_arr) == len(data_arr)):
            return

        _ensure_scipy()

        # 判断是否为规则格网
        unique_lats = np.unique(lat_arr)
        unique_lons = np.unique(lon_arr)
        n_lat = len(unique_lats)
        n_lon = len(unique_lons)

        if n_lat * n_lon == len(lat_arr):
            # 规则格网：重组为二维数组，用 RegularGridInterpolator
            try:
                # 按 (lat, lon) 排列数据
                grid = np.full((n_lat, n_lon), np.nan)
                lat_idx = {v: i for i, v in enumerate(unique_lats)}
                lon_idx = {v: i for i, v in enumerate(unique_lons)}
                for la, lo, d in zip(lat_arr, lon_arr, data_arr):
                    grid[lat_idx[la], lon_idx[lo]] = d

                # RegularGridInterpolator 要求坐标单调递增
                if unique_lats[0] > unique_lats[-1]:
                    unique_lats = unique_lats[::-1]
                    grid = grid[::-1, :]
                if unique_lons[0] > unique_lons[-1]:
                    unique_lons = unique_lons[::-1]
                    grid = grid[:, ::-1]

                self._interp = _RegularGridInterpolator(
                    (unique_lats, unique_lons), grid,
                    method='linear', bounds_error=False,
                    fill_value=None)
                self._mode = 'regular'
                self._ok = True
                return
            except Exception:
                pass   # 回退到散点插值

        # 散点插值（通用）
        try:
            self._interp = _LinearNDInterpolator(
                list(zip(lat_arr, lon_arr)), data_arr)
            self._mode = 'scatter'
            self._ok = True
        except Exception:
            pass

    def query(self, lat: float, lon: float) -> Optional[float]:
        """插值查询。失败返回 None。"""
        self._load()
        if not self._ok or self._interp is None:
            return None
        try:
            if self._mode == 'regular':
                val = float(self._interp([[lat, lon]])[0])
            else:
                val = float(self._interp([[lat, lon]])[0])
            return val if not math.isnan(val) else None
        except Exception:
            return None

    @property
    def available(self) -> bool:
        self._load()
        return self._ok


# ══════════════════════════════════════════════════════════
#  全局数据集实例（惰性加载）
# ══════════════════════════════════════════════════════════

_ds_r001  = _TriFileDataset(_R001_LAT,  _R001_LON,  _R001_DATA,  "P.837 R001")
_ds_hrain = _TriFileDataset(_HRAIN_LAT, _HRAIN_LON, _HRAIN_DATA, "P.839-4 H_R")


# ══════════════════════════════════════════════════════════
#  对外接口
# ══════════════════════════════════════════════════════════

def get_city_coords(city_name: str) -> Tuple[float, float]:
    """返回 (纬度, 经度)"""
    for name, lat, lon in CHINA_CITIES:
        if name == city_name:
            return lat, lon
    return 35.0, 105.0


def get_rain_height(city_name: str) -> float:
    """从 P.839-4 数据获取雨顶高度 H_R (km)"""
    lat, lon = get_city_coords(city_name)
    val = _ds_hrain.query(lat, lon)
    if val is not None and 0.5 < val < 15.0:
        return round(val, 2)
    return _FALLBACK.get(city_name, (4.0, 50.0))[0]


def get_r001(city_name: str) -> float:
    """从 P.837 数据获取 R_0.01 (mm/h)"""
    lat, lon = get_city_coords(city_name)
    val = _ds_r001.query(lat, lon)
    if val is not None and 0 < val < 500:
        return round(val, 2)
    return _FALLBACK.get(city_name, (4.0, 50.0))[1]


def get_city_rain_params(city_name: str) -> dict:
    """一次返回城市所有雨衰相关参数"""
    lat, lon = get_city_coords(city_name)
    return {
        "city":           city_name,
        "lat":            lat,
        "lon":            lon,
        "rain_height_km": get_rain_height(city_name),
        "R001_mmh":       get_r001(city_name),
    }


def data_source_info() -> str:
    """返回当前数据来源（有格点文件 or 内置回退）"""
    h_src = "P.839-4 格点文件" if _ds_hrain.available else "内置回退值"
    r_src = "P.837 格点文件"   if _ds_r001.available  else "内置回退值"
    return f"雨高: {h_src}  |  R₀.₀₁: {r_src}"