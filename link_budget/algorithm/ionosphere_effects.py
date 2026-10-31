"""
电离层效应模块 — ITU-R P.531-16
§4.1 电子总含量（TEC）
§4.3 法拉第旋转
§4.4 电离层群时延
"""

import math


# ══════════════════════════════════════════════════════════
#  §4.4 电离层群时延
#  t = 1.345 × N_T / f² × 10⁻⁷  (s)
#  N_T : TEC (el/m²)，典型范围 10¹⁶ ~ 10¹⁸ el/m²
#  f   : 频率 (Hz)
# ══════════════════════════════════════════════════════════

def calc_ionospheric_group_delay(
    freq_ghz: float,
    N_T: float = 1e17,
) -> float:
    """
    电离层群时延 t (s)，ITU-R P.531-16 §4.4。

    t = 1.345 × N_T / f² × 10⁻⁷

    参数
    ----
    freq_ghz : 工作频率 (GHz)
    N_T      : 电子总含量 TEC (el/m²)
               典型值：低活动期 10¹⁶，平均 10¹⁷，高活动期 10¹⁸

    返回
    ----
    t_s : 群时延 (s)
    t_ns: 群时延 (ns，工程上更常用)
    """
    f_hz = freq_ghz * 1e9
    t_s  = 1.345e-7 * N_T / (f_hz ** 2)
    return t_s, t_s * 1e9   # (s, ns)


# ══════════════════════════════════════════════════════════
#  §4.3 法拉第旋转
#  Ω = 2.36 × 10⁻¹⁴ × B_av × N_T / f²  (rad)
#  B_av : 平均地球磁场 (T 或 Wb/m²)，典型值 50 μT = 5×10⁻⁵ T
#  f    : 频率 (Hz)
# ══════════════════════════════════════════════════════════

def calc_faraday_rotation(
    freq_ghz: float,
    N_T: float  = 1e17,
    B_av: float = 50e-6,
) -> tuple[float, float]:
    """
    法拉第旋转角 Ω，ITU-R P.531-16 §4.3。

    Ω = 2.36 × 10⁻¹⁴ × B_av × N_T / f²  (rad)

    参数
    ----
    freq_ghz : 工作频率 (GHz)
    N_T      : TEC (el/m²)，默认 10¹⁷
    B_av     : 平均地球磁场 (T)，默认 50 μT = 5×10⁻⁵ T

    返回
    ----
    (Omega_rad, Omega_deg)
    """
    f_hz      = freq_ghz * 1e9
    Omega_rad = 2.36e-14 * B_av * N_T / (f_hz ** 2)
    return Omega_rad, math.degrees(Omega_rad)