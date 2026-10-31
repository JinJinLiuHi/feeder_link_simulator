"""
雨衰计算模块 — ITU-R P.838-3 + P.618-14
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Union

Number = Union[float, int, np.ndarray]

# ══════════════════════════════════════════════════════════
#  P.838-3 系数表
# ══════════════════════════════════════════════════════════
_KH_A  = [-5.33980,-0.35351,-0.23789,-0.94158]
_KH_B  = [-0.10008, 1.26970, 0.86036, 0.64552]
_KH_C  = [ 1.13098, 0.45400, 0.15354, 0.16817]
_KH_MK = -0.18961;  _KH_CK =  0.71147

_KV_A  = [-3.80595,-3.44965,-0.39902, 0.50167]
_KV_B  = [ 0.56934,-0.22911, 0.73042, 1.07319]
_KV_C  = [ 0.81061, 0.51059, 0.11899, 0.27195]
_KV_MK = -0.16398;  _KV_CK =  0.63297

_AH_A  = [-0.14318, 0.29591, 0.32177,-5.37610,16.1721]
_AH_B  = [ 1.82442, 0.77564, 0.63773,-0.96230,-3.29980]
_AH_C  = [-0.55187, 0.19822, 0.13164, 1.47828, 3.43990]
_AH_MA =  0.67849;  _AH_CA = -1.95537

_AV_A  = [-0.07771, 0.56727,-0.20238,-48.2991, 48.5833]
_AV_B  = [ 2.33840, 0.95545, 1.14520, 0.791669,0.791459]
_AV_C  = [-0.76284, 0.54039, 0.26809, 0.116226,0.116479]
_AV_MA = -0.053739; _AV_CA =  0.83433


def _calc_k(f, a, b, c, mk, ck):
    lgf = math.log10(f)
    s = sum(aj*math.exp(-((lgf-bj)/cj)**2) for aj,bj,cj in zip(a,b,c))
    return 10**(s + mk*lgf + ck)

def _calc_alpha(f, a, b, c, ma, ca):
    lgf = math.log10(f)
    s = sum(aj*math.exp(-((lgf-bj)/cj)**2) for aj,bj,cj in zip(a,b,c))
    return s + ma*lgf + ca

def calc_p838_coeffs(freq_ghz):
    kH = _calc_k(freq_ghz,     _KH_A,_KH_B,_KH_C,_KH_MK,_KH_CK)
    aH = _calc_alpha(freq_ghz, _AH_A,_AH_B,_AH_C,_AH_MA,_AH_CA)
    kV = _calc_k(freq_ghz,     _KV_A,_KV_B,_KV_C,_KV_MK,_KV_CK)
    aV = _calc_alpha(freq_ghz, _AV_A,_AV_B,_AV_C,_AV_MA,_AV_CA)
    return kH, aH, kV, aV

def calc_specific_attenuation(freq_ghz, rain_rate, elevation_deg, polarization_deg):
    """P.838-3 完整极化公式，返回 (gamma_R数组, k, alpha)"""
    rain_rate  = np.atleast_1d(np.asarray(rain_rate, dtype=float))
    kH,aH,kV,aV = calc_p838_coeffs(freq_ghz)
    theta      = math.radians(elevation_deg)
    tau        = math.radians(polarization_deg)
    cos2_theta = math.cos(theta)**2
    cos_2tau   = math.cos(2*tau)
    k = (kH+kV+(kH-kV)*cos2_theta*cos_2tau)/2.0
    if k > 0:
        alpha = (kH*aH+kV*aV+(kH*aH-kV*aV)*cos2_theta*cos_2tau)/(2.0*k)
    else:
        alpha = (aH+aV)/2.0
    return k*rain_rate**alpha, k, alpha


def calc_effective_path_length(rain_height_km, station_altitude_km,
                               elevation_deg, R001, polarization_deg,lat=35.0,
                               freq_ghz=20.0, gamma_R=None):
    """
    P.618-14 §2.2.1.1 有效路径长度 L_eff (km)。

    完整 10 步流程（θ ≥ 5°）：
      步骤2  L_S = (H_R - h_s) / sin θ
      步骤3  L_G = L_S · cos θ
      步骤5  γ_R = k·R001^α  （用 R001 作降雨强度）
      步骤6  r_001 = 1 / (1 + 0.78·√(L_G·γ_R/f) - 0.38·(1-exp(-2·L_G)))
      步骤7  ζ = arctan((H_R-h_s)/(L_G·r_001))
             L_R = L_G·r_001/cosθ  (ζ>θ)，否则 (H_R-h_s)/sinθ
             χ = max(36-|lat|, 0)
             v_001 = 1/(1 + √sinθ·(31·(1-exp(-θ/(1+χ)))·√(L_R·γ_R)/R001^0.45 - 0.45))
      步骤8  L_eff = L_R · v_001

    参数
    ----
    freq_ghz : 工作频率（GHz），用于步骤6 r_001 计算
    gamma_R  : 雨比衰减（dB/km）；若为 None 则由函数内部用 R001 计算
    """
    theta   = math.radians(max(elevation_deg, 5.0))
    sin_t   = math.sin(theta)
    cos_t   = math.cos(theta)
    delta_h = rain_height_km - station_altitude_km
    if delta_h <= 0:
        return 0.0

    # 步骤2-3: 斜路径长度和水平投影
    L_S = delta_h / sin_t
    L_G = L_S * cos_t

    # 步骤5: γ_R（用 R001 计算，或由外部传入）
    if gamma_R is None:
        # 调用 P.838-3 计算 γ_R(R001)，极化取 45°（圆极化，保守值）
        g_arr, _, _ = calc_specific_attenuation(
            freq_ghz, np.array([R001]), elevation_deg, polarization_deg)
        gamma_R = float(g_arr[0])
    gamma_R = max(gamma_R, 1e-6)

    # 步骤6: 水平换算系数 r_001
    # r_001 = 1 / (1 + 0.78·√(L_G·γ_R/f) - 0.38·(1-exp(-2·L_G)))
    r_num = 1.0 + 0.78 * math.sqrt(L_G * gamma_R / freq_ghz)                 - 0.38 * (1.0 - math.exp(-2.0 * L_G))
    r_001 = 1.0 / max(r_num, 1e-6)
    r_001 = max(min(r_001, 1.0), 0.01)

    # 步骤7: 修正路径长度 L_R
    zeta_deg = math.degrees(math.atan(delta_h / (L_G * r_001 + 1e-12)))
    if zeta_deg > elevation_deg:
        L_R = L_G * r_001 / cos_t
    else:
        L_R = delta_h / sin_t

    # 步骤7: 垂直调整系数 v_001
    # χ = max(36-|lat|, 0)
    # v_001 = 1/(1+√sinθ·(31·(1-exp(-θ_deg/(1+χ)))·√(L_R·γ_R)/R001^0.45 - 0.45))
    chi   = max(36.0 - abs(lat), 0.0)
    inner = (31.0 * (1.0 - math.exp(-elevation_deg / (1.0 + chi)))
             * math.sqrt(L_R * gamma_R)
             / (freq_ghz ** 2)
             - 0.45)
    v_001 = 1.0 / (1.0 + math.sqrt(sin_t) * inner)

    # 步骤8
    return max(L_R * v_001, 0.0)


# ══════════════════════════════════════════════════════════
#  P.618-14 式8：由 A(0.01%) 外推至任意超越概率 p%
# ══════════════════════════════════════════════════════════

def scale_attenuation_by_probability(A_001, p, elevation_deg, lat):
    if A_001 <= 0 or p <= 0:
        return 0.0
    if p == 0.01:
        return A_001
    elif p >= 1.0 or abs(lat) >= 36.0:
        beta = 0.0
    elif p < 1.0 and abs(lat) < 36.0 and elevation_deg >= 25.0:
        beta = -0.005*(abs(lat) - 36.0)
    else:
        beta = -0.005*(abs(lat) - 36.0) + 1.8 -4.25*math.sin(math.radians(elevation_deg))
    theta_r  = math.radians(elevation_deg)
    exponent = -(0.655 + 0.033*math.log(p)
                 - 0.045*math.log(A_001)
                 - beta*(1.0 - p)*math.sin(theta_r))
    return max(A_001 * (p/0.01)**exponent, 0.0)


# ══════════════════════════════════════════════════════════
#  结果 dataclass
# ══════════════════════════════════════════════════════════

@dataclass
class RainAttenuationResult:
    freq_ghz: float;  rain_rate: np.ndarray
    rain_height_km: float;  elevation_deg: float;  polarization_deg: float
    kH: float;  alphaH: float;  kV: float;  alphaV: float
    k:  float;  alpha:  float
    gamma_R: np.ndarray;  L_eff: np.ndarray;  A_rain: np.ndarray

@dataclass
class RainStatisticsResult:
    """P.618-14 多超越概率统计雨衰"""
    city: str;  freq_ghz: float
    elevation_deg: float;  rain_height_km: float
    R001: float;  lat: float
    probabilities: list   # [0.001, 0.01, 0.1, 1.0, 5.0] %
    gamma_R_vals:  list   # dB/km（对应各概率的 γ_R）
    R_p_vals:      list   # mm/h（对应各概率的 R_p）
    A_p_vals:      list   # dB（路径衰减）


# ══════════════════════════════════════════════════════════
#  主计算入口
# ══════════════════════════════════════════════════════════

def compute_rain_attenuation(
    freq_ghz, rain_rate, rain_height_km,
    elevation_deg=10.0, station_alt_km=0.0, polarization_deg=45.0,
    R001=None, lat=35.0,
) -> RainAttenuationResult:
    """
    单点雨衰（向后兼容）。
    R001 不为 None 时用 P.618-14 完整路径公式；否则用简化斜路径。
    """
    rain_rate = np.atleast_1d(np.asarray(rain_rate, dtype=float))
    kH,aH,kV,aV = calc_p838_coeffs(freq_ghz)
    gamma_R, k, alpha = calc_specific_attenuation(
        freq_ghz, rain_rate, elevation_deg, polarization_deg)

    if R001 is not None and R001 > 0:
        # 用 R001 对应的 gamma_R 传入路径公式（步骤5-7 均需要 gamma_R 和 freq）
        g_001, _, _ = calc_specific_attenuation(
            freq_ghz, np.array([R001]), elevation_deg, polarization_deg)
        gamma_R_001 = float(g_001[0])
        L_val = calc_effective_path_length(
            rain_height_km, station_alt_km, elevation_deg, polarization_deg, R001, lat,
            freq_ghz=freq_ghz, gamma_R=gamma_R_001)
    else:
        theta  = math.radians(max(elevation_deg, 5.0))
        delta_h = max(rain_height_km - station_alt_km, 0.0)
        L_val  = delta_h / math.sin(theta)

    L_eff  = np.full_like(rain_rate, L_val)
    A_rain = gamma_R * L_eff

    return RainAttenuationResult(
        freq_ghz=freq_ghz, rain_rate=rain_rate,
        rain_height_km=rain_height_km,
        elevation_deg=elevation_deg, polarization_deg=polarization_deg,
        kH=kH,alphaH=aH,kV=kV,alphaV=aV,k=k,alpha=alpha,
        gamma_R=gamma_R, L_eff=L_eff, A_rain=A_rain)


def compute_rain_statistics(
    freq_ghz, elevation_deg, polarization_deg,
    rain_height_km, R001, lat,
    station_alt_km=0.0, city="",
) -> RainStatisticsResult:
    """
    P.618-14 §2.2.1.1 完整流程：计算 5 个超越概率的路径衰减。
    超越概率：0.001% / 0.01% / 0.1% / 1% / 5%
    L_eff 通过 P.618-14 步骤5-8 完整公式计算，依赖 freq_ghz 和 gamma_R。
    """
    PROBS = [0.001, 0.01, 0.1, 1.0, 5.0]

    # 步骤5: γ_R(R001)
    g_arr, k, alpha = calc_specific_attenuation(
        freq_ghz, np.array([R001]), elevation_deg, polarization_deg)
    gamma_001 = float(g_arr[0])

    # 步骤6-8: L_eff（依赖 freq_ghz 和 gamma_R）
    L_eff = calc_effective_path_length(
        rain_height_km, station_alt_km, elevation_deg, polarization_deg, R001, lat,
        freq_ghz=freq_ghz, gamma_R=gamma_001)

    # 步骤9: A(0.01%) = γ_R(R001) × L_eff
    A_001 = gamma_001 * L_eff

    gamma_R_vals, R_p_vals, A_p_vals = [], [], []

    for p in PROBS:
        # 路径衰减（式8外推）
        A_p = scale_attenuation_by_probability(A_001, p, elevation_deg, lat)
        A_p_vals.append(round(A_p, 4))

        # 反推 R_p：γ_p = A_p/L_eff → R_p = (γ_p/k)^(1/alpha)
        gamma_p = A_p / (L_eff + 1e-12) if L_eff > 0 else 0.0
        # gamma_R_vals.append(round(gamma_p), 4)
        R_p = (gamma_p/k)**(1.0/alpha) if (gamma_p > 0 and k > 0 and alpha > 0) else 0.0
        R_p_vals.append(round(R_p, 2))

        # 对应的 γ_R
        if R_p > 0:
            g2, _, _ = calc_specific_attenuation(
                freq_ghz, np.array([R_p]), elevation_deg, polarization_deg)
            gamma_R_vals.append(round(float(g2[0]), 4))
        else:
            gamma_R_vals.append(0.0)

    return RainStatisticsResult(
        city=city, freq_ghz=freq_ghz,
        elevation_deg=elevation_deg, rain_height_km=rain_height_km,
        R001=R001, lat=lat,
        probabilities=PROBS,
        gamma_R_vals=gamma_R_vals,
        R_p_vals=R_p_vals,
        A_p_vals=A_p_vals)


def rain_attenuation_db(freq_ghz, rain_rate_mmh, rain_height_km,
                        elevation_deg=10.0, station_alt_km=0.0,
                        polarization_deg=45.0, R001=None, lat=35.0):
    res = compute_rain_attenuation(
        freq_ghz, rain_rate_mmh, rain_height_km,
        elevation_deg, station_alt_km, polarization_deg, R001, lat)
    return float(res.A_rain[0])