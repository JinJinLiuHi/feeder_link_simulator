"""
云雾衰减模块 — ITU-R P.840-9
公式（P.840-9 式 12）：

    K_L(f) = K_l(f, T=273.15 K) × (A1·exp(-(f-f1)²/σ1)
                                   + A2·exp(-(f-f2)²/σ2)
                                   + A3)

    A_c(f) = K_L(f) × L / sin(θ)   (dB)

其中：
    A1 = 0.1522,  A2 = 11.51,   A3 = -10.4912
    f1 = -23.9589 GHz,  f2 = 219.2096 GHz
    σ1 = 3.2991×10³,   σ2 = 2.7595×10⁶

    L  : 地表液态水柱含量 (kg/m² 或 mm)
    θ  : 地面站仰角 (°)，适用范围 5°-90°
"""

import math
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════
#  P.840-9 K_l(f, T) — 双德拜介电常数模型（式 2-11）
# ══════════════════════════════════════════════════════════

def _calc_Kl_base(freq_ghz: float, temperature_k: float = 273.75) -> float:
    """
    基础质量吸收系数 K_l(f, T)，(dB/km)/(g/m³)
    供式 12 的 K_L 拟合修正使用，固定 T =  273.75 K。
    """
    f     = freq_ghz
    T     = temperature_k
    theta = 300.0 / T - 1.0

    eps0 = 77.66 + 103.3 * theta
    eps1 = 0.0671 * eps0
    eps2 = 3.52

    fp = 20.20 - 146.0 * theta + 316.0 * theta ** 2
    fs = 39.8 * fp

    x  = f / fp
    y  = f / fs
    er = ((eps0 - eps1) / (1.0 + x ** 2)
          + (eps1 - eps2) / (1.0 + y ** 2)
          + eps2)
    ei = (f * (eps0 - eps1) / (fp * (1.0 + x ** 2))
          + f * (eps1 - eps2) / (fs * (1.0 + y ** 2)))

    eta = (2.0 + er) / ei
    return 0.819 * f / (ei * (1.0 + eta ** 2))


# ══════════════════════════════════════════════════════════
#  P.840-9 式 12：K_L(f) 拟合修正系数
# ══════════════════════════════════════════════════════════

# 拟合参数（P.840-9 式 12）
_A1 = 0.1522
_A2 = 11.51
_A3 = -10.4912
_f1 = -23.9589      # GHz
_f2 = 219.2096      # GHz
_s1 = 3.2991e3      # σ₁
_s2 = 2.7595e6      # σ₂


def calc_KL(freq_ghz: float) -> float:
    """
    云雾比衰减系数 K_L(f)，(dB/km)/(g/m³)，即 (dB/km)/(kg/m²·km⁻¹)
    P.840-9 式 12：
        K_L(f) = K_l(f, 273.15) × (A1·exp(-(f-f1)²/σ1)
                                   + A2·exp(-(f-f2)²/σ2) + A3)
    """
    Kl_base = _calc_Kl_base(freq_ghz,  273.75)
    f = freq_ghz
    correction = (_A1 * math.exp(-(f - _f1) ** 2 / _s1)
                  + _A2 * math.exp(-(f - _f2) ** 2 / _s2)
                  + _A3)
    return Kl_base * correction


# ══════════════════════════════════════════════════════════
#  斜路径衰减（P.840-9 §3）
# ══════════════════════════════════════════════════════════

def calc_cloud_attenuation(
    freq_ghz: float,
    L_kg_m2: float,
    elevation_deg: float,
) -> tuple[float, float]:
    """
    云雾斜路径衰减。

    A_c = K_L(f) × L / sin(θ)   (dB)，适用 5° ≤ θ ≤ 90°

    低仰角（< 5°）用余弦定理修正斜路径（等效云高 2 km）：
        L_path = 2H / (sqrt(sin²θ + 2H/Re) + sinθ)
        A_c = K_L × (L / H) × L_path
    其中 Re = 8500 km。

    参数
    ----
    L_kg_m2      : 地表液态水柱含量 L (kg/m² 或 mm)
                   典型值：薄云 0.1-0.3，中等云 0.3-1.0，厚积云 1.0-3.0
                   北京年超越概率 1% 约 0.3 kg/m²

    返回
    ----
    (K_L, A_c_dB)
    """
    KL    = calc_KL(freq_ghz)
    theta = math.radians(elevation_deg)


    A_c = KL * L_kg_m2 / math.sin(theta)

    return KL, A_c


# ══════════════════════════════════════════════════════════
#  结果 dataclass
# ══════════════════════════════════════════════════════════

@dataclass
class CloudAttenuationResult:
    freq_ghz:     float
    elevation_deg:float
    L_kg_m2:      float
    KL:           float   # 云雾比衰减系数 (dB/km)/(g/m³)
    A_cloud:      float   # 斜路径衰减 dB


# ══════════════════════════════════════════════════════════
#  统一入口
# ══════════════════════════════════════════════════════════

def compute_cloud_attenuation(
    freq_ghz:     float,
    L_kg_m2:      float,
    elevation_deg:float = 10.0,
) -> CloudAttenuationResult:
    """
    计算云雾斜路径衰减（P.840-9）。

    参数
    ----
    freq_ghz     : 工作频率 (GHz)，有效范围 1-200 GHz
    L_kg_m2      : 地表液态水柱含量 (kg/m²)
    elevation_deg: 地面站仰角 (°)
    """
    KL, A_c = calc_cloud_attenuation(freq_ghz, L_kg_m2, elevation_deg)
    return CloudAttenuationResult(
        freq_ghz=freq_ghz,
        elevation_deg=elevation_deg,
        L_kg_m2=L_kg_m2,
        KL=KL,
        A_cloud=A_c,
    )