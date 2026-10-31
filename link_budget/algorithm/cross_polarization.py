"""
交叉极化效应模块
1. 法拉第旋转导致的 XPD（电离层，P.531-16）
2. 雨衰导致的 XPD（P.618-14）

XPD（交叉极化鉴别度）：被测极化方向的主极化功率与交叉极化功率之比（dB），
越大越好，典型要求 > 25 dB。
"""

import math


# ══════════════════════════════════════════════════════════
#  1. 法拉第旋转导致的 XPD
#
#  线极化天线在法拉第旋转角 Ω 下，交叉极化功率为：
#    XPD_faraday = -20 × log10(|tan(Ω)|)  (dB)
#  当 Ω 很小时 tan(Ω) ≈ Ω，XPD 约为 -20log10(Ω)。
#  圆极化不受法拉第旋转影响（XPD → ∞）。
# ══════════════════════════════════════════════════════════

def calc_xpd_faraday(
    faraday_rotation_rad: float,
    polarization: str = "linear",
) -> float:
    """
    法拉第旋转导致的交叉极化鉴别度 XPD (dB)。

    参数
    ----
    faraday_rotation_rad : 法拉第旋转角 Ω (rad)
    polarization         : 'linear'（线极化）或 'circular'（圆极化）

    返回
    ----
    XPD_faraday (dB)，正值表示主极化优于交叉极化
    """
    if polarization == "circular":
        return float("inf")   # 圆极化免疫法拉第旋转

    Omega = abs(faraday_rotation_rad)
    if Omega < 1e-12:
        return 99.9   # 旋转角极小，XPD 趋于无穷

    # XPD = -20 log10(|tan Ω|)
    xpd = -20.0 * math.log10(abs(math.tan(Omega)))
    return xpd


# ══════════════════════════════════════════════════════════
#  2. 雨衰导致的 XPD（P.618-14 §4.1）
#
#  适用范围：6 ≤ f ≤ 55 GHz，θ ≤ 60°
#
#  步骤：
#   Step 1: C_f  — 频率相关项
#   Step 2: C_A  — 雨衰相关项
#   Step 3: C_τ  — 极化倾角改善因子
#   Step 4: C_θ  — 仰角相关项
#   Step 5: C_σ  — 雨滴伪角项
#   Step 6: XPD_rain = C_f - C_A + C_τ + C_θ + C_σ
# ══════════════════════════════════════════════════════════

def calc_xpd_rain(
    A_p: float,
    freq_ghz: float,
    elevation_deg: float,
    tau_deg: float = 45.0,
    sigma_deg: float = 5.0,
) -> dict:
    """
    雨衰导致的交叉极化鉴别度 XPD（P.618-14 §4.1）。

    参数
    ----
    A_p           : 链路雨衰（共极化衰减 CPA）(dB)
    freq_ghz      : 工作频率 (GHz)，6-55 GHz 有效
    elevation_deg : 仰角 θ (°)，≤ 60° 有效
    tau_deg       : 极化倾角 τ (°)
                    水平 0°，垂直 90°，圆极化 45°（默认）
                    C_τ 在 τ=45° 时为 0（最差），τ=0° 或 90° 时最大 15 dB
    sigma_deg     : 雨滴伪角标准差 σ (°)
                    P.618 推荐：1% → 0°，0.1% → 5°，0.01% → 10°，0.001% → 15°
                    默认 5°（对应约 0.1% 超越概率）

    返回
    ----
    dict：
      C_f, C_A, C_tau, C_theta, C_sigma : 各步骤中间项 (dB)
      XPD_rain : 雨衰 XPD (dB)
    """
    f     = freq_ghz
    theta = elevation_deg
    tau   = tau_deg
    sigma = sigma_deg

    # Step 1: 频率相关项 C_f
    if 6 <= f < 9:
        C_f = 60.0 * math.log10(f) - 28.3
    elif 9 <= f < 36:
        C_f = 26.0 * math.log10(f) + 4.1
    elif 36 <= f <= 55:
        C_f = 35.9 * math.log10(f) - 11.3
    else:
        # 超出有效范围，外推
        C_f = 26.0 * math.log10(f) + 4.1

    # Step 2: 雨衰相关项 C_A
    # C_A = V(f) × log10(A_p)
    # V(f): 6-9 GHz → 30.8f^-0.21, 9-20 GHz → 12.8f^0.19,
    #       20-40 GHz → 22.6, 40-55 GHz → 13f^0.15
    if A_p <= 0:
        C_A = 0.0
    else:
        if 6 <= f < 9:
            V = 30.8 * f ** (-0.21)
        elif 9 <= f < 20:
            V = 12.8 * f ** 0.19
        elif 20 <= f < 40:
            V = 22.6
        elif 40 <= f <= 55:
            V = 13.0 * f ** 0.15
        else:
            V = 22.6
        C_A = V * math.log10(A_p)

    # Step 3: 极化改善因子 C_τ
    # C_τ = -10 log10(1 - 0.484(1 + cos4τ))
    tau_r = math.radians(tau)
    C_tau = -10.0 * math.log10(1.0 - 0.484 * (1.0 + math.cos(4.0 * tau_r)))

    # Step 4: 仰角相关项 C_θ
    # C_θ = -40 log10(cos θ)  for θ ≤ 60°
    theta_r = math.radians(min(theta, 60.0))
    C_theta = -40.0 * math.log10(math.cos(theta_r))

    # Step 5: 雨滴伪角项 C_σ
    # C_σ = 0.0053 σ²
    C_sigma = 0.0053 * sigma ** 2

    # Step 6: XPD_rain
    XPD_rain = C_f - C_A + C_tau + C_theta + C_sigma

    return {
        "C_f":     round(C_f,     4),
        "C_A":     round(C_A,     4),
        "C_tau":   round(C_tau,   4),
        "C_theta": round(C_theta, 4),
        "C_sigma": round(C_sigma, 4),
        "XPD_rain":round(XPD_rain,4),
    }