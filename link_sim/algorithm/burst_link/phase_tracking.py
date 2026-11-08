"""帧头辅助相位跟踪，不负责同步、均衡、解调或译码。"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class PhaseTrackingResult:
    """同长度复符号、逐符号相位、幅度增益及相位误差。

    首个参考之前的符号保留输入，轨迹为 NaN；相位单位为弧度。
    帧头跟踪法的 error 仅在非零已知帧头符号上有值，数据段为 NaN。
    输入和输出的符号索引不变；幅度增益为接收幅度/参考幅度。
    """

    symbols: torch.Tensor
    phase: torch.Tensor
    gain: torch.Tensor
    error: torch.Tensor


def _validate_header_spans(value: Iterable[tuple[int, int]], length: int,
           header_length: int) -> tuple[tuple[int, int], ...]:
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise TypeError("header_spans 必须是区间序列") from exc
    if not raw:
        raise ValueError("至少需要一个帧头")
    result = []
    previous_end = 0
    for span in raw:
        try:
            a, b = span
            if isinstance(a, bool) or isinstance(b, bool):
                raise TypeError
            a, b = operator.index(a), operator.index(b)
        except (TypeError, ValueError) as exc:
            raise ValueError("帧头区间必须是整数起止对") from exc
        if a < previous_end or b - a != header_length or b > length:
            raise ValueError("帧头区间必须有序、不重叠、位于输入内且长度与参考一致")
        result.append((a, b))
        previous_end = b
    return tuple(result)


class DecisionDirectedPhaseTracker:
    """利用已知帧头和数据临时判决的一符号更新二阶相位环。

    kp、ki 分别为比例与积分增益。环路状态在单次调用中跨帧头、
    跨码字连续保持；每次调用均重新初始化，不跨突发保存状态。
    数据判决采用传入调制器的 constellation_points(device)。
    每个帧头估计恒定幅度，增益在该帧头及后续数据段内保持。
    首个帧头相位由整段已知帧头相关初始化，因而为离线块处理；
    它要求帧头内相位变化不大，不能用于未知帧头或快速失锁场景。

    跟踪使用输入张量原设备，逐符号存在递归依赖，当前为直观实现；
    CUDA 上会启动大量小算子，不承诺高速，后续可用专门内核优化。
    """

    def __init__(self, *, kp: float = 0.05, ki: float = 0.0005) -> None:
        self.kp = self._gain(kp, "kp", positive=True)
        self.ki = self._gain(ki, "ki", positive=False)

    @staticmethod
    def _gain(value: Any, name: str, *, positive: bool) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} 必须是有限实数")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} 必须是有限实数") from exc
        if not math.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError(f"{name} 必须是{'正' if positive else '非负'}有限数")
        return value

    @staticmethod
    def _spans(value: Iterable[tuple[int, int]], length: int,
               header_length: int) -> tuple[tuple[int, int], ...]:
        return _validate_header_spans(value, length, header_length)

    @torch.no_grad()
    def __call__(self, symbols: torch.Tensor, *, header_spans: Iterable[tuple[int, int]],
                 header_symbols: torch.Tensor, modulation: Any) -> PhaseTrackingResult:
        """跟踪一个突发。所有输入均在同一设备；输出与输入等长。"""
        if not isinstance(symbols, torch.Tensor) or symbols.ndim != 1 or not symbols.is_complex():
            raise TypeError("symbols 必须是一维复数张量")
        if symbols.dtype not in (torch.complex64, torch.complex128):
            raise TypeError("symbols 必须为 complex64 或 complex128")
        if not bool(torch.isfinite(symbols).all().item()):
            raise ValueError("symbols 必须有限")
        if (not isinstance(header_symbols, torch.Tensor) or header_symbols.ndim != 1
                or not header_symbols.is_complex() or header_symbols.numel() == 0):
            raise TypeError("header_symbols 必须是非空一维复数张量")
        if header_symbols.device != symbols.device:
            raise ValueError("header_symbols 与 symbols 必须在同一设备")
        reference = header_symbols.to(dtype=symbols.dtype)
        if not bool(torch.isfinite(reference).all().item()):
            raise ValueError("帧头参考必须有限")
        energy = reference.abs().square().sum()
        if not bool((energy > 0).item()):
            raise ValueError("帧头参考能量必须大于零")
        spans = self._spans(header_spans, symbols.numel(), reference.numel())
        get_points = getattr(modulation, "constellation_points", None)
        if not callable(get_points):
            raise TypeError("调制器须提供 constellation_points(device)")
        points = get_points(symbols.device)
        if (not isinstance(points, torch.Tensor) or points.ndim != 1 or points.numel() < 2
                or not points.is_complex() or points.device != symbols.device):
            raise ValueError("星座须为当前设备上的一维复数张量，至少含两个点")
        points = points.to(dtype=symbols.dtype)
        if not bool(torch.isfinite(points).all().item()):
            raise ValueError("星座点必须有限")

        out = symbols.clone()
        real_dtype = symbols.real.dtype
        phase_trace = torch.full((symbols.numel(),), float("nan"), dtype=real_dtype, device=symbols.device)
        gain_trace = torch.full_like(phase_trace, float("nan"))
        error_trace = torch.full_like(phase_trace, float("nan"))
        first_start, first_end = spans[0]
        first_corr = torch.sum(symbols[first_start:first_end] * reference.conj())
        tiny = torch.finfo(real_dtype).tiny
        if not bool((first_corr.abs() > tiny).item()):
            raise ValueError("首个帧头相关幅度过小，无法初始化相位和增益")
        phase = torch.angle(first_corr)
        frequency = torch.zeros((), device=symbols.device, dtype=real_dtype)

        for frame_index, (header_start, header_end) in enumerate(spans):
            # 一个帧头估计一个幅度，以对应的已知参考能量归一化。
            corr = torch.sum(symbols[header_start:header_end] * reference.conj())
            gain = corr.abs() / energy
            if not bool((gain > tiny).item()):
                raise ValueError(f"第 {frame_index + 1} 个帧头的增益估计无效")
            next_header_start = spans[frame_index + 1][0] if frame_index + 1 < len(spans) else symbols.numel()
            for index in range(header_start, next_header_start):
                corrected = symbols[index] / gain * torch.exp(-1j * phase)
                if index < header_end:
                    decided = reference[index - header_start]
                else:
                    decided = points[torch.argmin(torch.abs(corrected - points))]
                # 零参考相位未定义，不据此更新；避免对自定义帧头作额外假设。
                error = torch.angle(corrected * decided.conj())
                out[index] = corrected
                phase_trace[index] = phase
                gain_trace[index] = gain
                error_trace[index] = error
                frequency = frequency + self.ki * error
                phase = phase + frequency + self.kp * error

        return PhaseTrackingResult(out, phase_trace, gain_trace, error_trace)


class PilotPhaseTracker:
    """只用已知同步头和帧头估计相位，并离线补偿数据符号。

    可选 sync_symbols 为对齐输入开头的已调制同步符号；提供时，同步头
    复相关提供额外相位参考，估计时间为参考符号能量加权中心。幅度
    仍仅由各帧头复相关估计。未提供同步头时沿用纯帧头估计，首帧头
    前的符号保持原样；提供时从同步头首符号开始补偿。

    先对各参考相位差取主值解缠，按各自能量加权中心插值。尾部按最后
    两个相位参考的斜率外推，只有一个相位参考时保持常值。幅度仅由
    帧头估计，在帧头中心之间线性插值，两端保持边界值。数据符号不
    参与估计；modulation 仅用于兼容已有调用接口，可以省略。

    整个同步头和帧头各只给出一个相位参考。块内快速相位变化会引起
    相干抵消或估计偏差；相邻参考之间真实相位差达到或超过 π 时存在
    解缠方向或整周歧义。尾部线性外推与相位插值均不保证恢复快速相噪。
    同步头相关数值近零时拒绝估计，但非零不代表统计上可靠。
    """

    def __init__(self, *, sync_symbols: torch.Tensor | None = None):
        if sync_symbols is not None:
            if (not isinstance(sync_symbols, torch.Tensor) or sync_symbols.ndim != 1
                    or not sync_symbols.is_complex() or sync_symbols.numel() == 0):
                raise TypeError("sync_symbols 必须是非空一维复数张量或 None")
            if sync_symbols.dtype not in (torch.complex64, torch.complex128):
                raise TypeError("sync_symbols 必须为 complex64 或 complex128")
            if not bool(torch.isfinite(sync_symbols).all().item()):
                raise ValueError("同步头参考必须有限")
            if not bool((sync_symbols.abs().square().sum() > 0).item()):
                raise ValueError("同步头参考能量必须大于零")
            sync_symbols = sync_symbols.detach().clone()
        self._sync_symbols = sync_symbols

    @torch.no_grad()
    def __call__(self, symbols: torch.Tensor, *, header_spans: Iterable[tuple[int, int]],
                 header_symbols: torch.Tensor, modulation: Any = None) -> PhaseTrackingResult:
        if not isinstance(symbols, torch.Tensor) or symbols.ndim != 1 or not symbols.is_complex():
            raise TypeError("symbols 必须是一维复数张量")
        if symbols.dtype not in (torch.complex64, torch.complex128):
            raise TypeError("symbols 必须为 complex64 或 complex128")
        if not bool(torch.isfinite(symbols).all().item()):
            raise ValueError("symbols 必须有限")
        if (not isinstance(header_symbols, torch.Tensor) or header_symbols.ndim != 1
                or not header_symbols.is_complex() or header_symbols.numel() == 0):
            raise TypeError("header_symbols 必须是非空一维复数张量")
        if header_symbols.device != symbols.device:
            raise ValueError("header_symbols 与 symbols 必须在同一设备")
        reference = header_symbols.to(dtype=symbols.dtype)
        if not bool(torch.isfinite(reference).all().item()):
            raise ValueError("帧头参考必须有限")
        ref_power = reference.abs().square()
        energy = ref_power.sum()
        if not bool((energy > 0).item()):
            raise ValueError("帧头参考能量必须大于零")
        spans = _validate_header_spans(header_spans, symbols.numel(), reference.numel())
        headers = torch.stack([symbols[start:end] for start, end in spans])
        correlations = (headers * reference.conj()).sum(dim=1) / energy
        gains = correlations.abs()
        tiny = torch.finfo(symbols.real.dtype).tiny
        if not bool((torch.isfinite(gains) & (gains > tiny)).all().item()):
            raise ValueError("帧头复增益无效或为零，不能估计相位及归一化幅度")

        weight = ref_power.to(torch.float64)
        center_offset = (torch.arange(reference.numel(), device=symbols.device,
                                      dtype=torch.float64) * weight).sum() / weight.sum()
        header_centers = torch.tensor([start for start, _ in spans], device=symbols.device,
                                      dtype=torch.float64) + center_offset
        phase_centers = header_centers
        phase_correlations = correlations
        active_start = spans[0][0]
        if self._sync_symbols is not None:
            if self._sync_symbols.device != symbols.device:
                raise ValueError("sync_symbols 与 symbols 必须在同一设备")
            sync_ref = self._sync_symbols.to(dtype=symbols.dtype)
            sync_length = sync_ref.numel()
            if sync_length > spans[0][0]:
                raise ValueError("同步头必须位于输入开头且不与首个帧头重叠")
            sync_power = sync_ref.abs().square().to(torch.float64)
            sync_energy = sync_power.sum()
            observed = symbols[:sync_length]
            cross = (observed * sync_ref.conj()).sum()
            observed_energy = observed.abs().square().to(torch.float64).sum()
            scale = torch.sqrt(observed_energy * sync_energy)
            # 只拒绝相对于输入/参考能量在数值精度内的近零相关，不是统计置信检验。
            numerical_floor = scale * (32 * torch.finfo(symbols.real.dtype).eps)
            if not bool((torch.isfinite(cross) & (cross.abs().to(torch.float64) > numerical_floor)).item()):
                raise ValueError("同步头相关幅度数值上接近零，无法可靠估计相位")
            sync_center = (torch.arange(sync_length, device=symbols.device, dtype=torch.float64)
                           * sync_power).sum() / sync_energy
            phase_centers = torch.cat((sync_center.reshape(1), header_centers))
            phase_correlations = torch.cat(((cross / sync_energy).reshape(1), correlations))
            active_start = 0

        phase0 = torch.angle(phase_correlations[:1]).to(torch.float64)
        if phase_correlations.numel() == 1:
            phases = phase0
        else:
            # 相邻真实增量达到 π 时存在方向歧义，超过则发生整周混叠。
            delta = torch.angle(phase_correlations[1:] * phase_correlations[:-1].conj()).to(torch.float64)
            phases = torch.cat((phase0, phase0 + torch.cumsum(delta, dim=0)))
        positions = torch.arange(symbols.numel(), device=symbols.device, dtype=torch.float64)
        if phases.numel() == 1:
            phase = phases[0].expand_as(positions)
        else:
            right = torch.searchsorted(phase_centers, positions, right=True).clamp(1, phases.numel() - 1)
            left = right - 1
            fraction = (positions - phase_centers[left]) / (phase_centers[right] - phase_centers[left])
            phase = phases[left] + fraction * (phases[right] - phases[left])
        # 不用同步头的增益：保持原先仅由帧头确定的归一化轨迹。
        if len(spans) == 1:
            gain = gains[0].to(torch.float64).expand_as(positions)
        else:
            right = torch.searchsorted(header_centers, positions, right=True).clamp(1, len(spans) - 1)
            left = right - 1
            fraction = ((positions - header_centers[left]) /
                        (header_centers[right] - header_centers[left])).clamp(0.0, 1.0)
            gain = gains[left].to(torch.float64) + fraction * (gains[right] - gains[left]).to(torch.float64)
        active = positions >= active_start
        correction = torch.polar(gain.reciprocal(), -phase).to(dtype=symbols.dtype)
        out = torch.where(active, symbols * correction, symbols)
        real_dtype = symbols.real.dtype
        missing = torch.full_like(positions, float("nan"), dtype=real_dtype)
        phase_trace = torch.where(active, phase.to(real_dtype), missing)
        gain_trace = torch.where(active, gain.to(real_dtype), missing)
        error_trace = missing.clone()
        ref_nonzero = reference.abs() > 0
        for start, end in spans:
            error = torch.angle(out[start:end] * reference.conj())
            error_trace[start:end] = torch.where(ref_nonzero, error, missing[start:end])
        return PhaseTrackingResult(out, phase_trace, gain_trace, error_trace)
