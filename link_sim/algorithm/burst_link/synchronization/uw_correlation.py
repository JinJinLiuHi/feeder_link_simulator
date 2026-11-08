"""对接收匹配滤波后的波形进行已知同步符号滑动相关峰值检测。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from link_sim.algorithm.burst_link.pulse_shaping import RRC
from link_sim.algorithm.burst_link.receive_filtering import RRCMatchedFilter


@dataclass(frozen=True)
class SynchronizationResult:
    """滑动相关结果；坐标以输入的匹配滤波波形采样序号为基准。

    peak_index 是本地完整同步符号模板起点；start_sample 是其对应的首个同步符号
    的整数采样位置。fractional_offset 是相关峰的三点抛物线拟合偏差；
    fractional_peak_index 与 fractional_start_sample 是对应的小数采样坐标。
    仅当 fractional_valid 为 True 时小数坐标可用；fractional_reason 解释状态。
    success 仍仅表示整数相关检测是否通过门限。未执行波形插值、符号抽取
    或频偏估计和补偿；只使用输入波形，不读取信道对象及其真实时延。
    """
    success: bool
    reason: str
    correlation: torch.Tensor
    peak_index: int | None
    start_sample: int | None
    peak_value: float | None
    fractional_valid: bool = False
    fractional_reason: str = "相关检测未成功，无法估计小数位置"
    fractional_offset: float | None = None
    fractional_peak_index: float | None = None
    fractional_start_sample: float | None = None


def _refine_peak(correlation: torch.Tensor, peak_index: int) -> tuple[float | None, str]:
    """对相关幅度的整数峰及两侧样点做抛物线拟合，返回采样间隔为单位的偏移。

    不在边界上外推；平顶、曲率在数据精度内不可区分或无效偏移不伪造小数结果。
    三个邻点一同从设备读取；拟合不需要信道或发射端的真实定时。
    """
    if peak_index <= 0 or peak_index >= correlation.numel() - 1:
        return None, "相关峰位于搜索边界，无法进行三点拟合"
    left, center, right = correlation[peak_index - 1:peak_index + 2].to(torch.float64).cpu().tolist()
    if not all(math.isfinite(value) for value in (left, center, right)):
        return None, "相关峰邻点存在非有限值"
    if center <= left or center <= right:
        return None, "相关峰为平顶或非严格局部最大值，无法确定小数位置"
    curvature = left - 2.0 * center + right
    tolerance = 8.0 * torch.finfo(correlation.dtype).eps * max(1.0, abs(left), abs(center), abs(right))
    if curvature >= -tolerance:
        return None, "相关峰曲率不足，无法稳定估计小数位置"
    offset = 0.5 * (left - right) / curvature
    if not math.isfinite(offset) or abs(offset) > 0.5 + 8.0 * torch.finfo(correlation.dtype).eps:
        return None, "拟合结果超出峰值相邻采样区间"
    return min(0.5, max(-0.5, offset)), "小数峰估计成功"


def _convolve(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """FFT 完整线性卷积，输出复数精度与设备跟随输入。"""
    length = x.numel() + h.numel() - 1
    nfft = 1 << (length - 1).bit_length()
    return torch.fft.ifft(torch.fft.fft(x, n=nfft) * torch.fft.fft(h, n=nfft))[:length]


class CorrelationSynchronizer:
    """使用已匹配滤波的单突发波形和已知同步符号模板进行相关峰搜索。

    shaper 是发射端使用的 RRC 实例；内部只对本地参考进行发送成型和
    接收匹配滤波，调用时不对输入波形重复滤波。
    threshold 是归一化相关幅度门限，取 (0, 1]；低于门限时返回峰值并标记失败。
    搜索输入内能容纳完整参考模板的位置，相关序号即该模板起点序号。
    """
    def __init__(self, *, sync_symbols: torch.Tensor, shaper: RRC, threshold: float = 0.5):
        if not isinstance(shaper, RRC):
            raise TypeError("shaper 必须是发送时使用的 RRC 实例")
        if not isinstance(sync_symbols, torch.Tensor) or sync_symbols.ndim != 1 or sync_symbols.numel() == 0:
            raise ValueError("sync_symbols 必须是非空的一维复数张量")
        if sync_symbols.dtype not in (torch.complex64, torch.complex128):
            raise TypeError("sync_symbols 必须为 complex64 或 complex128")
        if not bool(torch.isfinite(sync_symbols).all().item()):
            raise ValueError("sync_symbols 不允许包含非有限值")
        if not bool((sync_symbols.abs() > 0).any().item()):
            raise ValueError("sync_symbols 不允许全为零")
        if isinstance(threshold, bool):
            raise ValueError("threshold 必须在 (0, 1] 范围内")
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("threshold 必须在 (0, 1] 范围内")
        self.shaper = shaper
        self.threshold = threshold
        self._sync_symbols = sync_symbols.detach().clone()
        self._matched_filter = RRCMatchedFilter(shaper)
        self._templates: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _template(self, waveform: torch.Tensor) -> torch.Tensor:
        key = (waveform.device, waveform.dtype)
        if key not in self._templates:
            reference = self._sync_symbols.to(device=waveform.device, dtype=waveform.dtype)
            self._templates[key] = self._matched_filter(self.shaper(reference))
        return self._templates[key]

    @torch.no_grad()
    def __call__(self, waveform: torch.Tensor) -> SynchronizationResult:
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 1 or waveform.numel() == 0:
            raise ValueError("waveform 必须是非空的一维复数张量")
        if waveform.dtype not in (torch.complex64, torch.complex128):
            raise TypeError("waveform 必须为 complex64 或 complex128")
        if not bool(torch.isfinite(waveform).all().item()):
            raise ValueError("waveform 不允许包含非有限值")
        template = self._template(waveform)
        size = template.numel()
        if waveform.numel() < size:
            empty = torch.empty(0, device=waveform.device, dtype=waveform.real.dtype)
            return SynchronizationResult(False, "接收窗口不足以容纳完整同步模板", empty, None, None, None)

        # 只允许模板完全位于输入采样窗口中的位置，相关数组索引即模板起点。
        numerator = _convolve(waveform, template.conj().flip(0))[size - 1:waveform.numel()]
        energy = waveform.abs().square().to(torch.float64)
        prefix = functional.pad(torch.cumsum(energy, dim=0), (1, 0))
        window_energy = (prefix[size:] - prefix[:-size]).clamp_min(0)
        template_energy = template.abs().square().to(torch.float64).sum()
        max_energy = window_energy.max()
        floor = max_energy * torch.finfo(waveform.real.dtype).eps
        denominator = torch.sqrt(window_energy * template_energy)
        correlation = torch.where(
            window_energy > floor,
            numerator.abs().to(torch.float64) / denominator.clamp_min(torch.finfo(torch.float64).tiny),
            torch.zeros_like(window_energy),
        ).clamp(0, 1).to(waveform.real.dtype)
        peak_index = int(torch.argmax(correlation).item())
        peak_value = float(correlation[peak_index].item())
        success = peak_value >= self.threshold
        reason = "检测成功" if success else "相关峰低于检测门限"
        # 模板经过 Tx 和 Rx RRC，其首个符号位于两者群时延之和处。
        start_sample = peak_index + int(round(2 * self.shaper.delay_samples))
        if not success:
            return SynchronizationResult(success, reason, correlation, peak_index, start_sample, peak_value)
        offset, refinement_reason = _refine_peak(correlation, peak_index)
        return SynchronizationResult(
            success, reason, correlation, peak_index, start_sample, peak_value,
            fractional_valid=offset is not None,
            fractional_reason=refinement_reason,
            fractional_offset=offset,
            fractional_peak_index=None if offset is None else peak_index + offset,
            fractional_start_sample=None if offset is None else start_sample + offset,
        )
