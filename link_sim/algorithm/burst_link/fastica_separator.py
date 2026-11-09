"""基于外部同步位置的双极化 FastICA 分离。"""
from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import torch

from .xpic_fastica import ComplexFastICA, FastICAResult


@dataclass(frozen=True)
class FastICASeparationResult:
    """排序后的两路波形、采样窗口及分离诊断。

    ``fit_start`` 和 ``fit_end`` 是输入共同采样时间轴上的整数坐标，
    窗口采用 ``[fit_start, fit_end)``。两路波形保留输入时间轴。
    """
    waveform_a: torch.Tensor
    waveform_b: torch.Tensor
    fastica: FastICAResult
    fit_start: int
    fit_end: int
    coefficient_shares: tuple[float, float]
    coefficient_margin: float
    index_a: int
    index_b: int


class FastICAPolarizationSeparator:
    """从共享采样时基上的两路匹配滤波波形中分离两个极化信源。

    调用方分别提供外部同步器测得的 A/B 同步起点，单位为输入采样点；
    以较晚起点加 ``window_offset_samples`` 为基准，向上取整后选择长度为
    ``window_length_samples`` 的训练窗口，FastICA 在该窗口拟合单个瞬时
    分离矩阵，再作用于完整波形。窗口偏移和长度应由调用方选择，须包含
    足够有效且近似独立的源信号；该对象不读取帧结构、同步头或信道真值。

    使用估计混合系数按“同极化路径更强且两路接收增益可比较”判定输出
    路序；区分度过低时拒绝输出。判路与迭代收敛并不保证分离质量。瞬时
    模型不能保证消除一般频率选择性耦合、卷积混合或路径差分时延。
    """

    def __init__(self, *, window_offset_samples: int, window_length_samples: int,
                 ica: ComplexFastICA | None = None, coefficient_margin: float = 0.1):
        self.window_offset_samples = self._integer(window_offset_samples, "window_offset_samples", 0)
        self.window_length_samples = self._integer(window_length_samples, "window_length_samples", 128)
        self.ica = ComplexFastICA() if ica is None else ica
        if not isinstance(self.ica, ComplexFastICA):
            raise TypeError("ica 必须是 ComplexFastICA")
        if isinstance(coefficient_margin, bool):
            raise TypeError("coefficient_margin 必须是 0 到 1 之间的有限数")
        self.coefficient_margin = float(coefficient_margin)
        if not math.isfinite(self.coefficient_margin) or not 0 <= self.coefficient_margin <= 1:
            raise ValueError("coefficient_margin 必须是 0 到 1 之间的有限数")

    @staticmethod
    def _integer(value: int, name: str, minimum: int) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{name} 必须是整数")
        try:
            value = operator.index(value)
        except TypeError as error:
            raise TypeError(f"{name} 必须是整数") from error
        if value < minimum:
            raise ValueError(f"{name} 必须大于等于 {minimum}")
        return value

    @staticmethod
    def _start(value: float, name: str, size: int) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} 必须是有限的同步起点")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value < size:
            raise ValueError(f"{name} 必须位于输入波形的采样范围内")
        return value

    @torch.no_grad()
    def __call__(self, waveform_a: torch.Tensor, waveform_b: torch.Tensor, *,
                 start_a: float, start_b: float, training_mask: torch.Tensor | None = None) -> FastICASeparationResult:
        """输入两路波形及外部同步起点，输出已判序的两路波形和诊断。

        可选 training_mask 仅选择训练窗口内的有效样本；分离矩阵仍作用于完整波形。
        """
        if (not isinstance(waveform_a, torch.Tensor)
                or not isinstance(waveform_b, torch.Tensor)
                or waveform_a.ndim != 1 or waveform_b.ndim != 1
                or waveform_a.numel() == 0 or waveform_a.numel() != waveform_b.numel()):
            raise ValueError("两路输入必须是等长非空一维张量，并具有共同采样时基")
        if (waveform_a.dtype not in (torch.complex64, torch.complex128)
                or waveform_b.dtype != waveform_a.dtype
                or waveform_a.device != waveform_b.device):
            raise ValueError("两路输入必须使用相同的复数精度和设备")
        start_a = self._start(start_a, "start_a", waveform_a.numel())
        start_b = self._start(start_b, "start_b", waveform_b.numel())
        fit_start = math.ceil(max(start_a, start_b) + self.window_offset_samples)
        fit_end = fit_start + self.window_length_samples
        if fit_end > waveform_a.numel():
            raise ValueError("FastICA 训练窗口超出输入波形范围")
        separated = self.ica(waveform_a, waveform_b, start=fit_start, end=fit_end, training_mask=training_mask)
        if not separated.converged:
            raise RuntimeError("复数 FastICA 未收敛，不能将输出当作已分离数据")

        # 行向量约定：sources = (received - mean) @ unmixing，逆矩阵
        # 的每一行给出一个分离分量在 A/B 接收支路的相对混合系数。
        power = torch.linalg.inv(separated.unmixing).abs().square()
        shares = power[:, 0] / power.sum(dim=1).clamp_min(torch.finfo(power.dtype).tiny)
        index_a = int(torch.argmax(shares).item())
        difference = float((shares.max() - shares.min()).item())
        if difference <= self.coefficient_margin:
            raise RuntimeError("混合系数不足以可靠判别两路")
        index_b = 1 - index_a
        return FastICASeparationResult(
            separated.components[index_a], separated.components[index_b],
            separated, fit_start, fit_end,
            tuple(float(value) for value in shares.detach().cpu().tolist()),
            difference, index_a, index_b,
        )
