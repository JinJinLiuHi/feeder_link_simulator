"""单突发复基带符号的整数倍上采样与根升余弦成型。"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
import torch.nn.functional as functional


class RRC:
    """整数倍插值和单位能量 RRC 成型，保留完整滤波尾部。

    参数以符号间隔为单位；输入输出为一维复数张量，输出保留输入设备与精度。
    """

    def __init__(
        self,
        *,
        samples_per_symbol: int = 4,
        rolloff: float = 0.25,
        span_symbols: int = 8,
    ) -> None:
        if isinstance(samples_per_symbol, bool) or not isinstance(samples_per_symbol, Integral) or samples_per_symbol < 2:
            raise ValueError("每符号采样数必须是大于等于 2 的整数")
        if isinstance(span_symbols, bool) or not isinstance(span_symbols, Integral) or span_symbols < 1:
            raise ValueError("滤波器跨度必须是正整数")
        if isinstance(rolloff, bool) or not isinstance(rolloff, Real) or not math.isfinite(float(rolloff)) or not 0 <= rolloff <= 1:
            raise ValueError("滚降系数必须是 0 到 1 之间的有限数值")

        self._samples_per_symbol = int(samples_per_symbol)
        self._rolloff = float(rolloff)
        self._span_symbols = int(span_symbols)
        self._coefficients_cpu = self._design_taps()
        self._device_taps: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    @property
    def samples_per_symbol(self) -> int:
        return self._samples_per_symbol

    @property
    def rolloff(self) -> float:
        return self._rolloff

    @property
    def span_symbols(self) -> int:
        return self._span_symbols

    @property
    def tap_count(self) -> int:
        return self.span_symbols * self.samples_per_symbol + 1

    @property
    def delay_samples(self) -> float:
        """RRC FIR 群延迟，单位为输出采样；可能是半个采样点。"""
        return (self.tap_count - 1) / 2

    @property
    def delay_symbols(self) -> float:
        return self.delay_samples / self.samples_per_symbol

    def output_length(self, symbol_count: int) -> int:
        """完整卷积的输出采样数；每次调用均使用零初始状态。"""
        if isinstance(symbol_count, bool) or not isinstance(symbol_count, Integral) or symbol_count < 1:
            raise ValueError("符号数必须是正整数")
        return (int(symbol_count) - 1) * self.samples_per_symbol + self.tap_count

    def _design_taps(self) -> torch.Tensor:
        alpha = self.rolloff
        pi = math.pi
        values = []
        for index in range(self.tap_count):
            t = (index - self.delay_samples) / self.samples_per_symbol
            if alpha == 0:
                value = 1.0 if t == 0 else math.sin(pi * t) / (pi * t)
            elif t == 0:
                value = 1 + alpha * (4 / pi - 1)
            elif math.isclose(abs(4 * alpha * t), 1.0, rel_tol=1e-12, abs_tol=1e-12):
                angle = pi / (4 * alpha)
                value = (alpha / math.sqrt(2)) * (
                    (1 + 2 / pi) * math.sin(angle) + (1 - 2 / pi) * math.cos(angle)
                )
            else:
                value = (
                    math.sin(pi * t * (1 - alpha)) + 4 * alpha * t * math.cos(pi * t * (1 + alpha))
                ) / (pi * t * (1 - (4 * alpha * t) ** 2))
            values.append(value)
        norm = math.sqrt(math.fsum(value * value for value in values))
        return torch.tensor([value / norm for value in values], dtype=torch.float64)

    def _taps_on(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = (device, dtype)
        if key not in self._device_taps:
            self._device_taps[key] = self._coefficients_cpu.to(device=device, dtype=dtype)
        return self._device_taps[key]

    def coefficients(
        self, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        """返回单位离散能量 RRC 抽头的独立副本。"""
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("抽头精度必须是 torch.float32 或 torch.float64")
        return self._taps_on(torch.device(device), dtype).clone()

    def __call__(self, symbols: torch.Tensor) -> torch.Tensor:
        if not isinstance(symbols, torch.Tensor):
            raise TypeError("输入必须是 PyTorch 复数张量")
        if symbols.ndim != 1 or symbols.numel() == 0:
            raise ValueError("输入必须是非空的一维复数符号张量")
        if symbols.dtype not in (torch.complex64, torch.complex128):
            raise ValueError("输入精度必须是 torch.complex64 或 torch.complex128")

        taps = self._taps_on(symbols.device, symbols.real.dtype)
        iq = torch.stack((symbols.real, symbols.imag)).unsqueeze(1)
        filtered = functional.conv_transpose1d(
            iq, taps.reshape(1, 1, -1), stride=self.samples_per_symbol
        )
        return torch.complex(filtered[0, 0], filtered[1, 0])

