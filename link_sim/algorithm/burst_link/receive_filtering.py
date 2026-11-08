"""对已过采样的接收波形进行独立 RRC 匹配滤波。"""

from __future__ import annotations

from numbers import Integral

import torch

from .pulse_shaping import RRC


class RRCMatchedFilter:
    """使用发射 RRC 的共轭翻转抽头进行完整卷积，不做上采样。

    输入为一维复数采样张量；每次调用视作独立突发，初始状态为零。
    输出包含完整滤波尾部，保持输入的 CPU/CUDA 设备和复数精度。
    """

    def __init__(self, shaper: RRC) -> None:
        if not isinstance(shaper, RRC):
            raise TypeError("shaper 必须是发射端使用的 RRC 实例")
        self.shaper = shaper

    @property
    def delay_samples(self) -> float:
        """接收 RRC 的群时延，单位为当前采样点。"""
        return self.shaper.delay_samples

    @property
    def tap_count(self) -> int:
        return self.shaper.tap_count

    def output_length(self, sample_count: int) -> int:
        """完整线性卷积的输出长度。"""
        if isinstance(sample_count, bool) or not isinstance(sample_count, Integral) or sample_count < 1:
            raise ValueError("输入采样数必须是正整数")
        return int(sample_count) + self.tap_count - 1

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("输入必须是 PyTorch 复数张量")
        if waveform.ndim != 1 or waveform.numel() == 0:
            raise ValueError("输入必须是非空的一维复数采样张量")
        if waveform.dtype not in (torch.complex64, torch.complex128):
            raise TypeError("输入精度必须是 torch.complex64 或 torch.complex128")
        if not bool(torch.isfinite(waveform).all().item()):
            raise ValueError("输入不能包含非有限值")
        taps = self.shaper.coefficients(device=waveform.device, dtype=waveform.real.dtype)
        taps = taps.flip(0).conj().to(waveform.dtype)
        out_length = self.output_length(waveform.numel())
        nfft = 1 << (out_length - 1).bit_length()
        return torch.fft.ifft(torch.fft.fft(waveform, n=nfft) * torch.fft.fft(taps, n=nfft))[:out_length]
