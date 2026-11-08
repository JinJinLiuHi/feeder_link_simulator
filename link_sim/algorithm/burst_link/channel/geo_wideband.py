"""单路 GEO 下行宽带等效信道；频响、时间及功率均按离散基带约定。

DLR 36 MHz OMUX 五阶 Chebyshev II 近似（阻带 38 dB，边缘 28.6 MHz）：
https://elib.dlr.de/99356/1/TWC2699185.pdf
名义带宽缩放参考 ETSI TR 102 376-2 第 4.4.1.2 节：
https://www.etsi.org/deliver/etsi_tr/102300_102399/10237602/01.02.01_60/tr_10237602v010201p.pdf
宽带缩放、双线性离散化和 IIR 冲激响应有限长度截断均为近似，并非具体卫星设备实测。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import signal
import torch
import torch.nn.functional as F


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限数值")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值")
    return value


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须为非负整数")
    return value


def _reference_omux(fs: float, bandwidth: float) -> np.ndarray:
    """36 MHz OMUX 缩放至指定名义带宽，稳定 IIR 截断为有限 FIR。"""
    edge = 28.6 / 36 * bandwidth
    if not 0 < edge < fs / 2:
        raise ValueError("缩放后的滤波器边缘频率必须大于 0 且低于采样率的一半")
    # scipy 的数字 cheby2 使用双线性变换；频率预畸变使给定边缘对齐。
    sos = signal.cheby2(5, 38, edge, fs=fs, output="sos")
    _, poles, _ = signal.sos2zpk(sos)
    largest = float(np.max(np.abs(poles)))
    length = max(128, math.ceil(math.log(1e-10) / math.log(largest)) + 64) if largest else 128
    if length > 32768:
        raise ValueError("参考滤波器尾部过长，请调整带宽或直接指定 FIR")
    impulse = np.zeros(length, dtype=np.float64)
    impulse[0] = 1.0
    return signal.sosfilt(sos, impulse)


def _validate_mask(mask):
    if mask is None:
        return None
    if not isinstance(mask, (tuple, list)) or len(mask) != 2:
        raise ValueError("phase_noise_mask 需为 (频偏 Hz 序列, 单边带相噪 dBc/Hz 序列)")
    freq = np.asarray(mask[0], dtype=np.float64)
    power = np.asarray(mask[1], dtype=np.float64)
    if freq.ndim != 1 or power.ndim != 1 or len(freq) != len(power) or len(freq) < 2:
        raise ValueError("相噪频偏与功率谱必须为等长、至少两个元素的一维序列")
    if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(power)) or np.any(freq <= 0) or np.any(np.diff(freq) <= 0):
        raise ValueError("相噪频偏须为递增正数，频偏和功率谱均须有限")
    return freq.copy(), power.copy()


def _phase_noise(count: int, fs: float, mask, device, generator):
    """有限时长谱合成：SSB L(f) 在小角度条件下对应双边相位 PSD=10**(L/10)。

    只合成掩模频率范围内的频率；频率间隔为 fs/count，不补造低于此间隔的相噪。
    """
    freq = np.fft.rfftfreq(count, 1.0 / fs)
    offsets, levels = mask
    valid = (freq >= offsets[0]) & (freq <= offsets[-1]) & (freq > 0)
    if not np.any(valid):
        raise ValueError(f"相噪掩模与当前采样窗口可表示的频率不相交；频率分辨率为 {fs / count:g} Hz")
    density = np.zeros_like(freq)
    density[valid] = 10.0 ** (np.interp(np.log10(freq[valid]), np.log10(offsets), levels) / 10.0)
    density = torch.as_tensor(density, dtype=torch.float64, device=device)
    scale = torch.sqrt(density * (count * fs / 2))
    re = torch.randn(scale.shape, device=device, dtype=torch.float64, generator=generator) * scale
    im = torch.randn(scale.shape, device=device, dtype=torch.float64, generator=generator) * scale
    re[0] = 0
    im[0] = 0
    if count % 2 == 0:
        re[-1] *= math.sqrt(2)
        im[-1] = 0
    return torch.fft.irfft(torch.complex(re, im), n=count)


def _convolve(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    length = x.numel() + h.numel() - 1
    nfft = 1 << (length - 1).bit_length()
    return torch.fft.ifft(torch.fft.fft(x, n=nfft) * torch.fft.fft(h, n=nfft))[:length]


class GEOWidebandChannel:
    """处理一个突发，输出含前后窗口的一维复采样，保留有限 FIR 的完整尾部。

    sample_rate_hz: 采样率。filter_bandwidth_hz: 缩放参考 OMUX 的名义带宽。
    filter_taps: 指定复 FIR 时替代参考响应；单位为本信道采样率。
    delay_samples: 相对接收窗口起点的整数+小数采样时延；不填充真实 GEO 长时延。
    absolute_delay_s: GEO 共同绝对传播时延，只记录为元信息，不产生海量零采样。
    frequency_offset_hz: 接收等效频偏；initial_phase_rad: 输出窗口 n=0 时确定性部分的初始相位；相噪另行叠加。
    phase_noise_mask: (递增正频偏 Hz, 单边带相噪 dBc/Hz)。对数频率线性插值，不外推。
    ebn0_db: 接收数据参考 Eb/N0，None 表示无接收噪声。
    energy_per_bit: 用户提供的接收信息位离散归一化能量 Eb；必须大于零。
       接收噪声复采样均方 = Eb / 10**(EbN0/10)，假设接收匹配滤波器单位能量。
       不使用包括同步头和窗口空白的突发平均能量估算 Eb。
    guard_before / guard_after: 输出的前后静默采样数，噪声覆盖整个输出。

    本对象不计算绝对链路预算；固定接收 Eb/N0 时重新依据衰减后信号调整噪声会
    抵消损耗造成的 SNR 变化。有限长度相噪只含 >= 1/T 且在掩模范围的频率。
    """

    def __init__(self, *, sample_rate_hz: float, filter_bandwidth_hz: float = 1e9,
                 filter_taps: Sequence[complex] | torch.Tensor | None = None,
                 delay_samples: float = 0.0, absolute_delay_s: float = 0.0,
                 frequency_offset_hz: float = 0.0, initial_phase_rad: float = 0.0,
                 phase_noise_mask=None, ebn0_db: float | None = None,
                 energy_per_bit: float | None = None, amplitude: float = 1.0,
                 guard_before: int = 64, guard_after: int = 64,
                 fractional_filter_half_length: int = 16):
        self.sample_rate_hz = _finite("sample_rate_hz", sample_rate_hz)
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz 必须大于 0")
        self.filter_bandwidth_hz = _finite("filter_bandwidth_hz", filter_bandwidth_hz)
        if filter_taps is None:
            taps = _reference_omux(self.sample_rate_hz, self.filter_bandwidth_hz)
            self.response_source = "DLR OMUX Chebyshev II 缩放近似，有限 FIR 截断"
        else:
            taps = filter_taps.detach().cpu().numpy() if isinstance(filter_taps, torch.Tensor) else np.asarray(filter_taps)
            if taps.ndim != 1 or taps.size == 0 or not np.all(np.isfinite(taps)):
                raise ValueError("filter_taps 必须是非空、有限的一维复数序列")
            taps = np.asarray(taps, dtype=np.complex128).copy()
            self.response_source = "调用方指定 FIR"
        self.filter_taps = torch.as_tensor(taps.copy())
        self.delay_samples = _finite("delay_samples", delay_samples)
        self._absolute_delay_s = _finite("absolute_delay_s", absolute_delay_s)
        if self.delay_samples < 0 or self.absolute_delay_s < 0:
            raise ValueError("delay_samples 和 absolute_delay_s 不可为负")
        self.frequency_offset_hz = _finite("frequency_offset_hz", frequency_offset_hz)
        self.initial_phase_rad = _finite("initial_phase_rad", initial_phase_rad)
        if abs(self.frequency_offset_hz) >= self.sample_rate_hz / 2:
            raise ValueError("频偏绝对值须小于采样率的一半")
        self.phase_noise_mask = _validate_mask(phase_noise_mask)
        self.ebn0_db = None if ebn0_db is None else _finite("ebn0_db", ebn0_db)
        self.energy_per_bit = None if energy_per_bit is None else _finite("energy_per_bit", energy_per_bit)
        if self.ebn0_db is not None and (self.energy_per_bit is None or self.energy_per_bit <= 0):
            raise ValueError("配置 ebn0_db 时必须提供正的 energy_per_bit")
        self.amplitude = _finite("amplitude", amplitude)
        if self.amplitude < 0:
            raise ValueError("amplitude 不可为负")
        self.guard_before = _nonnegative_int("guard_before", guard_before)
        self.guard_after = _nonnegative_int("guard_after", guard_after)
        self.fractional_filter_half_length = _nonnegative_int("fractional_filter_half_length", fractional_filter_half_length)
        if self.fractional_filter_half_length < 2 and not self.delay_samples.is_integer():
            raise ValueError("小数时延需要至少为 2 的 fractional_filter_half_length")
        self.fractional_filter_delay_samples = self.fractional_filter_half_length if not self.delay_samples.is_integer() else 0

    @property
    def absolute_delay_s(self) -> float:
        """共同传播时延，只作为只读时间信息，不添加采样点。"""
        return self._absolute_delay_s

    def frequency_response(self, points: int = 2048) -> tuple[np.ndarray, np.ndarray]:
        """返回线性滤波器的频率 Hz 与复响应，不含时延、频偏和相噪。"""
        if isinstance(points, bool) or not isinstance(points, int) or points < 2:
            raise ValueError("points 须是不小于 2 的整数")
        frequencies = np.fft.fftshift(np.fft.fftfreq(points, 1 / self.sample_rate_hz))
        _, response = signal.freqz(self.filter_taps.numpy(), worN=2 * np.pi * frequencies / self.sample_rate_hz)
        return frequencies, response

    def __call__(self, waveform: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 1 or waveform.numel() == 0:
            raise ValueError("waveform 须为非空的一维 torch.Tensor")
        if waveform.dtype not in (torch.complex64, torch.complex128):
            raise TypeError("waveform 须为 complex64 或 complex128")
        if generator is not None:
            generator_device = torch.device(generator.device)
            if generator_device.type == "cuda" and generator_device.index is None:
                generator_device = torch.device("cuda", torch.cuda.current_device())
            if generator_device != waveform.device:
                raise ValueError("generator 必须与波形位于同一设备")
        taps = self.filter_taps.to(device=waveform.device, dtype=waveform.dtype)
        out = _convolve(waveform, taps)
        integer = math.floor(self.delay_samples)
        fraction = self.delay_samples - integer
        if fraction:
            half = self.fractional_filter_half_length
            k = torch.arange(-half, half + 1, dtype=waveform.real.dtype, device=waveform.device)
            window = torch.hann_window(2 * half + 1, periodic=False, dtype=k.dtype, device=k.device)
            h = torch.sinc(k - fraction) * window
            out = _convolve(out, (h / h.sum()).to(waveform.dtype))
        out = F.pad(out, (self.guard_before + integer, self.guard_after))
        # 以 float64 计算相位，避免 float32 长序列时间精度不足。
        n = torch.arange(out.numel(), dtype=torch.float64, device=out.device)
        phase = self.initial_phase_rad + 2 * math.pi * self.frequency_offset_hz / self.sample_rate_hz * n
        if self.phase_noise_mask is not None:
            phase = phase + _phase_noise(out.numel(), self.sample_rate_hz, self.phase_noise_mask, out.device, generator)
        out = out * torch.polar(torch.ones_like(phase), phase).to(waveform.dtype) * self.amplitude
        if self.ebn0_db is not None:
            sigma = math.sqrt(self.energy_per_bit * 10 ** (-self.ebn0_db / 10) / 2)
            nr = torch.randn(out.shape, dtype=out.real.dtype, device=out.device, generator=generator)
            ni = torch.randn(out.shape, dtype=out.real.dtype, device=out.device, generator=generator)
            out = out + torch.complex(nr, ni) * sigma
        return out
