"""双输入双输出的静态双极化等效信道。"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from .geo_wideband import GEOWidebandChannel, _finite, _nonnegative_int, _reference_omux


def _pair(value, name: str) -> tuple[float, float]:
    """标量或二元序列依次对应发射 A、发射 B。"""
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} 必须是数值或长度为 2 的数值序列")
    try:
        values = tuple(value)
    except TypeError:
        values = (value, value)
    if len(values) != 2:
        raise ValueError(f"{name} 须为标量或两个数值")
    return tuple(_finite(name, item) for item in values)


def _receiver_pair(value, name: str, *, allow_none: bool = False):
    """接收 A/B 两路；标量表示共用，允许指定的量逐路关闭。"""
    if value is None and allow_none:
        return None, None
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} 必须是数值或两个数值")
    try:
        values = tuple(value)
    except TypeError:
        values = (value, value)
    if len(values) != 2:
        raise ValueError(f"{name} 必须是数值或接收 A/B 的两个值")
    return tuple(None if allow_none and item is None else _finite(name, item)
                 for item in values)


def _tap_array(value, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        taps = np.asarray(value, dtype=np.complex128)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} 必须是一维复数抽头") from exc
    if taps.ndim != 1 or taps.size == 0 or not np.all(np.isfinite(taps)):
        raise ValueError(f"{name} 必须是非空、有限的一维复数抽头")
    return taps.copy()


class DualPolarizationChannel:
    """一个双极化突发的静态 2×2 线性信道。

    行是接收极化，列是发射极化：
      y_A = h_AA*x_A + h_AB*x_B
      y_B = h_BA*x_A + h_BB*x_B
    其中 * 表示完整线性卷积。波形 A/B 必须在同一采样时间轴上。

    未指定 path_taps 时，两个同极化路径使用同一 filter_taps（或现有
    GEO_WIDEBAND 的参考响应）。xpd_db 可为标量或 (A, B)：按单路发射
    定义，A 是 10log10(|h_AA|² / |h_BA|²)，B 是
    10log10(|h_BB|² / |h_AB|²)，默认交叉响应分别是对应同极化响应
    的固定复比例。xpd_db=None 表示两交叉路径为零，不表示实际隔离度。
    cross_phase_rad 同样依次对应 A→B、B→A。因路径形状相同，默认配置
    的 XPD 不随频率变化；自定义四条复 FIR 可表达更一般的频率响应，
    其 XPD 由实际频响决定。不能同时使用 path_taps 和 xpd_db。

    source_delay_samples 为 (发射 A, 发射 B) 两路相对采样延时，分别
    同时作用于同一发射极化对应的两条路径，不能表示四条路径独立时延。
    两路仍共享接收窗口和采样时钟；以采样点计，可为非负整数或小数，
    默认 (0, 0) 不改变原有模型。小数延时由 Hann 窗 sinc FIR 近似，
    任一路需要小数延时时两路均增加相同的公共因果滤波器时延
    source_delay_filter_delay_samples（不影响相对延时），并保留完整尾部。
    这些相对延时是模型参数，不代表真实馈电设备或卫星链路的实测值；
    它们不同于只存为元信息的长传播时延，也不应被用于展开 GEO 长时延。

    四路径分别卷积并按接收支路合并后，使用两个独立接收支路模型。
    delay_samples、frequency_offset_hz、initial_phase_rad、ebn0_db、
    energy_per_bit 和 amplitude 可为标量或接收 (A,B) 双值；这些是
    合并后的接收支路参数，不能表达每个发射源独立频偏或相噪。
    接收时延为短的窗口内偏差；如有小数延时，两路添加相同的因果
    滤波延迟，并补足共同输出窗口的尾部，使两路仍处于同一采样时轴。
    phase_noise_mask 保留单个共享掩模接口；receiver_phase_noise_masks
    显式给出 (mask_A, mask_B)，各项可为 None，不得与共享掩模同时用。
    两接收支路的相噪及 AWGN 分别独立生成；这不表示共享振荡器模型。
    绝对 GEO 传播时延只保留为元信息，不展开巨长采样。

    ebn0_db 与 energy_per_bit 遵守 GEOWidebandChannel 的相同口径；
    噪声只在交叉信号合并后加入一次，与另一极化输入是否全零无关。
    每次调用一个突发，不承担跨突发状态或 XPIC/均衡。
    """

    def __init__(self, *, sample_rate_hz: float, filter_bandwidth_hz: float = 1e9,
                 filter_taps: Sequence[complex] | torch.Tensor | None = None,
                 path_taps: Sequence | None = None,
                 xpd_db: float | Sequence[float] | None = None,
                 cross_phase_rad: float | Sequence[float] = 0.0,
                 delay_samples: float | Sequence[float] = 0.0, absolute_delay_s: float = 0.0,
                 source_delay_samples: float | Sequence[float] = (0.0, 0.0),
                 frequency_offset_hz: float | Sequence[float] = 0.0,
                 initial_phase_rad: float | Sequence[float] = 0.0,
                 phase_noise_mask=None, receiver_phase_noise_masks=None,
                 ebn0_db: float | Sequence[float | None] | None = None,
                 energy_per_bit: float | Sequence[float | None] | None = None,
                 amplitude: float | Sequence[float] = 1.0,
                 guard_before: int = 64, guard_after: int = 64,
                 fractional_filter_half_length: int = 16):
        fs = _finite("sample_rate_hz", sample_rate_hz)
        if fs <= 0:
            raise ValueError("sample_rate_hz 必须大于 0")
        phases = _pair(cross_phase_rad, "cross_phase_rad")
        if path_taps is not None:
            if xpd_db is not None or filter_taps is not None or phases != (0.0, 0.0):
                raise ValueError("自定义 path_taps 时不能同时设置 xpd_db、filter_taps 或非零 cross_phase_rad")
            try:
                rows = tuple(tuple(row) for row in path_taps)
            except TypeError as exc:
                raise ValueError("path_taps 必须是 2×2 复 FIR 抽头序列") from exc
            if len(rows) != 2 or any(len(row) != 2 for row in rows):
                raise ValueError("path_taps 必须是 2×2 复 FIR 抽头序列")
            paths = tuple(tuple(_tap_array(rows[r][c], f"path_taps[{r}][{c}]")
                                for c in range(2)) for r in range(2))
            self.xpd_db = None
            self.response_source = "调用方指定的四条 FIR 路径"
        else:
            bw = _finite("filter_bandwidth_hz", filter_bandwidth_hz)
            if filter_taps is None:
                direct = _reference_omux(fs, bw).astype(np.complex128)
                self.response_source = "GEO 参考频响与固定交叉比例"
            else:
                direct = _tap_array(filter_taps, "filter_taps")
                self.response_source = "调用方指定主路径频响与固定交叉比例"
            if xpd_db is None:
                if phases != (0.0, 0.0):
                    raise ValueError("配置 cross_phase_rad 需要同时配置 xpd_db")
                cross_a = cross_b = 0j
                self.xpd_db = None
            else:
                xpds = _pair(xpd_db, "xpd_db")
                try:
                    cross_a = 10.0 ** (-xpds[0] / 20.0) * np.exp(1j * phases[0])
                    cross_b = 10.0 ** (-xpds[1] / 20.0) * np.exp(1j * phases[1])
                except OverflowError as exc:
                    raise ValueError("xpd_db 数值过大导致复增益溢出") from exc
                if not np.isfinite(cross_a) or not np.isfinite(cross_b):
                    raise ValueError("xpd_db 产生了无效交叉增益")
                self.xpd_db = xpds
            paths = ((direct.copy(), direct * cross_b),
                     (direct * cross_a, direct.copy()))
        # 每个发射源对应矩阵的一列；同一列的两条接收路径使用同一延时。
        source_delays = _pair(source_delay_samples, "source_delay_samples")
        if any(delay < 0 for delay in source_delays):
            raise ValueError("source_delay_samples 不能为负；请将窗口起点设为较早路径")
        half = _nonnegative_int("fractional_filter_half_length", fractional_filter_half_length)
        fractions = tuple(delay - math.floor(delay) for delay in source_delays)
        has_fraction = any(fraction != 0 for fraction in fractions)
        if has_fraction and half < 2:
            raise ValueError("小数时延需要至少为 2 的 fractional_filter_half_length")
        self.source_delay_samples = source_delays
        # 若任一路需要小数时延，另一条路也补上相同的 h 个样点，使相对
        # 时延保持正确；这 h 是实现滤波器的公共因果时延，不是传播时延。
        self.source_delay_filter_delay_samples = half if has_fraction else 0
        if has_fraction or any(source_delays):
            delays = []
            for delay, fraction in zip(source_delays, fractions):
                integer = math.floor(delay)
                if has_fraction:
                    if fraction:
                        positions = np.arange(-half, half + 1, dtype=np.float64)
                        kernel = np.sinc(positions - fraction) * np.hanning(2 * half + 1)
                        kernel /= kernel.sum()
                    else:
                        kernel = np.zeros(2 * half + 1, dtype=np.float64)
                        kernel[half] = 1.0
                else:
                    kernel = np.ones(1, dtype=np.float64)
                delays.append(np.pad(kernel, (integer, 0)))
            paths = tuple(tuple(np.convolve(paths[row][col], delays[col])
                                for col in range(2)) for row in range(2))
        self.sample_rate_hz = fs
        self._path_taps = paths
        self.max_tap_length = max(taps.size for row in paths for taps in row)
        bank = np.zeros((2, 2, self.max_tap_length), dtype=np.complex128)
        for row in range(2):
            for col in range(2):
                taps = paths[row][col]
                bank[row, col, :taps.size] = taps
        self._tap_bank = torch.from_numpy(bank)
        # A/B 是接收支路；单值沿用旧行为，二元组允许分别指定。
        receiver_delay = _receiver_pair(delay_samples, "delay_samples")
        receiver_frequency = _receiver_pair(frequency_offset_hz, "frequency_offset_hz")
        receiver_phase = _receiver_pair(initial_phase_rad, "initial_phase_rad")
        receiver_ebn0 = _receiver_pair(ebn0_db, "ebn0_db", allow_none=True)
        receiver_energy = _receiver_pair(energy_per_bit, "energy_per_bit", allow_none=True)
        receiver_amplitude = _receiver_pair(amplitude, "amplitude")
        if any(delay < 0 for delay in receiver_delay):
            raise ValueError("delay_samples 不能为负")
        if receiver_phase_noise_masks is not None:
            if phase_noise_mask is not None:
                raise ValueError("receiver_phase_noise_masks 与共享 phase_noise_mask 不能同时设置")
            if isinstance(receiver_phase_noise_masks, (str, bytes)):
                raise TypeError("receiver_phase_noise_masks 必须包含两个掩模或 None")
            try:
                receiver_masks = tuple(receiver_phase_noise_masks)
            except TypeError as error:
                raise TypeError("receiver_phase_noise_masks 必须包含两个掩模或 None") from error
            if len(receiver_masks) != 2:
                raise ValueError("receiver_phase_noise_masks 必须包含接收 A/B 的两个掩模或 None")
        else:
            receiver_masks = (phase_noise_mask, phase_noise_mask)

        # GEOWideband 的小数延时 FIR 自带 half 个采样因果延迟。
        # 若只有一路为小数，则为整数延时路加相同的共同延迟，避免无意中
        # 给两路引入 half 个采样的额外相对偏差。
        before = _nonnegative_int("guard_before", guard_before)
        after = _nonnegative_int("guard_after", guard_after)
        fractions = tuple(d != math.floor(d) for d in receiver_delay)
        if any(fractions) and half < 2:
            raise ValueError("接收小数时延需要至少为 2 的 fractional_filter_half_length")
        common_filter_delay = half if any(fractions) else 0
        extra_before = tuple(0 if fractional else common_filter_delay for fractional in fractions)
        extra_length = tuple(math.floor(d) + (2 * half if frac else extra)
                             for d, frac, extra in zip(receiver_delay, fractions, extra_before))
        max_extra = max(extra_length)
        receives = []
        for i in range(2):
            # 用额外 guard_after 而不是对输出补零，以确保接收噪声覆盖整个窗口。
            receives.append(GEOWidebandChannel(
                sample_rate_hz=fs, filter_bandwidth_hz=filter_bandwidth_hz,
                filter_taps=[1], delay_samples=receiver_delay[i],
                absolute_delay_s=absolute_delay_s,
                frequency_offset_hz=receiver_frequency[i], initial_phase_rad=receiver_phase[i],
                phase_noise_mask=receiver_masks[i], ebn0_db=receiver_ebn0[i],
                energy_per_bit=receiver_energy[i], amplitude=receiver_amplitude[i],
                guard_before=before + extra_before[i],
                guard_after=after + max_extra - extra_length[i],
                fractional_filter_half_length=fractional_filter_half_length))
        self._receive_a, self._receive_b = receives
        self.receiver_delay_samples = receiver_delay
        self.receiver_fractional_filter_delay_samples = tuple(
            receives[i].fractional_filter_delay_samples + extra_before[i] for i in range(2))
        self.receiver_frequency_offset_hz = receiver_frequency
        self.receiver_initial_phase_rad = receiver_phase
        self.receiver_ebn0_db = receiver_ebn0
        self.receiver_energy_per_bit = receiver_energy
        self.receiver_amplitude = receiver_amplitude
        self.delay_samples = receiver_delay[0] if receiver_delay[0] == receiver_delay[1] else receiver_delay
        self.fractional_filter_delay_samples = common_filter_delay

    @property
    def absolute_delay_s(self) -> float:
        return self._receive_a.absolute_delay_s

    @torch.no_grad()
    def __call__(self, waveform_a: torch.Tensor, waveform_b: torch.Tensor,
                 *, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """两路对齐复采样 → 两路对齐复采样，保持设备与复数精度。"""
        if not isinstance(waveform_a, torch.Tensor) or not isinstance(waveform_b, torch.Tensor):
            raise TypeError("两路波形均须为 torch.Tensor")
        if waveform_a.ndim != 1 or waveform_b.ndim != 1 or waveform_a.numel() == 0:
            raise ValueError("两路波形须为等长的非空一维张量")
        if waveform_a.numel() != waveform_b.numel():
            raise ValueError("两路波形须等长，并采用同一采样时间基准")
        if waveform_a.dtype not in (torch.complex64, torch.complex128) or waveform_b.dtype != waveform_a.dtype:
            raise TypeError("两路波形须为相同精度的 complex64 或 complex128")
        if waveform_a.device != waveform_b.device:
            raise ValueError("两路波形须位于同一设备")
        if not bool(torch.isfinite(waveform_a).all().item()) or not bool(torch.isfinite(waveform_b).all().item()):
            raise ValueError("两路波形不能包含非有限值")
        output_len = waveform_a.numel() + self.max_tap_length - 1
        nfft = 1 << (output_len - 1).bit_length()
        inputs = torch.stack((waveform_a, waveform_b))
        frequencies = torch.fft.fft(inputs, n=nfft, dim=-1)
        path_bank = self._tap_bank.to(device=inputs.device, dtype=inputs.dtype)
        response = torch.fft.fft(path_bank, n=nfft, dim=-1)
        mixed = torch.fft.ifft((response * frequencies.unsqueeze(0)).sum(dim=1), n=nfft, dim=-1)[..., :output_len]
        return (self._receive_a(mixed[0], generator=generator),
                self._receive_b(mixed[1], generator=generator))
