"""Dedicated Sionna CP-OFDM generation, loaded only when a waveform is requested.

The 50 MHz preset matches the visible MATLAB OFDM parameters (not a complete
NR downlink channel allocation). No post-generation FFT truncation/filtering.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OFDMPreset:
    spacing: float
    fft_size: int
    active_carriers: int
    cp_length: int
    symbols: int
    oversampling: int
    normal_cp: bool = False


PRESETS = {
    5e6: OFDMPreset(30e3, 256, 132, 18, 30, 7),
    15e6: OFDMPreset(30e3, 1024, 456, 72, 30, 7),
    40e6: OFDMPreset(30e3, 2048, 1272, 144, 30, 7),
    50e6: OFDMPreset(15e3, 4096, 3240, 288, 140, 1, True),
    100e6: OFDMPreset(30e3, 4096, 3276, 288, 30, 7),
}


def parameters(bandwidth, oversampling=None):
    if bandwidth not in PRESETS:
        raise ValueError("支持的 OFDM 带宽为 5 / 15 / 40 / 50 / 100 MHz")
    preset = PRESETS[bandwidth]
    factor = preset.oversampling if oversampling is None else oversampling
    if int(factor) != factor or not 1 <= factor <= 32:
        raise ValueError("OFDM 过采样倍数必须为 1–32 的整数")
    factor = int(factor)
    cp = np.full(preset.symbols, preset.cp_length, dtype=np.int64)
    if preset.normal_cp:
        # 15 kHz normal CP at NFFT=4096: 320 samples every half-subframe,
        # 288 otherwise. 140 symbols give exactly 10 ms at 61.44 MHz.
        cp[::7] = 320
    return dict(engine="NVIDIA Sionna", bandwidthHz=float(bandwidth),
                subcarrierSpacingHz=preset.spacing, fftSize=preset.fft_size * factor,
                activeCarriers=preset.active_carriers, numSymbols=preset.symbols,
                cyclicPrefixLengths=cp * factor, oversampling=factor,
                sampleRate=preset.spacing * preset.fft_size * factor,
                occupiedGridHz=preset.active_carriers * preset.spacing,
                qamOrder=64, dcNull=False, windowing=False, filtering=False,
                cpMode="normal" if preset.normal_cp else "constant")


def _modulate(bandwidth, oversampling=None, seed=12345):
    """Unscaled samples and frequency grid; shared with round-trip tests."""
    info = parameters(bandwidth, oversampling)
    from link_budget.algorithm.ber_core.diagnostics import configure_runtime_logging
    configure_runtime_logging()
    try:
        import torch
        from sionna.phy.mapping import Mapper
        from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator
    except (ImportError, OSError) as exc:
        import sys
        raise RuntimeError(f"OFDM 依赖导入失败：{exc}\n当前 Python：{sys.executable}\n"
                           "请在该环境检查 torch 和 sionna 的安装及版本。") from exc
    size, count = info["fftSize"], info["activeCarriers"]
    # Oversampling is implemented in the OFDM grid: larger FFT and longer CP
    # at unchanged subcarrier spacing. This does not low-pass the serialized
    # symbol transitions, whose natural sidelobes must remain in the signal.
    guard = size - count
    resource_grid = ResourceGrid(
        num_ofdm_symbols=info["numSymbols"], fft_size=size,
        subcarrier_spacing=info["subcarrierSpacingHz"],
        num_guard_carriers=(guard // 2, guard - guard // 2),
        dc_null=False, pilot_pattern="empty", precision="double", device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        bits = torch.randint(0, 2, (1, 1, 1, int(resource_grid.num_data_symbols) * 6),
                             generator=generator, device="cpu")
        symbols = Mapper("qam", 6, precision="double", device="cpu")(bits)
        grid = ResourceGridMapper(resource_grid, precision="double", device="cpu")(symbols)
        modulator = OFDMModulator(cyclic_prefix_length=info["cyclicPrefixLengths"],
                                  precision="double", device="cpu")
        samples = modulator(grid).numpy().reshape(-1)
    return samples, grid.numpy().reshape(info["numSymbols"], size), info


def generate_ofdm(bandwidth, oversampling=None, seed=12345, *, return_reference=False):
    """Return peak-normalized complex samples, sample rate, symbol count."""
    samples, grid, info = _modulate(bandwidth, oversampling, seed)
    peak = np.max(abs(samples))
    if peak:
        samples = samples / peak
    if return_reference:
        # Unitary FFT convention matches Sionna; scale reference exactly like x.
        reference = dict(grid=grid / peak if peak else grid, **info)
        return samples, info["sampleRate"], reference
    return samples, info["sampleRate"], info["numSymbols"]
