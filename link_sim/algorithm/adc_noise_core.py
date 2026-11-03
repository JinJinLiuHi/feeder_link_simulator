"""Use the ORIGINAL ADC model's estimated FFT noise floor for link noise.

The original editor/function/schema are unchanged. Its noise_floor_dbfs metric
is the mean per-positive-FFT-bin amplitude-squared level, excluding DC, the test
tone and configured spur bins. In the complex-envelope convention (full-scale
tone magnitude 1), density = 10**(floor/10)/(model_fs/model_nfft).

For complex I/Q at Fs, the total sample band is [-Fs/2, Fs/2], width B=Fs.
Pn = density * B and each real Gaussian branch has variance Pn/2. No spur
tones, calibration-tone residuals or extra quantization are added to the link.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ADCNoiseEstimate:
    floor_dbfs: float
    bin_width_hz: float

    @property
    def density(self):
        return 10**(self.floor_dbfs/10)/self.bin_width_hz

    @property
    def density_dbfs_hz(self):
        return self.floor_dbfs-10*np.log10(self.bin_width_hz)

    def noise_power(self, bandwidth_hz):
        if not np.isfinite(bandwidth_hz) or bandwidth_hz <= 0:
            raise ValueError("链路采样带宽须为有限正数")
        return float(self.density*bandwidth_hz)


def calibrate_model_noise(model):
    """Read exactly the same estimated noise-floor metric shown by the editor."""
    model = model.validated()
    if model.kind != "adda":
        raise ValueError("请选择原 ADC/DAC 器件模型")
    # Match the desktop entry point's Qt-before-Matplotlib DLL loading order.
    from PySide6 import QtCore  # noqa: F401
    from .ad_model import run_adc_model
    keys = ("n_bits", "fs", "nfft", "f_sig", "spur_freqs", "spur_dbfs", "noise_density_dbfs_hz")
    result = run_adc_model(**{key: model.parameters[key] for key in keys})
    floor = float(result["metrics"]["noise_floor_dbfs"])
    if not np.isfinite(floor):
        raise ValueError("原 ADC 模型没有可用的估计噪声基底，请检查原模型参数")
    return ADCNoiseEstimate(floor, float(result["bin_width_hz"]))


def generate_model_noise(estimate, count, waveform_fs, seed=2025):
    """Gaussian equivalent noise from the model's measured floor, not its spurs."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 4:
        raise ValueError("等效噪声序列至少需要 4 点")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 2**32-1:
        raise ValueError("内部噪声种子无效")
    power = estimate.noise_power(waveform_fs)
    sigma = np.sqrt(power/2)
    rng = np.random.default_rng(seed)
    noise = sigma*(rng.standard_normal(count)+1j*rng.standard_normal(count))
    return noise, power
