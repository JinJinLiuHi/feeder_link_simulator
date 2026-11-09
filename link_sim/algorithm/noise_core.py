"""Complex Gaussian noise at an absolute spectral density."""
import numpy as np

def generate_noise(count, fs, density_dbfs_hz, seed=2025):
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(density_dbfs_hz) or not -300 <= density_dbfs_hz <= 0:
        raise ValueError("采样率须为正，底噪密度须为 -300 至 0 dBFS/Hz")
    rng = np.random.default_rng(seed)
    power = fs * 10**(density_dbfs_hz/10)
    return np.sqrt(power/2)*(rng.standard_normal(count)+1j*rng.standard_normal(count))

