"""Two-sided complex-baseband amplitude spectra with coherent window correction."""

import numpy as np

from .pa_simulation_core import positive_rate, waveform


def amplitude_spectrum(x, sample_rate):
    """Return Hz and dB relative to unit complex-tone amplitude (not dB/Hz)."""
    x = waveform(x)
    fs = positive_rate(sample_rate)
    n = len(x)
    window = np.hanning(n+1)[:-1] if n > 2 else np.ones(n)
    amplitude = np.abs(np.fft.fft(x*window)) / window.sum()
    frequency = np.fft.fftshift(np.fft.fftfreq(n, d=1/fs))
    return frequency, 20*np.log10(np.maximum(np.fft.fftshift(amplitude), 1e-15))


def signal_statistics(x):
    """Normalized I/Q power; energy_sum is a discrete sum, without physical units."""
    power = abs(waveform(x))**2
    mean = float(np.mean(power))
    return dict(mean_power=mean, rms=float(np.sqrt(mean)),
                papr_db=float(10*np.log10(np.max(power)/mean)) if mean > 0 else 0.,
                energy_sum=float(np.sum(power)))
