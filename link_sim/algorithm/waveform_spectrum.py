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
