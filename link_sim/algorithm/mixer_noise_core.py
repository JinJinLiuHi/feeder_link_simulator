"""Generate real LO phase noise from single-sideband dBc/Hz anchors."""

import numpy as np


def phase_noise_psd(anchors, fs, n_samples):
    """Return the nonnegative frequency grid and its two-sided phase PSD."""
    offsets = np.fft.rfftfreq(n_samples, 1.0 / fs)
    anchor_offsets, anchor_levels = np.asarray(anchors, dtype=float).T
    levels = np.empty_like(offsets)
    # Interpolate level in dB against log frequency, holding the end levels outside.
    levels[1:] = np.interp(np.log10(offsets[1:]), np.log10(anchor_offsets), anchor_levels)
    levels[0] = levels[1] if len(levels) > 1 else anchor_levels[0]
    return offsets, levels, 10 ** (levels / 10.0)


def generate_phase_noise(fs, n_samples, anchors, seed=2025):
    """Shape unit-variance real white noise, then return zero-mean phase in radians."""
    offsets, levels, psd = phase_noise_psd(anchors, fs, n_samples)
    white = np.random.default_rng(seed).standard_normal(n_samples)
    # White noise has two-sided PSD 1/fs; negative frequencies mirror the same weights.
    weights = np.sqrt(psd * fs)
    mirror = weights[-2:0:-1] if n_samples % 2 == 0 else weights[-1:0:-1]
    phi = np.fft.ifft(np.fft.fft(white) * np.concatenate((weights, mirror))).real
    phi -= phi.mean()
    return phi, offsets, levels, psd
