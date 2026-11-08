"""Offline PA/DPD evaluation on the current link's time grid.

The native coefficients and measured sample interval are retained. Fractional
delays extend the discrete model using bandlimited interpolation of the input;
this is an approximation, not new wideband characterization of the hardware.
"""

import numpy as np

from .pa_simulation_core import positive_rate, simulate, waveform


INTERPOLATION_RADIUS = 32


def simulate_at_rate(model, x, sample_rate):
    rate = positive_rate(sample_rate)
    if model.sample_rate is None or np.isclose(rate, model.sample_rate, rtol=1e-9, atol=1e-6):
        return simulate(model, x, rate)
    x = waveform(x)
    ratio = rate/model.sample_rate
    terms = list(model.terms())
    delays = {d for _, _, m, ell, p in terms for d in ((m, ell) if p else (m,))}
    offsets = {}
    radius = INTERPOLATION_RADIUS
    for delay in delays:
        samples = delay*ratio
        nearest = round(samples)
        if np.isclose(samples, nearest, rtol=0., atol=1e-10):
            offsets[delay] = (int(nearest), None)
        else:
            integer = int(np.floor(samples))
            fraction = samples-integer
            grid = np.arange(-radius, radius+1)
            weights = np.sinc(grid+fraction)*np.kaiser(2*radius+1, 8.6)
            offsets[delay] = (integer, weights/weights.sum())
    start = max(0, max(d+(radius if weights is not None else 0) for d, weights in offsets.values()))
    stop = min(len(x), min(len(x)+d-(radius if weights is not None else 0) for d, weights in offsets.values()))
    if stop-start < 4:
        raise ValueError("波形过短，跨采样率记忆延迟和插值边界后不足 4 个有效点")
    envelope_keys = {(ell, p) for row, col, _, ell, p in terms if p and model.coefficients[row, col] != 0}
    if (stop-start)*(16*len(delays)+8*len(envelope_keys)) > 512*1024**2:
        raise ValueError("跨采样率插值临时数据超过 512 MiB，请缩短波形或减少模型记忆")
    valid = slice(start, stop)
    delayed = {}
    for delay, (integer, weights) in offsets.items():
        if weights is None:
            delayed[delay] = x[start-integer:stop-integer]
        else:
            values = np.zeros(stop-start, dtype=complex)
            for offset, weight in zip(range(-radius, radius+1), weights):
                values += weight*x[start-integer+offset:stop-integer+offset]
            delayed[delay] = values
    y = np.zeros(len(x), dtype=complex)
    envelopes = {}
    with np.errstate(over="raise", invalid="raise"):
        for row, col, delay, envelope_delay, power in terms:
            coefficient = model.coefficients[row, col]
            if coefficient == 0:
                continue
            term = coefficient*delayed[delay]
            if power:
                key = envelope_delay, power
                if key not in envelopes:
                    envelopes[key] = np.abs(delayed[envelope_delay])**power
                term *= envelopes[key]
            y[valid] += term
    return x, y, valid
