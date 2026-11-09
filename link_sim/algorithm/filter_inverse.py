"""Causal, gain-limited inverse FIR for the paired converter low-pass filter."""

import hashlib
import json
import numpy as np
from scipy import signal


def validate_lowpass(model):
    model = model.validated()
    if model.kind != "filter" or model.parameters.get("band") != "低通":
        raise ValueError("请选择滤波器模型库中的低通模型")
    return model


def filter_fingerprint(model):
    model = validate_lowpass(model)
    digest = hashlib.sha256(json.dumps(model.parameters, sort_keys=True).encode())
    for key, values in sorted(model.arrays.items()):
        digest.update(key.encode())
        digest.update(np.asarray(values, dtype="<c16").tobytes())
    return digest.hexdigest()


def inverse_taps(model):
    model = validate_lowpass(model)
    arrays = model.arrays
    length = len(arrays.get("h", arrays.get("b", [])))
    # IIR h is a preview; the actual frequency response uses the recurrence.
    if model.parameters["impl"] == "IIR":
        length = max(len(arrays["b"]), len(arrays["a"]))
    if length > 8192:
        raise ValueError("预补偿当前支持最多 8192 抽头的低通模型，请降低低通阶数")
    taps = max(129, 2*length+1)
    size = 1 << (max(4096, 8*taps)-1).bit_length()
    if "sos" in arrays:
        _, response = signal.sosfreqz(arrays["sos"], worN=size, whole=True)
    elif "b" in arrays:
        _, response = signal.freqz(arrays["b"], arrays.get("a", [1.]), worN=size, whole=True)
    else:
        response = np.fft.fft(arrays["h"], size)
    if not np.all(np.isfinite(response)) or np.max(abs(response)) == 0:
        raise ValueError("低通模型的频响无效或全为零，无法预补偿")
    inverse = response.conj()/(abs(response)**2 + 1e-6*np.max(abs(response)**2))
    inverse *= np.minimum(1., 10./np.maximum(abs(inverse), 1e-30))
    impulse = np.fft.fftshift(np.fft.ifft(inverse))
    delay = taps//2
    h = impulse[size//2-delay:size//2+delay+1]*np.kaiser(taps, 8.6)
    if not np.any(abs(h) > 1e-15):
        raise ValueError("无法得到有效预补偿抽头，请检查低通模型")
    peak = np.max(abs(np.fft.fft(h, size)))
    if peak > 10.:
        h *= 10./peak
    return np.real_if_close(h), delay
