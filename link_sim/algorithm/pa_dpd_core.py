"""Offline indirect-learning DPD, matched evaluation and model interchange."""

from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.signal import welch, correlate, correlation_lags
from .pa_simulation_core import PAModel, fit_model, waveform, positive_rate, load_model, save_model


def signature(model):
    digest = hashlib.sha256(model.coefficients.astype('<c16').tobytes())
    digest.update(str((model.model_type, model.degree, model.cross_depth, model.sample_rate)).encode())
    return digest.hexdigest()


def check_rate(pa, rate):
    rate = positive_rate(rate)
    if pa.role != "power_amplifier":
        raise ValueError("请选择功放正向模型")
    if pa.sample_rate is not None and not np.isclose(pa.sample_rate, rate, rtol=1e-9):
        raise ValueError("输入信号与功放模型采样率不一致")
    return rate


def check_gain(gain):
    gain = complex(gain)
    if not np.isfinite(gain) or abs(gain) < 1e-12:
        raise ValueError("目标线性增益必须为非零有限值")
    return gain


def cascade(pa, dpd, x):
    """Both model guards accumulate; zero padding never enters scored outputs."""
    x = waveform(x)
    first = dpd.valid_slice(len(x))
    second = pa.valid_slice(len(x))
    valid = slice(first.start + second.start, first.stop - (len(x) - second.stop))
    if valid.stop - valid.start < 4:
        raise ValueError("信号太短，无法去除预失真器和功放的级联边界")
    u = dpd.predict(x)
    return u, pa.predict(u), valid


def relative_error(reference, actual):
    denominator = np.linalg.norm(reference)
    if denominator <= 0:
        raise ValueError("参考信号能量必须大于零")
    return float(np.linalg.norm(actual - reference) / denominator)


def align_pair(u, y, max_delay=1024):
    """Align integer acquisition delay, retaining physical gain and phase."""
    u, y = waveform(u), waveform(y)
    if len(u) != len(y):
        raise ValueError("实测输入输出长度必须一致")
    if not np.any(abs(u)) or not np.any(abs(y)):
        raise ValueError("实测输入输出能量必须大于零")
    correlation = correlate(y, u, mode="full", method="fft")
    lags = correlation_lags(len(y), len(u))
    selected = abs(lags) <= min(max_delay, len(u) - 4)
    delay = int(lags[selected][np.argmax(abs(correlation[selected]))])
    if delay >= 0:
        return u[:len(u)-delay], y[delay:], delay
    return u[-delay:], y[:len(y)+delay], delay


def partition_capture(x, reference=None):
    """Disjoint 50/25/25 blocks; preserve whole OFDM symbols and their references."""
    x = waveform(x)
    if reference is None:
        cuts = [0, len(x)//2, 3*len(x)//4, len(x)]
        return [(waveform(x[a:b]), None) for a, b in zip(cuts, cuts[1:])]
    cp = np.asarray(reference["cyclicPrefixLengths"])
    symbols = len(cp)
    cuts = [0, symbols//2, 3*symbols//4, symbols]
    offsets = np.r_[0, np.cumsum(cp + int(reference["fftSize"]))]
    if offsets[-1] != len(x) or len(reference["grid"]) != symbols or symbols < 8:
        raise ValueError("OFDM 波形与参考符号长度不一致或符号数不足")
    blocks = []
    for a, b in zip(cuts, cuts[1:]):
        ref = dict(reference, grid=reference["grid"][a:b], cyclicPrefixLengths=cp[a:b], numSymbols=b-a)
        blocks.append((x[offsets[a]:offsets[b]], ref))
    return blocks


@dataclass
class DPDResult:
    model: PAModel
    gain: complex
    history: list
    baseline_error: float
    validation_error: float


def train_ila(pa, train_x, validation_x, sample_rate, gain, *, model_type="memPoly",
              memory=3, degree=5, cross_depth=1, iterations=3, measured_pair=None,
              peak_limit=None, progress=None):
    """Train on one capture, select on another; no final-test samples enter here."""
    rate, gain = check_rate(pa, sample_rate), check_gain(gain)
    train_x, validation_x = waveform(train_x), waveform(validation_x)
    if not 1 <= int(iterations) <= 10 or int(iterations) != iterations:
        raise ValueError("ILA 迭代次数须为 1–10")
    if peak_limit is not None and (not np.isfinite(peak_limit) or peak_limit <= 0):
        raise ValueError("功放输入峰值上限必须为正数")
    if peak_limit is not None and max(abs(train_x).max(), abs(validation_x).max()) > peak_limit * (1 + 1e-9):
        raise ValueError("训练或验证输入超过功放辨识幅度范围，请增加输入回退或修改峰值上限")
    best = PAModel([[1]], "memPoly", 1, sample_rate=rate, role="predistorter")
    # Score every candidate and the baseline over a fixed common interval.
    edge = cross_depth if model_type == "gmp" and degree > 1 else 0
    pa_guard = pa.valid_slice(len(validation_x))
    valid = slice(pa_guard.start + memory - 1 + edge, pa_guard.stop - edge)
    if valid.stop - valid.start < 4:
        raise ValueError("验证信号长度不足以覆盖级联记忆")
    baseline = pa.predict(validation_x)
    baseline_error = relative_error(gain * validation_x[valid], baseline[valid])
    best_error, history, current = baseline_error, [], best
    captured = None
    if measured_pair is not None:
        captured = align_pair(*measured_pair)
        if peak_limit is not None and abs(captured[0]).max() > peak_limit * (1 + 1e-9):
            raise ValueError("实测训练输入超过设定的功放输入峰值上限")
    for iteration in range(1, (1 if captured else int(iterations)) + 1):
        if progress:
            progress(f"ILA 第 {iteration} 轮：辨识后置逆模型…")
        if captured:
            u, y, delay = captured
        else:
            u, y, guard = cascade(pa, current, train_x)
            u, y, delay = u[guard], y[guard], 0
        candidate = fit_model(y / gain, u, model_type, memory, degree, cross_depth, rate)
        candidate.role = "predistorter"
        try:
            candidate_u, output, guard = cascade(pa, candidate, validation_x)
            peak = float(abs(candidate_u[candidate.valid_slice(len(validation_x))]).max())
            train_drive = candidate.predict(train_x)
            peak = max(peak, float(abs(train_drive[candidate.valid_slice(len(train_x))]).max()))
            error = relative_error(gain * validation_x[valid], output[valid])
            in_range = peak_limit is None or peak <= peak_limit * (1 + 1e-9)
            accepted = in_range and error <= best_error * (1 + 1e-8) + 1e-14
            history.append(dict(iteration=iteration, rmsPercent=error * 100,
                                peak=peak, accepted=bool(accepted), inRange=bool(in_range)))
            if accepted:
                best, best_error, current = candidate, error, candidate
        except (FloatingPointError, ValueError) as exc:
            history.append(dict(iteration=iteration, accepted=False, reason=str(exc)))
            break
    best.fit_info.update(
        algorithm="ILA", gainReal=gain.real, gainImag=gain.imag, paSignature=signature(pa),
        history=history, validationRmsPercent=100 * best_error,
        baselineValidationRmsPercent=100 * baseline_error, peakLimit=peak_limit,
        trainingSamples=len(captured[0]) if captured else len(train_x),
        validationSamples=len(validation_x), acquisitionDelay=captured[2] if captured else 0,
        trainingSource="measured" if captured else "model",
        improved=bool(best_error < baseline_error * (1 - 1e-6)),
    )
    return DPDResult(best, gain, history, baseline_error, best_error)


def channel_acpr(signal, sample_rate, main_bandwidth, adjacent_bandwidth,
                 adjacent_offset, nfft=4096):
    fs = positive_rate(sample_rate)
    b, a, offset = float(main_bandwidth), float(adjacent_bandwidth), float(adjacent_offset)
    if not all(np.isfinite(v) and v > 0 for v in (b, a, offset)):
        raise ValueError("主信道/邻道带宽及偏移必须为正数")
    if offset < (b + a) / 2 or offset + a / 2 > fs / 2 or b > fs:
        raise ValueError("主邻道不得重叠，且上下邻道须完全位于采样带宽内")
    signal = waveform(signal)
    nfft = min(int(nfft), len(signal))
    f, density = welch(signal, fs=fs, nperseg=nfft, noverlap=nfft//2,
                       detrend=False, return_onesided=False, scaling="density")
    def power(center, width):
        mask = (f >= center - width/2) & (f < center + width/2)
        if mask.sum() < 2:
            raise ValueError("信道内频谱点数不足，请增加 FFT 长度或信号长度")
        return float(density[mask].sum() * fs/nfft)
    main = power(0, b)
    if main <= 0:
        raise ValueError("主信道功率为零，无法计算 ACPR")
    return tuple(float(10*np.log10(max(power(c, a)/main, 1e-30))) for c in (-offset, offset))


def ofdm_evm(signal, reference, gain, valid):
    """Fixed-gain EVM on known active QAM carriers, complete valid symbols only."""
    gain = check_gain(gain)
    grid = np.asarray(reference["grid"])
    size = int(reference["fftSize"])
    if grid.ndim != 2 or grid.shape[1] != size:
        raise ValueError("OFDM 参考栅格维度不正确")
    position, errors, powers, count = 0, 0.0, 0.0, 0
    for row, cp in zip(grid, reference["cyclicPrefixLengths"]):
        start, stop = position + int(cp), position + int(cp) + size
        if start >= valid.start and stop <= valid.stop:
            actual = np.fft.fftshift(np.fft.fft(signal[start:stop] / gain)) / np.sqrt(size)
            mask = abs(row) > 0
            errors += float(np.sum(abs(actual[mask] - row[mask])**2))
            powers += float(np.sum(abs(row[mask])**2))
            count += 1
        position = stop
    if not count or powers == 0:
        raise ValueError("有效区间内没有完整 OFDM 符号，无法计算 EVM")
    return float(100*np.sqrt(errors/powers)), count


def evaluate_dpd(pa, dpd, x, sample_rate, gain, *, main_bandwidth,
                 adjacent_bandwidth, adjacent_offset, nfft=4096, reference_load=100,
                 reference=None, peak_limit=None):
    rate, gain = check_rate(pa, sample_rate), check_gain(gain)
    if dpd.role != "predistorter":
        raise ValueError("请选择预失真器模型")
    if dpd.sample_rate is not None and not np.isclose(dpd.sample_rate, rate, rtol=1e-9):
        raise ValueError("DPD 模型与输入信号采样率不一致")
    if not np.isfinite(reference_load) or reference_load <= 0:
        raise ValueError("参考阻抗必须为正数")
    recorded = dpd.fit_info.get("paSignature")
    if recorded is not None and recorded != signature(pa):
        raise ValueError("DPD 对应的功放模型与当前选择不一致，请重新训练")
    x = waveform(x)
    u, y1, valid = cascade(pa, dpd, x)
    y0, target = pa.predict(x), gain*x
    metrics = {}
    for name, output in (("before", y0), ("after", y1)):
        ref, actual = target[valid], output[valid]
        error = relative_error(ref, actual)
        lower, upper = channel_acpr(actual, rate, main_bandwidth, adjacent_bandwidth, adjacent_offset, nfft)
        mask = abs(ref) > abs(ref).max()*1e-4
        item = dict(rmsPercent=error*100, nmseDb=float(20*np.log10(max(error, 1e-15))),
                    lowerAcprDb=lower, upperAcprDb=upper,
                    powerDbm=float(10*np.log10(max(np.mean(abs(actual)**2)/reference_load/1e-3, 1e-30))),
                    amRmsPercent=float(100*np.linalg.norm(abs(actual)-abs(ref))/np.linalg.norm(ref)),
                    pmRmsDeg=float(np.sqrt(np.mean(np.angle(actual[mask]*ref[mask].conj())**2))*180/np.pi),
                    evmPercent=None)
        if reference is not None:
            item["evmPercent"], item["evmSymbols"] = ofdm_evm(output, reference, gain, valid)
        metrics[name] = item
    dpd_valid = dpd.valid_slice(len(x))
    power = np.mean(abs(u[dpd_valid])**2)
    peak = float(abs(u[dpd_valid]).max())
    limit = peak_limit if peak_limit is not None else dpd.fit_info.get("peakLimit")
    metrics["drive"] = dict(rms=float(np.sqrt(power)), peak=peak,
        paprDb=float(10*np.log10(max(peak**2/max(power, 1e-30), 1e-30))),
        peakLimit=limit, outOfRangePercent=float(100*np.mean(abs(u[dpd_valid]) > limit)) if limit else None)
    metrics["lowerAcprImprovementDb"] = metrics["before"]["lowerAcprDb"] - metrics["after"]["lowerAcprDb"]
    metrics["upperAcprImprovementDb"] = metrics["before"]["upperAcprDb"] - metrics["after"]["upperAcprDb"]
    return dict(x=x, u=u, y0=y0, y1=y1, target=target, validStart=valid.start,
                validStop=valid.stop, sampleRate=rate, gainReal=gain.real, gainImag=gain.imag,
                mainBandwidth=main_bandwidth, adjacentBandwidth=adjacent_bandwidth,
                adjacentOffset=adjacent_offset, referenceLoad=reference_load, nfft=nfft, metrics=metrics)


def save_dpd(path, model):
    if model.role != "predistorter":
        raise ValueError("只可导出已训练的预失真器")
    save_model(path, model)


def load_dpd(path):
    model = load_model(path, expected_role="predistorter")
    if not all(key in model.fit_info for key in ("gainReal", "gainImag", "paSignature")):
        raise ValueError("DPD 文件缺少目标增益或对应功放信息")
    check_gain(complex(model.fit_info["gainReal"], model.fit_info["gainImag"]))
    return model
