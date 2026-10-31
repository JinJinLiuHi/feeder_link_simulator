"""Portable PA models and offline complex-envelope simulation (no UI dependencies).

GMP columns: aligned powers 0..D-1, then lagging envelope offsets 1..L,
then leading offsets -1..-L; each offset contains powers 1..D-1.
All rows are signal delays m=0..M-1. A term is x[n-m]|x[n-m-offset]|**p.
The leading-envelope form is offline; valid_slice excludes unavailable history
and look-ahead samples. Legacy ctMemPoly files retain their original meaning.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import re

import numpy as np
from scipy.linalg import lstsq
from scipy.signal import welch


def positive_rate(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("采样率必须是有限正数（Hz）")
    return value


def waveform(values):
    values = np.asarray(values)
    if values.ndim == 2 and 1 in values.shape:
        values = values.ravel()
    if values.ndim != 1 or values.size < 4:
        raise ValueError("波形须为至少 4 点的一维复数序列")
    values = np.asarray(values, dtype=np.complex128)
    if not np.all(np.isfinite(values)):
        raise ValueError("波形包含 NaN 或无穷大")
    return values


@dataclass
class PAModel:
    coefficients: np.ndarray
    model_type: str
    degree: int
    cross_depth: int = 0
    sample_rate: float | None = None
    fit_info: dict = field(default_factory=dict)
    role: str = "power_amplifier"

    def __post_init__(self):
        if self.role not in ("power_amplifier", "predistorter"):
            raise ValueError("模型用途必须为功放或预失真器")
        self.coefficients = np.array(self.coefficients, dtype=np.complex128, copy=True)
        if self.model_type not in ("memPoly", "gmp", "ctMemPoly"):
            raise ValueError(f"不支持的功放模型：{self.model_type}")
        if self.coefficients.ndim != 2 or min(self.coefficients.shape) < 1:
            raise ValueError("系数必须为非空二维复数矩阵")
        if not np.all(np.isfinite(self.coefficients)):
            raise ValueError("模型系数包含 NaN 或无穷大")
        if int(self.degree) != self.degree or self.degree < 1:
            raise ValueError("多项式阶数必须为正整数")
        if int(self.cross_depth) != self.cross_depth or self.cross_depth < 0:
            raise ValueError("交叉包络偏移必须为非负整数")
        self.degree, self.cross_depth = int(self.degree), int(self.cross_depth)
        if self.model_type != "gmp" and self.cross_depth:
            raise ValueError("只有 GMP 使用交叉包络偏移参数")
        columns = self.degree
        if self.model_type == "ctMemPoly":
            columns = 1 + self.memory * (self.degree - 1)
        elif self.model_type == "gmp":
            columns += 2 * self.cross_depth * (self.degree - 1)
        if self.coefficients.shape[1] != columns:
            raise ValueError("系数矩阵维度与模型阶数、记忆长度或交叉偏移不一致")
        if self.sample_rate is not None:
            self.sample_rate = positive_rate(self.sample_rate)

    @property
    def memory(self):
        return self.coefficients.shape[0]

    @property
    def name(self):
        return {"memPoly": "MP", "gmp": "GMP", "ctMemPoly": "Cross-Term Memory"}[self.model_type]

    @property
    def edge(self):
        return self.cross_depth if self.model_type == "gmp" and self.degree > 1 else 0

    def valid_slice(self, size):
        start, stop = self.memory - 1 + self.edge, size - self.edge
        if stop - start < 4:
            raise ValueError("波形过短：去除模型记忆和超前包络边界后须至少剩余 4 点")
        return slice(start, stop)

    def terms(self):
        """Yield (row, column, signal_delay, envelope_delay, envelope_power)."""
        for m in range(self.memory):
            if self.model_type == "ctMemPoly":
                yield m, 0, m, m, 0
                for p in range(1, self.degree):
                    for ell in range(self.memory):
                        yield m, 1 + (p - 1) * self.memory + ell, m, ell, p
            else:
                for p in range(self.degree):
                    yield m, p, m, m, p
                if self.model_type == "gmp":
                    offsets = list(range(1, self.cross_depth + 1)) + list(range(-1, -self.cross_depth - 1, -1))
                    for block, offset in enumerate(offsets):
                        for p in range(1, self.degree):
                            col = self.degree + block * (self.degree - 1) + p - 1
                            yield m, col, m, m + offset, p

    def predict(self, x):
        x = waveform(x)
        valid = self.valid_slice(len(x))
        y = np.zeros(len(x), dtype=np.complex128)
        active = [term for term in self.terms() if self.coefficients[term[0], term[1]] != 0]
        with np.errstate(over="raise", invalid="raise"):
            amplitude = np.abs(x)
            envelopes = {p: amplitude ** p for p in {term[4] for term in active} if p}
            for m, col, delay, env_delay, power in active:
                c = self.coefficients[m, col]
                term = c * x[valid.start - delay:valid.stop - delay]
                if power:
                    term *= envelopes[power][valid.start - env_delay:valid.stop - env_delay]
                y[valid] += term
        return y


def fit_model(x, y, model_type, memory, degree, cross_depth=0, sample_rate=None, *, rcond=1e-4):
    """Column-normalized truncated SVD; reject unidentifiable coefficient directions.

    rcond is relative to the largest singular value after column normalization.
    It limits noise amplification for correlated oversampled MP/GMP regressors;
    it does not filter the signal or constrain an imported model's prediction.
    """
    if not np.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError("奇异值截断阈值须在 0 与 1 之间")
    x, y = waveform(x), waveform(y)
    if len(x) != len(y):
        raise ValueError("输入和输出波形长度必须相同")
    if int(memory) != memory or memory < 1:
        raise ValueError("记忆长度必须为正整数")
    if model_type != "gmp":
        cross_depth = 0
    cols = degree + 2 * cross_depth * (degree - 1)
    if model_type == "ctMemPoly":
        cols = 1 + memory * (degree - 1)
    model = PAModel(np.zeros((memory, cols)), model_type, degree, cross_depth, sample_rate)
    valid = model.valid_slice(len(x))
    n = np.arange(valid.start, valid.stop)
    count = model.coefficients.size
    if len(n) < count:
        raise ValueError("拟合样本少于系数数量，请增加样本或降低阶数/记忆长度")
    # Bound the design matrix before allocating it in the GUI process.
    if len(n) * count * 16 > 512 * 1024**2:
        raise ValueError("拟合矩阵超过 512 MiB，请减少数据量、阶数或记忆长度")
    terms = list(model.terms())
    with np.errstate(over="raise", invalid="raise"):
        design = np.column_stack([x[n - m] * np.abs(x[n - ell]) ** p
                                  for _, _, m, ell, p in terms])
    scale = np.linalg.norm(design, axis=0)
    scale[scale == 0] = 1.0
    design /= scale
    scaled_coef, _, rank, singular = lstsq(design, y[valid], cond=rcond, lapack_driver="gelsd")
    if not rank:
        raise ValueError("实测输入能量为零，无法辨识功放")
    coef = scaled_coef / scale
    denominator = np.linalg.norm(y[valid])
    if denominator == 0:
        raise ValueError("实测输出能量为零，无法辨识功放")
    model.fit_info = dict(
        solver="column_scaled_tsvd", rcond=float(rcond), rank=int(rank), terms=count,
        trainingRmsPercent=float(100 * np.linalg.norm(design @ scaled_coef - y[valid]) / denominator),
        inputRms=float(np.sqrt(np.mean(abs(x[valid]) ** 2))), inputPeak=float(np.max(abs(x[valid]))),
        conditionNumber=float(singular[0] / singular[-1]) if singular[-1] > singular[0] * np.finfo(float).tiny else None,
    )
    for value, (m, col, *_rest) in zip(coef, terms):
        model.coefficients[m, col] = value
    return model


def save_model(path, model):
    """Write self-describing NPZ or commented CSV/TXT; no object arrays/pickle."""
    path = Path(path)
    metadata = dict(modType=model.model_type, memLen=model.memory, degLen=model.degree,
                    crossDepth=model.cross_depth, formatVersion=1, modelRole=model.role)
    if model.sample_rate is not None:
        metadata["sampleRate"] = model.sample_rate
    if model.fit_info:
        metadata["fitInfo"] = json.dumps(model.fit_info, ensure_ascii=False, allow_nan=False)
    if path.suffix.lower() == ".npz":
        np.savez(path, fitCoefMatMem=model.coefficients, **metadata)
    elif path.suffix.lower() in (".csv", ".txt"):
        with path.open("w", encoding="utf-8") as stream:
            stream.write("# 功放模型：复系数及参数\n")
            for key, value in metadata.items():
                stream.write(f"# {key}, {value}\n")
            for row in model.coefficients:
                stream.write(",".join(f"{v.real:.17g}{v.imag:+.17g}j" for v in row) + "\n")
    else:
        raise ValueError("模型文件须为 .npz、.csv 或 .txt")


def load_model(path, *, expected_role="power_amplifier"):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return _load_unified_model(path, expected_role)
    metadata = {}
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "modelManifest" in data:
                return _load_unified_model(path, expected_role)
            if "fitCoefMatMem" not in data:
                raise ValueError("缺少 fitCoefMatMem 系数矩阵")
            coefficients = np.array(data["fitCoefMatMem"])
            for key in ("modType", "memLen", "degLen", "crossDepth", "sampleRate", "formatVersion", "fitInfo", "modelRole"):
                if key in data:
                    metadata[key] = data[key].item()
    elif path.suffix.lower() in (".csv", ".txt"):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("#") and "," in line:
                key, value = line[1:].split(",", 1)
                metadata[key.strip()] = value.strip()
        if "modType" not in metadata:
            match = re.search(r"\((memPoly|ctMemPoly)\)", metadata.get("模型类型", ""))
            if match:
                metadata["modType"] = match[1]
        coefficients = np.loadtxt(path, delimiter=",", dtype=complex, ndmin=2, encoding="utf-8-sig")
    else:
        raise ValueError("模型文件须为 .npz、.csv 或 .txt")
    if int(metadata.get("formatVersion", 1)) != 1:
        raise ValueError("不支持该模型文件版本")
    if any(key not in metadata for key in ("modType", "memLen", "degLen")):
        raise ValueError("模型缺少类型、记忆长度或阶数，请导入本程序导出的模型文件")
    if int(metadata["memLen"]) != coefficients.shape[0]:
        raise ValueError("模型记忆长度与系数矩阵不一致")
    if metadata["modType"] == "gmp" and "crossDepth" not in metadata:
        raise ValueError("GMP 文件缺少交叉包络偏移 crossDepth")
    fit_info = json.loads(metadata.get("fitInfo", "{}"))
    if not isinstance(fit_info, dict):
        raise ValueError("模型辨识信息格式不正确")
    role = metadata.get("modelRole", "power_amplifier")
    if expected_role is not None and role != expected_role:
        raise ValueError("模型用途不匹配：请将功放与预失真器文件分别导入对应位置")
    return PAModel(coefficients, str(metadata["modType"]), int(metadata["degLen"]),
                   int(metadata.get("crossDepth", 0)), metadata.get("sampleRate"), fit_info, role)


def _load_unified_model(path, expected_role):
    from .device_models import read_model_file, to_pa
    model = to_pa(read_model_file(path))
    if expected_role is not None and model.role != expected_role:
        raise ValueError("模型用途不匹配：请分别导入功放或 DPD 模型")
    return model


def load_waveform(path):
    """NPY: complex vector; NPZ/MAT: x (+ sampleRate); CSV/TXT: I,Q columns."""
    path = Path(path)
    rate = None
    suffix = path.suffix.lower()
    if suffix == ".npy":
        x = np.load(path, allow_pickle=False)
    elif suffix in (".npz", ".mat", ""):
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if "x" not in data:
                    raise ValueError("波形文件缺少 x 数组")
                x = np.array(data["x"])
                if "sampleRate" in data:
                    rate = data["sampleRate"].item()
        else:
            from scipy.io import loadmat
            data = loadmat(path, appendmat=False)
            if "x" in data:
                x = data["x"]
            elif "results" in data and "InputWaveform" in (data["results"].dtype.names or ()):
                x = data["results"][0, 0]["InputWaveform"]
            else:
                raise ValueError("MAT 波形文件须包含 x 或 results.InputWaveform，可选 sampleRate（Hz）")
            if "sampleRate" in data:
                rate = data["sampleRate"].item()
    elif suffix in (".csv", ".txt"):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        rows = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
        if rows and rows[0].replace(" ", "").lower() == "i,q":
            rows = rows[1:]
        iq = np.loadtxt(rows, delimiter=",", ndmin=2)
        if iq.shape[1] != 2:
            raise ValueError("CSV/TXT 波形须为 I,Q 两列，可带 I,Q 表头")
        x = iq[:, 0] + 1j * iq[:, 1]
    else:
        raise ValueError("不支持的波形格式")
    return waveform(x), positive_rate(rate) if rate is not None else None


def simulate(model, x, sample_rate):
    rate = positive_rate(sample_rate)
    if model.sample_rate is not None and not np.isclose(rate, model.sample_rate, rtol=1e-9, atol=1e-6):
        raise ValueError(f"信号采样率与模型采样率 {model.sample_rate / 1e6:g} MHz 不一致；请提供匹配的输入波形")
    x = waveform(x)
    y = model.predict(x)
    valid = model.valid_slice(len(x))
    return x, y, valid


def spectrum_pair(x, y, sample_rate, valid, nfft=2048, reference_load=None):
    """Matched two-sided Welch PSD; no noise, peak scaling or gain removal.

    Default is dB relative to one amplitude-unit squared per Hz. Supplying R
    explicitly treats samples as RMS-equivalent complex voltages: P=|x|²/R.
    """
    rate = positive_rate(sample_rate)
    x, y = waveform(x)[valid], waveform(y)[valid]
    if len(x) != len(y) or len(x) < 4:
        raise ValueError("用于频谱比较的有效输入输出长度必须相同且至少为 4")
    if int(nfft) != nfft or nfft < 4:
        raise ValueError("FFT 长度至少为 4")
    if reference_load is not None and (not np.isfinite(reference_load) or reference_load <= 0):
        raise ValueError("参考阻抗必须为有限正数")
    segment = min(int(nfft), len(x))
    spectra = []
    for signal in (x, y):
        freq, density = welch(signal, fs=rate, window="hann", nperseg=segment,
                              noverlap=segment // 2, nfft=segment, detrend=False,
                              return_onesided=False, scaling="density")
        if reference_load is not None:
            density = density / reference_load / 1e-3
        spectra.append(10 * np.log10(np.maximum(np.fft.fftshift(density), 1e-30)))
    return np.fft.fftshift(freq), spectra[0], spectra[1]


def power_spectrum(x, sample_rate, nfft=1024, ref_load=100.0, noise_floor_dbm=None):
    """Characterization-compatible power per FFT bin, in dBm.

    This preserves the existing Hann-window/50%-overlap and total-power
    normalization convention. Optional measurement noise affects plotting only;
    x is never modified. Both PA pages call this same estimator.
    """
    x = waveform(x)
    rate = positive_rate(sample_rate)
    if not np.isfinite(ref_load) or ref_load <= 0:
        raise ValueError("参考阻抗必须为有限正数")
    nfft = min(int(nfft), len(x))
    if nfft < 4:
        raise ValueError("FFT 长度至少为 4")
    if noise_floor_dbm is not None:
        if not np.isfinite(noise_floor_dbm):
            raise ValueError("测量噪声底必须为有限数值")
        sigma = np.sqrt(10**(noise_floor_dbm / 10) * 1e-3 * nfft * ref_load)
        rng = np.random.RandomState(12345)
        x = x + (rng.randn(len(x)) + 1j * rng.randn(len(x))) * sigma / np.sqrt(2)
    frequency, power = welch(x, fs=rate, window=np.hanning(nfft), nperseg=nfft,
                             noverlap=nfft // 2, detrend=False,
                             return_onesided=False, scaling="density")
    total = np.sum(power)
    bins = power / total * np.mean(abs(x)**2) if total > 0 else np.zeros_like(power)
    return np.fft.fftshift(frequency) / 1e6, 10 * np.log10(np.fft.fftshift(bins) / ref_load / 1e-3 + 1e-30)
