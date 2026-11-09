"""Import measured impulse responses and oscillator phase-noise tables."""

from pathlib import Path
import numpy as np
from scipy.io import loadmat
from .device_models import DeviceModel, FILTER_DEFAULTS


def _table(path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    lines = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError("实测数据文件为空")
    delimiter = "," if "," in lines[0] else None
    # Permit one descriptive header; malformed numeric rows still fail.
    tokens = lines[0].split(delimiter)
    try:
        [complex(token) for token in tokens]
    except ValueError:
        try:
            complex(tokens[0])
        except ValueError:
            pass
        else:
            raise ValueError("实测数据首行包含无效数值")
        lines = lines[1:]
    data = np.loadtxt(lines, delimiter=delimiter, ndmin=2)
    if not data.size or not np.all(np.isfinite(data)):
        raise ValueError("实测数据必须为非空有限数值")
    return data


def load_measured_filter(path, fs):
    """MAT/NPZ: h or b/a/sos and optional fs; CSV/TXT: h or f, dB, degrees."""
    path = Path(path)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("采样率必须为有限正数（Hz）")
    if path.suffix.lower() in (".mat", ".npz"):
        if path.suffix.lower() == ".mat":
            data = loadmat(path)
        else:
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
        fs = float(np.asarray(data.get("fs", fs)).item())
        arrays = {key: np.atleast_1d(np.asarray(data[key]).squeeze()) for key in ("h", "b", "a") if key in data}
        if "sos" in data:
            from scipy.signal import sos2tf
            arrays["sos"] = np.atleast_2d(data["sos"])
            arrays["b"], arrays["a"] = sos2tf(arrays["sos"])
        elif "b" in arrays and "a" not in arrays:
            arrays["a"] = np.array([1.])
    else:
        data = _table(path)
        if data.shape[1] == 1:
            arrays = {"h": data[:, 0]}
        elif data.shape[1] == 3:
            f, db, phase = data.T
            if len(f) < 2 or np.any(np.diff(f) <= 0) or f[0] < 0 or f[-1] > fs/2:
                raise ValueError("频率须递增、无重复，范围为 0 至 fs/2（Hz）")
            response = 10**(db/20) * np.exp(1j*np.deg2rad(phase))
            grid = np.fft.rfftfreq(4096, 1/fs)
            spectrum = (np.interp(grid, f, response.real, left=0, right=0)
                        + 1j*np.interp(grid, f, response.imag, left=0, right=0))
            arrays = {"h": np.fft.irfft(spectrum, n=4096)}
        else:
            raise ValueError("滤波器表格需要单列 h，或三列：频率 Hz、增益 dB、相位 deg")
    return DeviceModel("filter", dict(FILTER_DEFAULTS, fs=fs), arrays, path.stem).validated()


def load_measured_mixer(path, parameters):
    """CSV/TXT: offset Hz, dBc/Hz; MAT: offsets and levels arrays."""
    path = Path(path)
    if path.suffix.lower() == ".mat":
        data = loadmat(path)
        if "offsets" not in data or "levels" not in data:
            raise ValueError("MAT 文件需要 offsets（Hz）与 levels（dBc/Hz）数组")
        offsets, levels = (np.asarray(data[key]).reshape(-1) for key in ("offsets", "levels"))
        if offsets.shape != levels.shape:
            raise ValueError("offsets 与 levels 数量必须一致")
        anchors = np.column_stack((offsets, levels))
    else:
        anchors = _table(path)
    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("相噪表格需要两列：频偏 Hz、电平 dBc/Hz")
    anchors = anchors[np.argsort(anchors[:, 0])]
    return DeviceModel("mixer", dict(parameters, anchors=anchors.tolist()), name=path.stem).validated()
