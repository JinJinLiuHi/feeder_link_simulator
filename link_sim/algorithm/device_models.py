"""Validated device-model snapshots and portable JSON/NPZ interchange (no UI)."""

from dataclasses import dataclass, field
import copy
import json
from pathlib import Path
import numpy as np
from .pa_simulation_core import PAModel, load_model

FORMAT = "feeder-link-device-model"
VERSION = 1
KINDS = {"pa": "功放", "dpd": "数字预失真", "adda": "ADC/DAC", "filter": "滤波器", "mixer": "混频器"}
MAX_BYTES = 32 * 1024**2
FILTER_DEFAULTS = dict(impl="外部系数", iir_type="Butterworth", fir_method="窗函数法", window="hamming",
    band="低通", fs=100e6, fc1=20e6, fc2=40e6, order=6, numtaps=65, rp=.5,
    rs=60., trans_bw=4e6, beta=.25, sps=8, span=10)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


@dataclass
class DeviceModel:
    kind: str
    parameters: dict
    arrays: dict = field(default_factory=dict)
    name: str = ""
    description: str = ""

    def validated(self):
        """Validate and return an independent snapshot before any write/apply."""
        if self.kind not in KINDS or not isinstance(self.parameters, dict) or not isinstance(self.arrays, dict):
            raise ValueError("模型类型或参数格式不正确")
        if not isinstance(self.name, str) or len(self.name) > 120 or not isinstance(self.description, str) or len(self.description) > 2000:
            raise ValueError("模型名称最多 120 字，说明最多 2000 字")
        try:
            parameters = json.loads(json_text(self.parameters))
        except (TypeError, ValueError) as exc:
            raise ValueError("模型参数须为有限数值及 JSON 数据") from exc
        arrays = {}
        size = 0
        for name, values in self.arrays.items():
            if not isinstance(name, str) or not name.isidentifier() or len(name) > 64:
                raise ValueError("系数数组名称无效")
            array = np.asarray(values)
            if array.dtype.kind not in "biufc" or array.ndim > 2 or array.size == 0 or not np.all(np.isfinite(array)):
                raise ValueError("系数须为非空有限数值数组，不支持对象或非数值数组")
            size += array.nbytes
            if size > MAX_BYTES:
                raise ValueError("模型系数超过 32 MiB，请勿将仿真波形作为模型保存")
            arrays[name] = array.copy()
        model = DeviceModel(self.kind, parameters, arrays, self.name.strip(), self.description)
        validate_behavior(model)
        return model


def number(p, key, low=None, high=None, integer=False, positive=False):
    v = p.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
        raise ValueError(f"{key} 必须为有限数值")
    if (integer and int(v) != v) or (low is not None and v < low) or (high is not None and v > high) or (positive and v <= 0):
        raise ValueError(f"{key} 超出允许范围")
    return v


def validate_behavior(model):
    p, a = model.parameters, model.arrays
    if model.kind in ("pa", "dpd"):
        pa = to_pa(model)
        if model.kind == "dpd":
            info = pa.fit_info
            if not all(key in info for key in ("gainReal", "gainImag", "paSignature")):
                raise ValueError("DPD 缺少目标增益或对应功放信息")
            gain = complex(info["gainReal"], info["gainImag"])
            if not np.isfinite(gain) or abs(gain) < 1e-12 or not isinstance(info["paSignature"], str):
                raise ValueError("DPD 目标增益或功放信息无效")
        if set(a) != {"coefficients"}:
            raise ValueError("功放/DPD 模型只应包含 coefficients 系数")
    elif model.kind == "adda":
        number(p, "n_bits", 2, 32, integer=True)
        fs = number(p, "fs", positive=True)
        number(p, "nfft", 256, 1048576, integer=True)
        f = number(p, "f_sig", positive=True)
        number(p, "noise_density_dbfs_hz", high=0)
        target = number(p, "sfdr_target", 0, 300)
        limits = {"ADC12DJ3200": (52, 69), "EV12DS130": (53, 68)}.get(p.get("device_profile"))
        if limits is not None and not limits[0] <= target <= limits[1]:
            raise ValueError("目标 SFDR 超出所选 ADC/DAC 器件预设范围；通用器件可使用自定义名称")
        if f >= fs/2 or not isinstance(p.get("device_profile"), str):
            raise ValueError("ADC/DAC 基波须低于 Nyquist 频率，器件名称须为文本")
        freq, levels = p.get("spur_freqs"), p.get("spur_dbfs")
        if not isinstance(freq, list) or not isinstance(levels, list) or not 1 <= len(freq) <= 32 or len(freq) != len(levels):
            raise ValueError("ADC/DAC 杂散须为 1–32 组频率/电平")
        for i, (f, db) in enumerate(zip(freq, levels)):
            number({"f": f}, "f", positive=True)
            number({"db": db}, "db", high=0)
            if f >= fs/2:
                raise ValueError(f"第 {i+1} 根杂散须低于 Nyquist 频率")
        if not np.isclose(levels[0], -target, atol=1e-9):
            raise ValueError("第 1 根杂散须与当前 ADC/DAC 模型的目标 SFDR 一致")
        if a:
            raise ValueError("ADC/DAC 行为模型以参数保存，不包含波形数组")
    elif model.kind == "mixer":
        fs = number(p, "fs", positive=True)
        duration = number(p, "sim_time", positive=True)
        f = number(p, "f_if_in", positive=True)
        if f >= fs/2 or not 256 <= fs*duration <= 1048576:
            raise ValueError("混频器中频须低于 Nyquist，采样点数须为 256–1048576")
        anchors = p.get("anchors")
        if not isinstance(anchors, list) or not 2 <= len(anchors) <= 256:
            raise ValueError("相噪锚点须为至少两组频偏/电平")
        previous = 0
        for pair in anchors:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("相噪锚点须为 [频偏 Hz, 电平 dBc/Hz]")
            offset = number({"f": pair[0]}, "f", positive=True)
            number({"db": pair[1]}, "db", high=0)
            if offset <= previous or offset >= fs/2:
                raise ValueError("相噪频偏须递增、无重复且低于 Nyquist")
            previous = offset
        if a:
            raise ValueError("混频器行为模型不包含仿真波形")
    elif model.kind == "filter":
        fs = number(p, "fs", positive=True)
        if p.get("impl") not in ("IIR", "FIR", "RRC 成形", "外部系数"):
            raise ValueError("未知滤波器实现类型")
        if p.get("band") not in ("低通", "高通", "带通", "带阻"):
            raise ValueError("未知滤波器带型")
        allowed = {"b", "a", "sos", "h"}
        if not set(a) <= allowed:
            raise ValueError("滤波器仅支持 b/a/sos/h 系数数组")
        if "h" in a:
            if a["h"].ndim != 1 or set(a) != {"h"}:
                raise ValueError("RRC 系数须为单个一维 h 数组")
            if p["impl"] != "RRC 成形":
                raise ValueError("h 数组仅用于 RRC；普通 FIR 请使用 b 和 a")
            if np.any(np.imag(a["h"]) != 0):
                raise ValueError("RRC 成形滤波器须使用实数系数")
        elif not {"b", "a"} <= set(a) or any(a[key].ndim != 1 for key in ("b", "a")) or a["a"][0] == 0:
            raise ValueError("滤波器须包含一维 b/a 系数且 a[0] 非零")
        if "sos" in a and (a["sos"].ndim != 2 or a["sos"].shape[1] != 6 or np.any(a["sos"][:, 3] == 0)):
            raise ValueError("SOS 须为 N×6 数组，各节 a0 非零")
        if p["impl"] != "外部系数":
            for key in FILTER_DEFAULTS:
                if key not in p:
                    raise ValueError(f"滤波器缺少设计参数 {key}")
            number(p, "order", 1, 128, integer=True)
            number(p, "numtaps", 3, 262144, integer=True)
            number(p, "rp", low=0)
            number(p, "rs", positive=True)
            number(p, "trans_bw", positive=True)
            number(p, "beta", 0, 1)
            number(p, "sps", 2, 256, integer=True)
            number(p, "span", 2, 4096, integer=True)
            if not p["impl"].startswith("RRC"):
                fc = number(p, "fc1", positive=True)
                if fc >= fs/2:
                    raise ValueError("滤波器截止频率须低于 Nyquist")
                if p["band"] in ("带通", "带阻") and not fc < number(p, "fc2") < fs/2:
                    raise ValueError("带通/带阻须满足 0 < fc1 < fc2 < fs/2")


def from_pa(pa, name="", description=""):
    return DeviceModel("dpd" if pa.role == "predistorter" else "pa",
        dict(model_type=pa.model_type, degree=pa.degree, cross_depth=pa.cross_depth,
             sample_rate=pa.sample_rate, fit_info=copy.deepcopy(pa.fit_info)),
        {"coefficients": pa.coefficients.copy()}, name, description).validated()


def to_pa(model):
    p = model.parameters
    if model.kind not in ("pa", "dpd") or "coefficients" not in model.arrays:
        raise ValueError("该记录不是功放或预失真器模型")
    number(p, "degree", 1, 64, integer=True)
    number(p, "cross_depth", 0, 256, integer=True)
    if not isinstance(p.get("fit_info"), dict):
        raise ValueError("功放/DPD 辨识信息必须为字典")
    return PAModel(model.arrays["coefficients"], p["model_type"], p["degree"], p["cross_depth"],
                   p.get("sample_rate"), copy.deepcopy(p["fit_info"]),
                   "predistorter" if model.kind == "dpd" else "power_amplifier")


def manifest(model):
    return dict(format=FORMAT, version=VERSION, kind=model.kind, name=model.name,
                description=model.description, parameters=model.parameters)


def unpack_manifest(data, arrays):
    if not isinstance(data, dict) or data.get("format") != FORMAT or data.get("version") != VERSION:
        raise ValueError("不是受支持的器件模型文件或格式版本")
    try:
        return DeviceModel(data["kind"], data["parameters"], arrays, data.get("name", ""), data.get("description", "")).validated()
    except (KeyError, TypeError) as exc:
        raise ValueError("模型文件缺少类型、参数或系数") from exc


def write_model_file(path, model):
    model = model.validated()
    path = Path(path)
    data = manifest(model)
    if path.suffix.lower() == ".npz":
        np.savez_compressed(path, modelManifest=json_text(data), **{"array_"+k: v for k, v in model.arrays.items()})
    elif path.suffix.lower() == ".json":
        data["arrays"] = {name: dict(dtype=a.dtype.str, shape=list(a.shape),
            real=a.real.tolist(), **({"imag": a.imag.tolist()} if np.iscomplexobj(a) else {}))
            for name, a in model.arrays.items()}
        path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    else:
        raise ValueError("统一模型导出支持 .json 和 .npz")


def read_model_file(path, expected_kind=None):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES*4:
        raise ValueError("模型文件过大，请勿导入仿真波形文件")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or not isinstance(data.get("arrays", {}), dict):
            raise ValueError("模型文件须为包含器件类型与参数的 JSON 对象")
        arrays = {}
        for name, item in data.get("arrays", {}).items():
            dtype = np.dtype(item["dtype"])
            if dtype.kind not in "biufc":
                raise ValueError("不支持该数组数据类型")
            values = np.asarray(item["real"])
            if "imag" in item:
                values = values + 1j*np.asarray(item["imag"])
            array = np.asarray(values, dtype=dtype)
            if list(array.shape) != item["shape"]:
                raise ValueError("模型系数形状与内容不一致")
            arrays[name] = array
        model = unpack_manifest(data, arrays)
    elif suffix == ".npz":
        import zipfile
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > MAX_BYTES*2:
                raise ValueError("解压后的模型文件过大")
        with np.load(path, allow_pickle=False) as data:
            if "modelManifest" in data:
                model = unpack_manifest(json.loads(data["modelManifest"].item()),
                    {k[6:]: np.array(data[k]) for k in data.files if k.startswith("array_")})
            elif "fitCoefMatMem" in data:
                model = from_pa(load_model(path, expected_role=None), path.stem)
            elif "fs" in data and ("b" in data or "sos" in data):
                arrays = {k: np.array(data[k]) for k in ("b", "a", "sos") if k in data}
                if "sos" in arrays and ("b" not in arrays or "a" not in arrays):
                    from scipy.signal import sos2tf
                    arrays["b"], arrays["a"] = sos2tf(arrays["sos"])
                elif "a" not in arrays:
                    arrays["a"] = np.array([1.0])
                model = DeviceModel("filter", dict(FILTER_DEFAULTS, fs=float(data["fs"].item())), arrays, path.stem).validated()
            else:
                raise ValueError("NPZ 缺少模型说明、功放系数或滤波器 b/a/sos 与 fs")
    elif suffix in (".csv", ".txt"):
        model = from_pa(load_model(path, expected_role=None), path.stem)
    else:
        raise ValueError("请导入 JSON/NPZ 模型；功放/DPD 还支持原 CSV/TXT 文件")
    if expected_kind is not None and model.kind != expected_kind:
        raise ValueError(f"模型类型不匹配：当前需要 {KINDS[expected_kind]}，文件为 {KINDS[model.kind]}")
    return model
