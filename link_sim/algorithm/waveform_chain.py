"""Finite complex-envelope flowgraphs, independent of Qt and the model editors.

Samples use normalized I/Q units; AD/DA errors come from the original model output.
carrier_hz is metadata: an RF carrier is not sampled at GHz rates. Frequency
offset blocks, unlike an upconverter's carrier setting, shift sampled spectra.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from scipy import signal

from .device_models import DeviceModel, manifest, unpack_manifest, to_pa
from .pa_simulation_core import load_waveform, positive_rate, simulate, waveform
from .adc_noise_core import calibrate_model_noise, generate_model_noise


MAX_SAMPLES = 262144
KINDS = {"source": "基带信号源", "dac": "AD/DA", "filter": "滤波器",
         "mixer": "上变频器", "pa": "功率放大器", "gain": "增益",
         "frequency": "频率偏移", "awgn": "加性白噪声"}
DEVICE_KINDS = {"dac": "adda", "filter": "filter", "mixer": "mixer", "pa": "pa"}


@dataclass
class Wave:
    x: np.ndarray
    fs: float
    carrier_hz: float = 0.
    start: int = 0
    stop: int | None = None
    notes: tuple = ()
    added_error: np.ndarray | None = None

    def __post_init__(self):
        self.x = waveform(self.x)
        self.fs = positive_rate(self.fs)
        if len(self.x) > MAX_SAMPLES:
            raise ValueError(f"单次波形最多 {MAX_SAMPLES:,} 点")
        self.stop = len(self.x) if self.stop is None else self.stop
        if not 0 <= self.start < self.stop <= len(self.x):
            raise ValueError("有效波形区间无效")
        if not np.isfinite(self.carrier_hz) or self.carrier_hz < 0:
            raise ValueError("载频须为有限非负数")
        if self.added_error is not None:
            self.added_error = waveform(self.added_error)
            if len(self.added_error) != len(self.x):
                raise ValueError("本级等效误差与输出长度不一致")

    @property
    def valid(self):
        return self.x[self.start:self.stop]


@dataclass
class Stage:
    key: str
    kind: str
    name: str
    settings: dict = field(default_factory=dict)
    model: DeviceModel | None = None


def _number(settings, key, default, low=None, high=None, integer=False):
    value = settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
        raise ValueError(f"{key} 须为有限数值")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{key} 超出允许范围 {low}–{high}")
    if integer and value != int(value):
        raise ValueError(f"{key} 须为整数")
    return int(value) if integer else float(value)


def default_settings(kind):
    return {
        "source": dict(fs=100e6, count=16384, sps=8, amplitude=.2, seed=2025, modulation="QPSK"),
        "dac": {},
        "filter": dict(cutoff=15e6, taps=65),
        "mixer": dict(carrier_hz=14e9, gain_db=0.),
        "pa": dict(gain_db=0.),
        "gain": dict(gain_db=0.),
        "frequency": dict(offset_hz=0.),
        "awgn": dict(snr_db=30., seed=2025),
    }[kind].copy()


def source_wave(settings):
    fs = _number(settings, "fs", 100e6, 1., 1e12)
    if "iq" in settings:
        a = np.asarray(settings["iq"], dtype=float)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError("导入波形须为 I/Q 两列")
        return Wave(a[:, 0] + 1j*a[:, 1], fs, notes=("外部 I/Q：保留原始幅度；编码/调制由文件提供",))
    count = _number(settings, "count", 16384, 256, MAX_SAMPLES, True)
    sps = _number(settings, "sps", 8, 2, 256, True)
    amplitude = _number(settings, "amplitude", .2, 1e-6, 1.)
    seed = _number(settings, "seed", 2025, 0, 2**32-1, True)
    mode = settings.get("modulation", "QPSK")
    rng = np.random.default_rng(seed)
    n = (count+sps-1)//sps
    if mode == "BPSK":
        symbols = 2*rng.integers(0, 2, n)-1
    elif mode == "QPSK":
        symbols = ((2*rng.integers(0, 2, n)-1) + 1j*(2*rng.integers(0, 2, n)-1))/np.sqrt(2)
    elif mode == "16QAM":
        symbols = ((2*rng.integers(0, 4, n)-3) + 1j*(2*rng.integers(0, 4, n)-3))/np.sqrt(10)
    else:
        raise ValueError("支持 BPSK、QPSK 和 16QAM")
    return Wave(amplitude*np.repeat(symbols, sps)[:count], fs,
                notes=(f"{mode} 未编码测试信号，矩形脉冲，{sps} 点/符号",))


def import_source(path, fs):
    x, file_rate = load_waveform(path)
    if len(x) > MAX_SAMPLES:
        raise ValueError(f"导入波形最多 {MAX_SAMPLES:,} 点，请先分段")
    return dict(fs=file_rate or positive_rate(fs), iq=np.column_stack((x.real, x.imag)).tolist(),
                filename=Path(path).name)


def order_graph(stages, edges):
    """Every node must be connected to one source; fan-out is allowed, no merges."""
    by_id = {stage.key: stage for stage in stages}
    if not by_id or len(by_id) != len(stages) or len(stages) > 64:
        raise ValueError("流程图须包含 1–64 个模块，且模块编号不能重复")
    if any(s.kind not in KINDS for s in stages):
        raise ValueError("流程图包含未知模块")
    sources = [s.key for s in stages if s.kind == "source"]
    if len(sources) != 1:
        raise ValueError("请保留且仅保留一个基带信号源")
    parents, children = {}, {key: [] for key in by_id}
    for src, dst in edges:
        if src not in by_id or dst not in by_id or dst == sources[0] or dst in parents:
            raise ValueError("连线无效：每个器件只接受一个输入，信号源不能接收输入")
        parents[dst] = src
        children[src].append(dst)
    queue, ordered = [sources[0]], []
    while queue:
        key = queue.pop(0)
        if key in ordered:
            raise ValueError("流程图不能形成环路")
        ordered.append(key)
        queue.extend(children[key])
    if len(ordered) != len(stages):
        raise ValueError("存在未连接的模块或环路，请将所有模块连接到信号源")
    return [by_id[key] for key in ordered], parents


def _check_model(stage, fs):
    if stage.model is None:
        return None
    model = stage.model.validated()
    if model.kind != DEVICE_KINDS.get(stage.kind):
        raise ValueError("所选模型的器件类型不匹配")
    # The ADC noise floor is converted from its calibration FFT bin width to
    # density; the input array's sampling grid is not replaced.
    rate = None if model.kind == "adda" else model.parameters.get("sample_rate" if model.kind == "pa" else "fs")
    if rate is not None and not np.isclose(fs, rate, rtol=1e-9, atol=1e-6):
        raise ValueError(f"模型采样率为 {rate/1e6:g} MHz，输入为 {fs/1e6:g} MHz；请修改信号源或更换模型")
    return model


def _phase_noise(fs, count, anchors, seed):
    # Same SSB L(f) convention as the existing mixer editor, without its Qt imports.
    offsets = np.fft.rfftfreq(count, 1/fs)
    frequency, level = np.array(anchors, dtype=float).T
    psd = 10**(np.interp(np.log10(np.maximum(offsets, offsets[1])),
                         np.log10(frequency), level)/10)
    weights = np.sqrt(psd*fs)
    mirrored = weights[-2:0:-1] if count % 2 == 0 else weights[-1:0:-1]
    white = np.random.default_rng(seed).standard_normal(count)
    phi = np.fft.ifft(np.fft.fft(white)*np.concatenate((weights, mirrored))).real
    return phi-phi.mean()


def process_stage(stage, incoming, backend="numpy", noise_stream=0):
    model = _check_model(stage, incoming.fs)
    p = stage.settings
    x, fs = incoming.x, incoming.fs
    start, stop = incoming.start, incoming.stop
    carrier = incoming.carrier_hz
    notes = []
    added_error = None
    if stage.kind == "filter":
        if model is None:
            cutoff = _number(p, "cutoff", 15e6, 1.)
            taps = _number(p, "taps", 65, 3, 4095, True)
            if cutoff >= fs/2:
                raise ValueError("截止频率必须低于采样率的一半")
            b, a, sos = signal.firwin(taps, cutoff, fs=fs), np.array([1.]), None
            notes.append("内置低通 FIR")
        else:
            b = model.arrays.get("h", model.arrays.get("b"))
            a = model.arrays.get("a", np.array([1.]))
            sos = model.arrays.get("sos")
            notes.append(f"模型：{model.name}")
        fir = sos is None and len(a) == 1
        if backend == "gnuradio" and fir:
            from .waveform_gnuradio import native_block
            y = native_block(x, "fir", taps=b/a[0])
            notes.append("GNU Radio 原生 FIR")
        else:
            y = signal.sosfilt(sos, x) if sos is not None else signal.lfilter(b, a, x)
        if fir:
            start += len(b)-1
            notes.append(f"FIR 起始 {len(b)-1} 点为启动瞬态；尾部未补零冲洗")
        else:
            notes.append("IIR 输出包含启动瞬态，未自动判定稳定时间")
    elif stage.kind == "dac":
        # Keep the historical node key 'dac' so saved chain diagrams still load.
        if model is None:
            y = x.copy()
            notes.append("AD/DA 未加载器件模型，当前为理想直通；双击从原 ADC/DAC 模型库加载")
        else:
            seed = 2025
            if noise_stream:
                seed = int(np.random.SeedSequence([seed, noise_stream]).generate_state(1)[0])
            reference = calibrate_model_noise(model)
            noise, power = generate_model_noise(reference, len(x), fs, seed=seed)
            y = x+noise
            added_error = noise
            measured = float(np.mean(abs(noise[start:stop])**2))
            notes.append(f"x=s+n；{model.name or model.parameters['device_profile']}；"
                         f"模型估计噪声基底 {reference.floor_dbfs:.2f} dBFS/频点；"
                         f"Pn={power:.6g}；理论噪声 RMS {np.sqrt(power):.4g}；"
                         f"实测 RMS {np.sqrt(measured):.4g}；自动匹配链路采样率，不叠加杂散")
    elif stage.kind == "mixer":
        carrier = _number(p, "carrier_hz", 14e9, 0., 1e15)
        gain = 10**(_number(p, "gain_db", 0., -100., 100.)/20)
        phi = _phase_noise(fs, len(x), model.parameters["anchors"], 2025) if model else 0.
        y = gain*x*np.exp(1j*phi)
        notes.append("复包络上变频：载频作为元数据；加载模型时应用相噪锚点；不含本振泄漏")
    elif stage.kind == "pa":
        if model:
            pa = to_pa(model)
            _, y, valid = simulate(pa, x, fs)
            start += valid.start
            stop -= pa.edge
            notes.append(f"{model.name}；按保存系数计算，未重新归一化")
        else:
            y = x*10**(_number(p, "gain_db", 0., -100., 100.)/20)
            notes.append("理想线性功放；双击加载辨识模型以模拟非线性")
    elif stage.kind == "gain":
        gain = 10**(_number(p, "gain_db", 0., -100., 100.)/20)
        if backend == "gnuradio":
            from .waveform_gnuradio import native_block
            y = native_block(x, "gain", gain=gain)
        else:
            y = x*gain
    elif stage.kind == "frequency":
        offset = _number(p, "offset_hz", 0., -fs/2, fs/2)
        if backend == "gnuradio":
            from .waveform_gnuradio import native_block
            y = native_block(x, "frequency", offset=offset, fs=fs)
        else:
            y = x*np.exp(2j*np.pi*offset*np.arange(len(x))/fs)
        notes.append("频偏移动采样频谱，请确保信号带宽加频偏未超过 Nyquist")
    elif stage.kind == "awgn":
        snr = _number(p, "snr_db", 30., -40., 200.)
        seed = _number(p, "seed", 2025, 0, 2**32-1, True)
        rng = np.random.default_rng(seed)
        sigma = np.sqrt(np.mean(abs(incoming.valid)**2)*10**(-snr/10)/2)
        y = x+sigma*(rng.standard_normal(len(x))+1j*rng.standard_normal(len(x)))
        notes.append(f"输入有效区间平均功率定义 SNR={snr:g} dB")
    else:
        raise ValueError("未知处理模块")
    if stop-start < 4:
        raise ValueError("波形过短，器件记忆/滤波器启动瞬态后不足 4 个有效点")
    return Wave(y, fs, carrier, start, stop, tuple(notes), added_error)


def run_graph(stages, edges, backend="numpy", progress=None, cancelled=None):
    if backend not in ("numpy", "gnuradio"):
        raise ValueError("未知计算后端")
    if backend == "gnuradio":
        from .waveform_gnuradio import require_runtime
        require_runtime()
    ordered, parents = order_graph(stages, edges)
    outputs = {}
    noise_stream = 0
    for index, stage in enumerate(ordered):
        if cancelled and cancelled():
            raise InterruptedError("仿真已取消")
        try:
            outputs[stage.key] = source_wave(stage.settings) if stage.kind == "source" else process_stage(
                stage, outputs[parents[stage.key]], backend, noise_stream=noise_stream)
            if stage.kind == "dac":
                noise_stream += 1
        except (ValueError, FloatingPointError) as exc:
            raise ValueError(f"{stage.name}：{exc}") from exc
        if progress:
            progress(index+1, len(ordered), stage.name)
    return outputs


def model_to_dict(model):
    if model is None:
        return None
    model = model.validated()
    data = manifest(model)
    data["arrays"] = {k: dict(real=a.real.tolist(), imag=a.imag.tolist()) for k, a in model.arrays.items()}
    return data


def model_from_dict(data):
    if data is None:
        return None
    arrays = {k: np.asarray(a["real"])+1j*np.asarray(a["imag"]) for k, a in data["arrays"].items()}
    # Preserve real filter coefficients; SciPy's SOS implementation accepts real SOS.
    arrays = {k: a.real if not np.any(a.imag) else a for k, a in arrays.items()}
    return unpack_manifest(data, arrays)


def save_chain(path, stages, edges, positions, backend="numpy"):
    order_graph(stages, edges)
    payload = dict(format="feeder-waveform-chain", version=1, backend=backend, edges=edges,
                   nodes=[dict(key=s.key, kind=s.kind, name=s.name, settings=s.settings,
                               model=model_to_dict(s.model), pos=positions.get(s.key, [0, 0])) for s in stages])
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def load_chain(path):
    path = Path(path)
    if path.stat().st_size > 128*1024**2:
        raise ValueError("链路方案超过 128 MiB")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("format") != "feeder-waveform-chain" or data.get("version") != 1:
        raise ValueError("不是受支持的全链路波形方案")
    if not isinstance(data.get("nodes"), list) or not 1 <= len(data["nodes"]) <= 64:
        raise ValueError("模块数量须为 1–64")
    stages, positions = [], {}
    for n in data["nodes"]:
        if not all(isinstance(n.get(k), str) for k in ("key", "kind", "name")) or not isinstance(n.get("settings"), dict):
            raise ValueError("模块说明无效")
        pos = np.asarray(n.get("pos", [0, 0]), dtype=float)
        if pos.shape != (2,) or not np.all(np.isfinite(pos)):
            raise ValueError("画布位置无效")
        stage = Stage(n["key"], n["kind"], n["name"], n["settings"], model_from_dict(n.get("model")))
        if stage.kind == "source":
            source_wave(stage.settings)
        elif stage.model and stage.model.kind != DEVICE_KINDS.get(stage.kind):
            raise ValueError("器件模型类型与模块不匹配")
        stages.append(stage)
        positions[stage.key] = pos.tolist()
    order_graph(stages, data["edges"])
    backend = data.get("backend", "numpy")
    if backend not in ("numpy", "gnuradio"):
        raise ValueError("未知计算后端")
    return stages, data["edges"], positions, backend


def export_wave(path, wave):
    """NPZ is compatible with the existing PA waveform importer (x/sampleRate)."""
    np.savez_compressed(path, x=wave.x, sampleRate=wave.fs, carrierHz=wave.carrier_hz,
                        validStart=wave.start, validStop=wave.stop, units="normalized IQ",
                        notes=np.asarray(wave.notes, dtype=str),
                        **({"addedError": wave.added_error} if wave.added_error is not None else {}))
