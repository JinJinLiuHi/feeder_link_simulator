"""Optional GNU Radio 3.10 runtime adapter; no GNU Radio source is vendored.

Finite source -> native DSP block -> vector sink. Existing fitted PA, DAC,
phase-noise-anchor and IIR models retain the project's numerical adapters.
GNU Radio's complex64 output may differ slightly from NumPy complex128.
"""

import numpy as np


def require_runtime():
    try:
        from gnuradio import gr, blocks, filter, analog
        return gr, blocks, filter, analog
    except (ImportError, OSError) as exc:
        raise RuntimeError("当前 Python 环境没有可用的 GNU Radio 运行库。请选择本地计算，或在安装 GNU Radio 的环境运行本项目。") from exc


def runtime_status():
    try:
        gr, *_ = require_runtime()
        return True, str(gr.version())
    except RuntimeError as exc:
        return False, str(exc)


def native_block(x, kind, **parameters):
    gr, blocks, filters, analog = require_runtime()
    graph = gr.top_block("Feeder finite waveform")
    source = blocks.vector_source_c(np.asarray(x, dtype=np.complex64).tolist(), False)
    sink = blocks.vector_sink_c()
    if kind == "fir":
        operation = filters.fir_filter_ccc(1, np.asarray(parameters["taps"], dtype=np.complex64).tolist())
        graph.connect(source, operation, sink)
    elif kind == "gain":
        operation = blocks.multiply_const_cc(parameters["gain"])
        graph.connect(source, operation, sink)
    elif kind == "frequency":
        oscillator = analog.sig_source_c(parameters["fs"], analog.GR_COS_WAVE, parameters["offset"], 1., 0.)
        operation = blocks.multiply_cc()
        # Bound both branches, including the oscillator, so graph.run() terminates.
        head = blocks.head(gr.sizeof_gr_complex, len(x))
        graph.connect(source, (operation, 0))
        graph.connect(oscillator, head, (operation, 1))
        graph.connect(operation, sink)
    else:
        raise ValueError("该模块尚未映射到 GNU Radio 原生块")
    graph.run()
    y = np.asarray(sink.data(), dtype=np.complex128)
    if len(y) != len(x):
        raise RuntimeError(f"GNU Radio 输出长度 {len(y)} 与输入 {len(x)} 不符")
    return y
