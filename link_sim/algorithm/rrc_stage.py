"""RRC design and saved filter taps owned by the independent RRC node."""

import numpy as np
import torch

from .burst_link.pulse_shaping import RRC


class SavedRRC(RRC):
    def __init__(self, model):
        model = model.validated()
        p = model.parameters
        if model.kind != "filter" or p.get("impl") != "RRC 成形":
            raise ValueError("RRC 模块需要滤波器库中的 RRC 成形模型")
        super().__init__(samples_per_symbol=int(p["sps"]), rolloff=p["beta"], span_symbols=int(p["span"]))
        taps = model.arrays.get("h")
        if taps is None:
            taps = model.arrays["b"] / model.arrays["a"][0]
        if len(taps) != self.tap_count or np.any(np.imag(taps)) or not np.allclose(taps, taps[::-1]):
            raise ValueError("RRC 抽头应为实数对称序列，长度为 span × sps + 1")
        if not np.sum(np.abs(taps)**2) > 0:
            raise ValueError("RRC 抽头能量必须大于零")
        self._coefficients_cpu = torch.tensor(np.real(taps), dtype=torch.float64)


def transmit_shaper(stage):
    if stage.model is None:
        raise ValueError("请先从模型库加载或导入 RRC 成形模型")
    return SavedRRC(stage.model)


def legacy_rrc_model(settings):
    """Convert explicitly saved pre-model RRC settings into a portable snapshot."""
    from .waveform_chain import _number
    from .device_models import DeviceModel, FILTER_DEFAULTS
    sps = _number(settings, "sps", 4, 2, 256, True)
    beta = _number(settings, "rolloff", .25, 0., 1.)
    span = _number(settings, "span", 8, 2, 4096, True)
    taps = RRC(samples_per_symbol=sps, rolloff=beta, span_symbols=span).coefficients().numpy()
    return DeviceModel("filter", dict(FILTER_DEFAULTS, impl="RRC 成形", sps=sps, beta=beta, span=span),
                       {"h": taps}, name="原方案 RRC 模型").validated()
