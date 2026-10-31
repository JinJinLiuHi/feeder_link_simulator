"""Explicit model/config mappings for the four existing device editors."""

import copy
import numpy as np
from PySide6.QtCore import QSignalBlocker
from link_sim.algorithm.device_models import DeviceModel, FILTER_DEFAULTS, from_pa, to_pa


def _numeric_fields(dialog, names):
    for name in names:
        try:
            value = float(getattr(dialog, name).text())
        except (TypeError, ValueError) as exc:
            raise ValueError("请先修正器件参数中的非数值输入") from exc
        if not np.isfinite(value):
            raise ValueError("器件参数不能包含 NaN 或无穷大")


def capture_device(dialog):
    model = _capture_device(dialog)
    identity = getattr(dialog, "_library_identity", None)
    if identity is not None and identity[0] == model.kind:
        model.name, model.description = identity[1:]
    return model


def _capture_device(dialog):
    kind = dialog.device_model_kind()
    if kind in ("pa", "dpd"):
        if kind == "dpd":
            page = dialog.dpd_page
            if page.busy:
                raise ValueError("DPD 正在训练，请完成后再保存")
            model = page.dpd
        else:
            model = dialog.fitted_model if dialog.tabs.currentIndex() == 0 else dialog.simulation_page.model
        if model is None:
            raise ValueError("请先拟合、训练或导入一个模型")
        return from_pa(model)
    if kind == "adda":
        _numeric_fields(dialog, ("e_bits", "e_fs", "e_nfft", "e_fsig", "e_noise_density", "e_sfdr_target", "e_spur_count"))
        for name, low, high in (("e_bits", 2, 32), ("e_nfft", 256, 1048576), ("e_spur_count", 1, 32)):
            value = float(getattr(dialog, name).text())
            if value != int(value) or not low <= value <= high:
                raise ValueError("位数、FFT 点数与杂散数量必须为允许范围内的整数")
        for frequency, level in dialog._spur_edits:
            if not all(np.isfinite(float(e.text())) for e in (frequency, level)):
                raise ValueError("杂散频率和电平须为有限数值")
        if int(float(dialog.e_spur_count.text())) != len(dialog._spur_edits):
            raise ValueError("请先点击“更新条目”，使杂散数量与条目一致")
        cfg = dialog._collect_cfg()
        cfg.update(device_profile=dialog.cb_device.currentText(), sfdr_target=float(dialog.e_sfdr_target.text()))
        return DeviceModel(kind, cfg).validated()
    if kind == "mixer":
        _numeric_fields(dialog, ("e_fs", "e_tsim", "e_f_if"))
        cfg = dialog._collect_cfg()
        entries = [s for s in dialog.e_anchors.text().replace("\n", ",").split(",") if s.strip()]
        if len(entries) != len(cfg["anchors"]):
            raise ValueError("请修正相噪锚点，所有条目都须为“频偏:电平”")
        return DeviceModel(kind, cfg).validated()
    if kind == "filter":
        _numeric_fields(dialog, ("e_fs", "e_fc1", "e_fc2", "e_order", "e_taps", "e_rp", "e_rs", "e_tb", "e_beta", "e_sps", "e_span"))
        if dialog._last is None:
            raise ValueError("请先设计或导入滤波器")
        tag, cfg, coefficients = dialog._last
        if cfg != dialog._collect_cfg():
            raise ValueError("滤波器参数已修改，请先重新设计，再保存实际系数")
        if tag == "rrc":
            arrays = {"h": coefficients}
        else:
            b, a, sos = coefficients
            arrays = {"b": b, "a": a}
            if sos is not None:
                arrays["sos"] = sos
        return DeviceModel(kind, cfg, arrays).validated()
    raise ValueError("该模块尚无可保存的器件模型")


def apply_device(dialog, model):
    model = model.validated()
    if dialog.MODEL_KIND == "pa":
        if model.kind not in ("pa", "dpd") or dialog.dpd_page.busy:
            raise ValueError("请选择功放/DPD 模型，并等待当前训练完成")
        pa = to_pa(model)
        if model.kind == "pa":
            dialog.simulation_page.set_model(pa, model.name or pa.name, "SQLite / 外部模型")
            dialog.tabs.setCurrentIndex(1)
        else:
            dialog.dpd_page.set_dpd_model(pa)
            dialog.tabs.setCurrentIndex(2)
        return
    if model.kind != dialog.MODEL_KIND:
        raise ValueError("模型类型与器件编辑器不匹配")
    p = model.parameters
    if model.kind == "adda":
        from link_sim.algorithm.ad_model import run_adc_model
        kwargs = {k: p[k] for k in ("n_bits", "fs", "nfft", "f_sig", "spur_freqs", "spur_dbfs", "noise_density_dbfs_hz")}
        result = run_adc_model(**kwargs)
        blocker = QSignalBlocker(dialog.cb_device)
        profile = p["device_profile"]
        if dialog.cb_device.findText(profile) < 0:
            dialog.cb_device.addItem(profile)
        dialog.cb_device.setCurrentText(profile)
        del blocker
        values = dict(e_bits=p["n_bits"], e_fs=p["fs"]/1e6, e_nfft=p["nfft"], e_fsig=p["f_sig"]/1e6,
                      e_noise_density=p["noise_density_dbfs_hz"], e_sfdr_target=p["sfdr_target"], e_spur_count=len(p["spur_freqs"]))
        for name, value in values.items():
            getattr(dialog, name).setText(f"{value:.17g}")
        dialog._rebuild_spur_rows()
        for (f, level), frequency, db in zip(dialog._spur_edits, p["spur_freqs"], p["spur_dbfs"]):
            f.setText(f"{frequency/1e6:.17g}")
            level.setText(f"{db:.17g}")
        dialog.device_hint.setText(f"已加载模型：{model.name or profile}")
        result["device_name"] = profile
        dialog._last_result = result
        dialog._plot_spectrum(result)
    elif model.kind == "mixer":
        from link_sim.algorithm.mixer_model import run_mixer_phase_noise
        result = run_mixer_phase_noise(**p)
        for field, value in ((dialog.e_fs, p["fs"]/1e6), (dialog.e_tsim, p["sim_time"]*1e3), (dialog.e_f_if, p["f_if_in"]/1e6)):
            field.setText(f"{value:.17g}")
        dialog.e_anchors.setText(", ".join(f"{f:.17g}:{db:.17g}" for f, db in p["anchors"]))
        dialog._last_result = result
        dialog._plot_spectrum(result)
    elif model.kind == "filter":
        cfg = dict(FILTER_DEFAULTS, **p)
        names = (("cb_impl", "impl"), ("cb_iir", "iir_type"), ("cb_firm", "fir_method"), ("cb_win", "window"), ("cb_band", "band"))
        blockers = [QSignalBlocker(getattr(dialog, name)) for name, _ in names]
        for name, key in names:
            combo = getattr(dialog, name)
            if combo.findText(cfg[key]) < 0:
                combo.addItem(cfg[key])
            combo.setCurrentText(cfg[key])
        fields = {"e_fs": ("fs", 1e6), "e_fc1": ("fc1", 1e6), "e_fc2": ("fc2", 1e6),
                  "e_order": ("order", 1), "e_taps": ("numtaps", 1), "e_rp": ("rp", 1),
                  "e_rs": ("rs", 1), "e_tb": ("trans_bw", 1e6), "e_beta": ("beta", 1),
                  "e_sps": ("sps", 1), "e_span": ("span", 1)}
        for name, (key, divisor) in fields.items():
            getattr(dialog, name).setText(f"{cfg[key]/divisor:.17g}")
        blockers.clear()
        dialog._on_impl_changed(dialog.cb_impl.currentIndex())
        dialog._on_band_changed(dialog.cb_band.currentIndex())
        # Freeze the loaded arrays. Changing only the view must not redesign them.
        dialog._library_filter = (dialog._collect_cfg(), {k: v.copy() for k, v in model.arrays.items()})
        dialog._do_run()

