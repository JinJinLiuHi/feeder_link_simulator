"""DPD training/evaluation workspace, sharing PA import and waveform controls."""

import copy
import json
from pathlib import Path
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QComboBox, QFormLayout, QHBoxLayout, QVBoxLayout,
    QFileDialog, QMessageBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QPlainTextEdit,
)
from link_sim.ui.numeric_inputs import integer, decimal
from .pa_simulation import PASimulationWidget
from .pa_simulation_core import power_spectrum, spectrum_pair, positive_rate
from .pa_ofdm import generate_ofdm, parameters
from .pa_plot import draw_pa_spectrum
from .pa_dpd_core import (
    train_ila, evaluate_dpd, partition_capture, load_dpd, save_dpd, check_rate,
)


# Keep workers alive if a user closes the window while a numerical solve runs.
_workers = set()


class _Worker(QThread):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, job):
        super().__init__()
        self.job = job

    def run(self):
        try:
            self.completed.emit(self.job(self.progress.emit))
        except Exception as exc:
            self.failed.emit(str(exc))


class PADPDWidget(PASimulationWidget):
    def __init__(self, parent=None):
        self.dpd = None
        self.busy = False
        self.reference = None
        self.measured_pair = None
        self.measured_rate = None
        super().__init__(parent)
        from .pa_model import _group
        controls = self.controls_scroll.widget().layout()
        current = QPushButton("使用当前已拟合功放")
        current.setMinimumHeight(30)
        current.clicked.connect(self.use_current_model)
        controls.itemAt(0).widget().layout().addWidget(current)
        group = _group("预失真训练与验证")
        form = QFormLayout()
        group.layout().addLayout(form)
        self.training_source = QComboBox()
        self.training_source.addItems(["由功放模型生成反馈", "导入实测输入输出"])
        form.addRow("训练反馈", self.training_source)
        self.measured_button = QPushButton("选择实测训练文件…")
        self.measured_button.setMinimumHeight(30)
        self.measured_button.setEnabled(False)
        self.measured_button.clicked.connect(self.import_measured)
        form.addRow(self.measured_button)
        self.measured_info = QLabel("实测文件取前半段训练；采集链路增益/相位须已校准，自动对齐整点时延。")
        self.measured_info.setWordWrap(True)
        form.addRow(self.measured_info)
        self.kind = QComboBox()
        self.kind.addItem("Memory Polynomial (MP)", "memPoly")
        self.kind.addItem("Generalized Memory Polynomial (GMP)", "gmp")
        self.kind.addItem("Cross-Term Memory", "ctMemPoly")
        form.addRow("DPD 模型", self.kind)
        self.memory = integer(3, 1, 32)
        self.degree = integer(5, 1, 15)
        self.cross = integer(1, 0, 16)
        self.cross.setEnabled(False)
        form.addRow("记忆长度", self.memory)
        form.addRow("多项式阶数", self.degree)
        form.addRow("GMP 包络偏移", self.cross)
        self.gain_db = decimal(29, -60, 100, .1)
        self.gain_phase = decimal(0, -180, 180, 1)
        self.gain_db.setDecimals(9)
        self.gain_phase.setDecimals(9)
        form.addRow("目标增益 (dB)", self.gain_db)
        form.addRow("目标相位 (°)", self.gain_phase)
        self.gain_note = QLabel("目标增益固定用于校正前后对比。")
        self.gain_note.setWordWrap(True)
        form.addRow(self.gain_note)
        self.iterations = integer(3, 1, 10)
        self.training_points = integer(16000, 512, 64000)
        form.addRow("ILA 迭代上限", self.iterations)
        form.addRow("训练/选模点数上限", self.training_points)
        self.limit_enabled = QCheckBox("限制在功放辨识输入范围内")
        self.peak_limit = decimal(2.4, .000001, 1e9, .1)
        self.peak_limit.setDecimals(12)
        self.peak_limit.setEnabled(False)
        form.addRow(self.limit_enabled)
        form.addRow("PA 输入峰值上限", self.peak_limit)
        self.main_bw = decimal(100, .001, 100000, 1)
        self.adj_bw = decimal(100, .001, 100000, 1)
        self.adj_offset = decimal(100, .001, 100000, 1)
        form.addRow("主信道带宽 (MHz)", self.main_bw)
        form.addRow("邻道带宽 (MHz)", self.adj_bw)
        form.addRow("邻道中心偏移 (MHz)", self.adj_offset)
        note = QLabel("数据按 50% 训练、25% 选模、25% 最终测试划分；OFDM 保留完整符号。指标不添加显示噪声。")
        note.setWordWrap(True)
        form.addRow(note)
        self.dpd_info = QLabel("尚未训练或导入 DPD")
        self.dpd_info.setWordWrap(True)
        form.addRow(self.dpd_info)
        buttons = QHBoxLayout()
        self.import_dpd_button = QPushButton("导入 DPD")
        self.export_dpd_button = QPushButton("导出 DPD")
        self.export_dpd_button.setEnabled(False)
        buttons.addWidget(self.import_dpd_button)
        buttons.addWidget(self.export_dpd_button)
        form.addRow(buttons)
        self.import_dpd_button.clicked.connect(self.import_dpd)
        self.export_dpd_button.clicked.connect(self.export_dpd)
        controls.insertWidget(2, group)
        self.run_button.setText("训练 DPD 并独立验证")
        self.evaluate_button = QPushButton("使用已有 DPD 验证")
        self.evaluate_button.setMinimumHeight(32)
        self.evaluate_button.setEnabled(False)
        self.evaluate_button.clicked.connect(lambda: self.start_job(False))
        controls.insertWidget(controls.indexOf(self.run_button)+1, self.evaluate_button)
        self.export_button.setText("导出级联波形与指标")
        # Keep the main actions visible while the longer DPD settings scroll.
        self.action_footer = QWidget()
        footer = QVBoxLayout(self.action_footer)
        footer.setContentsMargins(0, 6, 8, 0)
        footer.setSpacing(6)
        for button in (self.run_button, self.evaluate_button):
            controls.removeWidget(button)
            footer.addWidget(button)
        for index in range(controls.count()):
            item = controls.itemAt(index)
            if item.layout() is not None and item.layout().indexOf(self.export_button) >= 0:
                footer.addLayout(controls.takeAt(index).layout())
                break
        split = self.controls_scroll.parentWidget()
        self.controls_panel = QWidget()
        self.controls_panel.setFixedWidth(390)
        panel_box = QVBoxLayout(self.controls_panel)
        panel_box.setContentsMargins(0, 0, 0, 0)
        panel_box.setSpacing(0)
        split.replaceWidget(0, self.controls_panel)
        panel_box.addWidget(self.controls_scroll, 1)
        panel_box.addWidget(self.action_footer)
        self.noise_floor.setChecked(False)
        self.noise_floor.setEnabled(False)
        self.noise_floor.setText("DPD 对比不添加显示噪声")
        self.nfft.setValue(4096)
        self.plot_mode = QComboBox()
        self.plot_mode.addItems(["功率谱对比", "AM-AM 对比", "AM-PM 对比"])
        right = self.canvas.parentWidget().layout()
        right.insertWidget(0, self.plot_mode)
        self.plot_mode.currentIndexChanged.connect(self.draw_result)
        self.table = QTableWidget(8, 3)
        self.table.setHorizontalHeaderLabels(["指标（最终测试段）", "校正前", "校正后"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setFixedHeight(295)
        right.insertWidget(2, self.table)
        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        self.history.setMaximumHeight(85)
        right.insertWidget(3, self.history)
        self.kind.currentIndexChanged.connect(self.kind_changed)
        self.training_source.currentIndexChanged.connect(self.training_source_changed)
        self.limit_enabled.toggled.connect(self.peak_limit.setEnabled)
        for field in (self.memory, self.degree, self.cross, self.gain_db, self.gain_phase,
                      self.iterations, self.training_points, self.peak_limit,
                      self.main_bw, self.adj_bw, self.adj_offset):
            field.valueChanged.connect(self.invalidate)
        self.limit_enabled.toggled.connect(self.invalidate)
        self.bandwidth.currentTextChanged.connect(self.update_channels)
        self.update_channels(self.bandwidth.currentText())
        self.status.setText("导入功放模型，选择激励后训练 DPD；也可导入已有 DPD 验证。")

    def invalidate(self, *_):
        super().invalidate()
        if hasattr(self, "table"):
            self.table.clearContents()
            self.history.clear()
            self.reference = None
            self.evaluate_button.setEnabled(self.dpd is not None and self.run_button.isEnabled())

    def units_changed(self, *_):
        super().units_changed()
        self.noise_floor.setChecked(False)
        self.noise_floor.setEnabled(False)

    def kind_changed(self):
        self.cross.setEnabled(self.kind.currentData() == "gmp")
        self.invalidate()

    def training_source_changed(self):
        measured = self.training_source.currentIndex() == 1
        self.measured_button.setEnabled(measured)
        self.iterations.setEnabled(not measured)
        self.invalidate()

    def update_channels(self, text):
        for field in (self.main_bw, self.adj_bw, self.adj_offset):
            field.setValue(float(text))

    def set_model(self, model, label="当前已拟合功放", location=""):
        super().set_model(model, label, location)
        if not hasattr(self, "gain_db"):
            return
        linear = complex(np.sum(model.coefficients[:, 0]))
        db = model.fit_info.get("linearGainDb")
        self.gain_db.setValue(float(db) if db is not None else 20*np.log10(max(abs(linear), 1e-12)))
        self.gain_phase.setValue(float(np.angle(linear, deg=True)))
        self.gain_note.setText("幅度来自实测线性增益，相位来自模型直流线性响应；可修改。" if db is not None
                               else "目标取模型小信号直流增益；请按工作频带确认或修改。")
        limit = model.fit_info.get("inputPeak")
        self.limit_enabled.setChecked(limit is not None)
        if limit is not None:
            self.peak_limit.setValue(limit)
        if model.sample_rate is not None:
            from .pa_ofdm import PRESETS
            for bandwidth, preset in PRESETS.items():
                if np.isclose(parameters(bandwidth)["sampleRate"], model.sample_rate, rtol=1e-9):
                    self.bandwidth.setCurrentText(f"{bandwidth/1e6:g}")
                    self.oversampling.setValue(preset.oversampling)
                    break

    def use_current_model(self):
        model = getattr(self.window(), "fitted_model", None)
        if model is None:
            QMessageBox.information(self, "使用当前功放", "请先在实测数据页面完成拟合")
            return
        self.set_model(copy.deepcopy(model))

    def import_measured(self):
        path, _ = QFileDialog.getOpenFileName(self, "实测训练输入输出", "", "实测数据 (*.npz *.mat);;所有文件 (*)")
        if not path:
            return
        try:
            if Path(path).suffix.lower() == ".npz":
                with np.load(path, allow_pickle=False) as data:
                    u = np.array(data["u"] if "u" in data else data["x"]).ravel()
                    y = np.array(data["y"]).ravel()
                    rate = positive_rate(data["sampleRate"].item())
            else:
                from .pa_model import load_pa_data
                data = load_pa_data(path)
                u, y, rate = data.input_wave, data.output_wave, data.sample_rate
            self.set_measured_pair(u, y, rate, Path(path).name)
        except Exception as exc:
            QMessageBox.warning(self, "实测数据读取失败", str(exc))

    def set_measured_pair(self, u, y, rate, label="实测数据"):
        from .pa_simulation_core import waveform
        u, y = waveform(u), waveform(y)
        if len(u) != len(y):
            raise ValueError("实测输入输出长度必须一致")
        half = len(u)//2
        self.measured_pair = (u[:half].copy(), y[:half].copy())
        self.measured_rate = positive_rate(rate)
        self.measured_info.setText(f"{label}：前半段 {half:,} 点用于训练，采样率 {rate/1e6:g} MHz；自动对齐整点时延。")
        self.invalidate()

    def run_simulation(self):
        self.start_job(True)

    def configuration(self, train):
        if self.model is None:
            raise ValueError("请先导入功放模型")
        generated = self.source.currentIndex() == 1
        bandwidth = float(self.bandwidth.currentText())*1e6
        rate = parameters(bandwidth, self.oversampling.value())["sampleRate"] if generated else self.rate.value()*1e6
        check_rate(self.model, rate)
        if not generated and self.input_wave is None:
            raise ValueError("请先导入输入波形")
        measured = train and self.training_source.currentIndex() == 1
        if measured and self.measured_pair is None:
            raise ValueError("请先选择实测训练输入输出文件")
        if measured and not np.isclose(self.measured_rate, rate, rtol=1e-9):
            raise ValueError("实测训练数据与功放模型采样率不一致")
        if not generated and self.rate.value() == self.rate.minimum():
            raise ValueError("请填写输入信号采样率")
        gain = 10**(self.gain_db.value()/20)*np.exp(1j*np.deg2rad(self.gain_phase.value()))
        if not train:
            if self.dpd is None:
                raise ValueError("请先训练或导入 DPD")
            saved = complex(self.dpd.fit_info["gainReal"], self.dpd.fit_info["gainImag"])
            if not np.isclose(gain, saved, rtol=1e-5, atol=1e-8):
                raise ValueError("目标增益已改变，请重新训练 DPD 或恢复导入文件中的目标增益")
        return dict(pa=copy.deepcopy(self.model), dpd=copy.deepcopy(self.dpd), generated=generated,
                    bandwidth=bandwidth, oversampling=self.oversampling.value(), rate=rate,
                    wave=None if generated else self.input_wave.copy(), amplitude=10**(self.level.value()/20),
                    gain=gain, model_type=self.kind.currentData(), memory=self.memory.value(),
                    degree=self.degree.value(), cross_depth=self.cross.value(), iterations=self.iterations.value(),
                    count=self.training_points.value(), measured_pair=copy.deepcopy(self.measured_pair) if measured else None,
                    peak_limit=self.peak_limit.value() if self.limit_enabled.isChecked() else None,
                    main_bandwidth=self.main_bw.value()*1e6, adjacent_bandwidth=self.adj_bw.value()*1e6,
                    adjacent_offset=self.adj_offset.value()*1e6, nfft=self.nfft.value(),
                    reference_load=self.load.value())

    @staticmethod
    def compute(config, train, progress):
        c = config
        reference = None
        if c["generated"]:
            progress("Sionna 正在生成独立 OFDM 符号及参考栅格…")
            x, _, reference = generate_ofdm(c["bandwidth"], c["oversampling"], return_reference=True)
            reference["grid"] = reference["grid"]*c["amplitude"]
        else:
            x = c["wave"]
        x = x*c["amplitude"]
        (training, _), (validation, _), (test, reference) = partition_capture(x, reference)
        count = c["count"]
        # Validate measurement settings before spending time fitting.
        from .pa_dpd_core import channel_acpr
        channel_acpr(test, c["rate"], c["main_bandwidth"], c["adjacent_bandwidth"], c["adjacent_offset"], c["nfft"])
        dpd = c["dpd"]
        if train:
            fitted = train_ila(c["pa"], training[:count], validation[:count], c["rate"], c["gain"],
                model_type=c["model_type"], memory=c["memory"], degree=c["degree"],
                cross_depth=c["cross_depth"], iterations=c["iterations"],
                measured_pair=tuple(v[:count] for v in c["measured_pair"]) if c["measured_pair"] else None,
                peak_limit=c["peak_limit"], progress=progress)
            dpd = fitted.model
        progress("计算最终测试段的级联频谱、ACPR 与误差…")
        result = evaluate_dpd(c["pa"], dpd, test, c["rate"], c["gain"],
            main_bandwidth=c["main_bandwidth"], adjacent_bandwidth=c["adjacent_bandwidth"],
            adjacent_offset=c["adjacent_offset"], nfft=c["nfft"], reference_load=c["reference_load"],
            reference=reference, peak_limit=c["peak_limit"])
        result["trainingSourceSamples"] = len(training)
        result["selectionSourceSamples"] = len(validation)
        return dpd, result, reference

    def start_job(self, train):
        if self.busy:
            return
        try:
            config = self.configuration(train)
        except Exception as exc:
            QMessageBox.warning(self, "DPD 配置", str(exc))
            return
        self.invalidate()
        self.busy = True
        self.controls_scroll.setEnabled(False)
        self.action_footer.setEnabled(False)
        self.status.setText("正在训练与验证…" if train else "正在验证已有 DPD…")
        worker = _Worker(lambda progress: self.compute(config, train, progress))
        self.worker = worker
        _workers.add(worker)
        worker.progress.connect(self.status.setText)
        worker.completed.connect(self.job_completed)
        worker.failed.connect(self.job_failed)
        worker.finished.connect(self.job_finished)
        worker.finished.connect(lambda: _workers.discard(worker))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def job_finished(self):
        self.busy = False
        self.controls_scroll.setEnabled(True)
        self.action_footer.setEnabled(True)

    def job_failed(self, error):
        self.status.setText(f"DPD 失败：{error}")
        QMessageBox.warning(self, "DPD 失败", error)

    def describe_dpd(self):
        info = self.dpd.fit_info
        self.dpd_info.setText(f"{self.dpd.name} · 记忆 {self.dpd.memory} · {self.dpd.degree} 阶\n"
                              f"选模验证 RMS {info.get('validationRmsPercent', 0):.3f}%")
        self.export_dpd_button.setEnabled(True)
        self.evaluate_button.setEnabled(self.run_button.isEnabled())

    def job_completed(self, payload):
        self.dpd, self.result, self.reference = payload
        self.describe_dpd()
        self.draw_result()
        self.has_plot = True
        self.save_button.setEnabled(True)
        self.export_button.setEnabled(True)
        metrics = self.result["metrics"]
        rows = [("波形 RMS 误差 (%)", "rmsPercent"), ("NMSE (dB)", "nmseDb"),
                ("下邻道 ACPR (dBc)", "lowerAcprDb"), ("上邻道 ACPR (dBc)", "upperAcprDb"),
                ("OFDM EVM (%)", "evmPercent"), ("输出功率 (dBm)", "powerDbm"),
                ("AM-AM 残差 (%)", "amRmsPercent"), ("AM-PM 残差 (°)", "pmRmsDeg")]
        for row, (label, key) in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(label))
            for col, stage in enumerate(("before", "after"), 1):
                value = metrics[stage][key]
                self.table.setItem(row, col, QTableWidgetItem(f"{value:.4f}" if value is not None else "无符号参考"))
        lines = []
        for entry in self.dpd.fit_info.get("history", []):
            if "rmsPercent" in entry:
                lines.append(f"第 {entry['iteration']} 轮：验证 RMS {entry['rmsPercent']:.4f}% · "
                             f"峰值 {entry['peak']:.4f} · " + ("采用" if entry['accepted'] else "未采用（越界或未改善）"))
            else:
                lines.append(f"第 {entry['iteration']} 轮未采用：{entry.get('reason', '')}")
        self.history.setPlainText("\n".join(lines))
        drive = metrics["drive"]
        note = f"下/上邻道改善 {metrics['lowerAcprImprovementDb']:.2f} / {metrics['upperAcprImprovementDb']:.2f} dB"
        note += f" | DPD 输出峰值 {drive['peak']:.4f} · PAPR {drive['paprDb']:.2f} dB"
        if drive["outOfRangePercent"]:
            note += f" | {drive['outOfRangePercent']:.3f}% 样本超出辨识范围，不能确认补偿有效"
        elif metrics["after"]["rmsPercent"] >= metrics["before"]["rmsPercent"] * (1-1e-6):
            note += " | 最终测试未改善，建议调整回退、模型参数或目标增益"
        self.status.setText(note)

    def draw_result(self, *_):
        if self.result is None:
            return
        result = self.result
        valid = slice(result["validStart"], result["validStop"])
        curves = [(result["target"], "目标线性输出", "#888888"),
                  (result["y0"], "未校正输出", "#3B8FE0"),
                  (result["y1"], "DPD 校正后输出", "#E8C84C")]
        mode = self.plot_mode.currentIndex()
        if mode == 0:
            spectra = []
            for signal, label, color in curves:
                if self.units.currentIndex() == 0:
                    f, p = power_spectrum(signal[valid], result["sampleRate"], self.nfft.value(), self.load.value())
                    ylabel = "功率 (dBm)"
                else:
                    load = self.load.value() if self.units.currentIndex() == 2 else None
                    f, p, _ = spectrum_pair(signal, signal, result["sampleRate"], valid, self.nfft.value(), load)
                    f = f/1e6
                    ylabel = "功率谱密度 (dBm/Hz)" if load else "功率谱密度 (dB re 1 unit²/Hz)"
                spectra.append((p, label, color))
            draw_pa_spectrum(self.figure, f, spectra, "数字预失真：最终测试段功率谱对比", ylabel, None)
        else:
            x, target = result["x"][valid], result["target"][valid]
            mask = abs(x) > abs(x).max()*1e-4
            chosen = np.flatnonzero(mask)[::max(1, int(mask.sum()/5000))]
            pin = 10*np.log10(abs(x[chosen])**2/result["referenceLoad"]/1e-3)
            low, high = float(pin.min()), float(pin.max())
            if high-low < 1e-9:
                low, high = low-.5, high+.5
            draw_pa_spectrum(self.figure, np.array([low, high]), [],
                             "AM-AM：整体级联" if mode == 1 else "AM-PM：相对目标的相位残差",
                             "输出功率 (dBm)" if mode == 1 else "相位残差 (°)", None)
            ax = self.figure.axes[0]
            for signal, label, color in curves:
                actual = signal[valid][chosen]
                value = 10*np.log10(np.maximum(abs(actual)**2/result["referenceLoad"]/1e-3, 1e-30)) if mode == 1 else np.angle(actual*target[chosen].conj(), deg=True)
                ax.scatter(pin, value, s=3, alpha=.4, color=color, label=label)
            ax.set_xlabel("输入功率 (dBm)")
            legend = ax.legend(facecolor="#091B29", edgecolor="#555555")
            for text in legend.get_texts():
                text.set_color("#17384A")
        self.canvas.draw_idle()

    def import_dpd(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入 DPD 模型", "", "DPD 模型 (*.json *.npz *.csv *.txt)")
        if not path:
            return
        try:
            model = load_dpd(path)
            self.set_dpd_model(model)
        except Exception as exc:
            QMessageBox.warning(self, "DPD 导入失败", str(exc))

    def set_dpd_model(self, model):
        if self.busy or model.role != "predistorter":
            raise ValueError("请选择 DPD 模型并等待当前训练完成")
        gain = complex(model.fit_info["gainReal"], model.fit_info["gainImag"])
        self.memory.setMaximum(max(self.memory.maximum(), model.memory))
        self.degree.setMaximum(max(self.degree.maximum(), model.degree))
        self.cross.setMaximum(max(self.cross.maximum(), model.cross_depth))
        gain_db = float(20*np.log10(abs(gain)))
        self.gain_db.setRange(min(self.gain_db.minimum(), gain_db), max(self.gain_db.maximum(), gain_db))
        self.kind.setCurrentIndex(self.kind.findData(model.model_type))
        self.memory.setValue(model.memory)
        self.degree.setValue(model.degree)
        self.cross.setValue(model.cross_depth)
        self.gain_db.setValue(20*np.log10(abs(gain)))
        self.gain_phase.setValue(np.angle(gain, deg=True))
        self.dpd = model
        self.invalidate()
        self.describe_dpd()

    def export_dpd(self):
        if self.dpd is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 DPD 模型", "dpd_model.npz", "NumPy (*.npz);;CSV (*.csv);;文本 (*.txt)")
        if path:
            try:
                save_dpd(path, self.dpd)
                self.status.setText(f"DPD 已导出：{path}")
            except Exception as exc:
                QMessageBox.warning(self, "DPD 导出失败", str(exc))

    def export_result(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 DPD 波形与指标", "dpd_validation.npz", "NumPy (*.npz)")
        if not path:
            return
        try:
            payload = dict(self.result)
            payload["metricsJson"] = json.dumps(payload.pop("metrics"), ensure_ascii=False, allow_nan=False)
            payload.update(dpdCoefficients=self.dpd.coefficients, dpdType=self.dpd.model_type,
                           dpdMemory=self.dpd.memory, dpdDegree=self.dpd.degree, dpdCrossDepth=self.dpd.cross_depth,
                           dpdInfoJson=json.dumps(self.dpd.fit_info, ensure_ascii=False, allow_nan=False),
                           paCoefficients=self.model.coefficients, paType=self.model.model_type,
                           paMemory=self.model.memory, paDegree=self.model.degree, paCrossDepth=self.model.cross_depth)
            if self.reference is not None:
                payload.update({"ofdm_"+key: value for key, value in self.reference.items()})
            np.savez(path, **payload)
            self.status.setText(f"级联波形、模型和指标已导出：{path}")
        except Exception as exc:
            QMessageBox.warning(self, "DPD 波形导出失败", str(exc))
