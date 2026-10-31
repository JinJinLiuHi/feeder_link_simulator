"""Independent model-import → waveform → PA output spectrum workspace."""

from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QComboBox, QSplitter, QScrollArea,
    QFileDialog, QMessageBox, QLineEdit, QCheckBox,
)
from .pa_simulation_core import load_model, load_waveform, simulate, spectrum_pair, power_spectrum
from .pa_plot import draw_pa_spectrum
from .pa_ofdm import PRESETS, parameters as ofdm_parameters, generate_ofdm
from link_sim.ui.numeric_inputs import IntegerInput, DecimalInput


class PASimulationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.input_wave = None
        self.input_rate = None
        self.result = None
        self.has_plot = False
        from .pa_model import _group, _ES, PlotCanvas
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(1)
        split.setStyleSheet("QSplitter::handle{background:#17384A;}")
        root.addWidget(split)
        left = QWidget()
        left.setMinimumWidth(310)
        left.setMaximumWidth(390)
        left.setStyleSheet(
            "QComboBox,QSpinBox,QDoubleSpinBox{background:#091B29;border:1px solid #1C4961;"
            "border-radius:3px;padding:3px 6px;font-size:10pt;color:#D9ECF7;}"
            "QComboBox:disabled,QSpinBox:disabled,QDoubleSpinBox:disabled,QLineEdit:disabled"
            "{background:#0D3042;color:#648A9C;border-color:#1C4961;}"
            "QLabel{font-size:9pt;color:#C9E6F2;} QLabel:disabled{color:#648A9C;}"
            "QPushButton{background:#091B29;color:#C9E6F2;border:1px solid #CCC;"
            "border-radius:5px;font-size:9.5pt;}"
            "QPushButton:hover{background:#0D3042;}"
            "QPushButton:disabled{color:#91AFC2;border-color:#DDD;background:#0D3042;}"
            "QComboBox QAbstractItemView{background:#091B29;color:#D9ECF7;selection-background-color:#12617B;}")
        controls = QVBoxLayout(left)
        controls.setContentsMargins(0, 0, 8, 0)
        controls.setSpacing(7)

        group = _group("① 导入功放模型")
        box = group.layout()
        model_row = QHBoxLayout()
        self.model_file = QLineEdit()
        self.model_file.setReadOnly(True)
        self.model_file.setStyleSheet(_ES)
        self.model_file.setPlaceholderText("选择导出的功放模型")
        button = QPushButton("浏览…")
        button.setFixedWidth(64)
        button.setFixedHeight(28)
        button.clicked.connect(self.import_model)
        model_row.addWidget(self.model_file)
        model_row.addWidget(button)
        library_button = QPushButton("模型库…")
        library_button.setFixedSize(70, 28)
        library_button.clicked.connect(self.import_library_model)
        model_row.addWidget(library_button)
        box.addLayout(model_row)
        self.model_info = QLabel("未导入模型（模型库 / JSON / NPZ / CSV / TXT）")
        self.model_info.setWordWrap(True)
        self.model_info.setStyleSheet("font-size:8.5pt;color:#648A9C;")
        box.addWidget(self.model_info)
        controls.addWidget(group)

        group = _group("② 输入信号")
        box = group.layout()
        self.source = QComboBox()
        self.source.addItems(["导入复基带信号", "生成 OFDM 测试波形"])
        source_form = QFormLayout()
        source_form.setContentsMargins(0, 0, 0, 0)
        source_form.addRow("x[n]：", self.source)
        box.addLayout(source_form)
        self.import_fields = QWidget()
        import_box = QVBoxLayout(self.import_fields)
        import_box.setContentsMargins(0, 0, 0, 0)
        file_row = QHBoxLayout()
        self.wave_file = QLineEdit()
        self.wave_file.setReadOnly(True)
        self.wave_file.setPlaceholderText("选择输入波形文件")
        self.wave_button = QPushButton("浏览…")
        self.wave_button.setFixedWidth(64)
        self.wave_button.setFixedHeight(28)
        self.wave_button.clicked.connect(self.import_wave)
        file_row.addWidget(self.wave_file)
        file_row.addWidget(self.wave_button)
        import_box.addLayout(file_row)
        self.wave_info = QLabel("支持 I/Q 文件和实测数据中的 InputWaveform")
        self.wave_info.setWordWrap(True)
        import_box.addWidget(self.wave_info)
        box.addWidget(self.import_fields)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self.bandwidth = QComboBox()
        self.bandwidth.addItems(["5", "15", "40", "50", "100"])
        self.bandwidth.setCurrentText("50")
        self.bandwidth.setEnabled(False)
        self.bandwidth_label = QLabel("OFDM 带宽 (MHz)")
        self.bandwidth_label.setEnabled(False)
        form.addRow(self.bandwidth_label, self.bandwidth)
        self.oversampling = IntegerInput()
        self.oversampling.setRange(1, 32)
        self.oversampling.setValue(1)
        self.oversampling.setEnabled(False)
        self.oversampling_label = QLabel("OFDM 过采样倍数")
        self.oversampling_label.setEnabled(False)
        form.addRow(self.oversampling_label, self.oversampling)
        self.ofdm_info = QLabel()
        self.ofdm_info.setWordWrap(True)
        self.ofdm_info.setStyleSheet("font-size:8.5pt;color:#648A9C;")
        self.ofdm_info.setVisible(False)
        form.addRow(self.ofdm_info)
        self.rate = DecimalInput()
        self.rate.setDecimals(9)
        self.rate.setRange(0.000000001, 1000000)
        self.rate.setValue(0.000000001)
        self.rate.setSpecialValueText("请填写")
        form.addRow("采样率 (MHz)", self.rate)
        self.level = DecimalInput()
        self.level.setRange(-120, 60)
        self.level.setDecimals(2)
        self.level.setSuffix(" dB")
        form.addRow("输入幅度调整", self.level)
        box.addLayout(form)
        self.scale_note = QLabel("0 dB 保留文件幅度。对比建模页时，请导入同一实测输入。")
        self.scale_note.setWordWrap(True)
        self.scale_note.setStyleSheet("font-size:8.5pt;color:#648A9C;")
        box.addWidget(self.scale_note)
        controls.addWidget(group)

        group = _group("③ 频谱设置")
        form = QFormLayout()
        group.layout().addLayout(form)
        self.nfft = IntegerInput()
        self.nfft.setRange(64, 65536)
        self.nfft.setValue(1024)
        self.nfft.setSingleStep(256)
        form.addRow("FFT 长度", self.nfft)
        self.units = QComboBox()
        self.units.addItems(["功率 dBm（同建模页）", "相对功率谱密度", "功率谱密度 dBm/Hz"])
        form.addRow("纵轴单位", self.units)
        self.load = DecimalInput()
        self.load.setRange(0.001, 1000000)
        self.load.setValue(100)
        self.load.setSuffix(" Ω")
        form.addRow("参考阻抗", self.load)
        self.noise_floor = QCheckBox("显示测量噪声底 −77 dBm（仅绘图）")
        self.noise_floor.setChecked(True)
        form.addRow(self.noise_floor)
        note = QLabel("默认与建模页一致：1024 点、100 Ω。\n电压标度约定 P = mean(|x|²) / R。")
        note.setWordWrap(True)
        note.setStyleSheet("font-size:8.5pt;color:#648A9C;")
        form.addRow(note)
        controls.addWidget(group)
        self.preview_button = QPushButton("查看输入功率谱")
        self.preview_button.setFixedHeight(30)
        self.preview_button.setEnabled(False)
        self.preview_button.clicked.connect(self.preview_input)
        controls.addWidget(self.preview_button)
        self.run_button = QPushButton("运行功放并绘制频谱")
        self.run_button.setFixedHeight(32)
        self.run_button.setEnabled(False)
        self.run_button.setStyleSheet("QPushButton{background:#0D6078;color:#FFF;border:1px solid #22C7E6;border-radius:5px;font-size:10pt;font-weight:bold;} QPushButton:hover{background:#168BA5;} QPushButton:disabled{background:#20333E;color:#648A9C;border-color:#1C4961;}")
        self.run_button.clicked.connect(self.run_simulation)
        controls.addWidget(self.run_button)
        actions = QHBoxLayout()
        self.export_button = QPushButton("导出 x[n]、y[n]")
        self.export_button.setFixedHeight(30)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_result)
        actions.addWidget(self.export_button)
        self.save_button = QPushButton("保存谱图")
        self.save_button.setFixedHeight(30)
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_plot)
        actions.addWidget(self.save_button)
        controls.addLayout(actions)
        controls.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setFixedWidth(390)
        self.controls_scroll = scroll
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        split.addWidget(scroll)

        right = QWidget()
        box = QVBoxLayout(right)
        box.setContentsMargins(6, 0, 0, 0)
        box.setSpacing(4)
        self.canvas = PlotCanvas()
        self.figure = self.canvas.fig
        box.addWidget(self.canvas, 1)
        self.status = QLabel("导入模型和输入信号后运行，无需加载实测输出。")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-size:9pt;color:#648A9C;")
        box.addWidget(self.status)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        self.source.currentIndexChanged.connect(self.source_changed)
        self.bandwidth.currentIndexChanged.connect(self.preset_changed)
        self.oversampling.valueChanged.connect(self.source_changed)
        self.units.currentIndexChanged.connect(self.units_changed)
        self.noise_floor.toggled.connect(self.invalidate)
        for field in (self.rate, self.level, self.nfft, self.load):
            field.valueChanged.connect(self.invalidate)
        self.empty_plot()

    def empty_plot(self):
        self.figure.clear()
        self.figure.patch.set_facecolor("#07131F")
        ax = self.figure.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "导入功放模型 → 输入 x[n] → 查看输入与输出频谱",
                ha="center", va="center", transform=ax.transAxes, color="#777", fontsize=12)
        self.canvas.draw_idle()

    def invalidate(self, *_):
        self.result = None
        self.has_plot = False
        self.export_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.run_button.setEnabled(self.model is not None and (self.source.currentIndex() == 1 or self.input_wave is not None))
        self.preview_button.setEnabled(self.source.currentIndex() == 1 or self.input_wave is not None)
        self.status.setText("配置已更新，请运行功放生成频谱。")
        self.empty_plot()

    def source_changed(self, *_):
        generated = self.source.currentIndex() == 1
        self.import_fields.setEnabled(not generated)
        self.bandwidth.setEnabled(generated)
        self.bandwidth_label.setEnabled(generated)
        self.oversampling.setEnabled(generated)
        self.oversampling_label.setEnabled(generated)
        self.ofdm_info.setVisible(generated)
        self.rate.setEnabled(not generated and self.input_rate is None)
        self.scale_note.setText("OFDM 初始峰值为 1，无加窗、无滤波。过采样改变采样率，不改变子载波间隔。" if generated
                                else "0 dB 保留文件幅度。对比建模页时，请导入同一实测输入。")
        if generated:
            info = ofdm_parameters(float(self.bandwidth.currentText()) * 1e6, self.oversampling.value())
            self.rate.setValue(info["sampleRate"] / 1e6)
            cp = "正常 CP，10 子帧" if info["cpMode"] == "normal" else f"固定 CP，{info['numSymbols']} 符号"
            self.ofdm_info.setText(f"Sionna · 64-QAM · {info['subcarrierSpacingHz']/1e3:g} kHz\n"
                                   f"{info['activeCarriers']//12} RB · {cp}\n"
                                   f"资源网格宽度 {info['occupiedGridHz']/1e6:g} MHz · FFT {info['fftSize']}")
        elif self.input_rate is not None:
            self.rate.setValue(self.input_rate / 1e6)
        elif self.model is not None and self.model.sample_rate is not None:
            self.rate.setValue(self.model.sample_rate / 1e6)
        self.invalidate()

    def preset_changed(self, *_):
        bandwidth = float(self.bandwidth.currentText()) * 1e6
        self.oversampling.setValue(PRESETS[bandwidth].oversampling)
        self.source_changed()

    def input_signal(self):
        metadata = {}
        if self.source.currentIndex() == 1:
            bandwidth = float(self.bandwidth.currentText()) * 1e6
            metadata = ofdm_parameters(bandwidth, self.oversampling.value())
            self.status.setText("Sionna 正在生成 OFDM 信号…")
            self.status.repaint()
            x, rate, _ = generate_ofdm(bandwidth, self.oversampling.value())
        else:
            if self.input_wave is None:
                raise ValueError("请先导入输入波形")
            if self.rate.value() == self.rate.minimum():
                raise ValueError("请填写输入信号采样率")
            x, rate = self.input_wave, self.rate.value() * 1e6
        return x * 10 ** (self.level.value() / 20), rate, metadata

    def preview_input(self):
        self.invalidate()
        try:
            x, rate, _ = self.input_signal()
            load = self.load.value() if self.units.currentIndex() != 1 else None
            if self.units.currentIndex() == 0:
                floor = -77.0 if self.noise_floor.isChecked() else None
                freq, power = power_spectrum(x, rate, self.nfft.value(), load, floor)
                ylabel, limits = "功率 (dBm)", None
            else:
                frequency, power, _ = spectrum_pair(x, x, rate, slice(0, len(x)), self.nfft.value(), load)
                freq = frequency / 1e6
                ylabel = "功率谱密度 (dBm/Hz)" if load else "功率谱密度 (dB re 1 unit²/Hz)"
                limits = None
            draw_pa_spectrum(self.figure, freq, [(power, "输入功率谱", "#E8C84C")],
                             "输入信号功率谱", ylabel, limits)
            self.canvas.draw()
            self.has_plot = True
            self.save_button.setEnabled(True)
            self.status.setText(f"输入功率谱：{len(x):,} 点，采样率 {rate/1e6:g} MHz；未经过功放。")
        except Exception as exc:
            self.status.setText(f"输入信号生成失败：{exc}")
            QMessageBox.warning(self, "输入信号生成失败", str(exc))

    def units_changed(self, *_):
        self.load.setEnabled(self.units.currentIndex() != 1)
        self.noise_floor.setEnabled(self.units.currentIndex() == 0)
        self.invalidate()

    def set_model_file(self, path):
        self.set_model(load_model(path), Path(path).name, str(path))

    def set_model(self, model, label="当前已拟合功放", location=""):
        self.model = model
        self.model_file.setText(label)
        self.model_file.setToolTip(location)
        rate = f"{model.sample_rate / 1e6:g} MHz" if model.sample_rate else "旧文件未记录，请确认输入采样率"
        cross = f"\n交叉包络偏移：±1…±{model.cross_depth} 点" if model.edge else ""
        info = model.fit_info
        diagnostic = "\n未记录辨识诊断"
        if info.get("solver") == "column_scaled_tsvd":
            diagnostic = f"\n稳定辨识：有效方向 {info['rank']} / {info['terms']}"
            if "validationRmsPercent" in info:
                diagnostic += f" · 验证误差 {info['validationRmsPercent']:.3f}%"
        self.model_info.setText(f"{model.name}  |  记忆 {model.memory} 点  |  {model.degree} 阶\n采样率：{rate}{cross}{diagnostic}")
        self.model_info.setStyleSheet("font-size:8.5pt;color:#3A7D44;")
        self.source_changed()

    def import_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入功放模型", "", "功放模型 (*.json *.npz *.csv *.txt)")
        if path:
            try:
                self.set_model_file(path)
            except Exception as exc:
                QMessageBox.warning(self, "模型导入失败", str(exc))

    def import_library_model(self):
        try:
            if getattr(self, "busy", False):
                raise ValueError("请等待当前训练完成")
            from link_sim.ui.model_library import ModelLibraryDialog
            from .device_models import to_pa
            dialog = ModelLibraryDialog(self, "pa")
            if dialog.exec() == dialog.DialogCode.Accepted:
                self.set_model(to_pa(dialog.selected_model), dialog.selected_model.name, "SQLite 模型库")
        except Exception as exc:
            QMessageBox.warning(self, "模型加载失败", str(exc))

    def set_wave_file(self, path):
        x, rate = load_waveform(path)
        self.input_wave, self.input_rate = x, rate
        self.wave_file.setText(Path(path).name)
        self.wave_file.setToolTip(str(path))
        self.wave_info.setText(f"{x.size:,} 点复基带信号" +
                               (f"，{rate / 1e6:g} MHz" if rate else "；请确认采样率"))
        self.source_changed()

    def import_wave(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入输入波形 x[n]", "", "复基带波形 (*.npz *.npy *.csv *.txt *.mat);;所有文件 (*)")
        if path:
            try:
                self.set_wave_file(path)
            except Exception as exc:
                QMessageBox.warning(self, "波形导入失败", str(exc))

    def run_simulation(self):
        self.invalidate()
        try:
            if self.model is None:
                raise ValueError("请先导入功放模型")
            if self.source.currentIndex() == 1 and self.model.sample_rate is not None:
                requested = ofdm_parameters(float(self.bandwidth.currentText()) * 1e6, self.oversampling.value())["sampleRate"]
                if not np.isclose(requested, self.model.sample_rate, rtol=1e-9, atol=1e-6):
                    raise ValueError(f"OFDM 采样率 {requested/1e6:g} MHz 与模型 {self.model.sample_rate/1e6:g} MHz 不一致，请调整过采样倍数；也可先查看输入功率谱")
            x, rate, metadata = self.input_signal()
            x, y, valid = simulate(self.model, x, rate)
            load = self.load.value() if self.units.currentIndex() != 1 else None
            floor = None
            if self.units.currentIndex() == 0:
                floor = -77.0 if self.noise_floor.isChecked() else None
                freq_mhz, px = power_spectrum(x[valid], rate, self.nfft.value(), load, floor)
                _, py = power_spectrum(y[valid], rate, self.nfft.value(), load, floor)
                freq = freq_mhz * 1e6
                ylabel = "功率 (dBm)"
                limits = (-80, max(30, max(px.max(), py.max()) + 3))
            else:
                freq, px, py = spectrum_pair(x, y, rate, valid, self.nfft.value(), load)
                ylabel = "功率谱密度 (dBm/Hz)" if load else "功率谱密度 (dB re 1 unit²/Hz)"
                limits = None
            draw_pa_spectrum(
                self.figure, freq / 1e6,
                [(px, "输入功率谱", "#3B8FE0"), (py, f"输出功率谱（{self.model.name}）", "#E8C84C")],
                f"功放输入与输出功率谱对比 — {self.model.name}",
                ylabel=ylabel, limits=limits)
            self.canvas.draw()
            self.result = dict(x=x.copy(), y=y.copy(), sampleRate=rate,
                               validStart=valid.start, validStop=valid.stop,
                               frequencyHz=freq, inputSpectrum=px, outputSpectrum=py,
                               spectrumUnit=ylabel, spectrumMode="power" if self.units.currentIndex() == 0 else "density",
                               displayNoiseFloorDbm=floor if floor is not None else np.nan,
                               nfft=min(self.nfft.value(), valid.stop - valid.start),
                               referenceLoad=load if load else np.nan,
                               inputAdjustmentDb=self.level.value(), modType=self.model.model_type,
                               fitCoefMatMem=self.model.coefficients.copy(),
                               memLen=self.model.memory, degLen=self.model.degree,
                               crossDepth=self.model.cross_depth)
            self.result.update({f"ofdm_{key}": value for key, value in metadata.items()})
            self.has_plot = True
            self.export_button.setEnabled(True)
            self.save_button.setEnabled(True)
            note = " | 测量噪声底 −77 dBm（仅绘图）" if floor is not None else ""
            self.status.setText(f"完成：{len(x):,} 点，采样率 {rate / 1e6:g} MHz | 输入与输出功率谱{note}")
        except Exception as exc:
            self.status.setText(f"仿真失败：{exc}")
            QMessageBox.warning(self, "仿真失败", str(exc))

    def export_result(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出功放仿真波形", "pa_simulation.npz", "NumPy 文件 (*.npz)")
        if path:
            try:
                if not path.lower().endswith(".npz"):
                    path += ".npz"
                np.savez(path, **self.result)
                self.status.setText(f"已导出输入、输出、采样率和频谱：{path}")
            except Exception as exc:
                QMessageBox.warning(self, "导出失败", str(exc))

    def save_plot(self):
        if not self.has_plot:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存输入输出谱图", "pa_input_output.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            try:
                self.figure.savefig(path, dpi=150)
                self.status.setText(f"已保存谱图：{path}")
            except Exception as exc:
                QMessageBox.warning(self, "保存失败", str(exc))
