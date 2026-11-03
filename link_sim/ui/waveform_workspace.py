"""NodeGraphQt transmitter canvas and per-node waveform observation."""

import copy
import os
import threading

import numpy as np
from scipy.signal import welch
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
    QPushButton, QSplitter, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import rcParams, font_manager

# Qt.py must select the same binding as the rest of this desktop application.
os.environ["QT_PREFERRED_BINDING"] = "PySide6"
try:
    from NodeGraphQt import BaseNode, NodeGraph
except ImportError as exc:
    raise ImportError("缺少 NodeGraphQt 画布依赖。请在启动本软件的 Python 环境执行："
                      "python -m pip install -r requirements-waveform.txt") from exc

from link_sim.algorithm.waveform_chain import (
    Stage, Wave, KINDS, DEVICE_KINDS, MAX_SAMPLES, default_settings, export_wave,
    import_source, load_chain, model_from_dict, model_to_dict, order_graph,
    run_graph, save_chain, source_wave,
)
from link_sim.algorithm.waveform_gnuradio import runtime_status
from link_sim.algorithm.device_models import read_model_file, json_text
from link_sim.ui.model_library import ModelLibraryDialog


class SignalNode(BaseNode):
    __identifier__ = "feeder.waveform"
    NODE_NAME = "模块"
    KIND = "gain"

    def __init__(self):
        super().__init__()
        self.create_property("settings", default_settings(self.KIND))
        self.create_property("device_model", None)
        if self.KIND != "source":
            self.add_input("I/Q 输入", color=(83, 191, 221))
        self.add_output("I/Q 输出", multi_output=True, color=(83, 222, 177))
        self.set_color(*(dict(source=(28, 112, 116), dac=(47, 91, 146),
                             filter=(52, 92, 142), mixer=(83, 74, 141), pa=(131, 75, 64))
                         .get(self.KIND, (55, 101, 115))))


class SourceNode(SignalNode):
    NODE_NAME, KIND = "基带信号源", "source"


class DACNode(SignalNode):
    NODE_NAME, KIND = "AD/DA", "dac"


class FilterNode(SignalNode):
    NODE_NAME, KIND = "滤波器", "filter"


class MixerNode(SignalNode):
    NODE_NAME, KIND = "上变频器", "mixer"


class PANode(SignalNode):
    NODE_NAME, KIND = "功率放大器", "pa"


class GainNode(SignalNode):
    NODE_NAME, KIND = "增益", "gain"


class FrequencyNode(SignalNode):
    NODE_NAME, KIND = "频率偏移", "frequency"


class NoiseNode(SignalNode):
    NODE_NAME, KIND = "加性白噪声", "awgn"


NODE_CLASSES = {cls.KIND: cls for cls in (
    SourceNode, DACNode, FilterNode, MixerNode, PANode, GainNode, FrequencyNode, NoiseNode)}


# key, label, divisor for display units, min, max, decimals
FIELDS = {
    "source": (("fs", "采样率 (MHz)", 1e6, .000001, 1e6, 6),
               ("count", "采样点数", 1, 256, MAX_SAMPLES, 0),
               ("sps", "每符号采样点", 1, 2, 256, 0),
               ("amplitude", "标称 RMS 幅度 (归一化)", 1, .000001, 1, 6),
               ("seed", "随机种子", 1, 0, 2**32-1, 0)),
    "dac": (),
    "filter": (("cutoff", "低通截止频率 (MHz)", 1e6, .000001, 1e6, 6),
               ("taps", "FIR 抽头数", 1, 3, 4095, 0)),
    "mixer": (("carrier_hz", "输出载频 (GHz，元数据)", 1e9, 0, 1e6, 6),
              ("gain_db", "变频增益 (dB)", 1, -100, 100, 3)),
    "pa": (("gain_db", "理想线性增益 (dB)", 1, -100, 100, 3),),
    "gain": (("gain_db", "增益 (dB)", 1, -100, 100, 3),),
    "frequency": (("offset_hz", "频偏 (MHz)", 1e6, -1e6, 1e6, 6),),
    "awgn": (("snr_db", "输入信噪比 (dB)", 1, -40, 200, 3),
             ("seed", "随机种子", 1, 0, 2**32-1, 0)),
}


class StageDialog(QDialog):
    def __init__(self, stage, parent=None):
        super().__init__(parent)
        self.stage = copy.deepcopy(stage)
        self.setWindowTitle("配置模块 · "+stage.name)
        self.resize(620, 590)
        root = QVBoxLayout(self)
        self.model_label = QLabel()
        self.model_label.setWordWrap(True)
        root.addWidget(self.model_label)
        buttons = QHBoxLayout()
        if stage.kind in DEVICE_KINDS:
            actions = [("从模型库加载…", self.choose_model), ("导入模型文件…", self.import_model)]
            if stage.kind != "dac":
                actions.append(("恢复内置模型", self.clear_model))
            for label, action in actions:
                button = QPushButton(label)
                button.clicked.connect(action)
                buttons.addWidget(button)
        elif stage.kind == "source":
            for label, action in (("导入已调制 I/Q…", self.import_iq), ("恢复测试信号", self.clear_iq)):
                button = QPushButton(label)
                button.clicked.connect(action)
                buttons.addWidget(button)
        root.addLayout(buttons)
        form = QFormLayout()
        self.fields = {}
        self.modulation = None
        if stage.kind == "source":
            self.modulation = QComboBox()
            self.modulation.addItems(["BPSK", "QPSK", "16QAM"])
            self.modulation.setCurrentText(stage.settings.get("modulation", "QPSK"))
            form.addRow("未编码测试调制", self.modulation)
        settings = dict(default_settings(stage.kind), **stage.settings)
        for key, label, scale, low, high, decimals in FIELDS[stage.kind]:
            edit = QDoubleSpinBox()
            edit.setDecimals(decimals)
            edit.setRange(low, high)
            edit.setValue(settings[key]/scale)
            edit.setGroupSeparatorShown(True)
            self.fields[key] = (edit, scale, decimals)
            form.addRow(label, edit)
        root.addLayout(form)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        root.addWidget(self.details, 1)
        hint = QLabel("等效噪声使用原模型的估计量化噪声基底，自动匹配链路采样率；不叠加杂散。" if stage.kind == "dac" else
                      "载频只记录射频位置，计算使用复包络。加载模型时以模型参数为准；灰色字段不可编辑。")
        hint.setWordWrap(True)
        root.addWidget(hint)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        box.button(QDialogButtonBox.StandardButton.Ok).setText("应用")
        box.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        root.addWidget(box)
        self.refresh_model()

    def refresh_model(self):
        stage = self.stage
        model = stage.model
        imported = stage.kind == "source" and "iq" in stage.settings
        if model:
            self.model_label.setText(f"已加载：{model.name or '未命名模型'}")
            arrays = "\n".join(f"{k}：{a.shape} {a.dtype}" for k, a in model.arrays.items())
            self.details.setPlainText(json_text(model.parameters)+"\n"+arrays)
        elif imported:
            self.model_label.setText(f"外部波形：{stage.settings.get('filename', '')} · {len(stage.settings['iq']):,} 点")
            self.details.setPlainText("保留文件 I/Q 幅度；文件中的 sampleRate 优先作为初始采样率。\n"
                                      "编码和调制已由外部信号源完成。本页不重新编码或归一化。")
        else:
            self.model_label.setText("未加载原 ADC/DAC 模型（理想直通）" if stage.kind == "dac" else
                                     "当前：内置测试信号" if stage.kind == "source" else "当前：内置模型")
            descriptions = {
                "source": "未编码随机符号，矩形脉冲。实际编码调制信号请通过 I/Q 文件导入。",
                "dac": "请从原 ADC/DAC 模型库加载器件模型。未加载时为理想直通。\n"
                       "读取模型的估计量化噪声基底，换算噪声功率；链路输出 = 链路输入 + 等效噪声。\n"
                       "自动匹配链路采样率，忽略杂散。原模型参数不在此修改。",
                "filter": "使用窗函数低通 FIR；加载库模型后使用其 b/a、SOS 或 RRC 系数。",
                "mixer": "理想变频＋增益；加载混频器模型后应用已保存的相噪锚点。暂不含本振泄漏。",
                "pa": "当前为理想线性增益。加载功放 MP/GMP 模型后使用辨识系数计算非线性与记忆效应。",
                "gain": "对应 GNU Radio Multiply Const 的复数增益处理。",
                "frequency": "对应信号源×复乘法的频移；请为频偏预留采样带宽。",
                "awgn": "按输入有效区间平均功率生成复高斯噪声；种子固定时可复现。",
            }
            self.details.setPlainText(descriptions[stage.kind])
        for key, (edit, _, _) in self.fields.items():
            disabled = (model is not None and stage.kind in ("filter", "pa") and key != "seed")
            disabled |= imported and key not in ("fs",)
            edit.setEnabled(not disabled)
        if stage.kind == "dac" and model:
            p = model.parameters
            from link_sim.algorithm.adc_noise_core import calibrate_model_noise
            estimate = calibrate_model_noise(model)
            self.details.setPlainText(f"原器件模型：{p['device_profile']}\n"
                                      f"量化位数：{p['n_bits']} bit；模型采样率：{p['fs']/1e6:g} MHz\n"
                                      f"校准基波：{p['f_sig']/1e6:g} MHz；FFT 点数：{p['nfft']}\n"
                                      f"模型 NSR：{p['noise_density_dbfs_hz']:g} dBFS/Hz\n\n"
                                      f"估计量化噪声基底：{estimate.floor_dbfs:.2f} dBFS/频点\n"
                                      f"FFT 频点宽度：{estimate.bin_width_hz:g} Hz\n"
                                      f"每 Hz 噪声功率密度 D_n：{estimate.density:.6e} 归一化功率/Hz\n"
                                      f"对应噪声密度：{estimate.density_dbfs_hz:.4f} dBFS/Hz\n\n"
                                      "使用模型噪声基底换算 Pn，生成链路 I/Q 等效噪声；不叠加杂散。")
        if self.modulation:
            self.modulation.setEnabled(not imported)

    def choose_model(self):
        try:
            dialog = ModelLibraryDialog(self, DEVICE_KINDS[self.stage.kind])
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.stage.model = dialog.selected_model.validated()
                self.stage.settings.pop("model_revision", None)
                self.refresh_model()
        except Exception as exc:
            QMessageBox.warning(self, "加载模型失败", str(exc))

    def import_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入器件模型", "", "器件模型 (*.json *.npz *.csv *.txt)")
        if path:
            try:
                self.stage.model = read_model_file(path, DEVICE_KINDS[self.stage.kind])
                self.stage.settings.pop("model_revision", None)
                self.refresh_model()
            except Exception as exc:
                QMessageBox.warning(self, "导入失败", str(exc))

    def clear_model(self):
        self.stage.model = None
        self.stage.settings.pop("model_revision", None)
        self.refresh_model()

    def import_iq(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入基带 I/Q 波形", "", "波形 (*.npy *.npz *.mat *.csv *.txt)")
        if path:
            try:
                imported = import_source(path, self.fields["fs"][0].value()*1e6)
                self.stage.settings.update(imported)
                self.fields["fs"][0].setValue(imported["fs"]/1e6)
                self.refresh_model()
            except Exception as exc:
                QMessageBox.warning(self, "波形导入失败", str(exc))

    def clear_iq(self):
        self.stage.settings.pop("iq", None)
        self.stage.settings.pop("filename", None)
        self.refresh_model()

    def accept(self):
        settings = dict(default_settings(self.stage.kind), **self.stage.settings)
        settings.pop("model_revision", None)
        if self.stage.kind == "dac":
            settings = {}
        for key, (edit, scale, decimals) in self.fields.items():
            value = edit.value()*scale
            settings[key] = int(value) if decimals == 0 else value
        if self.modulation:
            settings["modulation"] = self.modulation.currentText()
        try:
            if self.stage.kind == "source":
                source_wave(settings)
            self.stage.settings = settings
        except Exception as exc:
            QMessageBox.warning(self, "参数无效", str(exc))
            return
        super().accept()


class WorkerSignals(QObject):
    done = Signal(object, str)
    progress = Signal(int, int, str)


class ChainWorker(QRunnable):
    def __init__(self, stages, edges, backend):
        super().__init__()
        self.signals = WorkerSignals()
        self.stages, self.edges, self.backend = copy.deepcopy(stages), list(edges), backend
        self.cancel = threading.Event()

    def run(self):
        try:
            result = run_graph(self.stages, self.edges, self.backend,
                               self.signals.progress.emit, self.cancel.is_set)
            self.signals.done.emit(result, "")
        except Exception as exc:
            self.signals.done.emit(None, str(exc))


class WaveformWorkspace(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        fonts = {f.name for f in font_manager.fontManager.ttflist}
        for family in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
            if family in fonts:
                rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
                rcParams["axes.unicode_minus"] = False
                break
        self.setWindowTitle("全链路波形仿真 · 发射端")
        self.resize(1300, 860)
        self.outputs = {}
        self.worker = None
        self.rebuilding = False
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        heading = QLabel("全链路波形仿真  /  发射端")
        heading.setStyleSheet("font-size:18px;font-weight:600;color:#d5f3fa;padding:5px")
        root.addWidget(heading)
        self.edit_row = QWidget()
        controls = QHBoxLayout(self.edit_row)
        controls.setContentsMargins(0, 0, 0, 0)
        self.palette = QComboBox()
        for kind, label in KINDS.items():
            self.palette.addItem(label, kind)
        controls.addWidget(self.palette)
        self._button(controls, "添加模块", self.add_selected_kind)
        self._button(controls, "配置所选…", self.configure_selected)
        self._button(controls, "删除所选", self.delete_selected)
        self._button(controls, "默认发射链路", self.reset_chain)
        self._button(controls, "适应画布", self.fit_canvas)
        controls.addStretch()
        self._button(controls, "打开方案…", self.open_scheme)
        self._button(controls, "保存方案…", self.save_scheme)
        root.addWidget(self.edit_row)
        run_row = QHBoxLayout()
        self.backend = QComboBox()
        self.backend.addItem("本地计算 · 现有器件模型", "numpy")
        available, version = runtime_status()
        self.backend.addItem("GNU Radio 原生块＋现有器件模型" if available else "GNU Radio（当前环境未安装）", "gnuradio")
        self.backend.model().item(1).setEnabled(available)
        self.backend.setToolTip("GNU Radio 后端用于 FIR、增益、频偏；DAC、相噪、功放及 IIR 仍使用本项目模型。\n"+version)
        run_row.addWidget(self.backend)
        self.run_button = self._button(run_row, "▶ 运行链路", self.run_chain)
        self.cancel_button = self._button(run_row, "取消", self.cancel_run)
        self.cancel_button.setEnabled(False)
        self.status = QLabel("拖动端口连线 · 双击模块加载模型 · 单击模块查看输出")
        self.status.setWordWrap(True)
        run_row.addWidget(self.status, 1)
        root.addLayout(run_row)
        self.graph = NodeGraph()
        self.graph.register_nodes(list(NODE_CLASSES.values()))
        self.graph.set_acyclic(True)
        self.graph.set_background_color(8, 21, 34)
        self.graph.set_grid_color(21, 43, 60)
        self.graph.set_pipe_style(1)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.graph.widget.setMinimumHeight(240)
        splitter.addWidget(self.graph.widget)
        observer = QWidget()
        observer_layout = QVBoxLayout(observer)
        observer_layout.setContentsMargins(0, 5, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("观察点"))
        self.point = QComboBox()
        self.point.setMinimumWidth(260)
        row.addWidget(self.point)
        self.export_button = self._button(row, "导出该级 I/Q…", self.export_selected)
        self.export_button.setEnabled(False)
        self.summary = QLabel("运行后可查看每个模块的输出")
        self.summary.setWordWrap(True)
        row.addWidget(self.summary, 1)
        observer_layout.addLayout(row)
        self.figure = Figure(figsize=(12, 2.8), dpi=100, facecolor="#091b29")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(200)
        observer_layout.addWidget(self.canvas, 1)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color:#8eafc3;padding:3px")
        observer_layout.addWidget(self.note)
        splitter.addWidget(observer)
        splitter.setSizes([360, 330])
        root.addWidget(splitter, 1)
        hint = QLabel("复包络 I/Q · 幅度为归一化单位 · 上变频载频作为元数据 · 首期不含接收端。"
                      "内置源为未编码测试信号；已编码/调制信号可双击信号源导入。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#7896ab;font-size:11px")
        root.addWidget(hint)
        self.graph.node_double_clicked.connect(self.configure_node)
        self.graph.node_selected.connect(self.select_node)
        for sig in (self.graph.port_connected, self.graph.port_disconnected,
                    self.graph.nodes_deleted, self.graph.node_created):
            sig.connect(self.invalidate)
        self.graph.property_changed.connect(self.property_changed)
        self.point.currentIndexChanged.connect(self.draw_output)
        self.backend.currentIndexChanged.connect(self.invalidate)
        self.reset_chain()

    def property_changed(self, node, name, value):
        # Selection, layout and node colors must not discard computed waveforms.
        if name in ("settings", "device_model", "disabled"):
            self.invalidate()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.fit_canvas)

    @staticmethod
    def _button(layout, text, action):
        button = QPushButton(text)
        button.clicked.connect(action)
        layout.addWidget(button)
        return button

    def fit_canvas(self):
        if self.graph.all_nodes():
            self.graph.clear_selection()
            self.graph.select_all()
            self.graph.fit_to_selection()
            self.graph.clear_selection()

    def stage_from_node(self, node):
        return Stage(node.id, node.KIND, node.name(), copy.deepcopy(node.get_property("settings")),
                     model_from_dict(node.get_property("device_model")))

    def snapshot(self):
        if any(not isinstance(n, SignalNode) for n in self.graph.all_nodes()):
            raise ValueError("当前流程图包含不支持的节点，请删除非波形模块")
        stages = [self.stage_from_node(n) for n in self.graph.all_nodes()]
        edges = [(p.node().id, node.id) for node in self.graph.all_nodes()
                 for port in node.input_ports() for p in port.connected_ports()]
        if any(n.disabled() for n in self.graph.all_nodes()):
            raise ValueError("请恢复禁用的模块，或删除该模块后重新连线；当前版本不支持旁路")
        return stages, edges

    def create_stage(self, kind, pos=None, stage=None):
        cls = NODE_CLASSES[kind]
        node = self.graph.create_node(cls.type_, name=stage.name if stage else KINDS[kind], pos=pos,
                                      push_undo=False)
        if stage:
            node.set_property("settings", copy.deepcopy(stage.settings), push_undo=False)
            node.set_property("device_model", model_to_dict(stage.model), push_undo=False)
        self.update_node_caption(node)
        return node

    def update_node_caption(self, node):
        data = node.get_property("device_model")
        if data:
            text = f"{KINDS[node.KIND]}\n{data['name'] or '库模型'}"
        elif node.KIND == "source" and "iq" in node.get_property("settings"):
            text = "基带信号源\n外部 I/Q"
        else:
            text = KINDS[node.KIND]
        node.set_name(text)

    def add_selected_kind(self):
        if len(self.graph.all_nodes()) >= 64:
            QMessageBox.warning(self, "模块数量", "最多添加 64 个模块")
            return
        self.create_stage(self.palette.currentData(), [100, 100])
        self.invalidate()

    def delete_selected(self):
        self.graph.delete_nodes(self.graph.selected_nodes())
        self.invalidate()

    def configure_selected(self):
        selected = self.graph.selected_nodes()
        if selected:
            self.configure_node(selected[0])
        else:
            self.status.setText("请先单击选择一个模块，也可直接双击模块进行配置")

    def configure_node(self, node):
        if self.worker:
            return
        try:
            dialog = StageDialog(self.stage_from_node(node), self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                node.set_property("settings", dialog.stage.settings)
                node.set_property("device_model", model_to_dict(dialog.stage.model))
                self.update_node_caption(node)
                self.invalidate()
        except Exception as exc:
            QMessageBox.warning(self, "模块配置失败", str(exc))

    def reset_chain(self):
        if self.worker:
            return
        self.rebuilding = True
        self.graph.clear_session()
        previous = None
        for index, kind in enumerate(("source", "dac", "filter", "mixer", "pa")):
            node = self.create_stage(kind, [index*245, 30])
            if previous:
                previous.output(0).connect_to(node.input(0), push_undo=False)
            previous = node
        self.rebuilding = False
        self.invalidate()
        QTimer.singleShot(0, self.fit_canvas)

    def invalidate(self, *_):
        if self.rebuilding:
            return
        self.outputs = {}
        self.point.clear()
        self.export_button.setEnabled(False)
        self.summary.setText("参数或连接已更新，请运行链路")
        self.note.setText("")
        self.figure.clear()
        self.canvas.draw_idle()

    def select_node(self, node):
        index = self.point.findData(node.id)
        if index >= 0:
            self.point.setCurrentIndex(index)

    def run_chain(self):
        if self.worker:
            return
        try:
            stages, edges = self.snapshot()
            order_graph(stages, edges)
        except Exception as exc:
            QMessageBox.warning(self, "请检查链路", str(exc))
            return
        self.invalidate()
        self.worker = ChainWorker(stages, edges, self.backend.currentData())
        self.worker.signals.done.connect(self.run_finished)
        self.worker.signals.progress.connect(self.run_progress)
        self.set_busy(True)
        self.status.setText("正在计算…")
        QThreadPool.globalInstance().start(self.worker)

    def set_busy(self, busy):
        self.edit_row.setEnabled(not busy)
        self.graph.widget.setEnabled(not busy)
        self.backend.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def run_progress(self, index, total, name):
        self.status.setText(f"正在计算 {index}/{total} · {name.replace(chr(10), ' ')}")

    def cancel_run(self):
        if self.worker:
            self.worker.cancel.set()
            self.status.setText("正在取消，将在当前模块结束后停止…")
            self.cancel_button.setEnabled(False)

    def run_finished(self, outputs, error):
        cancelled = self.worker.cancel.is_set()
        self.worker = None
        self.set_busy(False)
        if cancelled:
            self.status.setText("仿真已取消")
            return
        if error:
            self.status.setText("仿真失败："+error)
            QMessageBox.warning(self, "仿真失败", error)
            return
        self.outputs = outputs
        for key in outputs:
            node = self.graph.get_node_by_id(key)
            self.point.addItem(node.name().replace("\n", " · ")+" → 输出", key)
            if outputs[key].added_error is not None:
                self.point.addItem(node.name().replace("\n", " · ")+" → 等效误差 n", key+"::error")
        self.point.setCurrentIndex(self.point.count()-1)
        self.export_button.setEnabled(bool(outputs))
        self.status.setText(f"完成 · {len(outputs)} 个观察点 · "+self.backend.currentText())
        self.draw_output()

    def observed_wave(self):
        key = self.point.currentData()
        if isinstance(key, str) and key.endswith("::error"):
            original = self.outputs.get(key[:-7])
            if original is not None and original.added_error is not None:
                return Wave(original.added_error, original.fs, original.carrier_hz,
                            original.start, original.stop, ("本级新增等效误差 n",)+original.notes)
        return self.outputs.get(key)

    def draw_output(self, *_):
        self.point.setToolTip(self.point.currentText())
        wave = self.observed_wave()
        if wave is None:
            return
        self.figure.clear()
        time_ax, spectrum_ax, iq_ax = self.figure.subplots(1, 3)
        x = wave.valid
        n = min(1024, len(x))
        t = (np.arange(n)+wave.start)/wave.fs*1e6
        time_ax.plot(t, x[:n].real, color="#47d7ee", linewidth=.8, label="I")
        time_ax.plot(t, x[:n].imag, color="#ffbb68", linewidth=.8, label="Q")
        time_ax.set(title="时域波形（有效区间）", xlabel="时间 (μs)", ylabel="归一化幅度")
        time_ax.legend(loc="upper right", fontsize=8)
        f, psd = welch(x, fs=wave.fs, nperseg=min(2048, len(x)), return_onesided=False, detrend=False)
        spectrum_ax.plot(np.fft.fftshift(f)/1e6, 10*np.log10(np.maximum(np.fft.fftshift(psd), 1e-30)),
                         color="#47d7ee", linewidth=.8)
        spectrum_ax.set(title="双边功率谱密度", xlabel="相对载频 (MHz)", ylabel="dB (归一化单位²/Hz)")
        stride = max(1, len(x)//4000)
        iq_ax.scatter(x[::stride].real, x[::stride].imag, s=3, c="#5ee0b2", alpha=.45)
        iq_ax.set(title="I/Q 散点（未做符号同步）", xlabel="I", ylabel="Q")
        iq_ax.set_aspect("equal", adjustable="datalim")
        for ax in (time_ax, spectrum_ax, iq_ax):
            ax.set_facecolor("#091b29")
            ax.tick_params(colors="#92b4c8", labelsize=8)
            ax.xaxis.label.set_color("#a6c9db")
            ax.yaxis.label.set_color("#a6c9db")
            ax.title.set_color("#d2eef7")
            for spine in ax.spines.values():
                spine.set_color("#28495f")
            ax.grid(alpha=.15, color="#6292a7")
        self.figure.tight_layout(pad=1.8)
        self.canvas.draw_idle()
        power = float(np.mean(abs(x)**2))
        peak = float(np.max(abs(x)**2))
        papr = 10*np.log10(peak/power) if power > 0 else 0.
        self.summary.setText(f"{wave.fs/1e6:g} MHz · {len(wave.x):,} 点 · RMS {np.sqrt(power):.4g} · "
                             f"PAPR {papr:.2f} dB · 载频 {wave.carrier_hz/1e9:g} GHz")
        self.note.setText(f"有效样本 [{wave.start}, {wave.stop})。"+"；".join(wave.notes))

    def export_selected(self):
        wave = self.observed_wave()
        if wave is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出模块输出", "waveform.npz", "I/Q 波形 (*.npz)")
        if path:
            try:
                export_wave(path if path.lower().endswith(".npz") else path+".npz", wave)
                self.status.setText("已导出 I/Q、采样率、载频和有效样本区间")
            except Exception as exc:
                QMessageBox.warning(self, "导出失败", str(exc))

    def save_scheme(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存全链路方案", "transmitter.chain.json", "链路方案 (*.json)")
        if not path:
            return
        try:
            stages, edges = self.snapshot()
            positions = {n.id: n.pos() for n in self.graph.all_nodes()}
            save_chain(path if path.lower().endswith(".json") else path+".json", stages, edges,
                       positions, self.backend.currentData())
            self.status.setText("方案已保存，包含模型系数快照和导入 I/Q，可独立重新打开")
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))

    def open_scheme(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开全链路方案", "", "链路方案 (*.json)")
        if not path:
            return
        try:
            stages, edges, positions, backend = load_chain(path)
            self.rebuilding = True
            self.graph.clear_session()
            nodes = {s.key: self.create_stage(s.kind, positions[s.key], s) for s in stages}
            for src, dst in edges:
                nodes[src].output(0).connect_to(nodes[dst].input(0), push_undo=False)
            if backend == "gnuradio" and not self.backend.model().item(1).isEnabled():
                self.backend.setCurrentIndex(0)
                message = "方案已加载；当前环境无 GNU Radio，已切换本地计算"
            else:
                self.backend.setCurrentIndex(self.backend.findData(backend))
                message = "方案已加载，请运行链路"
            self.status.setText(message)
        except Exception as exc:
            QMessageBox.warning(self, "打开方案失败", str(exc))
        finally:
            self.rebuilding = False
            self.invalidate()
            self.fit_canvas()

    def closeEvent(self, event):
        if self.worker:
            self.worker.cancel.set()
        super().closeEvent(event)
