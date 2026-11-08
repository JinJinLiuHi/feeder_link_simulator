"""误码率分析：与馈电链路平台一致的参数、图表和统计界面。"""

import copy
import csv
import json
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QSplitter,
    QScrollArea,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QAbstractItemView,
)
from matplotlib import rcParams, font_manager
from pathlib import Path

font_path = Path("C:/Windows/Fonts/msyh.ttc")
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    rcParams["font.family"] = "Microsoft YaHei"
rcParams["axes.unicode_minus"] = False
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from link_budget.ui.base_dialog import ModuleDialog
from link_budget.ui.styles import APP_STYLE
from link_budget.ui.ber_editors import APSKEditor, CodingEditor, integer, decimal
from .ber_core.catalog import FAMILIES
from .ber_core.session import ProcessSession


class BERAnalysisDialog(ModuleDialog):
    TITLE = "误码率分析"
    MIN_WIDTH = 1060
    MIN_HEIGHT = 680
    ACCENT_COLOR = "#00CFE8"

    def build_content(self, layout):
        self.setStyleSheet(APP_STYLE + """
            QPushButton:disabled{color:#39728A;background:#07131F;border-color:#193F55;}
        """)
        self.codes = []
        self.runs = []
        self.active = None
        self.current_rows = []
        self.editors = []
        self.ready = False
        self.session = ProcessSession(self)
        self.session.message.connect(self.handle_event)
        self.session.ended.connect(self.finished_run)
        layout.setSpacing(12)
        toolbar = QHBoxLayout()
        heading = QLabel("误码率分析")
        heading.setStyleSheet("font-size:14pt;font-weight:600;color:#DFF8FF;")
        toolbar.addWidget(heading)
        description = QLabel("调制编码 · AWGN 信道")
        description.setStyleSheet("color:#648A9C;")
        toolbar.addWidget(description)
        toolbar.addStretch()
        self.export_button = QPushButton("导出数据")
        self.image_button = QPushButton("导出曲线")
        self.clear_button = QPushButton("清空结果")
        for button in (self.export_button, self.image_button, self.clear_button):
            toolbar.addWidget(button)
        self.export_button.clicked.connect(self.export_data)
        self.image_button.clicked.connect(self.export_image)
        self.clear_button.clicked.connect(self.clear_results)
        layout.addLayout(toolbar)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle{background:#193F55;}")
        layout.addWidget(splitter, 1)
        left = QWidget()
        left.setObjectName("berParameterContent")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 12, 0)
        lv.setSpacing(10)
        scroll = QScrollArea()
        scroll.setObjectName("berParameterScroll")
        scroll.viewport().setObjectName("berScrollViewport")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(left)
        left_panel = QWidget()
        left_panel.setObjectName("berParameterPanel")
        left_panel.setMinimumWidth(360)
        left_panel.setMaximumWidth(450)
        panel_layout = QVBoxLayout(left_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll, 1)
        splitter.addWidget(left_panel)
        modulation = QGroupBox("调制方式")
        mf = QFormLayout(modulation)
        self.modulation = QComboBox()
        mf.addRow("调制配置", self.modulation)
        actions = QHBoxLayout()
        self.add_mod = QPushButton("新增 APSK")
        self.edit_mod = QPushButton("修改")
        self.del_mod = QPushButton("删除")
        for button in (self.add_mod, self.edit_mod, self.del_mod):
            actions.addWidget(button)
        mf.addRow(actions)
        lv.addWidget(modulation)
        coding = QGroupBox("信道编码")
        cf = QFormLayout(coding)
        self.family = QComboBox()
        self.coding = QComboBox()
        cf.addRow("编码类别", self.family)
        cf.addRow("编码配置", self.coding)
        self.code_info = QLabel("正在加载本地编码库…")
        self.code_info.setWordWrap(True)
        self.code_info.setStyleSheet("color:#648A9C;font-size:9pt;")
        cf.addRow(self.code_info)
        ca = QHBoxLayout()
        self.add_code = QPushButton("新增编码")
        self.edit_code = QPushButton("修改")
        self.del_code = QPushButton("删除")
        for button in (self.add_code, self.edit_code, self.del_code):
            ca.addWidget(button)
        cf.addRow(ca)
        self.iterations = integer(20, 1, 200)
        cf.addRow("译码迭代", self.iterations)
        lv.addWidget(coding)
        sweep = QGroupBox("信噪比扫描")
        sf = QFormLayout(sweep)
        self.snr_start = decimal(0, -30, 60, 0.5)
        self.snr_stop = decimal(6, -30, 60, 0.5)
        self.snr_step = decimal(1, 0.001, 60, 0.1)
        for label, widget in [
            ("起始 Eb/N0 (dB)", self.snr_start),
            ("终止 Eb/N0 (dB)", self.snr_stop),
            ("步进 (dB)", self.snr_step),
        ]:
            sf.addRow(label, widget)
        lv.addWidget(sweep)
        limits = QGroupBox("统计与运行")
        lf = QFormLayout(limits)
        self.budget = integer(100000, 1, 2000000000)
        self.error_target = integer(200, 0, 2000000000)
        self.batch = integer(0, 0, 1024)
        self.batch.setSpecialValueText("自动")
        self.device = QComboBox()
        for label, key in [("自动选择", "auto"), ("CPU", "cpu"), ("CUDA 显卡", "cuda:0")]:
            self.device.addItem(label, key)
        for label, widget in [
            ("每点信息比特上限", self.budget),
            ("目标错误比特数", self.error_target),
            ("每批帧数上限", self.batch),
            ("计算设备", self.device),
        ]:
            lf.addRow(label, widget)
        note = QLabel("错误目标为 0 时仅按比特上限停止。\n按有效信息位统计，末帧补位不计入 BER。")
        note.setStyleSheet("color:#648A9C;font-size:9pt;")
        note.setWordWrap(True)
        lf.addRow(note)
        lv.addWidget(limits)
        for form in (mf, cf, sf, lf):
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        run = QHBoxLayout()
        self.start_button = QPushButton("开始仿真")
        self.start_button.setStyleSheet(
            "QPushButton{background:#00CFE8;border-color:#00CFE8;color:white;} QPushButton:disabled{background:#1C4961;border-color:#1C4961;}"
        )
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(False)
        run.addWidget(self.start_button, 1)
        run.addWidget(self.stop_button)
        panel_layout.addLayout(run)
        lv.addStretch()
        self.start_button.clicked.connect(self.start_run)
        self.stop_button.clicked.connect(self.stop_run)
        right = QWidget()
        right.setObjectName("berResults")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(12, 0, 0, 0)
        rv.setSpacing(10)
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane{border:1px solid #193F55;background:white;border-radius:6px;} QTabBar::tab{padding:9px 18px;color:#91AFC2;background:#103C50;} QTabBar::tab:selected{background:white;color:#185FA5;}"
        )
        self.figure = Figure(facecolor="#07131F", layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.tabs.addTab(self.canvas, "误码率曲线")
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            [
                "仿真配置",
                "Eb/N0 (dB)",
                "BER",
                "FER",
                "错误比特",
                "信息比特",
                "错误帧",
                "总帧数",
                "耗时 (s)",
                "停止原因",
            ]
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.table, "逐点统计")
        rv.addWidget(self.tabs, 1)
        curve_note = QLabel("每条曲线仅在首次零误码处显示三角（95% 上界），并与前一点连线。")
        curve_note.setStyleSheet("color:#648A9C;font-size:9pt;")
        rv.addWidget(curve_note)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setStyleSheet(
            "QProgressBar{border:1px solid #193F55;border-radius:5px;background:white;text-align:center;min-height:18px;} QProgressBar::chunk{background:#12617B;border-radius:4px;}"
        )
        rv.addWidget(self.progress)
        self.status = QLabel("正在读取本地编码配置…")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#91AFC2;font-size:9pt;")
        rv.addWidget(self.status)
        self.statistics = QLabel("每个信噪比点独立采样；零误码的统计值保留 BER=0。")
        self.statistics.setWordWrap(True)
        self.statistics.setStyleSheet("color:#648A9C;font-size:9pt;")
        rv.addWidget(self.statistics)
        splitter.addWidget(right)
        splitter.setSizes([390, 1000])
        self.modulation.currentIndexChanged.connect(self.mod_changed)
        self.family.currentIndexChanged.connect(self.family_changed)
        self.coding.currentIndexChanged.connect(self.code_changed)
        self.add_mod.clicked.connect(lambda: self.open_mod_editor())
        self.edit_mod.clicked.connect(lambda: self.open_mod_editor(True))
        self.del_mod.clicked.connect(self.remove_mod)
        self.add_code.clicked.connect(lambda: self.open_code_editor())
        self.edit_code.clicked.connect(lambda: self.open_code_editor(True))
        self.del_code.clicked.connect(self.remove_code)
        self.draw_curves()
        try:
            from .ber_core.catalog import session_catalog

            self.catalog_loaded(session_catalog())
        except Exception as error:
            self.catalog_failed(str(error))

    def catalog_loaded(self, catalog):
        self.codes = catalog["codes"]
        for mod in catalog["modulations"]:
            self.modulation.addItem(mod["name"], mod)
        self.modulation.setCurrentIndex(self.modulation.findText("QPSK"))
        self.populate_families("dvb")
        self.ready = True
        self.start_button.setEnabled(True)
        self.status.setText(
            f"就绪 · {sum(c['family']=='dvb' for c in self.codes)} 份 DVB-S2 矩阵 · {sum(c['family']=='nr' for c in self.codes)} 项 NR 配置"
        )
        self.mod_changed()

    def catalog_failed(self, message):
        self.status.setText("编码库加载失败：" + message)

    def populate_families(self, preferred=None):
        selected = preferred or self.family.currentData()
        self.family.blockSignals(True)
        self.family.clear()
        for key, label in FAMILIES.items():
            if any(c["family"] == key for c in self.codes):
                self.family.addItem(label, key)
        index = self.family.findData(selected)
        self.family.setCurrentIndex(index if index >= 0 else 0)
        self.family.blockSignals(False)
        self.family_changed()

    def family_changed(self):
        self.coding.blockSignals(True)
        self.coding.clear()
        for code in self.codes:
            if code["family"] == self.family.currentData():
                self.coding.addItem(code["name"], code)
        if self.family.currentData() == "dvb":
            for i in range(self.coding.count()):
                code = self.coding.itemData(i)
                if code.get("rate") == "1/2" and "short" in code["id"]:
                    self.coding.setCurrentIndex(i)
                    break
        self.coding.blockSignals(False)
        self.code_changed()

    def code_changed(self):
        code = self.coding.currentData()
        if not code:
            return
        if "k" in code:
            text = f"K = {code['k']} bit · N = {code['n']} bit\n实际码率 K/N = {code['k']/code['n']:.6f}"
        else:
            text = "实际 K/N 在准备编码器时由矩阵求秩确定。"
        if code["family"] == "dvb":
            text += "\n采用矩阵对应的 LDPC 编码，不含 BCH 外码。"
        self.code_info.setText(text)
        self.coding.setToolTip(code["name"])
        custom = code.get("custom", False)
        self.edit_code.setEnabled(custom)
        self.del_code.setEnabled(custom)
        self.coding.setEnabled(code["family"] != "none")
        self.iterations.setEnabled(code["family"] not in ("none", "conv"))
        if code["family"] == "turbo":
            self.iterations.setValue(6)

    def mod_changed(self):
        mod = self.modulation.currentData()
        custom = bool(mod and mod.get("custom"))
        self.edit_mod.setEnabled(custom)
        self.del_mod.setEnabled(custom)
        if mod:
            self.modulation.setToolTip(mod["name"])

    def open_mod_editor(self, edit=False):
        original = copy.deepcopy(self.modulation.currentData()) if edit else None
        dialog = APSKEditor(self, original)

        def save(mod):
            mod["custom"] = True
            if edit:
                self.modulation.setItemText(self.modulation.currentIndex(), mod["name"])
                self.modulation.setItemData(self.modulation.currentIndex(), mod)
            else:
                self.modulation.addItem(mod["name"], mod)
                self.modulation.setCurrentIndex(self.modulation.count() - 1)
            self.mod_changed()
            self.share_catalog()

        dialog.saved.connect(save)
        self.editors.append(dialog)
        dialog.showMaximized()

    def remove_mod(self):
        if self.modulation.currentData().get("custom"):
            self.modulation.removeItem(self.modulation.currentIndex())
            self.share_catalog()

    def share_catalog(self):
        from .ber_core.catalog import update_session_catalog
        update_session_catalog([self.modulation.itemData(i) for i in range(self.modulation.count())], self.codes)

    def open_code_editor(self, edit=False):
        original = copy.deepcopy(self.coding.currentData()) if edit else None
        dialog = CodingEditor(self, original)

        def save(code):
            if original:
                self.codes = [c for c in self.codes if c["id"] != original["id"]]
            if not any(c["id"] == code["id"] for c in self.codes):
                self.codes.append(code)
            self.populate_families(code["family"])
            for i in range(self.coding.count()):
                if self.coding.itemData(i)["id"] == code["id"]:
                    self.coding.setCurrentIndex(i)
                    break
            self.share_catalog()

        dialog.saved.connect(save)
        self.editors.append(dialog)
        dialog.showMaximized()

    def remove_code(self):
        code = self.coding.currentData()
        if code and code.get("custom"):
            self.codes = [c for c in self.codes if c["id"] != code["id"]]
            self.populate_families()
            self.share_catalog()

    def request(self):
        from .ber_core.simulation import snr_grid

        request = dict(
            mod=copy.deepcopy(self.modulation.currentData()),
            code=copy.deepcopy(self.coding.currentData()),
            start=self.snr_start.value(),
            stop=self.snr_stop.value(),
            step=self.snr_step.value(),
            budget=self.budget.value(),
            errors=self.error_target.value(),
            iterations=self.iterations.value(),
            batch=self.batch.value(),
            device=self.device.currentData(),
        )
        snr_grid(request)
        if request["mod"] is None or request["code"] is None:
            raise ValueError("请选择有效的调制和编码配置")
        return request

    def start_run(self):
        if not self.ready or self.session.process is not None:
            return
        try:
            request = self.request()
            self.active = dict(
                request=request,
                label=f"{request['mod']['name']} / {FAMILIES[request['code']['family']]} · {request['code']['name']}",
                rows=[],
            )
            self.session.start("simulate", request)
            self.runs.append(self.active)
            self.set_running(True)
            self.progress.setValue(0)
            self.status.setText("正在启动仿真…")
        except Exception as error:
            QMessageBox.warning(self, "无法开始仿真", str(error))

    def set_running(self, running):
        self.start_button.setEnabled(self.ready and not running)
        self.stop_button.setEnabled(running)
        for widget in (
            self.modulation,
            self.family,
            self.coding,
            self.iterations,
            self.snr_start,
            self.snr_stop,
            self.snr_step,
            self.budget,
            self.error_target,
            self.batch,
            self.device,
            self.add_mod,
            self.edit_mod,
            self.del_mod,
            self.add_code,
            self.edit_code,
            self.del_code,
            self.clear_button,
        ):
            widget.setEnabled(not running)
        if not running:
            self.code_changed()
            self.mod_changed()

    def stop_run(self):
        self.session.stop()
        self.stop_button.setEnabled(False)
        self.status.setText("正在停止，已完成的批次将保留…")

    def handle_event(self, event):
        kind = event["type"]
        if kind == "status":
            self.status.setText(event["message"])
        elif kind == "progress":
            completed = event["index"] + min(1, event["bits"] / self.active["request"]["budget"])
            self.progress.setValue(int(100 * completed / event["points"]))
            self.statistics.setText(
                f"Eb/N0 = {event['ebno']:g} dB　信息比特 {event['bits']:,}　错误比特 {event['bit_errors']:,}　错误帧 {event['frame_errors']:,}"
            )
        elif kind == "point":
            self.active["rows"].append(event["row"])
            self.add_table_row(self.active, event["row"])
            self.draw_curves()
        elif kind == "done":
            self.status.setText(
                "仿真已停止，已完成统计已保留。" if event["cancelled"] else "仿真完成。"
            )
            if not event["cancelled"]:
                self.progress.setValue(100)
        elif kind == "error":
            self.status.setText("仿真失败：" + event["message"])
            self.status.setToolTip(event.get("detail", event["message"]))

    def finished_run(self):
        self.set_running(False)

    def add_table_row(self, run, row):
        values = [
            run["label"],
            f"{row['ebno']:g}",
            f"{row['ber']:.4g}",
            f"{row['fer']:.4g}",
            row["bit_errors"],
            row["bits"],
            row["frame_errors"],
            row["frames"],
            f"{row['seconds']:.2f}",
            row["reason"],
        ]
        i = self.table.rowCount()
        self.table.insertRow(i)
        for j, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setToolTip(str(value))
            self.table.setItem(i, j, item)
        self.table.scrollToBottom()

    def draw_curves(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        colors = ["#22C7E6", "#3DDC97", "#4D9EFF", "#8BC6FF", "#F2C14E"]
        plotted = False
        displayed = []
        for i, run in enumerate(self.runs):
            rows = run["rows"]
            if not rows:
                continue
            first_zero = next((row for row in rows if row["bit_errors"] == 0), None)
            rows = [row for row in rows if row["bit_errors"] > 0 or row is first_zero]
            color = colors[i % len(colors)]
            plotted = True
            y = [r["ber"] if r["bit_errors"] else r["upper"] for r in rows]
            displayed.extend(y)
            ax.semilogy(
                [r["ebno"] for r in rows],
                y,
                "-",
                color=color,
                label=f"{i+1}. {run['label']}",
            )
            for zero, marker in ((False, "o"), (True, "v")):
                points = [r for r in rows if (r["bit_errors"] == 0) == zero]
                if points:
                    ax.semilogy(
                        [r["ebno"] for r in points],
                        [r["upper"] if zero else r["ber"] for r in points],
                        marker,
                        linestyle="None",
                        color=color,
                        markerfacecolor="white" if zero else color,
                        markersize=6 if zero else 4,
                    )
        ax.set_yscale("log")
        lower = min(1e-7, min(displayed) / 2) if displayed else 1e-7
        ax.set(xlabel="Eb/N0 (dB)", ylabel="BER", ylim=(max(1e-15, lower), 1))
        if not plotted:
            ax.set_xlim(
                self.snr_start.value(), max(self.snr_start.value() + 1, self.snr_stop.value())
            )
        ax.grid(True, which="both", alpha=0.15)
        if plotted:
            ax.legend(fontsize=8, loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                "当前结果未观测到误码，详情见逐点统计" if self.runs else "配置参数后开始仿真",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#648A9C",
                fontsize=12,
            )
        self.canvas.draw_idle()

    def clear_results(self):
        if self.session.process is not None:
            return
        self.runs = []
        self.active = None
        self.table.setRowCount(0)
        self.draw_curves()
        self.progress.setValue(0)
        self.status.setText("结果已清空。")

    def export_data(self):
        if not any(run["rows"] for run in self.runs):
            QMessageBox.information(self, "导出数据", "暂无可导出的仿真结果。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出误码统计", "误码率分析.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "配置",
                        "Eb/N0_dB",
                        "BER",
                        "FER",
                        "错误比特",
                        "有效信息比特",
                        "错误帧",
                        "总帧数",
                        "95%零误码上界",
                        "耗时_s",
                        "停止原因",
                        "参数JSON",
                    ]
                )
                for run in self.runs:
                    for r in run["rows"]:
                        writer.writerow(
                            [
                                run["label"],
                                r["ebno"],
                                r["ber"],
                                r["fer"],
                                r["bit_errors"],
                                r["bits"],
                                r["frame_errors"],
                                r["frames"],
                                r["upper"],
                                r["seconds"],
                                r["reason"],
                                json.dumps(run["request"], ensure_ascii=False),
                            ]
                        )
            self.status.setText("已导出误码统计：" + path)
        except OSError as error:
            QMessageBox.warning(self, "导出失败", str(error))

    def export_image(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出曲线", "误码率曲线.png", "PNG (*.png);;PDF (*.pdf)"
        )
        if path:
            try:
                self.figure.savefig(path, dpi=200)
            except OSError as error:
                QMessageBox.warning(self, "导出失败", str(error))

    def closeEvent(self, event):
        # Matplotlib's deferred timer must not redraw a canvas after its dialog is destroyed.
        self.canvas._draw_pending = False
        for editor in self.editors:
            editor.reject()
        self.session.dispose()
        super().closeEvent(event)
