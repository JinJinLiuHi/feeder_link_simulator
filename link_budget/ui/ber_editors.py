"""Platform-style parameter editors written for the BER module."""

import copy
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QComboBox,
    QCheckBox,
    QPushButton,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSplitter,
    QGroupBox,
    QAbstractItemView,
)
from link_budget.ui.styles import APP_STYLE, light_palette
from link_budget.algorithm.ber_core.catalog import apsk_definition
from link_budget.ui.numeric_inputs import integer, decimal
from link_budget.ui.apsk_canvas import APSKCanvas
from link_budget.ui.coding_editor import CodingEditor


class APSKEditor(QDialog):
    saved = Signal(object)
    PRESETS = {
        4: ([4], [1]),
        8: ([4, 4], [1, 2]),
        16: ([4, 12], [1, 2.85]),
        32: ([4, 12, 16], [1, 2.84, 5.27]),
        64: ([4, 12, 20, 28], [1, 2.7, 4.5, 6.4]),
        128: ([8, 16, 24, 36, 44], [1, 2.3, 3.8, 5.5, 7.2]),
        256: ([16, 32, 48, 64, 96], [1, 2.2, 3.6, 5.2, 7]),
    }

    def __init__(self, parent=None, definition=None):
        super().__init__(parent)
        self.setPalette(light_palette())
        self.setWindowTitle("修改 APSK" if definition else "新增 APSK")
        self.setStyleSheet(APP_STYLE)
        self.setMinimumSize(940, 620)
        self.resize(1180, 780)
        self._filling = False
        self.draft = None
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)
        title = QLabel(self.windowTitle())
        title.setStyleSheet("font-size:14pt;font-weight:600;")
        root.addWidget(title)
        subtitle = QLabel("修改参数或拖动星座点调整整环，确认后保存到调制库。")
        subtitle.setStyleSheet("color:#648A9C;")
        root.addWidget(subtitle)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)
        parameters = QGroupBox("星座参数")
        parameters.setMinimumWidth(400)
        parameters.setMaximumWidth(580)
        layout = QVBoxLayout(parameters)
        form = QFormLayout()
        self.order = QComboBox()
        self.order.addItems(map(str, self.PRESETS))
        form.addRow("星座阶数", self.order)
        self.ring_count = integer(2, 1, 16)
        reset = QPushButton("恢复默认")
        ring_row = QHBoxLayout()
        ring_row.addWidget(self.ring_count, 1)
        ring_row.addWidget(reset)
        form.addRow("环数", ring_row)
        layout.addLayout(form)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["点数", "半径比", "相位 (°)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setMinimumHeight(160)
        layout.addWidget(self.table, 1)
        self.validation = QLabel()
        self.validation.setWordWrap(True)
        layout.addWidget(self.validation)
        self.name_label = QLabel()
        self.name_label.setWordWrap(True)
        self.name_label.setStyleSheet("color:#91AFC2;font-size:9pt;")
        layout.addWidget(self.name_label)
        hint = QLabel(
            "调整环数会重新分配点数、半径和相位。\n内环半径固定为 1，相位从正 I 轴逆时针计量。\n环序采用 Gray 标记；平均符号能量归一化为 1。\n预置参数为研究起点，不代表 DVB 标准映射。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#648A9C;font-size:9pt;")
        layout.addWidget(hint)
        splitter.addWidget(parameters)
        preview = QGroupBox("实时星座")
        pv = QVBoxLayout(preview)
        options = QHBoxLayout()
        options.addWidget(QLabel("I / Q · 平均符号能量 Es = 1"))
        options.addStretch()
        self.show_bits = QCheckBox("显示比特")
        self.show_bits.setChecked(True)
        options.addWidget(self.show_bits)
        pv.addLayout(options)
        self.canvas = APSKCanvas()
        pv.addWidget(self.canvas, 1)
        instructions = QLabel("拖动点调整整环 · 指向局部滚轮缩放 · 双击复位")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setStyleSheet("color:#648A9C;font-size:9pt;")
        pv.addWidget(instructions)
        splitter.addWidget(preview)
        splitter.setSizes([450, 700])
        splitter.setStretchFactor(1, 1)
        buttons = QHBoxLayout()
        note = QLabel("仅保存在本次会话中。")
        note.setStyleSheet("color:#648A9C;font-size:9pt;")
        buttons.addWidget(note)
        buttons.addStretch()
        cancel = QPushButton("取消")
        self.save_button = QPushButton("保存修改" if definition else "加入调制库")
        self.save_button.setStyleSheet(
            "QPushButton{background:#00CFE8;border-color:#00CFE8;color:white;} QPushButton:disabled{background:#1C4961;border-color:#1C4961;}"
        )
        for button in (cancel, self.save_button):
            button.setAutoDefault(False)
            buttons.addWidget(button)
        root.addLayout(buttons)
        cancel.clicked.connect(self.reject)
        self.save_button.clicked.connect(self.save)
        self.order.currentTextChanged.connect(self.preset)
        self.ring_count.valueChanged.connect(self.change_ring_count)
        reset.clicked.connect(self.preset)
        self.show_bits.toggled.connect(self.canvas.set_bits_visible)
        self.canvas.ring_moved.connect(self.move_ring)
        self.canvas.ring_selected.connect(lambda row: self.table.setCurrentCell(row, 0))
        self.table.currentCellChanged.connect(self.select_ring)
        if definition:
            self.order.blockSignals(True)
            self.order.setCurrentText(str(1 << definition["bits"]))
            self.order.blockSignals(False)
            scale = definition["radii"][0]
            self.fill(
                definition["counts"],
                [radius / scale for radius in definition["radii"]],
                definition["phases"],
            )
        else:
            self.order.blockSignals(True)
            self.order.setCurrentText("16")
            self.order.blockSignals(False)
            self.preset()

    def add_ring(self, count=4, radius=None, phase=0):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setRowHeight(row, 48)
        controls = (
            integer(count, 1, 256),
            decimal(radius if radius is not None else row + 1, 0.001, 1000, 0.05),
            decimal(phase, -36000, 36000, 1),
        )
        for column, control in enumerate(controls):
            self.table.setCellWidget(row, column, control)
            control.valueChanged.connect(self.update_draft)
        if row == 0:
            controls[1].setValue(1)
            controls[1].setEnabled(False)
        self.table.setVerticalHeaderItem(row, QTableWidgetItem(f"环 {row+1}"))

    def fill(self, counts, radii, phases):
        self._filling = True
        self.table.setRowCount(0)
        for values in zip(counts, radii, phases):
            self.add_ring(*values)
        self.ring_count.setMaximum(min(16, int(self.order.currentText())))
        self.ring_count.setValue(len(counts))
        self._filling = False
        self.canvas.reset_view()
        self.update_draft()

    def preset(self, *args):
        counts, radii = self.PRESETS[int(self.order.currentText())]
        self.fill(counts, radii, [180 / count for count in counts])

    def change_ring_count(self, count):
        if self._filling:
            return
        order = int(self.order.currentText())
        denominator = count * (count + 1) // 2
        counts = [1 + (order - count) * (ring + 1) // denominator for ring in range(count)]
        for extra in range(order - sum(counts)):
            counts[count - 1 - extra % count] += 1
        self.fill(
            counts, [1 + 1.1 * ring for ring in range(count)], [180 / points for points in counts]
        )

    def definition(self):
        values = [
            [self.table.cellWidget(row, column).value() for column in range(3)]
            for row in range(self.table.rowCount())
        ]
        return apsk_definition(
            int(self.order.currentText()),
            [v[0] for v in values],
            [v[1] for v in values],
            [v[2] for v in values],
        )

    def update_draft(self, *args):
        if self._filling:
            return
        try:
            draft = self.definition()
            # Force the same label/normalization checks used by the actual modem.
            from link_budget.algorithm.ber_core.catalog import custom_points

            custom_points(draft)
            self.draft = draft
            self.canvas.set_definition(draft)
            self.save_button.setEnabled(True)
            self.validation.setText(
                f"{sum(draft['counts'])} 个星座点 · {len(draft['counts'])} 环 · 参数有效"
            )
            self.validation.setStyleSheet("color:#1D9E75;")
            self.name_label.setText(draft["name"])
        except ValueError as error:
            self.draft = None
            self.save_button.setEnabled(False)
            self.canvas.set_definition(None, str(error))
            self.validation.setText(str(error))
            self.validation.setStyleSheet("color:#D85A30;")
            self.name_label.clear()

    def select_ring(self, row, *args):
        if row >= 0:
            self.canvas.selected_ring = row
            self.canvas.update()

    def move_ring(self, row, radius, phase):
        if self.draft is None:
            return
        self._filling = True
        self.table.cellWidget(row, 1).setValue(radius)
        self.table.cellWidget(row, 2).setValue(phase)
        self._filling = False
        self.update_draft()

    def save(self):
        self.ring_count.interpretText()
        self._filling = True
        try:
            for row in range(self.table.rowCount()):
                for column in range(3):
                    self.table.cellWidget(row, column).interpretText()
        finally:
            self._filling = False
        self.update_draft()
        if self.draft is not None:
            self.saved.emit(copy.deepcopy(self.draft))
            self.accept()
