"""Editable per-business-frame modulation/coding table; no compute runtime loading."""

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QHeaderView, QLabel, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QSizePolicy,
)

from link_budget.algorithm.ber_core.catalog import FAMILIES, session_catalog
from link_sim.algorithm.frame_config import default_business_frame, frame_plan, frame_summary


class FrameSourceEditor(QWidget):
    def __init__(self, settings, fields, parent=None):
        super().__init__(parent)
        self.fields = fields
        self.plan = frame_plan(settings)
        self.catalog = session_catalog()
        for group in self.plan:
            for item in group:
                if not any(m["name"] == item["mod"]["name"] for m in self.catalog["modulations"]):
                    self.catalog["modulations"].append(copy.deepcopy(item["mod"]))
                if not any(c["id"] == item["code"]["id"] for c in self.catalog["codes"]):
                    self.catalog["codes"].append(copy.deepcopy(item["code"]))
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(QLabel("业务帧配置（每个业务帧承载一个编码信息块）"))
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["帧", "业务帧", "调制", "编码类别", "编码配置", "信息 K / bit", "编码 E / bit", "符号数"])
        self.table.horizontalHeaderItem(7).setToolTip(
            "该业务帧编码、调制后的复符号数量 = 向上取整(E / 每符号位数)。\n"
            "例如 1024 个编码 bit 使用 QPSK（2 bit/符号），得到 512 个符号。\n"
            "此列不计同步头和帧头。")
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(120)
        root.addWidget(self.table, 1)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.summary)
        self.error = None
        self.combos = []
        for key in ("segments", "words"):
            fields[key][0].valueChanged.connect(self.rebuild)
        fields["symbol_rate"][0].valueChanged.connect(self.update_summary)
        self.rebuild()

    def settings(self):
        result = {}
        for key, (field, scale, decimals) in self.fields.items():
            value = field.value()*scale
            result[key] = int(value) if decimals == 0 else value
        result["business_frames"] = copy.deepcopy(self.plan)
        return result

    def rebuild(self, *_):
        segments = int(self.fields["segments"][0].value())
        words = int(self.fields["words"][0].value())
        old = self.plan
        self.plan = [[copy.deepcopy(old[i][j] if i < len(old) and j < len(old[i]) else default_business_frame())
                      for j in range(words)] for i in range(segments)]
        self.table.setUpdatesEnabled(False)
        self.table.clearContents()
        self.table.setRowCount(segments*words)
        self.combos = []
        for i, group in enumerate(self.plan):
            for j, item in enumerate(group):
                row = i*words+j
                self.table.setItem(row, 0, QTableWidgetItem(str(i+1)))
                self.table.setItem(row, 1, QTableWidgetItem(str(j+1)))
                mod, family, code = QComboBox(), QComboBox(), QComboBox()
                mod.setMinimumWidth(100)
                family.setMinimumWidth(135)
                code.setMinimumWidth(215)
                code.setMinimumContentsLength(16)
                mod.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                code.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                for definition in self.catalog["modulations"]:
                    mod.addItem(definition["name"], definition)
                mod.setCurrentIndex(mod.findText(item["mod"]["name"]))
                mod.setToolTip(item["mod"]["name"])
                for key, label in FAMILIES.items():
                    if any(c["family"] == key for c in self.catalog["codes"]):
                        family.addItem(label, key)
                family.setCurrentIndex(family.findData(item["code"]["family"]))
                for column, widget in ((2, mod), (3, family), (4, code)):
                    self.table.setCellWidget(row, column, widget)
                self.populate_codes(i, j, family, code, item["code"])
                mod.currentIndexChanged.connect(lambda _, a=i, b=j, widget=mod: self.set_modulation(a, b, widget))
                family.currentIndexChanged.connect(lambda _, a=i, b=j, f=family, c=code: self.populate_codes(a, b, f, c))
                code.currentIndexChanged.connect(lambda _, a=i, b=j, widget=code: self.set_code(a, b, widget))
                self.combos.append((mod, family, code))
        self.table.setUpdatesEnabled(True)
        self.update_summary()

    def populate_codes(self, i, j, family, combo, selected=None):
        combo.blockSignals(True)
        combo.clear()
        for code in self.catalog["codes"]:
            if code["family"] == family.currentData():
                definition = selected if selected and code["id"] == selected["id"] else code
                combo.addItem(definition["name"], definition)
        if selected:
            for index in range(combo.count()):
                if combo.itemData(index)["id"] == selected["id"]:
                    combo.setCurrentIndex(index)
                    break
        combo.setToolTip(combo.currentText())
        combo.blockSignals(False)
        if selected is None:
            self.set_code(i, j, combo)

    def set_modulation(self, i, j, combo):
        self.plan[i][j]["mod"] = copy.deepcopy(combo.currentData())
        combo.setToolTip(combo.currentText())
        self.update_summary()

    def set_code(self, i, j, combo):
        self.plan[i][j]["code"] = copy.deepcopy(combo.currentData())
        combo.setToolTip(combo.currentText())
        self.update_summary()

    def update_summary(self, *_):
        try:
            info = frame_summary(self.settings())
            for row, frame in enumerate(info["frames"]):
                for column, key in ((5, "information_bits"), (6, "coded_bits"), (7, "symbols")):
                    self.table.setItem(row, column, QTableWidgetItem(f"{frame[key]:,}"))
            rate = info["symbol_rate"]
            self.summary.setText(
                f"原始业务数据：{info['information_bits']:,} bit　｜　编码后业务：{info['coded_bits']:,} bit\n"
                f"同步头＋训练头：{info['known_header_bits']:,} bit（BPSK）；业务调制补位：{info['modulation_padding_bits']:,} bit\n"
                f"输出：{info['total_symbols']:,} 个复符号，{rate/1e6:g} Mbaud，时长 {info['total_symbols']/rate*1e6:.6g} μs。\n"
                "同步头为 128 符号序列重复 8 次；每帧为 64 符号帧头＋业务码字。")
            self.error = None
        except (ValueError, KeyError, OSError) as exc:
            self.error = str(exc)
            self.summary.setText("配置需要调整："+self.error)
