"""One shared business-frame definition, fixed payload size and editable header."""

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFormLayout, QHeaderView, QLabel, QLineEdit,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QSizePolicy,
)

from link_budget.algorithm.ber_core.catalog import FAMILIES, session_catalog
from link_sim.algorithm.frame_config import (
    editor_settings, frame_summary, code_dimensions, modulation_bits, header_hex,
)


class FrameSourceEditor(QWidget):
    def __init__(self, settings, fields, parent=None):
        super().__init__(parent)
        self.fields = fields
        settings = editor_settings(settings)
        self.definition = settings["business_frame"]
        self.catalog = session_catalog()
        for collection, key, identity in (("modulations", "mod", "name"), ("codes", "code", "id")):
            saved = self.definition[key]
            if not any(item[identity] == saved[identity] for item in self.catalog[collection]):
                self.catalog[collection].append(copy.deepcopy(saved))
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.header = QLineEdit(settings["header_hex"])
        self.header.setMaxLength(280)
        self.header.setPlaceholderText("1ACFFC1D")
        self.header.setToolTip("高位在前，每个十六进制字符生成 4 个 BPSK 符号。1ACFFC1D 对应 32 符号，不重复。")
        form.addRow("帧头（十六进制）", self.header)
        root.addLayout(form)
        root.addWidget(QLabel("业务帧配置（所有帧共用；每帧承载一个编码信息块）"))
        self.table = QTableWidget(1, 6)
        self.table.setHorizontalHeaderLabels(["调制", "编码类别", "编码配置", "信息 K / bit", "编码 E / bit", "符号数"])
        self.table.horizontalHeaderItem(5).setToolTip(
            "编码调制后的实际符号数 = 向上取整(E / 每符号位数)。\n"
            "不足业务数据符号数时，在后面补复数零；超过时不能应用。")
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(42)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(110)
        self.table.setMaximumHeight(130)
        root.addWidget(self.table)
        mod, family, code = QComboBox(), QComboBox(), QComboBox()
        for widget, width in ((mod, 110), (family, 135), (code, 215)):
            widget.setMinimumWidth(width)
        for widget in (mod, code):
            widget.setMinimumContentsLength(16)
            widget.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        for definition in self.catalog["modulations"]:
            mod.addItem(definition["name"], definition)
        mod.setCurrentIndex(mod.findText(self.definition["mod"]["name"]))
        for key, label in FAMILIES.items():
            if any(c["family"] == key for c in self.catalog["codes"]):
                family.addItem(label, key)
        family.setCurrentIndex(family.findData(self.definition["code"]["family"]))
        self.combos = [(mod, family, code)]
        for column, widget in enumerate(self.combos[0]):
            self.table.setCellWidget(0, column, widget)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.summary)
        root.addStretch()
        self.error = None
        self.populate_codes(self.definition["code"])
        mod.currentIndexChanged.connect(self.set_modulation)
        family.currentIndexChanged.connect(lambda _: self.populate_codes())
        code.currentIndexChanged.connect(self.set_code)
        for field, _, _ in fields.values():
            field.valueChanged.connect(self.update_summary)
        self.header.textChanged.connect(self.update_summary)
        self.update_summary()

    def settings(self):
        result = {}
        for key, (field, scale, decimals) in self.fields.items():
            value = field.value()*scale
            result[key] = int(value) if decimals == 0 else value
        result["business_frame"] = copy.deepcopy(self.definition)
        result["header_hex"] = header_hex({"header_hex": self.header.text()})
        return result

    def populate_codes(self, selected=None):
        _, family, combo = self.combos[0]
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
        combo.blockSignals(False)
        self.set_code()

    def set_modulation(self, *_):
        combo = self.combos[0][0]
        self.definition["mod"] = copy.deepcopy(combo.currentData())
        combo.setToolTip(combo.currentText())
        self.update_summary()

    def set_code(self, *_):
        combo = self.combos[0][2]
        self.definition["code"] = copy.deepcopy(combo.currentData())
        combo.setToolTip(combo.currentText())
        self.update_summary()

    def update_summary(self, *_):
        try:
            k, n = code_dimensions(self.definition["code"])
            m = modulation_bits(self.definition["mod"])
            for column, value in ((3, k), (4, n), (5, (n+m-1)//m)):
                self.table.setItem(0, column, QTableWidgetItem(f"{value:,}"))
            info = frame_summary(self.settings())
            rate = info["symbol_rate"]
            per_padding = info["zero_padding_symbols"]//int(self.fields["segments"][0].value())
            self.summary.setText(
                f"原始业务数据：{info['information_bits']:,} bit　｜　编码后业务：{info['coded_bits']:,} bit\n"
                f"每帧业务区：{int(self.fields['payload_symbols'][0].value()):,} 符号，其中编码调制 {(n+m-1)//m:,} 符号、尾部补零 {per_padding:,} 符号\n"
                f"输出：{info['total_symbols']:,} 个复符号，{rate/1e6:g} Mbaud，时长 {info['total_symbols']/rate*1e6:.6g} μs。\n"
                f"同步头为 128 符号序列重复 8 次；每帧为 {info['header_symbols']} 符号 BPSK 帧头＋业务区。\n"
                "各帧业务数据独立生成；种子 0 每次随机，非零可复现。补零不增加业务 bit。")
            self.error = None
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.error = str(exc)
            self.summary.setText("配置需要调整："+self.error)
