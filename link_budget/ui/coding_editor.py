"""Type-specific coding editor following the reference workflow, in platform style."""

import copy
import hashlib
import json
import re
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QComboBox,
    QCompleter,
    QLineEdit,
    QPlainTextEdit,
    QCheckBox,
    QLabel,
    QPushButton,
    QStackedWidget,
    QScrollArea,
    QFileDialog,
    QMessageBox,
)
from link_budget.ui.styles import APP_STYLE, light_palette
from link_budget.ui.numeric_inputs import integer
from link_budget.algorithm.ber_core.session import ProcessSession
from link_budget.algorithm.ber_core.simulation import _polynomials

QPP_LENGTHS = (
    tuple(range(40, 513, 8))
    + tuple(range(528, 1025, 16))
    + tuple(range(1056, 2049, 32))
    + tuple(range(2112, 6145, 64))
)


def parse_polynomials(text):
    tokens = re.split(r"[\s,，;；]+", text.strip())
    if not tokens or any(not re.fullmatch(r"[0-7]{1,3}", p) for p in tokens):
        raise ValueError("请输入八进制多项式，用逗号或空格分隔")
    return tokens


def parse_permutation(text, k):
    raw = text.strip()
    if not raw or len(raw) > 100000:
        raise ValueError("请填写完整的自定义交织排列")
    if "," in raw and any(not part.strip() for part in raw.split(",")):
        raise ValueError("排列中含空逗号字段")
    tokens = re.split(r"[\s,]+", raw)
    if len(tokens) != k or any(not re.fullmatch(r"[0-9]+", token) for token in tokens):
        raise ValueError(f"排列必须恰好包含 K={k} 个非负整数")
    values = list(map(int, tokens))
    if sorted(values) != list(range(k)):
        raise ValueError("排列须为 0～K−1 的完整排列，不得重复、缺失或越界")
    return values


class TrellisFields(QGroupBox):
    changed = Signal()

    def __init__(self, turbo=False, parent=None):
        super().__init__("Turbo 分量码参数" if turbo else "前馈卷积码参数", parent)
        self.turbo = turbo
        form = QFormLayout(self)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.k = integer(1024, 2 if turbo else 1, 8192)
        self.k.setToolTip("每帧信息位数 K，不包含终止尾比特")
        self.length = integer(4 if turbo else 7, 3, 6 if turbo else 8)
        self.terminate = QCheckBox("零尾终止（每帧从全零状态开始）")
        self.terminate.setChecked(True)
        if turbo:
            self.qpp_k = QComboBox()
            self.qpp_k.setEditable(True)
            self.qpp_k.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            self.qpp_k.addItems(map(str, QPP_LENGTHS))
            self.qpp_k.setCurrentText("1024")
            self.qpp_k.setMaxVisibleItems(12)
            self.qpp_k.completer().setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
            self.qpp_k.completer().setFilterMode(Qt.MatchFlag.MatchContains)
            self.qpp_k.setToolTip("QPP 仅接受 188 个标准信息长度，不自动修正输入")
            self.k_stack = QStackedWidget()
            self.k_stack.addWidget(self.qpp_k)
            self.k_stack.addWidget(self.k)
            self.k_stack.setFixedHeight(max(self.k.minimumHeight(), self.qpp_k.sizeHint().height()))
            form.addRow("信息长度 K", self.k_stack)
            form.addRow("分量码约束长度 L", self.length)
            self.feedback = QLineEdit("13")
            self.feedforward = QLineEdit("15")
            self.feedback.setMaxLength(3)
            self.feedforward.setMaxLength(3)
            form.addRow("反馈多项式（八进制）", self.feedback)
            form.addRow("前馈多项式（八进制）", self.feedforward)
            self.rate = QComboBox()
            self.rate.addItem("1/3", "1/3")
            self.rate.addItem("1/2（固定打孔）", "1/2")
            self.interleaver = QComboBox()
            for label, key in [
                ("QPP（标准长度）", "qpp"),
                ("随机交织", "random"),
                ("自定义排列", "custom"),
            ]:
                self.interleaver.addItem(label, key)
            form.addRow("设计码率", self.rate)
            form.addRow("交织方式", self.interleaver)
            self.custom_panel = QWidget()
            custom = QVBoxLayout(self.custom_panel)
            custom.setContentsMargins(0, 0, 0, 0)
            custom.addWidget(QLabel("交织后[i] = 原始信息[排列[i]]，索引从 0 开始"))
            self.permutation = QPlainTextEdit()
            self.permutation.setMinimumHeight(110)
            self.permutation.setMaximumHeight(220)
            self.permutation.setFont(QFont("Consolas", 10))
            self.permutation.setPlaceholderText(
                "填写恰好 K 个 0～K−1 的整数；支持空格、逗号、换行。"
            )
            custom.addWidget(self.permutation)
            form.addRow(self.custom_panel)
            for field in (self.feedback, self.feedforward):
                field.textChanged.connect(self.changed)
            self.rate.currentIndexChanged.connect(self.changed)
            self.interleaver.currentIndexChanged.connect(self.switch_interleaver)
            self.qpp_k.currentTextChanged.connect(self.changed)
            self.permutation.textChanged.connect(self.changed)
            self.switch_interleaver()
        else:
            form.addRow("信息长度 K", self.k)
            form.addRow("约束长度 L", self.length)
            self.polys = QLineEdit("171, 133")
            self.polys.setMaxLength(64)
            self.polys.setPlaceholderText("例如 171, 133；2 或 3 个八进制多项式")
            form.addRow("生成多项式（八进制）", self.polys)
            self.polys.textChanged.connect(self.changed)
        form.addRow("终止方式", self.terminate)
        hint = QLabel(
            "最低位表示当前输入 D⁰；最高次数与 L−1 一致。尾比特参与传输，不计入信息位 BER。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#648A9C;font-size:9pt;")
        form.addRow(hint)
        for field in (self.k, self.length):
            field.valueChanged.connect(self.changed)
            field.lineEdit().textEdited.connect(self.changed)
        self.terminate.toggled.connect(self.changed)

    def switch_interleaver(self, *args):
        kind = self.interleaver.currentData()
        self.k_stack.setCurrentIndex(0 if kind == "qpp" else 1)
        self.custom_panel.setVisible(kind == "custom")
        self.changed.emit()

    def commit(self):
        self.k.interpretText()
        self.length.interpretText()

    def definition(self):
        family = "turbo" if self.turbo else "conv"
        k = self.k.value()
        if self.turbo and self.interleaver.currentData() == "qpp":
            text = self.qpp_k.currentText().strip()
            if not text.isdecimal() or int(text) not in QPP_LENGTHS:
                raise ValueError("QPP 信息长度 K 必须选择标准列表中的长度")
            k = int(text)
        code = dict(
            family=family,
            custom=True,
            k=k,
            length=self.length.value(),
            terminate=self.terminate.isChecked(),
            polys=(
                [self.feedback.text().strip(), self.feedforward.text().strip()]
                if self.turbo
                else parse_polynomials(self.polys.text())
            ),
        )
        _polynomials(code)
        if self.turbo:
            code.update(rate=self.rate.currentData(), interleaver=self.interleaver.currentData())
            if code["interleaver"] == "custom":
                code["permutation"] = parse_permutation(self.permutation.toPlainText(), k)
        return code

    def load(self, code):
        self.k.setValue(code["k"])
        self.length.setValue(code["length"])
        self.terminate.setChecked(code["terminate"])
        if self.turbo:
            self.feedback.setText(code["polys"][0])
            self.feedforward.setText(code["polys"][1])
            self.rate.setCurrentIndex(self.rate.findData(code["rate"]))
            self.qpp_k.setCurrentText(str(code["k"]))
            self.interleaver.setCurrentIndex(self.interleaver.findData(code["interleaver"]))
            self.permutation.setPlainText(" ".join(map(str, code.get("permutation", []))))
        else:
            self.polys.setText(", ".join(code["polys"]))


class MatrixFields(QGroupBox):
    changed = Signal()

    def __init__(self, qc=False, parent=None):
        super().__init__("QC 基矩阵" if qc else "完整校验矩阵 / alist", parent)
        self.qc = qc
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        if qc:
            self.z = integer(336, 1, 8192)
            self.direction = QComboBox()
            self.direction.addItem("右移：H[i,(i+s) mod Z] = 1", 1)
            self.direction.addItem("左移：H[i,(i−s) mod Z] = 1", -1)
            form.addRow("提升因子 Z", self.z)
            form.addRow("循环移位方向", self.direction)
            self.z.valueChanged.connect(self.changed)
            self.z.lineEdit().textEdited.connect(self.changed)
            self.direction.currentIndexChanged.connect(self.changed)
        else:
            self.format = QComboBox()
            self.format.addItem("完整 H（0/1 二元矩阵）", "h")
            self.format.addItem("alist（完整行、列邻接表）", "alist")
            form.addRow("输入格式", self.format)
            self.format.currentIndexChanged.connect(self.update_hint)
            self.format.currentIndexChanged.connect(self.changed)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#648A9C;font-size:9pt;")
        layout.addWidget(self.hint)
        read = QPushButton("读取矩阵文件…")
        read.setAutoDefault(False)
        read.clicked.connect(self.read_file)
        layout.addWidget(read, alignment=Qt.AlignmentFlag.AlignLeft)
        self.text = QPlainTextEdit()
        self.text.setFont(QFont("Consolas", 10))
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setMinimumHeight(240)
        layout.addWidget(self.text, 1)
        self.text.textChanged.connect(self.changed)
        self.update_hint()

    def update_hint(self, *args):
        if self.qc:
            message = "−1 表示全零块，0～Z−1 表示循环置换块。可选首行 Z 声明须与输入一致；不裁剪、不补齐。"
        elif self.format.currentData() == "h":
            message = "每行仅含 0/1，列数一致，无维度头。K 按 GF(2) 实际秩计算，保留冗余校验行。"
        else:
            message = "依次提供 N M、最大列/行度数、列度数、行度数、N 行列邻接、M 行行邻接；索引从 1 开始。"
        self.hint.setText(message)
        self.text.setPlaceholderText(message) if hasattr(self, "text") else None

    def read_file(self):
        name, _ = QFileDialog.getOpenFileName(
            self, "读取矩阵", "", "矩阵文件 (*.txt *.csv *.base *.alist);;全部文件 (*)"
        )
        if name:
            try:
                path = Path(name)
                if path.stat().st_size > 16 * 1024 * 1024:
                    raise ValueError("文件超过 16 MiB")
                self.text.setPlainText(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, "无法读取", str(error))

    def commit(self):
        if self.qc:
            self.z.interpretText()

    def definition(self):
        text = self.text.toPlainText()
        if not text.strip():
            raise ValueError("请先粘贴或读取原始矩阵")
        if len(text.encode("utf-8")) > 16 * 1024 * 1024:
            raise ValueError("矩阵文本超过 16 MiB")
        code = dict(
            family="qc" if self.qc else "h",
            custom=True,
            text=text,
            format="qc" if self.qc else self.format.currentData(),
        )
        if self.qc:
            code.update(z=self.z.value(), direction=self.direction.currentData())
        return code

    def load(self, code):
        self.text.setPlainText(code["text"])
        if self.qc:
            self.z.setValue(code["z"])
            self.direction.setCurrentIndex(self.direction.findData(code["direction"]))
        else:
            self.format.setCurrentIndex(self.format.findData(code["format"]))


class CodingEditor(QDialog):
    saved = Signal(object)

    def __init__(self, parent=None, definition=None):
        super().__init__(parent)
        self.setPalette(light_palette())
        self.setStyleSheet(APP_STYLE)
        self.setWindowTitle("修改编码" if definition else "新增编码")
        self.setMinimumSize(760, 580)
        self.resize(980, 800)
        self.initializing = True
        self.verified = None
        self.draft = None
        self.editing = definition is not None
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)
        heading = QLabel(self.windowTitle())
        heading.setStyleSheet("font-size:14pt;font-weight:600;")
        root.addWidget(heading)
        category = QHBoxLayout()
        category.addWidget(QLabel("编码类型"))
        self.kind = QComboBox()
        for name, key in [
            ("卷积码 · 前馈", "conv"),
            ("Turbo", "turbo"),
            ("LDPC · QC 基矩阵", "qc"),
            ("LDPC · H / alist", "h"),
        ]:
            self.kind.addItem(name, key)
        category.addWidget(self.kind, 1)
        root.addLayout(category)
        self.pages = {
            "conv": TrellisFields(),
            "turbo": TrellisFields(True),
            "qc": MatrixFields(True),
            "h": MatrixFields(),
        }
        self.stack = QStackedWidget()
        self.stack.setObjectName("codingPages")
        for page in self.pages.values():
            self.stack.addWidget(page)
            page.changed.connect(self.invalidate)
        scroll = QScrollArea()
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        body = QWidget()
        body.setObjectName("codingParameterContent")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addWidget(self.stack)
        body_layout.addStretch()
        scroll.setWidget(body)
        root.addWidget(scroll, 1)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color:#91AFC2;font-size:10pt;")
        root.addWidget(self.summary)
        self.status = QLabel("先检查参数，再验证编译码。通过后可加入本次会话的编码库。")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        footer = QHBoxLayout()
        self.validate = QPushButton("验证编译码")
        self.cancel_job = QPushButton("取消验证")
        self.cancel_job.setEnabled(False)
        self.save_button = QPushButton("保存修改" if definition else "加入编码库")
        self.save_button.setEnabled(False)
        close = QPushButton("取消")
        for button in (self.validate, self.cancel_job):
            footer.addWidget(button)
        footer.addStretch()
        footer.addWidget(close)
        footer.addWidget(self.save_button)
        root.addLayout(footer)
        for button in (self.validate, self.cancel_job, self.save_button, close):
            button.setAutoDefault(False)
        self.session = ProcessSession(self)
        self.session.message.connect(self.on_event)
        self.session.ended.connect(self.on_end)
        self.validate.clicked.connect(self.run_validation)
        self.cancel_job.clicked.connect(self.cancel_validation)
        self.save_button.clicked.connect(self.save)
        close.clicked.connect(self.reject)
        self.kind.currentIndexChanged.connect(self.update_fields)
        if definition:
            self.kind.setCurrentIndex(self.kind.findData(definition["family"]))
            self.pages[definition["family"]].load(definition)
            self.kind.setEnabled(False)
        self.initializing = False
        self.update_fields()

    @property
    def k(self):
        return self.pages["turbo" if self.kind.currentData() == "turbo" else "conv"].k

    @property
    def length(self):
        return self.pages["turbo" if self.kind.currentData() == "turbo" else "conv"].length

    @property
    def polys(self):
        return self.pages["conv"].polys

    @property
    def terminate(self):
        return self.pages["turbo" if self.kind.currentData() == "turbo" else "conv"].terminate

    @property
    def interleaver(self):
        return self.pages["turbo"].interleaver

    @property
    def rate(self):
        return self.pages["turbo"].rate

    @property
    def permutation(self):
        return self.pages["turbo"].permutation

    @property
    def z(self):
        return self.pages["qc"].z

    @property
    def direction(self):
        return self.pages["qc"].direction

    @property
    def format(self):
        return self.pages["h"].format

    @property
    def text(self):
        return self.pages["qc" if self.kind.currentData() == "qc" else "h"].text

    def update_fields(self, *args):
        self.stack.setCurrentWidget(self.pages[self.kind.currentData()])
        self.invalidate()

    def needs_validation(self):
        family = self.kind.currentData()
        return (
            self.editing
            or family in ("qc", "h")
            or (family == "turbo" and self.interleaver.currentData() == "custom")
        )

    def invalidate(self, *args):
        if self.initializing:
            return
        self.verified = None
        self.save_button.setEnabled(False)
        if self.session.process is not None:
            self.cancel_validation()
        required = self.needs_validation()
        self.validate.setVisible(required)
        self.cancel_job.setVisible(required)
        self.status.setText(
            "参数已修改，请重新验证。" if required else "参数有效，可直接加入编码库。"
        )
        self.status.setStyleSheet("color:#648A9C;")
        try:
            self.draft = self.definition()
            code = self.draft
            if code["family"] in ("conv", "turbo"):
                memory = code["length"] - 1
                n = (
                    (code["k"] + ((4 * memory + 2) // 3 if code["terminate"] else 0))
                    * (3 if code.get("rate") == "1/3" else 2)
                    if code["family"] == "turbo"
                    else (code["k"] + (memory if code["terminate"] else 0)) * len(code["polys"])
                )
                code["n"] = n
                decoder = "Log-MAP" if code["family"] == "turbo" else "软判决 Viterbi"
                self.summary.setText(
                    f"K = {code['k']:,} · N = {n:,} · 实际码率 K/N = {code['k']/n:.6f}\n译码：{decoder}；终止尾比特不计入信息位 BER。"
                )
                self.validate.setText("验证编译码")
            else:
                self.summary.setText("验证矩阵后显示 GF(2) 秩、实际信息长度 K、码长 N 和码率。")
                self.validate.setText("验证矩阵")
            self.validate.setEnabled(True)
            self.save_button.setEnabled(not required)
        except ValueError as error:
            self.draft = None
            self.summary.setText(str(error))
            self.validate.setEnabled(False)

    def definition(self):
        return self.pages[self.kind.currentData()].definition()

    def run_validation(self):
        self.pages[self.kind.currentData()].commit()
        try:
            definition = self.definition()
            self.session.start("validate", definition)
            self.validate.setEnabled(False)
            self.cancel_job.setEnabled(True)
            self.save_button.setEnabled(False)
            self.status.setText("正在后台核验实际输出长度、校验约束与无噪声恢复…")
            self.status.setStyleSheet("color:#91AFC2;")
        except (ValueError, RuntimeError) as error:
            self.status.setText(str(error))
            self.status.setStyleSheet("color:#D85A30;")

    def on_event(self, event):
        if event["type"] == "status":
            self.status.setText(event["message"])
        elif event["type"] == "validated":
            self.verified = event["code"]
            c = self.verified
            rank = f" · GF(2) 秩 = {c['n']-c['k']:,}" if c["family"] in ("qc", "h") else ""
            self.summary.setText(
                f"K = {c['k']:,} · N = {c['n']:,} · 实际码率 {c['k']/c['n']:.6f}{rank}"
            )
            self.status.setText("验证通过 · 全零、全一、单脉冲、随机信息无噪声恢复通过。")
            self.status.setStyleSheet("color:#1D9E75;")
            self.save_button.setEnabled(True)
        elif event["type"] == "error":
            self.status.setText("验证失败：" + event["message"])
            self.status.setStyleSheet("color:#D85A30;")
            self.status.setToolTip(event.get("detail", ""))

    def on_end(self):
        self.validate.setEnabled(self.draft is not None)
        self.cancel_job.setEnabled(False)

    def cancel_validation(self):
        self.session.dispose()
        self.on_end()
        self.verified = None
        self.save_button.setEnabled(False)
        self.status.setText("验证已取消。")

    def save(self):
        self.pages[self.kind.currentData()].commit()
        if not self.needs_validation():
            self.invalidate()
        source = self.verified if self.needs_validation() else self.draft
        if source is None:
            return
        code = copy.deepcopy(source)
        fingerprint = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()[:8]
        code["id"] = "custom_" + fingerprint
        code["name"] = f"自定义 · K{code['k']} · N{code['n']} · {fingerprint}"
        self.saved.emit(code)
        self.accept()

    def done(self, result):
        self.session.dispose()
        super().done(result)
