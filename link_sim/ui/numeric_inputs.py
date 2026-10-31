"""Numeric inputs with dedicated, stacked step buttons and reliable hit areas."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractSpinBox, QSpinBox, QDoubleSpinBox, QToolButton


class StepButtons:
    def setRange(self, minimum, maximum):
        super().setRange(minimum, maximum)
        if hasattr(self, "up_button"):
            self.sync_step_buttons()

    def setMinimum(self, minimum):
        super().setMinimum(minimum)
        if hasattr(self, "up_button"):
            self.sync_step_buttons()

    def setMaximum(self, maximum):
        super().setMaximum(maximum)
        if hasattr(self, "up_button"):
            self.sync_step_buttons()

    def install_step_buttons(self):
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        # Two 18 px hit areas plus the 2 px top/bottom margins.
        self.setMinimumHeight(40)
        self.setKeyboardTracking(False)
        self.up_button = QToolButton(self)
        self.down_button = QToolButton(self)
        for button, arrow, direction, label in (
            (self.up_button, Qt.ArrowType.UpArrow, 1, "增加数值"),
            (self.down_button, Qt.ArrowType.DownArrow, -1, "减小数值"),
        ):
            button.setArrowType(arrow)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setAutoRepeat(True)
            button.setAccessibleName(label)
            button.setToolTip(label)
            button.setStyleSheet("""
                QToolButton {background:#F8F8F6;border:1px solid #1C4961;
                             border-radius:2px;padding:0;color:#C9E6F2;}
                QToolButton:hover {background:#E6F1FB;border-color:#00CFE8;}
                QToolButton:pressed {background:#12617B;}
                QToolButton:disabled {color:#39728A;background:#07131F;}
            """)
            button.clicked.connect(lambda checked=False, delta=direction: self.stepBy(delta))
        self.valueChanged.connect(self.sync_step_buttons)
        self.position_step_buttons()
        self.sync_step_buttons()

    def position_step_buttons(self):
        if not hasattr(self, "up_button"):
            return
        width = max(30, self.fontMetrics().height() + 12)
        half = max(18, (self.height() - 4) // 2)
        self.up_button.setGeometry(self.width() - width - 2, 2, width, half)
        self.down_button.setGeometry(
            self.width() - width - 2, 2 + half, width, self.height() - 4 - half
        )
        self.lineEdit().setTextMargins(0, 0, width + 4, 0)
        self.up_button.raise_()
        self.down_button.raise_()

    def sync_step_buttons(self, *args):
        flags = self.stepEnabled()
        self.up_button.setEnabled(bool(flags & QAbstractSpinBox.StepEnabledFlag.StepUpEnabled))
        self.down_button.setEnabled(bool(flags & QAbstractSpinBox.StepEnabledFlag.StepDownEnabled))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.position_step_buttons()


class IntegerInput(StepButtons, QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.install_step_buttons()


class DecimalInput(StepButtons, QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.install_step_buttons()


def integer(value, low, high):
    box = IntegerInput()
    box.setRange(low, high)
    box.setValue(value)
    return box


def decimal(value, low, high, step=0.1):
    box = DecimalInput()
    box.setRange(low, high)
    box.setDecimals(3)
    box.setSingleStep(step)
    box.setValue(value)
    return box
