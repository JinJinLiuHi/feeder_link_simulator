"""Editable numeric fields with native validation and hidden step buttons."""

from PySide6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QSpinBox


def hide_step_buttons(box):
    # Keep the edit field usable with styled Qt 6.12 spin boxes.
    box.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    box.setStyleSheet("""
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
            width: 0px; height: 0px; border: none;
        }
        QSpinBox QLineEdit, QDoubleSpinBox QLineEdit {
            background: transparent; border: none; padding: 0px;
        }
    """)
    box.lineEdit().setTextMargins(8, 0, 8, 0)


class IntegerInput(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        hide_step_buttons(self)
        self.setKeyboardTracking(False)


class DecimalInput(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        hide_step_buttons(self)
        self.setKeyboardTracking(False)


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
