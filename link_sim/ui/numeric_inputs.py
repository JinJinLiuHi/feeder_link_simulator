"""Editable numeric fields with native validation and hidden step buttons."""

from PySide6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QSpinBox


class IntegerInput(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setKeyboardTracking(False)


class DecimalInput(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
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
