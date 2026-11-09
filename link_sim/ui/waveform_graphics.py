"""Readable waveform blocks and direct canvas panning for NodeGraphQt."""

import html
from PySide6.QtCore import QEvent, QObject, QPoint, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPen, QTextOption
from PySide6.QtWidgets import QGraphicsTextItem
from NodeGraphQt.qgraphics.node_base import NodeItem


class WaveformNodeItem(NodeItem):
    WIDTH, BODY_HEIGHT = 144., 144.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.applied = False
        self.model_caption = QGraphicsTextItem(self)
        self.model_caption.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.model_caption.setDefaultTextColor(QColor("#aad6e5"))
        for item, bold in ((self.text_item, True), (self.model_caption, False)):
            font = item.font()
            font.setPointSize(16 if bold else 14)
            font.setBold(bold)
            item.setFont(font)
            item.setTextWidth(self.WIDTH-(20 if bold else 8))
            item.document().setDocumentMargin(0)
            option = QTextOption(Qt.AlignmentFlag.AlignCenter)
            option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
            item.document().setDefaultTextOption(option)
        self.text_item.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.text_item.set_locked(True)
        self.text_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def calc_size(self, *args, **kwargs):
        caption = getattr(self, "model_caption", None)
        extra = caption.boundingRect().height()+12 if caption and caption.toPlainText() else 0
        return self.WIDTH, self.BODY_HEIGHT+extra

    def _set_base_size(self, *args, **kwargs):
        self.prepareGeometryChange()
        self._width, self._height = self.calc_size()

    def align_label(self, *args, **kwargs):
        self.text_item.setPos(10, (self.BODY_HEIGHT-self.text_item.boundingRect().height())/2)

    def _draw_node_horizontal(self):
        self._set_base_size()
        self.align_label()
        self._icon_item.hide()
        self._x_item.hide()
        self.model_caption.setPos(4, self.BODY_HEIGHT+8)
        for items, ports, is_input in ((self._input_items, self.inputs, True), (self._output_items, self.outputs, False)):
            for label in items.values():
                label.hide()
            visible = [p for p in ports if p.isVisible()]
            for index, port in enumerate(visible):
                width, height = port.boundingRect().width(), port.boundingRect().height()
                y = self.BODY_HEIGHT/2 if len(visible) == 1 else self.BODY_HEIGHT/2-16+index*32
                port.setPos(-width/2 if is_input else self.WIDTH-width/2, y-height/2)
        self._tooltip_disable(self.disabled)
        self.update()

    def auto_switch_mode(self):
        self.text_item.show()
        self.model_caption.setVisible(bool(self.model_caption.toPlainText()))
        self._x_item.hide()

    def _tooltip_disable(self, state):
        name = html.escape(self.name)
        self.setToolTip(name+("<br>Bypass" if state else ""))

    def set_caption(self, text, applied):
        self.applied = applied
        self.model_caption.setPlainText(text)
        self.draw_node()

    def paint(self, painter, option, widget):
        self.auto_switch_mode()
        painter.save()
        fill = QColor(*self.color).darker(160 if self.disabled else 115)
        painter.setBrush(fill)
        color = "#ffcb70" if self.selected else "#28debe" if self.applied and not self.disabled else "#426578"
        pen = QPen(QColor(color), 2.5 if self.applied or self.selected else 1.)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawRoundedRect(QRectF(1, 1, self.WIDTH-2, self.BODY_HEIGHT-2), 6, 6)
        if self.disabled:
            painter.setPen(QColor("#f0c77a"))
            painter.setFont(QFont(self.text_item.font().family(), 11, QFont.Weight.Bold))
            painter.drawText(QRectF(4, self.BODY_HEIGHT-23, self.WIDTH-8, 20), Qt.AlignmentFlag.AlignCenter, "Bypass")
        painter.restore()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.viewer().node_double_clicked.emit(self.id)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


class CanvasPanFilter(QObject):
    def __init__(self, viewer):
        super().__init__(viewer)
        self.viewer = viewer
        self.dragging = False
        self.last = QPoint()
        self.moved = False
        viewer.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress:
            point = event.position().toPoint()
            empty = self.viewer.itemAt(point) is None
            if event.button() == Qt.MouseButton.RightButton or (
                    event.button() == Qt.MouseButton.LeftButton and empty and not event.modifiers()):
                self.dragging, self.moved, self.last = True, False, point
                self.viewer.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                return True
        elif kind == QEvent.Type.MouseMove and self.dragging:
            point = event.position().toPoint()
            delta = self.viewer.mapToScene(self.last)-self.viewer.mapToScene(point)
            rect = self.viewer.scene_rect()
            self.viewer.set_scene_rect([rect[0]+delta.x(), rect[1]+delta.y(), rect[2], rect[3]])
            self.moved |= (point-self.last).manhattanLength() > 1
            self.last = point
            return True
        elif kind == QEvent.Type.MouseButtonRelease and self.dragging:
            self.dragging = False
            self.viewer.viewport().unsetCursor()
            return True
        elif kind == QEvent.Type.ContextMenu and self.moved:
            self.moved = False
            return True
        return super().eventFilter(obj, event)
