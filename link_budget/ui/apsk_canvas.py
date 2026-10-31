"""Interactive APSK preview sharing the simulation's label-indexed constellation."""

import copy
import math
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget, QToolTip
from link_budget.algorithm.ber_core.catalog import custom_points


class APSKCanvas(QWidget):
    ring_moved = Signal(int, float, float)
    ring_selected = Signal(int)
    COLORS = ("#22C7E6", "#3DDC97", "#4D9EFF", "#8BC6FF", "#F2C14E")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(350, 340)
        self.setMouseTracking(True)
        self.definition = None
        self.points = []
        self.zoom = 1.0
        self.center = 0j
        self.show_bits = True
        self.selected_ring = 0
        self.drag = None
        self.hover = None
        self.message = "调整参数以预览星座"

    def set_definition(self, definition, message=""):
        previous = (
            None
            if self.definition is None
            else (self.definition["bits"], self.definition["counts"])
        )
        self.definition = copy.deepcopy(definition)
        self.selected_ring = (
            min(self.selected_ring, len(definition["counts"]) - 1) if definition else 0
        )
        self.points = custom_points(definition) if definition else []
        self.message = message or "参数无效，请检查左侧设置"
        topology = None if definition is None else (definition["bits"], definition["counts"])
        if previous != topology and self.drag is None:
            self.reset_view()
        self.update()

    def reset_view(self):
        self.zoom = 1.0
        self.center = 0j
        self.update()

    def set_bits_visible(self, visible):
        self.show_bits = visible
        self.update()

    def plot_rect(self):
        side = max(40.0, min(self.width() - 100, self.height() - 100))
        return QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)

    def energy_scale(self):
        d = self.definition
        return math.sqrt(sum(n * r * r for n, r in zip(d["counts"], d["radii"])) / sum(d["counts"]))

    def bound(self):
        base = self.drag[3] if self.drag else max((abs(p) for p in self.points), default=1) * 1.25
        return base / self.zoom

    def to_screen(self, point):
        rect = self.plot_rect()
        bound = self.bound()
        return QPointF(
            rect.center().x() + (point.real - self.center.real) * rect.width() / (2 * bound),
            rect.center().y() - (point.imag - self.center.imag) * rect.height() / (2 * bound),
        )

    def to_world(self, position):
        rect = self.plot_rect()
        bound = self.bound()
        return self.center + complex(
            (position.x() - rect.center().x()) * 2 * bound / rect.width(),
            (rect.center().y() - position.y()) * 2 * bound / rect.height(),
        )

    def point_geometry(self):
        if not self.definition:
            return []
        d = self.definition
        items = []
        index = 0
        # Keep the coordinate scale stable during a drag, then return to Es=1 on release.
        ratio = self.energy_scale() / self.drag[2] if self.drag else 1
        for ring, count in enumerate(d["counts"]):
            for ordinal in range(count):
                label = index ^ (index >> 1)
                items.append((self.points[label] * ratio, label, ring, ordinal))
                index += 1
        return items

    def nearest(self, position):
        nearest = None
        distance = 14.0
        for item in self.point_geometry():
            pixel = self.to_screen(item[0])
            d = math.hypot(pixel.x() - position.x(), pixel.y() - position.y())
            if d < distance:
                nearest = item
                distance = d
        return nearest

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#091B29"))
        if not self.definition:
            painter.setPen(QColor("#648A9C"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            return
        rect = self.plot_rect()
        bound = self.bound()
        painter.save()
        painter.setClipRect(rect)
        step = 10 ** math.floor(math.log10(2 * bound / 5))
        if 2 * bound / step > 10:
            step *= 2
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.setPen(QPen(QColor("#EEF0F1"), 1))
        for axis in (0, 1):
            center = self.center.real if axis == 0 else self.center.imag
            low = math.ceil((center - bound) / step)
            high = math.floor((center + bound) / step)
            for tick in range(low, high + 1):
                value = tick * step
                if axis == 0:
                    p = self.to_screen(complex(value, self.center.imag))
                    painter.drawLine(QPointF(p.x(), rect.top()), QPointF(p.x(), rect.bottom()))
                else:
                    p = self.to_screen(complex(self.center.real, value))
                    painter.drawLine(QPointF(rect.left(), p.y()), QPointF(rect.right(), p.y()))
        origin = self.to_screen(0j)
        painter.setPen(QPen(QColor("#CCD9E5"), 1.3))
        painter.drawLine(QPointF(rect.left(), origin.y()), QPointF(rect.right(), origin.y()))
        painter.drawLine(QPointF(origin.x(), rect.top()), QPointF(origin.x(), rect.bottom()))
        scale = self.drag[2] if self.drag else self.energy_scale()
        for ring, radius in enumerate(self.definition["radii"]):
            color = QColor(self.COLORS[ring % len(self.COLORS)])
            color.setAlpha(110)
            painter.setPen(
                QPen(color, 2 if ring == self.selected_ring else 1, Qt.PenStyle.DashLine)
            )
            pixels = radius / scale * rect.width() / (2 * bound)
            painter.drawEllipse(origin, pixels, pixels)
        items = self.point_geometry()
        point_boxes = []
        for point, label, ring, _ in items:
            pixel = self.to_screen(point)
            point_boxes.append(QRectF(pixel.x() - 6, pixel.y() - 6, 12, 12))
            if self.hover and self.hover[1] == label:
                color = QColor(self.COLORS[ring % len(self.COLORS)])
                color.setAlpha(35)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(pixel, 12, 12)
            painter.setPen(QPen(QColor("white"), 1.5))
            painter.setBrush(QColor(self.COLORS[ring % len(self.COLORS)]))
            painter.drawEllipse(pixel, 5, 5)
        if self.show_bits:
            painter.setFont(QFont("Consolas", 10))
            metrics = painter.fontMetrics()
            used = []
            painter.setPen(QColor("#C9E6F2"))
            for point, label, ring, _ in items:
                pixel = self.to_screen(point)
                text = format(label, f"0{self.definition['bits']}b")
                width = metrics.horizontalAdvance(text) + 4
                height = metrics.height() + 2
                for dx, dy in (
                    (9, -height - 4),
                    (9, 5),
                    (-width - 9, -height - 4),
                    (-width - 9, 5),
                ):
                    box = QRectF(pixel.x() + dx, pixel.y() + dy, width, height)
                    if rect.contains(box) and not any(
                        box.intersects(other) for other in used + point_boxes
                    ):
                        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
                        used.append(box)
                        break
        painter.restore()
        painter.setPen(QColor("#648A9C"))
        painter.setFont(QFont("Microsoft YaHei", 9))
        for tick in range(-2, 3):
            p = self.to_screen(complex(tick, 0))
            if rect.left() <= p.x() <= rect.right():
                painter.drawText(
                    QRectF(p.x() - 20, rect.bottom() + 8, 40, 20),
                    Qt.AlignmentFlag.AlignCenter,
                    str(tick),
                )
            p = self.to_screen(complex(0, tick))
            if rect.top() <= p.y() <= rect.bottom():
                painter.drawText(
                    QRectF(rect.left() - 38, p.y() - 10, 30, 20),
                    Qt.AlignmentFlag.AlignCenter,
                    str(tick),
                )
        painter.drawText(
            QRectF(rect.center().x() - 20, rect.bottom() + 29, 40, 20),
            Qt.AlignmentFlag.AlignCenter,
            "I",
        )
        painter.drawText(
            QRectF(rect.left() - 36, rect.top() - 27, 30, 20), Qt.AlignmentFlag.AlignCenter, "Q"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.definition:
            item = self.nearest(event.position())
            if item:
                self.selected_ring = item[2]
                self.ring_selected.emit(item[2])
                self.drag = (item[2], item[3], self.energy_scale(), self.bound() * self.zoom)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                self.update()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag and self.definition:
            ring, ordinal, scale, _ = self.drag
            world = self.to_world(event.position())
            radius = abs(world) * scale
            radii = self.definition["radii"]
            radius = (
                1.0
                if ring == 0
                else min(
                    (radii[ring + 1] - 0.001) if ring + 1 < len(radii) else 1000,
                    max(radii[ring - 1] + 0.001, radius),
                )
            )
            phase = (
                math.degrees(math.atan2(world.imag, world.real))
                - 360 * ordinal / self.definition["counts"][ring]
            ) % 360
            self.ring_moved.emit(ring, round(radius, 4), round(phase, 3))
            event.accept()
            return
        self.hover = self.nearest(event.position())
        self.setCursor(Qt.CursorShape.OpenHandCursor if self.hover else Qt.CursorShape.ArrowCursor)
        if self.hover:
            point, label, ring, _ = self.hover
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"环 {ring+1} · {label:0{self.definition['bits']}b}\nI={point.real:.5f}  Q={point.imag:.5f}",
                self,
            )
        else:
            QToolTip.hideText()
        self.update()

    def mouseReleaseEvent(self, event):
        self.drag = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.hover = None
        QToolTip.hideText()
        self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        if self.definition and self.drag is None and self.plot_rect().contains(event.position()):
            before = self.to_world(event.position())
            self.zoom = min(32, max(1, self.zoom * 1.2 ** (event.angleDelta().y() / 120)))
            after = self.to_world(event.position())
            self.center += before - after
            if self.zoom == 1:
                self.center = 0j
            self.update()
            event.accept()
            return
        super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.drag = None
        self.reset_view()
        event.accept()
