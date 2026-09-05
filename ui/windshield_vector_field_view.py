"""
camera_calibrator.ui.windshield_vector_field_view
======================================================

Windshield Baseline의 SpatialErrorMap을 dx/dy 방향 화살표(vector field)로
그린다(사용자 스펙 8번 "Spatial dx/dy Map"이 "단순 heatmap뿐 아니라 가능하면
이미지 위치별 vector field로 확인" 요구). 기존 코드베이스 어디에도
SpatialErrorMap을 화면에 그리는 곳이 없었다 - calibration/spatial_error_map.py는
ASCII 포맷터(format_spatial_error_map)만 갖고 있었다. 이 위젯은 그 ASCII
그리드와 같은 데이터(SpatialErrorCell.mean_dx/mean_dy)를 실제 화살표로
그린다. ui/radial_profile_view.py::RadialProfileChartWidget과 같은 방식
(QPainter 직접 그리기, matplotlib 등 외부 라이브러리 의존 없음)을 따른다.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from calibration.types import SpatialErrorMap
from ui.theme import Theme

_ARROW_COLOR_LOW = QColor(Theme.GOOD)
_ARROW_COLOR_HIGH = QColor(Theme.BAD)
_AXIS_COLOR = QColor(Theme.GRAPH_AXIS)
_GRID_COLOR = QColor(Theme.GRAPH_GRID)


def _lerp_color(t: float) -> QColor:
    """0(작은 오차, 초록) ~ 1(큰 오차, 빨강) 선형 보간 - radial_profile_view.py
    의 _lerp_color와 동일한 관례."""
    t = max(0.0, min(1.0, t))
    r = int(_ARROW_COLOR_LOW.red() + (_ARROW_COLOR_HIGH.red() - _ARROW_COLOR_LOW.red()) * t)
    g = int(_ARROW_COLOR_LOW.green() + (_ARROW_COLOR_HIGH.green() - _ARROW_COLOR_LOW.green()) * t)
    b = int(_ARROW_COLOR_LOW.blue() + (_ARROW_COLOR_HIGH.blue() - _ARROW_COLOR_LOW.blue()) * t)
    return QColor(r, g, b)


class VectorFieldChartWidget(QWidget):
    """Windshield로 인한 픽셀 변위(dx,dy)를 이미지 grid 위 화살표로 표시.

        ↖ ↑ ↑ ↗
        ← ← · → →
        ↙ ↓ ↓ ↘

    형태를 실제 벡터(길이=오차 크기 비례, 방향=atan2(dy,dx))로 그린다.
    SpatialErrorCell.mean_dx/mean_dy/rms를 그대로 쓴다 - 새 계산 없음.
    """

    _MARGIN = 24

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._map: SpatialErrorMap | None = None
        self.setMinimumHeight(280)

    def set_spatial_error_map(self, smap: SpatialErrorMap | None) -> None:
        self._map = smap
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, QColor(Theme.GRAPH_BG))

        if not self._map or not self._map.cells:
            painter.setPen(QPen(_AXIS_COLOR))
            painter.drawText(rect, Qt.AlignCenter, "표시할 데이터가 없습니다.\n(Baseline 계산 후 표시됩니다)")
            painter.end()
            return

        rows, cols = self._map.rows, self._map.cols
        plot_rect = QRectF(
            self._MARGIN, self._MARGIN,
            rect.width() - 2 * self._MARGIN, rect.height() - 2 * self._MARGIN,
        )
        if rows <= 0 or cols <= 0 or plot_rect.width() <= 0 or plot_rect.height() <= 0:
            painter.end()
            return
        cell_w = plot_rect.width() / cols
        cell_h = plot_rect.height() / rows

        grid = {(c.row, c.col): c for c in self._map.cells}
        magnitudes = [
            math.hypot(c.mean_dx, c.mean_dy)
            for c in self._map.cells
            if c.num_points > 0 and c.mean_dx is not None and c.mean_dy is not None
        ]
        max_mag = max(magnitudes) if magnitudes else 1.0
        max_mag = max(max_mag, 1e-6)
        max_arrow_len = min(cell_w, cell_h) * 0.42

        painter.setPen(QPen(_GRID_COLOR))
        for r in range(rows + 1):
            y = plot_rect.top() + r * cell_h
            painter.drawLine(QPointF(plot_rect.left(), y), QPointF(plot_rect.right(), y))
        for c in range(cols + 1):
            x = plot_rect.left() + c * cell_w
            painter.drawLine(QPointF(x, plot_rect.top()), QPointF(x, plot_rect.bottom()))

        for r in range(rows):
            for c in range(cols):
                cell = grid.get((r, c))
                cx = plot_rect.left() + (c + 0.5) * cell_w
                cy = plot_rect.top() + (r + 0.5) * cell_h

                if cell is None or cell.num_points == 0 or cell.mean_dx is None or cell.mean_dy is None:
                    painter.setPen(QPen(QColor(Theme.TEXT_DISABLED)))
                    painter.drawText(QRectF(cx - cell_w / 2, cy - 8, cell_w, 16), Qt.AlignCenter, "·")
                    continue

                mag = math.hypot(cell.mean_dx, cell.mean_dy)
                t = mag / max_mag
                color = _lerp_color(t)
                arrow_len = t * max_arrow_len
                if mag > 1e-9:
                    ux, uy = cell.mean_dx / mag, cell.mean_dy / mag
                else:
                    ux, uy = 0.0, 0.0

                tip = QPointF(cx + ux * arrow_len, cy + uy * arrow_len)
                self._draw_arrow(painter, QPointF(cx, cy), tip, color)

                painter.setPen(QPen(QColor(Theme.TEXT_VALUE)))
                painter.drawText(
                    QRectF(cx - cell_w / 2, cy + cell_h / 2 - 16, cell_w, 14),
                    Qt.AlignCenter,
                    f"{mag:.2f}px",
                )

        painter.end()

    @staticmethod
    def _draw_arrow(painter: QPainter, start: QPointF, tip: QPointF, color: QColor) -> None:
        painter.setPen(QPen(color, 2))
        painter.drawLine(start, tip)
        if (tip - start).manhattanLength() < 1e-6:
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(start, 2.5, 2.5)
            return
        angle = math.atan2(tip.y() - start.y(), tip.x() - start.x())
        head_len = 6.0
        head_angle = math.radians(28)
        p1 = QPointF(
            tip.x() - head_len * math.cos(angle - head_angle),
            tip.y() - head_len * math.sin(angle - head_angle),
        )
        p2 = QPointF(
            tip.x() - head_len * math.cos(angle + head_angle),
            tip.y() - head_len * math.sin(angle + head_angle),
        )
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(QPolygonF([tip, p1, p2]))
