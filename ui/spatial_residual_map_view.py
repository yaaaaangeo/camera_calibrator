"""
camera_calibrator.ui.spatial_residual_map_view
==============================================================

Paper Evidence 단계 - Hold-out spatial residual(calibration/validation.py::
compute_holdout_spatial_evidence가 만든 SpatialErrorMap)을 heatmap(+평균
오차 방향 화살표)으로 그린다. 이 프로젝트는 matplotlib를 의도적으로 쓰지
않는다(requirements.txt/ui/windshield_vector_field_view.py 참고 - QPainter로
직접 그리는 게 기존 컨벤션) - 여기서도 그 컨벤션을 그대로 따른다.

계산은 전혀 하지 않는다 - 이미 계산된 SpatialErrorMap(칸별 RMS/평균
dx,dy)을 받아 그리기만 한다. 화면에 표시하지 않고도(QApplication 인스턴스만
있으면) QPixmap.save(path)로 PNG를 바로 저장할 수 있어 headless export에도
쓸 수 있다.

여러 모델을 비교할 때 반드시 지켜야 하는 것: 같은 color/arrow scale을
공유해야 Fisheye 같은 모델이 시각적으로 과장/축소되지 않는다 -
render_model_comparison()이 value_range를 모델들 중 공통 max로 강제한다.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap

from calibration.types import SpatialErrorMap


def shared_magnitude_range(maps: list[SpatialErrorMap]) -> tuple[float, float]:
    """여러 모델의 SpatialErrorMap에서 공통으로 쓸 (min, max) RMS 범위.
    논문 비교 그림에서는 각 모델을 따로 그리지 말고 이 범위를 전부에 동일하게
    넘겨야 한다 - 그래야 "Fisheye가 시각적으로 더 나빠 보인다"가 실제
    magnitude 차이인지 단순히 그 모델만 다른 scale을 썼기 때문인지 헷갈리지
    않는다.
    """
    values = [c.rms for m in maps for c in m.cells if c.rms is not None]
    if not values:
        return (0.0, 1.0)
    return (0.0, max(values))


def _color_for_value(value: float | None, vmin: float, vmax: float) -> QColor:
    if value is None:
        return QColor(230, 230, 230)
    span = max(vmax - vmin, 1e-9)
    t = max(0.0, min(1.0, (value - vmin) / span))
    # 외부 colormap 라이브러리 없이 파랑(낮음) -> 노랑 -> 빨강(높음) 2단 보간.
    if t < 0.5:
        u = t / 0.5
        r, g, b = 30 + u * (255 - 30), 80 + u * (200 - 80), 200 - u * 150
    else:
        u = (t - 0.5) / 0.5
        r, g, b = 255.0, 200 - u * 150, 50 - u * 50
    clamp = lambda v: max(0, min(255, int(v)))  # noqa: E731
    return QColor(clamp(r), clamp(g), clamp(b))


def render_spatial_error_map(
    smap: SpatialErrorMap,
    *,
    value_range: tuple[float, float] | None = None,
    title: str = "",
    canvas_size: tuple[int, int] = (420, 340),
    arrow_scale: float | None = None,
) -> QPixmap:
    """SpatialErrorMap 하나를 grid heatmap(+평균 오차 방향 화살표)으로 그린
    QPixmap을 반환한다. 화면에 보여줄 수도, pixmap.save(path)로 PNG 저장도
    가능하다. value_range를 명시하지 않으면 이 map 자신의 min~max로
    정규화되므로 - 다른 모델과 비교하려면 반드시 같은 value_range를 넘겨야
    한다(shared_magnitude_range() 또는 render_model_comparison() 참고).
    """
    w, h = canvas_size
    pixmap = QPixmap(w, h)
    pixmap.fill(Qt.white)
    painter = QPainter(pixmap)
    try:
        title_h = 22 if title else 0
        margin = 24
        top = margin + title_h
        plot_w = w - 2 * margin
        plot_h = h - margin - top

        if title:
            painter.setFont(QFont("Sans", 10, QFont.Bold))
            painter.setPen(QPen(QColor(0, 0, 0)))
            painter.drawText(QRectF(0, 2, w, title_h), Qt.AlignCenter, title)

        vmin, vmax = value_range or shared_magnitude_range([smap])
        cols = max(smap.cols, 1)
        rows = max(smap.rows, 1)
        cell_w = plot_w / cols
        cell_h = plot_h / rows

        for cell in smap.cells:
            x0 = margin + cell.col * cell_w
            y0 = top + cell.row * cell_h
            color = _color_for_value(cell.rms, vmin, vmax)
            painter.fillRect(QRectF(x0, y0, cell_w, cell_h), color)
            painter.setPen(QPen(QColor(120, 120, 120)))
            painter.drawRect(QRectF(x0, y0, cell_w, cell_h))
            if cell.rms is not None:
                painter.setPen(QPen(QColor(20, 20, 20)))
                painter.setFont(QFont("Sans", 8))
                painter.drawText(QRectF(x0, y0, cell_w, cell_h), Qt.AlignCenter, f"{cell.rms:.2f}")
            if cell.mean_dx is not None and cell.mean_dy is not None:
                scale = arrow_scale if arrow_scale and arrow_scale > 0 else (vmax if vmax > 0 else 1.0)
                cx, cy = x0 + cell_w / 2, y0 + cell_h / 2
                dx = cell.mean_dx / scale * min(cell_w, cell_h) * 0.4
                dy = cell.mean_dy / scale * min(cell_w, cell_h) * 0.4
                painter.setPen(QPen(QColor(10, 10, 10), 1.5))
                painter.drawLine(QPointF(cx, cy), QPointF(cx + dx, cy + dy))

        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawRect(QRectF(margin, top, plot_w, plot_h))
    finally:
        painter.end()
    return pixmap


def render_model_comparison(
    maps_by_label: dict[str, SpatialErrorMap],
    canvas_size_each: tuple[int, int] = (360, 300),
) -> QPixmap:
    """여러 모델(라벨 -> SpatialErrorMap)을 가로로 나란히, 반드시 같은
    color/arrow scale로 그린 하나의 QPixmap을 만든다 - "Brown-Conrady /
    Rational / Fisheye Hold-out Spatial Residual"을 한 그림으로 비교할 때
    쓴다. 개별로 render_spatial_error_map()을 각자 다른 scale로 부르면
    Fisheye가 시각적으로 과장/축소될 수 있으므로 이 함수를 우선 쓴다.
    """
    vmin, vmax = shared_magnitude_range(list(maps_by_label.values()))
    tiles = [
        render_spatial_error_map(
            smap,
            value_range=(vmin, vmax),
            title=f"{label} Hold-out Spatial Residual",
            canvas_size=canvas_size_each,
            arrow_scale=vmax if vmax > 0 else 1.0,
        )
        for label, smap in maps_by_label.items()
    ]
    if not tiles:
        return QPixmap(1, 1)
    total_w = sum(t.width() for t in tiles)
    total_h = max(t.height() for t in tiles)
    combined = QPixmap(total_w, total_h)
    combined.fill(Qt.white)
    painter = QPainter(combined)
    try:
        x = 0
        for t in tiles:
            painter.drawPixmap(x, 0, t)
            x += t.width()
    finally:
        painter.end()
    return combined
