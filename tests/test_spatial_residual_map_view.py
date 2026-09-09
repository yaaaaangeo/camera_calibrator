"""
tests/test_spatial_residual_map_view.py
====================================

Paper Evidence 단계 - ui/spatial_residual_map_view.py(QPainter 기반 held-out
spatial residual heatmap 렌더러, matplotlib 미사용) 회귀 테스트.

계산 로직이 없는 순수 렌더러이므로 여기서는 "여러 모델이 같은 color/arrow
scale을 공유하는지"와 "PNG로 실제 저장되는지"만 확인한다.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets", reason="PySide6.QtWidgets is not importable in this environment")

from PySide6.QtWidgets import QApplication

from calibration.types import SpatialErrorCell, SpatialErrorMap


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_map(base_rms: float, rows: int = 3, cols: int = 3) -> SpatialErrorMap:
    cells = [
        SpatialErrorCell(row=r, col=c, num_points=5, rms=base_rms + r * 0.1 + c * 0.05, mean_dx=0.2, mean_dy=-0.1)
        for r in range(rows) for c in range(cols)
    ]
    return SpatialErrorMap(cells=cells, rows=rows, cols=cols)


def test_shared_magnitude_range_uses_max_across_all_maps(qapp):
    from ui.spatial_residual_map_view import shared_magnitude_range

    low = _make_map(1.0)
    high = _make_map(3.0)
    vmin, vmax = shared_magnitude_range([low, high])
    assert vmin == 0.0
    expected_max = max(c.rms for c in high.cells)
    assert vmax == pytest.approx(expected_max)


def test_shared_magnitude_range_empty_maps_returns_default():
    from ui.spatial_residual_map_view import shared_magnitude_range

    empty_map = SpatialErrorMap(cells=[SpatialErrorCell(row=0, col=0)], rows=1, cols=1)
    vmin, vmax = shared_magnitude_range([empty_map])
    assert (vmin, vmax) == (0.0, 1.0)


def test_render_single_map_produces_nonempty_pixmap(qapp, tmp_path):
    from ui.spatial_residual_map_view import render_spatial_error_map

    smap = _make_map(1.5)
    pixmap = render_spatial_error_map(smap, title="Brown-Conrady Hold-out Spatial Residual")
    assert not pixmap.isNull()
    path = tmp_path / "single.png"
    assert pixmap.save(str(path))
    assert path.stat().st_size > 0


def test_render_model_comparison_uses_shared_scale_and_stacks_tiles(qapp, tmp_path):
    from ui.spatial_residual_map_view import render_model_comparison

    maps = {
        "Brown-Conrady": _make_map(1.0),
        "Rational": _make_map(2.0),
        "Fisheye": _make_map(0.5),
    }
    combined = render_model_comparison(maps, canvas_size_each=(300, 250))
    assert combined.width() == 300 * 3
    assert combined.height() == 250
    path = tmp_path / "combined.png"
    assert combined.save(str(path))
    assert path.stat().st_size > 0
