"""
camera_calibrator.ui.coverage_view
======================================

설계 문서 5번(Coverage Map), 7번(자세 다양성)을 시각화.
백엔드 계산(calibration/quality.py)의 결과(CoverageCell, DiversityScores)를
받아서 그리기만 한다 - 이 파일 안에서 coverage_score 등을 재계산하지 않는다.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from calibration.types import CoverageCell, DiversityScores


def _score_to_color(score: float) -> QColor:
    """0(빨강, 부족) ~ 1(초록, 충분) 그라데이션."""
    score = max(0.0, min(1.0, score))
    r = int(255 * (1 - score))
    g = int(200 * score)
    return QColor(r, g, 60)


class CoverageGridWidget(QWidget):
    """4x4(가변) 격자를 색상 셀로 표시. 셀 안에는 코너 수를 숫자로 표기."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._layout = QGridLayout(self)
        self._layout.setSpacing(2)
        self._cell_labels: dict[tuple[int, int], QLabel] = {}

    def set_cells(self, cells: list[CoverageCell], rows: int, cols: int) -> None:
        # 기존 위젯 정리
        for lbl in self._cell_labels.values():
            self._layout.removeWidget(lbl)
            lbl.deleteLater()
        self._cell_labels.clear()

        grid = {(c.row, c.col): c for c in cells}
        for r in range(rows):
            for c in range(cols):
                cell = grid.get((r, c))
                score = cell.coverage_score if cell else 0.0
                count = cell.corner_count if cell else 0

                label = QLabel(str(count))
                label.setAlignment(Qt.AlignCenter)
                label.setMinimumSize(60, 45)
                color = _score_to_color(score)
                text_color = "#000000" if score > 0.4 else "#ffffff"
                label.setStyleSheet(
                    f"background-color: {color.name()}; color: {text_color}; "
                    f"border: 1px solid #333; font-weight: bold;"
                )
                self._layout.addWidget(label, r, c)
                self._cell_labels[(r, c)] = label


class DiversityBarsWidget(QWidget):
    """설계 문서 7번 ASCII 바 차트를 QProgressBar로."""

    _ROWS = [
        ("position_coverage", "Position Coverage"),
        ("distance_diversity", "Distance Diversity"),
        ("rotation_diversity", "Rotation Diversity"),
        ("edge_coverage", "Edge Coverage"),
    ]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._bars: dict[str, QProgressBar] = {}

        for key, label in self._ROWS:
            row = QHBoxLayout()
            row.addWidget(QLabel(label), stretch=0)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            row.addWidget(bar, stretch=1)
            self._bars[key] = bar
            layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Overall Dataset Quality"))
        self._overall_bar = QProgressBar()
        self._overall_bar.setRange(0, 100)
        row.addWidget(self._overall_bar, stretch=1)
        layout.addLayout(row)

    def set_diversity(self, diversity: DiversityScores) -> None:
        values = {
            "position_coverage": diversity.position_coverage,
            "distance_diversity": diversity.distance_diversity,
            "rotation_diversity": diversity.rotation_diversity,
            "edge_coverage": diversity.edge_coverage,
        }
        for key, bar in self._bars.items():
            bar.setValue(round(values[key] * 100))
        self._overall_bar.setValue(round(diversity.overall * 100))


class CoverageView(QWidget):
    """그리드 + 다양성 바 + 경고 목록을 한 화면에."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        grid_group = QGroupBox("Coverage Map")
        grid_layout = QVBoxLayout(grid_group)
        self.grid_widget = CoverageGridWidget()
        grid_layout.addWidget(self.grid_widget)
        layout.addWidget(grid_group)

        diversity_group = QGroupBox("Dataset Diversity")
        diversity_layout = QVBoxLayout(diversity_group)
        self.diversity_widget = DiversityBarsWidget()
        diversity_layout.addWidget(self.diversity_widget)
        layout.addWidget(diversity_group)

        warning_group = QGroupBox("경고")
        warning_layout = QVBoxLayout(warning_group)
        self.warning_list = QListWidget()
        warning_layout.addWidget(self.warning_list)
        layout.addWidget(warning_group)

    def set_quality(
        self,
        cells: list[CoverageCell],
        diversity: DiversityScores,
        warnings: list[str],
        rows: int = 4,
        cols: int = 4,
    ) -> None:
        self.grid_widget.set_cells(cells, rows, cols)
        self.diversity_widget.set_diversity(diversity)
        self.warning_list.clear()
        if warnings:
            self.warning_list.addItems(warnings)
        else:
            self.warning_list.addItem("경고 없음 (커버리지 양호)")
