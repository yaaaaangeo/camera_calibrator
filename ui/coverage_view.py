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
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from calibration.types import CoverageCell, DatasetQualityScore, DiversityScores
from ui.theme import Theme


def _score_to_color(score: float) -> QColor:
    """Dark status surface: insufficient red -> sufficient NVIDIA green."""
    score = max(0.0, min(1.0, score))
    low = QColor(Theme.COVERAGE_LOW)
    high = QColor(Theme.COVERAGE_HIGH)
    return QColor(
        int(low.red() + (high.red() - low.red()) * score),
        int(low.green() + (high.green() - low.green()) * score),
        int(low.blue() + (high.blue() - low.blue()) * score),
    )


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
                text_color = Theme.TEXT_VALUE
                border_color = Theme.GOOD if score >= 0.7 else (Theme.WARNING if score >= 0.35 else Theme.BAD)
                label.setStyleSheet(
                    f"background-color: {color.name()}; color: {text_color}; "
                    f"border: 1px solid {border_color}; font-weight: bold;"
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

        # 라벨 텍스트 길이가 제각각이면(Position Coverage vs Edge Coverage 등)
        # 그 뒤에 붙는 막대(stretch=1)의 시작 x좌표가 줄마다 달라져서 막대
        # 크기가 서로 다르게 보인다 - 라벨 너비를 고정해서 막대 시작점을 맞춘다.
        _LABEL_WIDTH = 170

        for key, label in self._ROWS:
            row = QHBoxLayout()
            label_widget = QLabel(label)
            label_widget.setFixedWidth(_LABEL_WIDTH)
            row.addWidget(label_widget, stretch=0)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            row.addWidget(bar, stretch=1)
            self._bars[key] = bar
            layout.addLayout(row)

        row = QHBoxLayout()
        overall_label = QLabel("Overall Dataset Quality")
        overall_label.setFixedWidth(_LABEL_WIDTH)
        row.addWidget(overall_label, stretch=0)
        self._overall_bar = QProgressBar()
        self._overall_bar.setRange(0, 100)
        row.addWidget(self._overall_bar, stretch=1)
        layout.addLayout(row)

    def set_diversity(self, diversity: DiversityScores | None) -> None:
        if diversity is None:
            # 아직 Coverage/Diversity 분석이 안 된 데이터셋(예: quality 분석 전에
            # 저장된 프로젝트를 불러온 경우) - 바를 0으로 비워두고 크래시하지 않는다.
            for bar in self._bars.values():
                bar.setValue(0)
            self._overall_bar.setValue(0)
            return
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
        outer_layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)

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

        # 설계 문서 4번 - Overall Dataset Score
        score_group = QGroupBox("Overall Dataset Score")
        score_layout = QVBoxLayout(score_group)
        self.dataset_score_label = QLabel("아직 계산되지 않았습니다.")
        self.dataset_score_label.setWordWrap(True)
        score_layout.addWidget(self.dataset_score_label)
        layout.addWidget(score_group)

        warning_group = QGroupBox("경고")
        warning_layout = QVBoxLayout(warning_group)
        self.warning_list = QListWidget()
        warning_layout.addWidget(self.warning_list)
        layout.addWidget(warning_group)

        scroll.setWidget(content)
        outer_layout.addWidget(scroll)
        self.scroll_area = scroll

    def set_quality(
        self,
        cells: list[CoverageCell],
        diversity: DiversityScores | None,
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

    def set_dataset_quality_score(self, score: DatasetQualityScore | None) -> None:
        if score is None:
            self.dataset_score_label.setText("아직 계산되지 않았습니다.")
            return
        self.dataset_score_label.setText(
            f"Total: {score.overall:.1f} ({score.grade.value})\n"
            f"Avg Frame Quality {score.avg_frame_quality:.1f}  |  "
            f"Detection Rate {score.detection_success_rate:.1f}%  |  "
            f"Coverage {score.coverage_score:.1f}  |  "
            f"Pose Diversity {score.diversity_score:.1f}  |  "
            f"Duplicate Penalty -{score.duplicate_penalty:.1f}"
        )
