"""
camera_calibrator.ui.dataset_view
=====================================

설계 문서 5번(Coverage Map), 6번(Frame Quality Score), 7번(자세 다양성) -
Dataset 탭 하나에 Coverage Map -> Dataset Diversity -> Batch(기존 이미지 목록)
순서로 모아 보여준다. 백엔드 계산(calibration/quality.py)의 결과를 받아서
그리기만 한다 - 이 파일 안에서 coverage_score 등을 재계산하지 않는다.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.types import (
    CoverageCell,
    Dataset,
    DatasetQualityScore,
    DiversityScores,
    FrameStatus,
    QualityGrade,
)
from ui.theme import Theme

_STATUS_LABEL = {
    FrameStatus.PENDING: "대기",
    FrameStatus.DETECTED: "검출됨",
    FrameStatus.DETECTION_FAILED: "검출 실패",
    FrameStatus.DISABLED_OUTLIER: "제외됨(이상치)",
    FrameStatus.DISABLED_MANUAL: "제외됨(수동)",
}

_STATUS_COLOR = {
    FrameStatus.PENDING: Theme.TEXT_SECONDARY,
    FrameStatus.DETECTED: Theme.GOOD,
    FrameStatus.DETECTION_FAILED: Theme.BAD,
    FrameStatus.DISABLED_OUTLIER: Theme.WARNING,
    FrameStatus.DISABLED_MANUAL: Theme.TEXT_DISABLED,
}

# 설계 문서 6번 - Frame Quality Score 등급 표시
_GRADE_LABEL = {
    QualityGrade.EXCELLENT: "✓ Excellent",
    QualityGrade.VERY_GOOD: "✓ Very Good",
    QualityGrade.GOOD: "✓ Good",
    QualityGrade.WARNING: "⚠ Warning",
    QualityGrade.POOR: "⚠ Poor",
    QualityGrade.REJECT: "✕ Reject",
}

_GRADE_COLOR = {
    QualityGrade.EXCELLENT: Theme.GOOD,
    QualityGrade.VERY_GOOD: Theme.GOOD,
    QualityGrade.GOOD: Theme.GOOD,
    QualityGrade.WARNING: Theme.WARNING,
    QualityGrade.POOR: Theme.BAD,
    QualityGrade.REJECT: Theme.BAD,
}


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


class DatasetView(QWidget):
    """Coverage Map -> Dataset Diversity -> Batch(이미지 목록 테이블) 통합 뷰."""

    def __init__(self, parent: QWidget | None = None, *, group_title: str = "Batch"):
        super().__init__(parent)
        outer_layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)

        # 1+2. Coverage Map(왼쪽) / Dataset Diversity(오른쪽) 한 줄에 배치
        top_row = QHBoxLayout()

        grid_group = QGroupBox("Coverage Map")
        grid_layout = QVBoxLayout(grid_group)
        self.grid_widget = CoverageGridWidget()
        grid_layout.addWidget(self.grid_widget)
        top_row.addWidget(grid_group, stretch=1)

        diversity_group = QGroupBox("Dataset Diversity")
        diversity_layout = QVBoxLayout(diversity_group)
        self.diversity_widget = DiversityBarsWidget()
        diversity_layout.addWidget(self.diversity_widget)
        self.dataset_score_label = QLabel("아직 계산되지 않았습니다.")
        self.dataset_score_label.setWordWrap(True)
        diversity_layout.addWidget(self.dataset_score_label)
        top_row.addWidget(diversity_group, stretch=1)

        layout.addLayout(top_row)

        # 3. Batch (기존 Dataset 이미지 목록)
        self.summary_label = QLabel("이미지를 불러오면 여기에 요약이 표시됩니다.")
        layout.addWidget(self.summary_label)

        batch_group = QGroupBox(group_title)
        batch_layout = QVBoxLayout(batch_group)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["파일", "상태", "코너 수", "선명도", "재투영 오차(px)", "품질 점수", "등급"]
        )
        # "파일"(짧은 이름)이 아니라 "상태"(검출 실패 이유 등 긴 문장이 들어감)가
        # 남는 공간을 가져가야 한다 - 예전엔 반대였어서 상태 텍스트가 좁은
        # 칸에 잘려 보였다. 숫자/등급 컬럼들은 내용 크기에 맞춰 고정폭.
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for col in range(2, 7):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.setColumnWidth(0, 140)
        # 상태 칸에 긴 실패 이유가 들어가면 줄바꿈 때문에 그 행만 검출 성공
        # 행보다 훨씬 높아져서 표가 들쭉날쭉해 보였다. 줄바꿈을 끄고 한 줄로
        # 말줄임(...) 처리해서 모든 행 높이를 통일하고, 전체 문구는 계속
        # tooltip(위 status_item.setToolTip)으로 볼 수 있게 유지한다.
        self.table.setWordWrap(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        batch_layout.addWidget(self.table)

        layout.addWidget(batch_group)

        scroll.setWidget(content)
        outer_layout.addWidget(scroll)
        self.scroll_area = scroll

    def set_dataset(self, dataset: Dataset) -> None:
        self.table.setRowCount(len(dataset.frames))
        for row, frame in enumerate(dataset.frames):
            det = frame.detection
            name_item = QTableWidgetItem(frame.image_info.image_id)

            status_item = QTableWidgetItem(_STATUS_LABEL.get(frame.status, frame.status.value))
            status_item.setForeground(_qcolor(_STATUS_COLOR.get(frame.status, Theme.INFO)))
            if frame.status == FrameStatus.DETECTION_FAILED and det and det.failure_reason:
                # 실패 이유를 셀에 바로 짧게 붙이고, 전체 문구는 tooltip으로 (칸이 좁아 잘릴 수 있음)
                status_item.setText(f"{_STATUS_LABEL[FrameStatus.DETECTION_FAILED]}: {det.failure_reason}")
                status_item.setToolTip(det.failure_reason)
            elif frame.status in (FrameStatus.DISABLED_OUTLIER, FrameStatus.DISABLED_MANUAL) and frame.disabled_reason:
                status_item.setToolTip(frame.disabled_reason)

            corners = str(det.num_corners) if det else "-"
            sharpness = f"{frame.image_info.sharpness:.0f}" if frame.image_info.sharpness else "-"
            error = f"{frame.reprojection_error:.3f}" if frame.reprojection_error is not None else "-"

            score_text = f"{frame.quality.overall_score:.0f}" if frame.quality else "-"
            grade_item = QTableWidgetItem(
                _GRADE_LABEL.get(frame.quality.grade, "-") if frame.quality else "-"
            )
            if frame.quality:
                grade_item.setForeground(_qcolor(_GRADE_COLOR.get(frame.quality.grade, Theme.INFO)))

            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, status_item)
            self.table.setItem(row, 2, QTableWidgetItem(corners))
            self.table.setItem(row, 3, QTableWidgetItem(sharpness))
            self.table.setItem(row, 4, QTableWidgetItem(error))
            self.table.setItem(row, 5, QTableWidgetItem(score_text))
            self.table.setItem(row, 6, grade_item)

        total = dataset.num_total
        detected = dataset.num_detected
        enabled = dataset.num_enabled
        self.summary_label.setText(
            f"총 {total}장  |  검출 성공 {detected}장  |  "
            f"현재 사용 중 {enabled}장 ({enabled/total*100:.0f}%)" if total else "이미지 없음"
        )
        # 줄바꿈이 꺼져 있어(setWordWrap(False)) 모든 행이 원래 한 줄 높이지만,
        # 폰트/DPI 차이에 따른 실제 한 줄 높이를 여기서 다시 맞춰준다.
        self.table.resizeRowsToContents()

    def refresh_errors(self, dataset: Dataset) -> None:
        """재계산 후 프레임별 오차/상태만 갱신 (테이블 구조는 그대로)."""
        self.set_dataset(dataset)

    def set_quality(
        self,
        cells: list[CoverageCell],
        diversity: DiversityScores | None,
        warnings: list[str],
        rows: int = 4,
        cols: int = 4,
    ) -> None:
        # warnings(예: "관측이 부족합니다" 류 커버리지 경고 문구)는 더 이상
        # 화면에 표시하지 않는다 - 계산(analyze_dataset_quality)은 그대로
        # 유지되므로 호출부 시그니처는 바꾸지 않는다.
        self.grid_widget.set_cells(cells, rows, cols)
        self.diversity_widget.set_diversity(diversity)

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


def _qcolor(hex_str: str):
    from PySide6.QtGui import QColor

    return QColor(hex_str)
