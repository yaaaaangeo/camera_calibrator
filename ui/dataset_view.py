"""
camera_calibrator.ui.dataset_view
=====================================

설계 문서 6번(Frame Quality Score) 스타일 - 사진 목록에 성공/실패뿐 아니라
검출 코너 수, 재투영 오차, 상태를 함께 보여준다.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.types import Dataset, FrameStatus, QualityGrade
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


class DatasetView(QWidget):
    """이미지 목록 테이블 + 요약 라벨."""

    def __init__(self, parent: QWidget | None = None, *, group_title: str = "Dataset"):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.summary_label = QLabel("이미지를 불러오면 여기에 요약이 표시됩니다.")
        layout.addWidget(self.summary_label)

        group = QGroupBox(group_title)
        group_layout = QVBoxLayout(group)

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
        group_layout.addWidget(self.table)

        layout.addWidget(group)

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


def _qcolor(hex_str: str):
    from PySide6.QtGui import QColor

    return QColor(hex_str)
