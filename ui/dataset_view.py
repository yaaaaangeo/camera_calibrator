"""
camera_calibrator.ui.dataset_view
=====================================

설계 문서 6번(Frame Quality Score) 스타일 - 사진 목록에 성공/실패뿐 아니라
검출 코너 수, 재투영 오차, 상태를 함께 보여준다.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.types import Dataset, FrameStatus

_STATUS_LABEL = {
    FrameStatus.PENDING: "대기",
    FrameStatus.DETECTED: "검출됨",
    FrameStatus.DETECTION_FAILED: "검출 실패",
    FrameStatus.DISABLED_OUTLIER: "제외됨(이상치)",
    FrameStatus.DISABLED_MANUAL: "제외됨(수동)",
}

_STATUS_COLOR = {
    FrameStatus.PENDING: "#888888",
    FrameStatus.DETECTED: "#2e7d32",
    FrameStatus.DETECTION_FAILED: "#c62828",
    FrameStatus.DISABLED_OUTLIER: "#ef6c00",
    FrameStatus.DISABLED_MANUAL: "#6d4c41",
}


class DatasetView(QWidget):
    """이미지 목록 테이블 + 요약 라벨."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.summary_label = QLabel("이미지를 불러오면 여기에 요약이 표시됩니다.")
        layout.addWidget(self.summary_label)

        group = QGroupBox("Dataset")
        group_layout = QVBoxLayout(group)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["파일", "상태", "코너 수", "선명도", "재투영 오차(px)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
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
            status_item.setForeground(_qcolor(_STATUS_COLOR.get(frame.status, "#000000")))

            corners = str(det.num_corners) if det else "-"
            sharpness = f"{frame.image_info.sharpness:.0f}" if frame.image_info.sharpness else "-"
            error = f"{frame.reprojection_error:.3f}" if frame.reprojection_error is not None else "-"

            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, status_item)
            self.table.setItem(row, 2, QTableWidgetItem(corners))
            self.table.setItem(row, 3, QTableWidgetItem(sharpness))
            self.table.setItem(row, 4, QTableWidgetItem(error))

        total = dataset.num_total
        detected = dataset.num_detected
        enabled = dataset.num_enabled
        self.summary_label.setText(
            f"총 {total}장  |  검출 성공 {detected}장  |  "
            f"현재 사용 중 {enabled}장 ({enabled/total*100:.0f}%)" if total else "이미지 없음"
        )

    def refresh_errors(self, dataset: Dataset) -> None:
        """재계산 후 프레임별 오차/상태만 갱신 (테이블 구조는 그대로)."""
        self.set_dataset(dataset)


def _qcolor(hex_str: str):
    from PySide6.QtGui import QColor

    return QColor(hex_str)
