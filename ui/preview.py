"""
camera_calibrator.ui.preview
================================

설계 문서 2번(실시간 왜곡 보정 토글), 17번 Step10(Undistortion 검증).

이 위젯은 cv2 계산을 직접 하지 않는다 - calibration.models.common.undistort_image()를
그대로 호출하고, 결과를 QPixmap으로 변환해서 보여주기만 한다.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from calibration.types import CalibrationResult, CameraConfig, CameraModelType, Dataset
from calibration.models.common import undistort_image

_MODEL_LABELS = {
    CameraModelType.PINHOLE: "Pinhole",
    CameraModelType.EXTENDED_PINHOLE: "Extended Pinhole",
    CameraModelType.FISHEYE: "Fisheye",
}
_MODEL_ORDER = [CameraModelType.PINHOLE, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE]

_PREVIEW_MAX_WIDTH = 480


def _cv_to_qpixmap(img_bgr: np.ndarray, max_width: int = _PREVIEW_MAX_WIDTH) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(qimg)
    if pixmap.width() > max_width:
        pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
    return pixmap


class PreviewView(QWidget):
    """이미지 선택 + 모델 선택 -> 원본/보정 이미지 나란히 표시."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._dataset: Dataset | None = None
        self._camera_config: CameraConfig | None = None
        self._calibration_results: dict[CameraModelType, CalibrationResult] = {}

        layout = QVBoxLayout(self)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("이미지:"))
        self.image_combo = QComboBox()
        control_row.addWidget(self.image_combo, stretch=1)

        control_row.addWidget(QLabel("모델:"))
        self.model_combo = QComboBox()
        for m in _MODEL_ORDER:
            self.model_combo.addItem(_MODEL_LABELS[m], userData=m)
        control_row.addWidget(self.model_combo)

        self.preview_button = QPushButton("미리보기 갱신")
        self.preview_button.clicked.connect(self._update_preview)
        control_row.addWidget(self.preview_button)
        layout.addLayout(control_row)

        images_row = QHBoxLayout()
        original_group = QGroupBox("원본")
        original_layout = QVBoxLayout(original_group)
        self.original_label = QLabel("이미지를 선택하고 [미리보기 갱신]을 누르세요.")
        self.original_label.setAlignment(Qt.AlignCenter)
        original_layout.addWidget(self.original_label)
        images_row.addWidget(original_group)

        undistorted_group = QGroupBox("왜곡 보정 후")
        undistorted_layout = QVBoxLayout(undistorted_group)
        self.undistorted_label = QLabel("-")
        self.undistorted_label.setAlignment(Qt.AlignCenter)
        undistorted_layout.addWidget(self.undistorted_label)
        images_row.addWidget(undistorted_group)

        layout.addLayout(images_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def set_context(
        self,
        dataset: Dataset,
        camera_config: CameraConfig,
        calibration_results: dict[CameraModelType, CalibrationResult],
    ) -> None:
        self._dataset = dataset
        self._camera_config = camera_config
        self._calibration_results = calibration_results

        self.image_combo.clear()
        for frame in dataset.enabled_frames:
            self.image_combo.addItem(frame.image_info.image_id, userData=frame.image_info.path)

    def select_model(self, model: CameraModelType) -> None:
        idx = self.model_combo.findData(model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

    def _update_preview(self) -> None:
        if self._camera_config is None or self.image_combo.count() == 0:
            self.status_label.setText("먼저 이미지를 불러오고 캘리브레이션을 실행하세요.")
            return

        image_path = self.image_combo.currentData()
        model = self.model_combo.currentData()
        result = self._calibration_results.get(model)

        if result is None or not result.success:
            self.status_label.setText(f"{_MODEL_LABELS.get(model, model)} 모델의 캘리브레이션 결과가 없습니다.")
            return

        img = cv2.imread(image_path)
        if img is None:
            self.status_label.setText(f"이미지를 읽을 수 없습니다: {image_path}")
            return

        try:
            undistorted = undistort_image(img, result, self._camera_config)
        except ValueError as e:
            self.status_label.setText(str(e))
            return

        self.original_label.setPixmap(_cv_to_qpixmap(img))
        self.undistorted_label.setPixmap(_cv_to_qpixmap(undistorted))
        self.status_label.setText(
            f"{_MODEL_LABELS.get(model, model)} 모델 기준 보정 결과입니다. "
            f"외곽부(모서리) 직선이 얼마나 곧아졌는지 확인하세요."
        )
