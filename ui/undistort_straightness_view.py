"""
camera_calibrator.ui.undistort_straightness_view
=====================================================

옛 "③ Error Analysis" 서브탭 중 Undistort Preview와 Straightness Map을
한 화면으로 합친 뷰 (Edge Error Map은 별도 요청으로 제거됨).

계산은 새로 만들지 않는다 - 원본/보정 이미지는 기존 PreviewView와 동일하게
calibration.models.common.undistort_image()를, 보정 이미지 위 Straightness
라인 오버레이는 ui.straightness_view.render_straightness_overlay()를 그대로
재사용한다. 두 결과를 "원본 | 보정 후+Straightness" 두 패널로 나란히 보여주고,
컨트롤(이미지/모델 선택)도 하나로 합쳐 중복 콤보박스를 없앤다.
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

from calibration.types import CalibrationResult, CameraConfig, CameraModelType, Dataset, PatternConfig
from calibration.models.common import undistort_image
from calibration.straightness import compute_straightness_improvement
from ui.straightness_view import render_straightness_overlay

_MODEL_LABELS = {
    CameraModelType.PINHOLE: "Ideal Pinhole",
    CameraModelType.BROWN_CONRADY: "Brown-Conrady",
    CameraModelType.EXTENDED_PINHOLE: "Rational",
    CameraModelType.FISHEYE: "Fisheye",
}
_MODEL_ORDER = [
    CameraModelType.PINHOLE,
    CameraModelType.BROWN_CONRADY,
    CameraModelType.EXTENDED_PINHOLE,
    CameraModelType.FISHEYE,
]

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


class UndistortStraightnessView(QWidget):
    """이미지 선택 + 모델 선택 -> 원본 / (보정 후 + Straightness 오버레이) 나란히 표시."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._dataset: Dataset | None = None
        self._camera_config: CameraConfig | None = None
        self._calibration_results: dict[CameraModelType, CalibrationResult] = {}
        self._pattern_config: PatternConfig | None = None

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
        self.original_label.setProperty("surface", "image")
        original_layout.addWidget(self.original_label)
        images_row.addWidget(original_group)

        undistorted_group = QGroupBox("왜곡 보정 후 + Straightness (초록=곧음, 빨강=많이 휨)")
        undistorted_layout = QVBoxLayout(undistorted_group)
        self.undistorted_label = QLabel("-")
        self.undistorted_label.setAlignment(Qt.AlignCenter)
        self.undistorted_label.setProperty("surface", "image")
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
        pattern_config: PatternConfig | None = None,
    ) -> None:
        self._dataset = dataset
        self._camera_config = camera_config
        self._calibration_results = calibration_results
        self._pattern_config = pattern_config

        self.image_combo.clear()
        for frame in dataset.enabled_frames:
            # path 대신 image_id를 userData로 쓴다 - straightness 계산에 Frame
            # 객체 자체(검출된 코너 등)가 필요해서, 콤보박스가 "어느 프레임인지"를
            # 들고 있어야 두 계산(원본/보정 표시 + straightness 오버레이)이 항상
            # 같은 프레임을 가리킨다.
            self.image_combo.addItem(frame.image_info.image_id, userData=frame.image_info.image_id)

    def select_model(self, model: CameraModelType) -> None:
        idx = self.model_combo.findData(model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

    def _update_preview(self) -> None:
        if self._camera_config is None or self.image_combo.count() == 0:
            self.status_label.setText("먼저 이미지를 불러오고 캘리브레이션을 실행하세요.")
            return

        image_id = self.image_combo.currentData()
        frame = next(
            (f for f in self._dataset.enabled_frames if f.image_info.image_id == image_id),
            None,
        ) if self._dataset is not None else None
        model = self.model_combo.currentData()
        result = self._calibration_results.get(model)

        if result is None or not result.success:
            self.status_label.setText(f"{_MODEL_LABELS.get(model, model)} 모델의 캘리브레이션 결과가 없습니다.")
            return
        if frame is None:
            self.status_label.setText("선택한 이미지를 데이터셋에서 찾을 수 없습니다.")
            return

        image_path = frame.image_info.path
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

        if self._pattern_config is not None:
            # undistort_image()와 정확히 같은 방식으로 target_K를 구해야 오버레이
            # 좌표가 화면에 보이는 이미지와 어긋나지 않는다 (fisheye는 K가 재추정됨).
            if model == CameraModelType.FISHEYE:
                size = (self._camera_config.width, self._camera_config.height)
                target_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    result.camera_matrix, result.distortion, size, np.eye(3), balance=0.0
                )
            else:
                target_K = result.camera_matrix
            canvas, _lines = render_straightness_overlay(
                undistorted, frame, self._pattern_config,
                result.camera_matrix, result.distortion, model, target_K,
            )
            self.undistorted_label.setPixmap(_cv_to_qpixmap(canvas))
        else:
            self.undistorted_label.setPixmap(_cv_to_qpixmap(undistorted))

        self.status_label.setText(self._build_status_text(frame, model, result))

    def _build_status_text(
        self, frame, model: CameraModelType, result: CalibrationResult,
    ) -> str:
        """Line Straightness Residual(calibration/straightness.py, Model
        Score에도 쓰이는 것과 동일한 지표)의 보정 전/후 숫자만 짧게 보여준다.
        """
        if self._pattern_config is None:
            return f"{_MODEL_LABELS.get(model, model)} 모델 기준 보정 결과입니다. (패턴 정보가 없어 정량적 straightness 계산은 생략됨 - 육안으로 확인하세요.)"

        raw_residual, corrected_residual = compute_straightness_improvement(
            frame, self._pattern_config, result.camera_matrix, result.distortion, model,
        )
        if raw_residual is None or corrected_residual is None:
            return f"{_MODEL_LABELS.get(model, model)} 모델 기준 보정 결과입니다. (이 이미지엔 straightness 계산에 필요한 격자 라인이 부족합니다.)"

        if raw_residual > 1e-9:
            improvement_pct = max(0.0, (1 - corrected_residual / raw_residual)) * 100
            improvement_note = f" (개선율 {improvement_pct:.0f}%)"
        else:
            improvement_note = ""

        return f"Line Straightness — 보정 전 {raw_residual:.3f}px → 보정 후 {corrected_residual:.3f}px{improvement_note}"
