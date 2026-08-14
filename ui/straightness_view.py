"""
camera_calibrator.ui.straightness_view
===========================================

설계 문서 3.4번 - Line Straightness Residual을 숫자 하나가 아니라 이미지
위에 겹쳐서 보여준다. Coverage Map(ui/coverage_view.py)이 "어디가 부족한지"를
그리드로 보여주듯, 여기는 "어느 줄이 휘어있는지"를 보드의 실제 행/열 선으로
직접 그려서 보여준다.

계산(어느 점이 어느 줄에 속하는지, 잔차가 얼마인지)은 전부
calibration/straightness.py가 하고, 여기는 그 결과를 undistort된 이미지
위에 cv2로 그리기만 한다 (백엔드/UI 분리 원칙).
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

from calibration.types import CalibrationResult, CameraConfig, CameraModelType, Dataset, Frame, PatternConfig
from calibration.models.common import undistort_image
from calibration.straightness import compute_frame_straightness_lines, sort_points_along_line

_MODEL_LABELS = {
    CameraModelType.PINHOLE: "Pinhole",
    CameraModelType.EXTENDED_PINHOLE: "Extended Pinhole",
    CameraModelType.FISHEYE: "Fisheye",
}
_MODEL_ORDER = [CameraModelType.PINHOLE, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE]

_PREVIEW_MAX_WIDTH = 900

# 설계 문서 3.1번과 같은 어휘(Excellent/Good/Warning/Poor)를 색으로 매핑.
# BGR 순서(cv2로 그리므로).
_COLOR_EXCELLENT = (80, 175, 76)    # 초록
_COLOR_WARNING = (0, 165, 255)      # 주황
_COLOR_POOR = (40, 40, 220)         # 빨강
_RESIDUAL_MAX_FOR_COLOR = 1.5       # 이 값 이상이면 완전히 빨강 (px)


def _residual_to_color(residual: float) -> tuple[int, int, int]:
    """0(완벽한 직선) ~ _RESIDUAL_MAX_FOR_COLOR(px) 사이를
    초록 -> 주황 -> 빨강으로 부드럽게 보간. 설계 문서 3.1번 등급 경계
    (0.3=Excellent, 1.0=Warning)를 색 변화의 기준점으로 삼는다.
    """
    r = max(0.0, min(1.0, residual / _RESIDUAL_MAX_FOR_COLOR))
    if r < 0.5:
        t = r / 0.5
        c1, c2 = _COLOR_EXCELLENT, _COLOR_WARNING
    else:
        t = (r - 0.5) / 0.5
        c1, c2 = _COLOR_WARNING, _COLOR_POOR
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _cv_to_qpixmap(img_bgr: np.ndarray, max_width: int = _PREVIEW_MAX_WIDTH) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(qimg)
    if pixmap.width() > max_width:
        pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
    return pixmap


def render_straightness_overlay(
    undistorted_image: np.ndarray,
    frame: Frame,
    pattern_config: PatternConfig,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
    target_K: np.ndarray,
) -> tuple[np.ndarray, list]:
    """undistort된 이미지 위에 행/열 라인을 잔차 색으로 그린다.

    Returns: (그려진 이미지, StraightnessLine 리스트) - 리스트는 요약 라벨用.
    """
    lines = compute_frame_straightness_lines(
        frame, pattern_config, camera_matrix, distortion, model, target_K=target_K
    )
    canvas = undistorted_image.copy()

    for line in lines:
        pts = sort_points_along_line(line.points).astype(np.int32)
        color = _residual_to_color(line.residual)
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=2)
        for x, y in pts:
            cv2.circle(canvas, (int(x), int(y)), 3, color, -1)

    return canvas, lines


class StraightnessView(QWidget):
    """이미지 선택 + 모델 선택 -> 행/열 라인을 잔차 색으로 겹쳐 그린 이미지."""

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

        self.refresh_button = QPushButton("지도 갱신")
        self.refresh_button.clicked.connect(self._update_map)
        control_row.addWidget(self.refresh_button)
        layout.addLayout(control_row)

        group = QGroupBox("Straightness Map (초록=곧음, 빨강=많이 휨)")
        group_layout = QVBoxLayout(group)
        self.image_label = QLabel("이미지를 선택하고 [지도 갱신]을 누르세요.")
        self.image_label.setAlignment(Qt.AlignCenter)
        group_layout.addWidget(self.image_label)
        layout.addWidget(group)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

    def set_context(
        self,
        dataset: Dataset,
        camera_config: CameraConfig,
        calibration_results: dict[CameraModelType, CalibrationResult],
        pattern_config: PatternConfig,
    ) -> None:
        self._dataset = dataset
        self._camera_config = camera_config
        self._calibration_results = calibration_results
        self._pattern_config = pattern_config

        self.image_combo.clear()
        for frame in dataset.enabled_frames:
            if frame.detection and frame.detection.success:
                self.image_combo.addItem(frame.image_info.image_id, userData=frame.image_info.image_id)

    def select_model(self, model: CameraModelType) -> None:
        idx = self.model_combo.findData(model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

    def _update_map(self) -> None:
        if self._camera_config is None or self.image_combo.count() == 0:
            self.summary_label.setText("먼저 이미지를 불러오고 캘리브레이션을 실행하세요.")
            return

        image_id = self.image_combo.currentData()
        frame = next(
            (f for f in self._dataset.enabled_frames if f.image_info.image_id == image_id), None
        )
        if frame is None:
            self.summary_label.setText("선택한 이미지를 찾을 수 없습니다.")
            return

        model = self.model_combo.currentData()
        result = self._calibration_results.get(model)
        if result is None or not result.success:
            self.summary_label.setText(f"{_MODEL_LABELS.get(model, model)} 모델의 캘리브레이션 결과가 없습니다.")
            return

        img = cv2.imread(frame.image_info.path)
        if img is None:
            self.summary_label.setText(f"이미지를 읽을 수 없습니다: {frame.image_info.path}")
            return

        try:
            undistorted = undistort_image(img, result, self._camera_config)
        except ValueError as e:
            self.summary_label.setText(str(e))
            return

        # undistort_image()와 정확히 같은 방식으로 target_K를 구해야 오버레이
        # 좌표가 화면에 보이는 이미지와 어긋나지 않는다 (fisheye는 K가 재추정됨).
        if model == CameraModelType.FISHEYE:
            size = (self._camera_config.width, self._camera_config.height)
            target_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                result.camera_matrix, result.distortion, size, np.eye(3), balance=0.0
            )
        else:
            target_K = result.camera_matrix

        canvas, lines = render_straightness_overlay(
            undistorted, frame, self._pattern_config,
            result.camera_matrix, result.distortion, model, target_K,
        )
        self.image_label.setPixmap(_cv_to_qpixmap(canvas))

        if not lines:
            self.summary_label.setText(
                f"[{_MODEL_LABELS.get(model, model)}] 이 이미지에서는 직선(4점 이상)을 만들 수 없습니다."
            )
            return

        residuals = [l.residual for l in lines]
        avg = float(np.mean(residuals))
        worst = max(lines, key=lambda l: l.residual)
        self.summary_label.setText(
            f"[{_MODEL_LABELS.get(model, model)}] 이 이미지 평균 {avg:.3f}px, "
            f"가장 휜 줄: {worst.line_type} {worst.line_index} ({worst.residual:.3f}px) — "
            f"{len(lines)}개 줄 기준. 초록에 가까울수록 왜곡 보정이 잘 된 것입니다."
        )
