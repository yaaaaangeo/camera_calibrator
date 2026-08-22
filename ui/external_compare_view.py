"""
camera_calibrator.ui.external_compare_view
===============================================

사용자 요청 - "예전에 다른 사람/다른 툴로 구한 파라미터"와 "지금 이 툴로
구한 파라미터" 중 뭐가 더 정확한지 정량적으로, 누구나 납득할 수 있게
비교하는 화면.

계산은 전부 calibration/external_compare.py가 한다 - 이 파일은 입력(외부
파라미터를 어떻게 받을지: OpenCV YAML 파일 또는 수동 입력)과 결과 표시
(비교표, 한 줄 평, 같은 이미지에 두 파라미터를 각각 적용한 직선성 오버레이
나란히 보기)만 담당한다 (백엔드/UI 분리 원칙, 다른 view들과 동일).
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.types import (
    CameraConfig,
    CameraModelType,
    Dataset,
    PatternConfig,
    ValidationResult,
)
from calibration.external_compare import (
    ComparisonSide,
    ExternalCameraParams,
    ExternalComparisonResult,
    compare_with_external_params,
)
from export.opencv import (
    detect_model_hint_from_opencv_yaml,
    load_camera_matrix_and_distortion_from_opencv_yaml,
)
from ui.straightness_view import render_straightness_overlay

_MODEL_LABELS = {
    CameraModelType.PINHOLE: "Pinhole",
    CameraModelType.EXTENDED_PINHOLE: "Extended Pinhole",
    CameraModelType.FISHEYE: "Fisheye",
}
_MODEL_ORDER = [CameraModelType.PINHOLE, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE]

_PANEL_MAX_WIDTH = 460


def _cv_to_qpixmap(img_bgr: np.ndarray, max_width: int = _PANEL_MAX_WIDTH) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(qimg)
    if pixmap.width() > max_width:
        pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
    return pixmap


def _parse_distortion_text(text: str) -> np.ndarray:
    """"-0.28, 0.10, 0, 0, 0" 같은 콤마 구분 문자열을 distortion 벡터로.
    사용자가 실수하기 쉬운 부분(빈 값, 공백, 세미콜론 등)을 최대한 관대하게 받는다.
    """
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("왜곡 계수를 하나 이상 입력해 주세요 (콤마로 구분).")
    try:
        values = [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"왜곡 계수를 숫자로 해석할 수 없습니다: {e}") from e
    return np.array(values, dtype=np.float64)


class ExternalCompareView(QWidget):
    """"⑦ 외부 결과 비교" 탭."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._dataset: Dataset | None = None
        self._camera_config: CameraConfig | None = None
        self._pattern_config: PatternConfig | None = None
        self._validation_results: dict[CameraModelType, ValidationResult] = {}
        self._use_rational_model = False
        self._last_result: ExternalComparisonResult | None = None
        self._loaded_yaml_path: str | None = None

        layout = QVBoxLayout(self)

        intro = QLabel(
            "예전에 구한 카메라 파라미터(다른 사람/다른 툴 결과)를 입력하면, "
            "내가 캘리브레이션 학습에 전혀 쓰지 않은 이미지들(Hold-out test 프레임)에서 "
            "두 파라미터를 완전히 동일한 방식으로 재평가해 비교합니다. "
            "어느 한쪽에 유리한 조건을 주지 않는 정량 비교입니다."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_run_row())
        layout.addWidget(self._build_result_group())
        layout.addWidget(self._build_visual_group(), stretch=1)

    # ------------------------------------------------------------------
    # 입력 영역
    # ------------------------------------------------------------------

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("비교할 외부 파라미터")
        outer = QVBoxLayout(group)

        form = QFormLayout()
        self.label_edit = QLineEdit("예전 결과")
        form.addRow("표시 이름", self.label_edit)

        self.external_model_combo = QComboBox()
        for m in _MODEL_ORDER:
            self.external_model_combo.addItem(_MODEL_LABELS[m], userData=m)
        form.addRow("카메라 모델 종류", self.external_model_combo)

        self.source_note_edit = QLineEdit()
        self.source_note_edit.setPlaceholderText("예: 2025-03 A업체 캘리브레이션 (선택 입력)")
        form.addRow("출처 메모", self.source_note_edit)
        outer.addLayout(form)

        yaml_row = QHBoxLayout()
        self.load_yaml_button = QPushButton("OpenCV YAML 불러오기...")
        self.load_yaml_button.clicked.connect(self._on_load_yaml)
        yaml_row.addWidget(self.load_yaml_button)
        self.yaml_status_label = QLabel("불러온 파일 없음 - 아래 수동 입력을 쓰거나 YAML을 불러오세요.")
        self.yaml_status_label.setWordWrap(True)
        yaml_row.addWidget(self.yaml_status_label, stretch=1)
        outer.addLayout(yaml_row)

        manual_group = QGroupBox("직접 입력 (YAML을 안 불러왔다면 여기 채우기)")
        manual_form = QFormLayout(manual_group)
        self.fx_spin = self._make_double_spin(500.0)
        self.fy_spin = self._make_double_spin(500.0)
        self.cx_spin = self._make_double_spin(960.0)
        self.cy_spin = self._make_double_spin(540.0)
        manual_form.addRow("fx", self.fx_spin)
        manual_form.addRow("fy", self.fy_spin)
        manual_form.addRow("cx", self.cx_spin)
        manual_form.addRow("cy", self.cy_spin)
        self.distortion_edit = QLineEdit()
        self.distortion_edit.setPlaceholderText("k1, k2, p1, p2, k3 (모델에 맞는 개수만큼, 콤마로 구분)")
        manual_form.addRow("왜곡 계수", self.distortion_edit)
        outer.addWidget(manual_group)
        self._manual_group = manual_group

        return group

    @staticmethod
    def _make_double_spin(default: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-100000.0, 100000.0)
        spin.setDecimals(4)
        spin.setValue(default)
        return spin

    def _build_run_row(self) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("비교할 내 모델:"))
        self.my_model_combo = QComboBox()
        for m in _MODEL_ORDER:
            self.my_model_combo.addItem(_MODEL_LABELS[m], userData=m)
        h.addWidget(self.my_model_combo)

        self.run_button = QPushButton("비교 실행")
        self.run_button.clicked.connect(self._on_run_comparison)
        h.addWidget(self.run_button)
        h.addStretch(1)
        return row

    # ------------------------------------------------------------------
    # 결과 표시 영역
    # ------------------------------------------------------------------

    def _build_result_group(self) -> QGroupBox:
        group = QGroupBox("정량 비교 결과 (Hold-out test 프레임 기준, 동일 조건)")
        v = QVBoxLayout(group)

        self.table = QTableWidget(4, 2)
        self.table.setHorizontalHeaderLabels(["내 결과", "외부 결과"])
        self.table.setVerticalHeaderLabels(
            ["Test RMS (px)", "Edge RMS - 외곽 (px)", "Straightness - 직선성 (px)", "프레임별 승 개수"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.table)

        self.verdict_label = QLabel("아직 비교를 실행하지 않았습니다.")
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setStyleSheet("font-weight: bold;")
        v.addWidget(self.verdict_label)

        self.caveats_label = QLabel("")
        self.caveats_label.setWordWrap(True)
        self.caveats_label.setStyleSheet("color: #666666; font-size: 11px;")
        v.addWidget(self.caveats_label)

        return group

    def _build_visual_group(self) -> QGroupBox:
        group = QGroupBox("이미지 비교 (같은 사진에 두 파라미터를 각각 적용 - 초록=곧음, 빨강=많이 휨)")
        v = QVBoxLayout(group)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("이미지:"))
        self.image_combo = QComboBox()
        control_row.addWidget(self.image_combo, stretch=1)
        self.refresh_visual_button = QPushButton("갱신")
        self.refresh_visual_button.clicked.connect(self._update_visual)
        control_row.addWidget(self.refresh_visual_button)
        v.addLayout(control_row)

        images_row = QHBoxLayout()
        mine_box = QGroupBox("내 결과")
        mine_layout = QVBoxLayout(mine_box)
        self.mine_image_label = QLabel("비교를 실행하면 표시됩니다.")
        self.mine_image_label.setAlignment(Qt.AlignCenter)
        mine_layout.addWidget(self.mine_image_label)
        self.mine_caption_label = QLabel("")
        self.mine_caption_label.setWordWrap(True)
        mine_layout.addWidget(self.mine_caption_label)
        images_row.addWidget(mine_box)

        external_box = QGroupBox("외부 결과")
        external_layout = QVBoxLayout(external_box)
        self.external_image_label = QLabel("비교를 실행하면 표시됩니다.")
        self.external_image_label.setAlignment(Qt.AlignCenter)
        external_layout.addWidget(self.external_image_label)
        self.external_caption_label = QLabel("")
        self.external_caption_label.setWordWrap(True)
        external_layout.addWidget(self.external_caption_label)
        images_row.addWidget(external_box)

        v.addLayout(images_row)
        return group

    # ------------------------------------------------------------------
    # 컨텍스트 갱신 (메인 창이 캘리브레이션 실행 후 호출)
    # ------------------------------------------------------------------

    def set_context(
        self,
        dataset: Dataset,
        camera_config: CameraConfig,
        pattern_config: PatternConfig,
        validation_results: dict[CameraModelType, ValidationResult],
        use_rational_model: bool = False,
    ) -> None:
        self._dataset = dataset
        self._camera_config = camera_config
        self._pattern_config = pattern_config
        self._validation_results = validation_results
        self._use_rational_model = use_rational_model

    # ------------------------------------------------------------------
    # 외부 YAML 불러오기
    # ------------------------------------------------------------------

    def _on_load_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "OpenCV Camera YAML 선택", "", "YAML (*.yaml *.yml)"
        )
        if not path:
            return
        try:
            load_camera_matrix_and_distortion_from_opencv_yaml(path)  # 유효성만 먼저 확인
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "불러오기 실패", str(e))
            return

        self._loaded_yaml_path = path
        hint = detect_model_hint_from_opencv_yaml(path)
        if hint is not None:
            idx = self.external_model_combo.findData(hint)
            if idx >= 0:
                self.external_model_combo.setCurrentIndex(idx)
            self.yaml_status_label.setText(
                f"불러옴: {path}\n(이 파일에 저장된 모델 종류 '{_MODEL_LABELS.get(hint, hint)}'를 자동 선택했습니다 - "
                "실제와 다르면 위에서 바꿔주세요.)"
            )
        else:
            self.yaml_status_label.setText(
                f"불러옴: {path}\n(이 파일에는 모델 종류 정보가 없습니다 - 위에서 직접 선택해 주세요.)"
            )

    def _external_params_from_manual_input(self) -> ExternalCameraParams:
        K = np.array([
            [self.fx_spin.value(), 0.0, self.cx_spin.value()],
            [0.0, self.fy_spin.value(), self.cy_spin.value()],
            [0.0, 0.0, 1.0],
        ])
        D = _parse_distortion_text(self.distortion_edit.text())
        model = self.external_model_combo.currentData()
        return ExternalCameraParams(
            label=self.label_edit.text().strip() or "예전 결과",
            model_name=model,
            camera_matrix=K,
            distortion=D,
            source_note=self.source_note_edit.text().strip(),
        )

    def _external_params_from_yaml(self) -> ExternalCameraParams:
        K, D = load_camera_matrix_and_distortion_from_opencv_yaml(self._loaded_yaml_path)
        model = self.external_model_combo.currentData()
        return ExternalCameraParams(
            label=self.label_edit.text().strip() or "예전 결과",
            model_name=model,
            camera_matrix=K,
            distortion=D,
            source_note=self.source_note_edit.text().strip() or self._loaded_yaml_path,
        )

    # ------------------------------------------------------------------
    # 비교 실행
    # ------------------------------------------------------------------

    def _on_run_comparison(self) -> None:
        if self._dataset is None or self._camera_config is None or self._pattern_config is None:
            QMessageBox.warning(self, "데이터 없음", "먼저 이미지를 불러오고 캘리브레이션을 실행하세요.")
            return

        my_model = self.my_model_combo.currentData()
        my_validation = self._validation_results.get(my_model)
        if my_validation is None or not my_validation.success:
            QMessageBox.warning(
                self, "Hold-out 결과 없음",
                f"{_MODEL_LABELS.get(my_model, my_model)} 모델의 Hold-out Validation 결과가 없습니다. "
                "먼저 [캘리브레이션 실행]을 완료해 주세요.",
            )
            return

        try:
            if self._loaded_yaml_path:
                external = self._external_params_from_yaml()
            else:
                external = self._external_params_from_manual_input()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "외부 파라미터 입력 오류", str(e))
            return

        result = compare_with_external_params(
            self._dataset, self._camera_config, self._pattern_config,
            my_model, my_validation, external,
            use_rational_model=self._use_rational_model,
        )
        self._last_result = result
        self._render_result(result)
        self._populate_image_combo(result)

    def _render_result(self, result: ExternalComparisonResult) -> None:
        self.table.setHorizontalHeaderLabels([result.mine.label, result.external.label])

        def _set_row(row: int, mine_val, external_val, fmt="{:.3f}"):
            mine_text = fmt.format(mine_val) if mine_val is not None else "N/A"
            ext_text = fmt.format(external_val) if external_val is not None else "N/A"
            self.table.setItem(row, 0, QTableWidgetItem(mine_text))
            self.table.setItem(row, 1, QTableWidgetItem(ext_text))

        if result.mine.success and result.external.success:
            _set_row(0, result.mine.test_rms, result.external.test_rms)
            _set_row(1, result.mine.edge_rms, result.external.edge_rms)
            _set_row(2, result.mine.straightness_residual, result.external.straightness_residual)
            _set_row(3, result.mine_win_count, result.external_win_count, fmt="{:.0f}")
        else:
            for row in range(4):
                self.table.setItem(row, 0, QTableWidgetItem("-"))
                self.table.setItem(row, 1, QTableWidgetItem("-"))

        self.verdict_label.setText(result.verdict)
        self.caveats_label.setText("\n".join(result.caveats))

    def _populate_image_combo(self, result: ExternalComparisonResult) -> None:
        self.image_combo.clear()
        if not (result.mine.success and result.external.success):
            return
        common_ids = sorted(set(result.mine.per_frame_error) & set(result.external.per_frame_error))
        for fid in common_ids:
            m = result.mine.per_frame_error[fid]
            e = result.external.per_frame_error[fid]
            self.image_combo.addItem(f"{fid}  (내 {m:.2f}px / 외부 {e:.2f}px)", userData=fid)
        if common_ids:
            self._update_visual()

    # ------------------------------------------------------------------
    # 이미지 비교 (직선성 오버레이) - 같은 프레임에 두 파라미터를 각각 적용
    # ------------------------------------------------------------------

    def _update_visual(self) -> None:
        result = self._last_result
        if result is None or not (result.mine.success and result.external.success):
            return
        if self.image_combo.count() == 0:
            return
        frame_id = self.image_combo.currentData()
        frame = next(
            (f for f in self._dataset.enabled_frames if f.image_info.image_id == frame_id), None
        )
        if frame is None:
            return

        img = cv2.imread(frame.image_info.path)
        if img is None:
            self.mine_caption_label.setText(f"이미지를 읽을 수 없습니다: {frame.image_info.path}")
            self.external_caption_label.setText("")
            return

        self._render_one_side(img, frame, result.mine, self.mine_image_label, self.mine_caption_label)
        self._render_one_side(img, frame, result.external, self.external_image_label, self.external_caption_label)

    def _render_one_side(
        self, img: np.ndarray, frame, side: ComparisonSide, image_label: QLabel, caption_label: QLabel,
    ) -> None:
        from calibration.models.common import undistort_image
        from calibration.types import CalibrationResult

        fake_result = CalibrationResult(
            model_name=side.model_name, camera_matrix=side.camera_matrix,
            distortion=side.distortion, success=True,
        )
        try:
            undistorted = undistort_image(img, fake_result, self._camera_config)
        except ValueError as e:
            caption_label.setText(str(e))
            return

        if side.model_name == CameraModelType.FISHEYE:
            size = (self._camera_config.width, self._camera_config.height)
            target_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                side.camera_matrix, side.distortion, size, np.eye(3), balance=0.0
            )
        else:
            target_K = side.camera_matrix

        canvas, lines = render_straightness_overlay(
            undistorted, frame, self._pattern_config,
            side.camera_matrix, side.distortion, side.model_name, target_K,
        )
        image_label.setPixmap(_cv_to_qpixmap(canvas))

        frame_id = frame.image_info.image_id
        per_frame = side.per_frame_error.get(frame_id)
        detail = f"{side.label} - 이 프레임 재투영 오차: {per_frame:.3f}px" if per_frame is not None else side.label
        if lines:
            avg_residual = float(np.mean([l.residual for l in lines]))
            detail += f" / 직선성 평균 {avg_residual:.3f}px ({len(lines)}개 줄)"
        caption_label.setText(detail)
