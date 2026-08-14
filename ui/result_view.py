"""
camera_calibrator.ui.result_view
====================================

설계 문서 8번(Model Comparison/Recommendation), 9번(Outlier), 11번(Export)를
한 화면에 모은 뷰. 이 위젯은 계산을 하지 않는다 - 버튼을 누르면 필요한
정보(선택된 모델 등)를 담아 signal만 emit하고, 실제 계산/파일저장은
main_window가 worker나 export 함수를 호출해서 처리한다.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.types import CalibrationResult, CameraModelType, ModelScore, OutlierResult, ValidationResult
from calibration.models.common import regional_edge_average

_MODEL_LABELS = {
    CameraModelType.PINHOLE: "Pinhole",
    CameraModelType.EXTENDED_PINHOLE: "Extended Pinhole",
    CameraModelType.FISHEYE: "Fisheye",
}
_MODEL_ORDER = [CameraModelType.PINHOLE, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE]


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


class ResultView(QWidget):
    # 사용자가 "이상치 제거 후 재계산"을 눌렀을 때 (기준 모델을 함께 전달)
    outlier_prune_requested = Signal(object)  # CameraModelType
    export_opencv_requested = Signal(object)  # CameraModelType
    export_ros_requested = Signal(object)     # CameraModelType
    export_report_requested = Signal(object)  # CameraModelType
    export_json_requested = Signal(object)    # CameraModelType
    export_csv_requested = Signal(object)     # CameraModelType

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._calibration_results: dict[CameraModelType, CalibrationResult] = {}

        # --- 비교/검증/추천 테이블 ---
        compare_group = QGroupBox("Model Comparison & Validation & Score")
        compare_layout = QVBoxLayout(compare_group)
        self.table = QTableWidget(6, 3)
        self.table.setHorizontalHeaderLabels([_MODEL_LABELS[m] for m in _MODEL_ORDER])
        self.table.setVerticalHeaderLabels(
            ["Train RMS", "Test RMS", "Edge RMS", "Straightness", "Score", "Recommend"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        compare_layout.addWidget(self.table)
        self.recommendation_label = QLabel("아직 계산되지 않았습니다.")
        self.recommendation_label.setWordWrap(True)
        compare_layout.addWidget(self.recommendation_label)
        layout.addWidget(compare_group)

        # --- Outlier ---
        outlier_group = QGroupBox("Outlier")
        outlier_layout = QVBoxLayout(outlier_group)
        self.outlier_button = QPushButton("선택 모델 기준 이상치 확인 후 제거·재계산")
        self.outlier_button.clicked.connect(
            lambda: self.outlier_prune_requested.emit(self.model_combo.currentData())
        )
        outlier_layout.addWidget(self.outlier_button)
        self.outlier_summary_label = QLabel("아직 이상치 검사를 실행하지 않았습니다.")
        self.outlier_summary_label.setWordWrap(True)
        outlier_layout.addWidget(self.outlier_summary_label)
        self.outlier_list = QListWidget()
        outlier_layout.addWidget(self.outlier_list)
        layout.addWidget(outlier_group)

        # --- 모델 선택 콤보 (Outlier와 Export 사이 - "어떤 모델을 이상치
        # 제거/export 대상으로 쓸지"가 바로 아래 Export 섹션에 적용되므로,
        # 그 사이에 둬야 "이 선택이 무엇에 영향을 주는지"가 헷갈리지 않는다) ---
        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("기준/대상 모델 (Outlier·Export 공통):"))
        self.model_combo = QComboBox()
        for m in _MODEL_ORDER:
            self.model_combo.addItem(_MODEL_LABELS[m], userData=m)
        self.model_combo.currentIndexChanged.connect(self._update_model_status)
        combo_row.addWidget(self.model_combo)
        combo_row.addStretch(1)
        layout.addLayout(combo_row)

        # 선택한 모델이 계산 실패/미계산 상태면 버튼을 누르기 전에 바로 알 수 있게
        # (전에는 눌러야만 경고가 떠서 "눌러도 반응이 없다"고 헷갈리기 쉬웠음).
        self.model_status_label = QLabel("")
        self.model_status_label.setWordWrap(True)
        layout.addWidget(self.model_status_label)

        # --- Export ---
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout(export_group)
        self.export_opencv_button = QPushButton("Export OpenCV YAML")
        self.export_opencv_button.clicked.connect(
            lambda: self.export_opencv_requested.emit(self.model_combo.currentData())
        )
        self.export_ros_button = QPushButton("Export ROS CameraInfo YAML")
        self.export_ros_button.clicked.connect(
            lambda: self.export_ros_requested.emit(self.model_combo.currentData())
        )
        self.export_report_button = QPushButton("Export HTML Report")
        self.export_report_button.clicked.connect(
            lambda: self.export_report_requested.emit(self.model_combo.currentData())
        )
        self.export_json_button = QPushButton("Export JSON")
        self.export_json_button.clicked.connect(
            lambda: self.export_json_requested.emit(self.model_combo.currentData())
        )
        self.export_csv_button = QPushButton("Export CSV (Dataset)")
        self.export_csv_button.clicked.connect(
            lambda: self.export_csv_requested.emit(self.model_combo.currentData())
        )
        export_layout.addWidget(self.export_opencv_button)
        export_layout.addWidget(self.export_ros_button)
        export_layout.addWidget(self.export_report_button)
        export_layout.addWidget(self.export_json_button)
        export_layout.addWidget(self.export_csv_button)
        layout.addWidget(export_group)

        self._update_model_status()

    # ------------------------------------------------------------------
    # 데이터 반영 (계산 없음, 표시만)
    # ------------------------------------------------------------------

    def _update_model_status(self) -> None:
        """선택된 모델이 실제로 export/이상치 제거에 쓸 수 있는 상태인지
        버튼을 누르기 전에 미리 보여준다. 계산이 아예 안 됐는지(아직 실행 전)
        와 계산은 했지만 실패했는지를 구분해서 알려준다.
        """
        model = self.model_combo.currentData()
        result = self._calibration_results.get(model) if model else None

        usable = bool(result and result.success)
        self.outlier_button.setEnabled(usable)
        self.export_opencv_button.setEnabled(usable)
        self.export_ros_button.setEnabled(usable)
        self.export_report_button.setEnabled(usable)
        self.export_json_button.setEnabled(usable)
        # CSV는 이미지별 데이터셋 통계라 특정 모델의 성공 여부와 무관하다 -
        # 계산이 아예 안 됐을 때만 막고, 어떤 모델이 실패했는지는 상관없다.
        self.export_csv_button.setEnabled(bool(self._calibration_results))

        if not self._calibration_results:
            self.model_status_label.setText("")
        elif result is None:
            self.model_status_label.setText(
                f"⚠ {_MODEL_LABELS.get(model, model)} 모델이 아직 계산되지 않았습니다."
            )
            self.model_status_label.setStyleSheet("color: #ef6c00;")
        elif not result.success:
            reason = f" ({result.error_message})" if result.error_message else ""
            self.model_status_label.setText(
                f"✕ {_MODEL_LABELS.get(model, model)} 모델은 캘리브레이션에 실패했습니다{reason}. "
                f"Export/이상치 제거를 쓸 수 없습니다 - 다른 모델을 선택하세요."
            )
            self.model_status_label.setStyleSheet("color: #c62828;")
        else:
            self.model_status_label.setText(
                f"✓ {_MODEL_LABELS.get(model, model)} 모델 사용 가능 (RMS {result.rms_error:.3f}px)"
            )
            self.model_status_label.setStyleSheet("color: #2e7d32;")

    def set_comparison(
        self,
        calibration_results: dict[CameraModelType, CalibrationResult],
        validation_results: dict[CameraModelType, ValidationResult],
        scores: list[ModelScore],
    ) -> None:
        self._calibration_results = calibration_results
        score_by_model = {s.model_name: s for s in scores}
        for col, m in enumerate(_MODEL_ORDER):
            cal = calibration_results.get(m)
            val = validation_results.get(m)
            score = score_by_model.get(m)

            train_rms = _fmt(cal.rms_error) if cal and cal.success else "FAIL"
            test_rms = _fmt(val.test_rms) if val and val.success else "N/A"
            if val and val.success and val.edge_rms is not None:
                edge_rms = _fmt(val.edge_rms)
            elif cal and cal.success and cal.regional_error:
                edge_rms = _fmt(regional_edge_average(cal.regional_error))
            else:
                edge_rms = "N/A"
            straightness = _fmt(val.straightness_residual) if val else "N/A"
            score_str = f"{score.score:.3f}" if score else "N/A"
            recommend = "⭐" if (score and score.is_recommended) else ""

            for row, value in enumerate(
                [train_rms, test_rms, edge_rms, straightness, score_str, recommend]
            ):
                self.table.setItem(row, col, QTableWidgetItem(value))

        self._update_model_status()

    def set_recommendation_message(self, message: str) -> None:
        self.recommendation_label.setText(message)

    def set_outlier_result(self, reference_model: CameraModelType, outlier_result: OutlierResult) -> None:
        label = _MODEL_LABELS[reference_model]
        self.outlier_list.clear()
        if not outlier_result.removed_frame_ids:
            self.outlier_summary_label.setText(f"[{label} 기준] 이상치로 판단된 프레임이 없습니다.")
            return

        before = _fmt(outlier_result.rms_before)
        after = _fmt(outlier_result.rms_after)
        self.outlier_summary_label.setText(
            f"[{label} 기준] {outlier_result.iterations}회 반복, "
            f"threshold={outlier_result.threshold_used:.3f}px, "
            f"RMS {before}px → {after}px 로 개선됨"
        )
        self.outlier_list.addItems(outlier_result.removed_frame_ids)

    def select_model(self, model: CameraModelType) -> None:
        idx = self.model_combo.findData(model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

    def prompt_save_path(self, default_name: str, filter_str: str) -> str | None:
        path, _ = QFileDialog.getSaveFileName(self, "저장 위치 선택", default_name, filter_str)
        return path or None
