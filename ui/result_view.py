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

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # --- 모델 선택 콤보 (Outlier 기준 / Export 대상 공용) ---
        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("기준/대상 모델:"))
        self.model_combo = QComboBox()
        for m in _MODEL_ORDER:
            self.model_combo.addItem(_MODEL_LABELS[m], userData=m)
        combo_row.addWidget(self.model_combo)
        combo_row.addStretch(1)
        layout.addLayout(combo_row)

        # --- 비교/검증/추천 테이블 ---
        compare_group = QGroupBox("Model Comparison & Validation & Score")
        compare_layout = QVBoxLayout(compare_group)
        self.table = QTableWidget(6, 3)
        self.table.setHorizontalHeaderLabels([_MODEL_LABELS[m] for m in _MODEL_ORDER])
        self.table.setVerticalHeaderLabels(
            ["Train RMS", "Test RMS", "Edge RMS", "Complexity", "Score", "Recommend"]
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
        export_layout.addWidget(self.export_opencv_button)
        export_layout.addWidget(self.export_ros_button)
        layout.addWidget(export_group)

    # ------------------------------------------------------------------
    # 데이터 반영 (계산 없음, 표시만)
    # ------------------------------------------------------------------

    def set_comparison(
        self,
        calibration_results: dict[CameraModelType, CalibrationResult],
        validation_results: dict[CameraModelType, ValidationResult],
        scores: list[ModelScore],
    ) -> None:
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
            complexity = {"pinhole": "★", "extended_pinhole": "★★", "fisheye": "★★★"}[m.value]
            score_str = f"{score.score:.3f}" if score else "N/A"
            recommend = "⭐" if (score and score.is_recommended) else ""

            for row, value in enumerate([train_rms, test_rms, edge_rms, complexity, score_str, recommend]):
                self.table.setItem(row, col, QTableWidgetItem(value))

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
