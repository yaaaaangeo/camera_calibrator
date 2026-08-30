"""
camera_calibrator.ui.result_view
====================================

설계 문서 8번(Model Comparison/Recommendation), 11번(Export)를 한 화면에
모은 뷰. 이 위젯은 계산을 하지 않는다 - 버튼을 누르면 필요한 정보(선택된
모델 등)를 담아 signal만 emit하고, 실제 계산/파일저장은 main_window가
worker나 export 함수를 호출해서 처리한다.
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
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.types import (
    CalibrationResult,
    CameraModelType,
    CrossDatasetValidationResult,
    ModelScore,
    ValidationResult,
)
from calibration.models.common import regional_edge_average
from ui.theme import Theme, qcolor, set_tone

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


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def _fmt_pct(v: float | None) -> str:
    return f"{v:.0f}%" if v is not None else "N/A"


def _p95(cal: CalibrationResult | None, val: ValidationResult | None) -> float | None:
    if val and val.test_residual_stats:
        return val.test_residual_stats.p95
    if cal and cal.residual_stats:
        return cal.residual_stats.p95
    return None


def _radial_edge(cal: CalibrationResult | None) -> float | None:
    profile = (cal.radial_bands or cal.radial_profile) if cal else None
    if not profile or not profile.bins:
        return None
    values = []
    for b in profile.bins:
        label = (b.label or "").lower()
        if label in ("outer", "edge", "corner"):
            value = b.p95_error if b.p95_error is not None else b.rms_error
            if value is not None:
                values.append(value)
    return max(values) if values else None


def _stability(cal: CalibrationResult | None) -> float | None:
    pu = (cal.param_uncertainty_bootstrap or cal.param_uncertainty) if cal else None
    return pu.overall_stability if pu else None


def _format_object_releasing_result(result: CalibrationResult | None) -> str:
    if result is None:
        return "Object-Releasing: Not run."
    title = "Object-Releasing Brown-Conrady"
    diagnostics = result.object_releasing_diagnostics or []
    accepted = sum(1 for d in diagnostics if d.get("accepted"))
    total = len(diagnostics)
    lines = [title]
    if not result.success:
        lines.append("Status: Failed")
        if result.error_message:
            lines.append(result.error_message)
        return "\n".join(lines)
    lines.append(f"Calibration RMS: {_fmt(result.rms_error)} px")
    if total:
        lines.append(f"Full-board frames: {accepted}/{total}")
    lines.append("Hold-out Validation: Not available for Object-Releasing yet.")
    geom = result.target_geometry_refinement or {}
    if geom:
        lines.append(
            "Geometry refinement: "
            f"mean={_fmt(geom.get('mean_displacement'))}, "
            f"p95={_fmt(geom.get('p95_displacement'))}, "
            f"max={_fmt(geom.get('max_displacement'))}"
        )
    return "\n".join(lines)


class ResultView(QWidget):
    export_opencv_requested = Signal(object)  # CameraModelType
    cross_dataset_requested = Signal()

    def __init__(self, parent: QWidget | None = None, *, standalone: bool = True):
        super().__init__(parent)
        self._calibration_results: dict[CameraModelType, CalibrationResult] = {}
        self._object_releasing_result: CalibrationResult | None = None

        self.model_comparison_widget = QWidget()
        model_layout = QVBoxLayout(self.model_comparison_widget)

        # --- 비교/추천 테이블 ---
        compare_group = QGroupBox("Model Comparison & Validation & Score")
        compare_layout = QVBoxLayout(compare_group)
        self._row_labels = [
            "Train RMS", "Test RMS", "Test P95", "Edge RMS", "Straightness",
            "Radial Edge", "AIC", "BIC", "Stability", "Observability",
            "Undistortion", "Model Score", "Selection Conf.", "Recommend",
        ]
        self.table = QTableWidget(len(self._row_labels), len(_MODEL_ORDER))
        self.table.setHorizontalHeaderLabels([_MODEL_LABELS[m] for m in _MODEL_ORDER])
        self.table.setVerticalHeaderLabels(self._row_labels)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        compare_layout.addWidget(self.table)
        self.recommendation_label = QLabel("아직 계산되지 않았습니다.")
        self.recommendation_label.setWordWrap(True)
        compare_layout.addWidget(self.recommendation_label)
        model_layout.addWidget(compare_group)
        advanced_group = QGroupBox("Advanced Calibration")
        advanced_layout = QVBoxLayout(advanced_group)
        self.object_releasing_label = QLabel("Object-Releasing: Not run.")
        self.object_releasing_label.setWordWrap(True)
        advanced_layout.addWidget(self.object_releasing_label)
        model_layout.addWidget(advanced_group)

        self.validation_widget = QWidget()
        validation_layout = QVBoxLayout(self.validation_widget)

        # --- Sanity Check (설계 문서 8번) ---
        sanity_group = QGroupBox("Sanity Check")
        sanity_layout = QVBoxLayout(sanity_group)
        self.sanity_label = QLabel("아직 계산되지 않았습니다.")
        self.sanity_label.setWordWrap(True)
        sanity_layout.addWidget(self.sanity_label)
        validation_layout.addWidget(sanity_group)

        # --- Cross-Dataset Validation ---
        cross_group = QGroupBox("Cross-Dataset Validation")
        cross_layout = QVBoxLayout(cross_group)
        self.cross_dataset_button = QPushButton("Dataset B/C로 Generalization 평가...")
        self.cross_dataset_button.clicked.connect(self.cross_dataset_requested.emit)
        cross_layout.addWidget(self.cross_dataset_button)
        self.cross_dataset_summary_label = QLabel("아직 외부 Dataset 평가를 실행하지 않았습니다.")
        self.cross_dataset_summary_label.setWordWrap(True)
        cross_layout.addWidget(self.cross_dataset_summary_label)
        self.cross_dataset_table = QTableWidget(0, 6)
        self.cross_dataset_table.setHorizontalHeaderLabels(["Source", "Target", "Model", "Test RMS", "P95", "Gap"])
        self.cross_dataset_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cross_dataset_table.setEditTriggers(QTableWidget.NoEditTriggers)
        cross_layout.addWidget(self.cross_dataset_table)
        validation_layout.addWidget(cross_group)

        self.export_widget = QWidget()
        export_page_layout = QVBoxLayout(self.export_widget)

        # --- 모델 선택 콤보 (바로 아래 Export 섹션이 이 선택을 그대로 씀) ---
        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("Export 대상 모델:"))
        self.model_combo = QComboBox()
        for m in _MODEL_ORDER:
            self.model_combo.addItem(_MODEL_LABELS[m], userData=m)
        self.model_combo.currentIndexChanged.connect(self._update_model_status)
        combo_row.addWidget(self.model_combo)
        combo_row.addStretch(1)
        export_page_layout.addLayout(combo_row)

        # 선택한 모델이 계산 실패/미계산 상태면 버튼을 누르기 전에 바로 알 수 있게
        # (전에는 눌러야만 경고가 떠서 "눌러도 반응이 없다"고 헷갈리기 쉬웠음).
        self.model_status_label = QLabel("")
        self.model_status_label.setWordWrap(True)
        export_page_layout.addWidget(self.model_status_label)

        # --- Export ---
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout(export_group)
        self.export_opencv_button = QPushButton("Export OpenCV YAML")
        self.export_opencv_button.clicked.connect(
            lambda: self.export_opencv_requested.emit(self.model_combo.currentData())
        )
        export_layout.addWidget(self.export_opencv_button)
        export_page_layout.addWidget(export_group)
        export_page_layout.addStretch(1)

        if standalone:
            layout = QVBoxLayout(self)
            tabs = QTabWidget()
            tabs.addTab(self.validation_widget, "Validation")
            tabs.addTab(self.model_comparison_widget, "Model Comparison")
            tabs.addTab(self.export_widget, "Export")
            layout.addWidget(tabs)

        self._update_model_status()

    # ------------------------------------------------------------------
    # 데이터 반영 (계산 없음, 표시만)
    # ------------------------------------------------------------------

    def _update_model_status(self) -> None:
        """선택된 모델이 실제로 export에 쓸 수 있는 상태인지 버튼을 누르기
        전에 미리 보여준다. 계산이 아예 안 됐는지(아직 실행 전)와 계산은
        했지만 실패했는지를 구분해서 알려준다.
        """
        model = self.model_combo.currentData()
        result = self._calibration_results.get(model) if model else None

        usable = bool(result and result.success)
        self.export_opencv_button.setEnabled(usable)
        self.cross_dataset_button.setEnabled(usable)

        if not self._calibration_results:
            self.model_status_label.setText("")
            set_tone(self.model_status_label, "muted")
        elif result is None:
            self.model_status_label.setText(
                f"⚠ {_MODEL_LABELS.get(model, model)} 모델이 아직 계산되지 않았습니다."
            )
            set_tone(self.model_status_label, "warning")
        elif not result.success:
            reason = f" ({result.error_message})" if result.error_message else ""
            self.model_status_label.setText(
                f"✕ {_MODEL_LABELS.get(model, model)} 모델은 캘리브레이션에 실패했습니다{reason}. "
                f"Export를 쓸 수 없습니다 - 다른 모델을 선택하세요."
            )
            set_tone(self.model_status_label, "bad")
        else:
            self.model_status_label.setText(
                f"✓ {_MODEL_LABELS.get(model, model)} 모델 사용 가능 (RMS {result.rms_error:.3f}px)"
            )
            if result.warning_message:
                self.model_status_label.setText(
                    self.model_status_label.text() + "\n" + result.warning_message
                )
            set_tone(self.model_status_label, "good")

    def set_comparison(
        self,
        calibration_results: dict[CameraModelType, CalibrationResult],
        validation_results: dict[CameraModelType, ValidationResult],
        scores: list[ModelScore],
        object_releasing_result: CalibrationResult | None = None,
    ) -> None:
        self._calibration_results = calibration_results
        self._object_releasing_result = object_releasing_result
        self.object_releasing_label.setText(_format_object_releasing_result(object_releasing_result))
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
            p95 = _fmt(_p95(cal, val))
            radial = _fmt(_radial_edge(cal))
            aic = f"{score.aic:.1f}" if score and score.aic is not None else "N/A"
            bic = f"{score.bic:.1f}" if score and score.bic is not None else "N/A"
            stability = _fmt_pct(_stability(cal))
            observability = (
                f"{cal.observability.observability_score:.0f}% {cal.observability.observability_grade or ''}".strip()
                if cal and cal.observability and cal.observability.observability_score is not None else "N/A"
            )
            undistortion = (
                f"{cal.undistortion_quality.quality_score:.0f}% {cal.undistortion_quality.quality_grade.value}".strip()
                if cal and cal.undistortion_quality else "N/A"
            )
            score_str = f"{score.score:.3f}" if score else "N/A"
            selection_conf = (
                f"{score.selection_confidence:.0f}% {score.selection_confidence_level}"
                if score and score.selection_confidence is not None and score.selection_confidence_level else "N/A"
            )
            recommend = "⭐" if (score and score.is_recommended) else ""

            for row, value in enumerate(
                [
                    train_rms, test_rms, p95, edge_rms, straightness,
                    radial, aic, bic, stability, observability,
                    undistortion, score_str, selection_conf, recommend,
                ]
            ):
                item = QTableWidgetItem(value)
                # RMSE/P95/Stability 등의 숫자는 기본 흰색을 유지하고 실제 추천
                # 판정 셀만 semantic GOOD 색으로 강조한다.
                if row == len(self._row_labels) - 1 and recommend:
                    item.setForeground(qcolor(Theme.GOOD))
                self.table.setItem(row, col, item)

        self._update_model_status()

    def set_sanity_checks(self, checks: list) -> None:
        """설계 문서 8번 - 모델별 sanity check 결과를 요약해서 보여준다.
        checks: list[calibration.sanity_check.SanityCheckResult]
        """
        if not checks:
            self.sanity_label.setText("아직 계산되지 않았습니다.")
            set_tone(self.sanity_label, "muted")
            return
        if not any(c.issues for c in checks):
            self.sanity_label.setText("✓ 모든 모델 이상 없음 (fx/fy, principal point, aspect ratio, distortion, FOV, RMS)")
            set_tone(self.sanity_label, "good")
            return
        lines = []
        has_error = False
        for c in checks:
            if not c.issues:
                continue
            if c.has_errors:
                has_error = True
            label = c.model_name.value if hasattr(c.model_name, "value") else str(c.model_name)
            for issue in c.issues:
                mark = "✖" if issue.severity.value == "error" else "⚠"
                lines.append(f"{mark} [{label}] {issue.message}")
        self.sanity_label.setText("\n".join(lines))
        set_tone(self.sanity_label, "bad" if has_error else "warning")

    def set_recommendation_message(self, message: str) -> None:
        self.recommendation_label.setText(message)

    def set_cross_dataset_results(self, results: list[CrossDatasetValidationResult]) -> None:
        self.cross_dataset_table.setRowCount(len(results))
        if not results:
            self.cross_dataset_summary_label.setText("아직 외부 Dataset 평가를 실행하지 않았습니다.")
            return

        ok = sum(1 for r in results if r.success)
        self.cross_dataset_summary_label.setText(
            f"외부 Dataset 평가 {len(results)}건 중 {ok}건 성공. Export HTML/JSON과 프로젝트 저장에 포함됩니다."
        )
        for row, r in enumerate(results):
            values = [
                r.source_dataset_id,
                r.target_dataset_id,
                r.model_name.value,
                _fmt(r.test_rms) if r.success else "FAIL",
                _fmt(r.test_p95) if r.success else "N/A",
                _fmt(r.generalization_gap) if r.success else (r.error_message or "N/A"),
            ]
            for col, value in enumerate(values):
                self.cross_dataset_table.setItem(row, col, QTableWidgetItem(str(value)))

    def select_model(self, model: CameraModelType) -> None:
        idx = self.model_combo.findData(model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self._update_model_status()

    def prompt_save_path(self, default_name: str, filter_str: str) -> str | None:
        path, _ = QFileDialog.getSaveFileName(self, "저장 위치 선택", default_name, filter_str)
        return path or None
