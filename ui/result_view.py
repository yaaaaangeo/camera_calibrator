"""
camera_calibrator.ui.result_view
====================================

설계 문서 8번(Model Comparison/Recommendation), 11번(Export)를 한 화면에
모은 뷰. 이 위젯은 계산을 하지 않는다 - 버튼을 누르면 필요한 정보(선택된
모델 등)를 담아 signal만 emit하고, 실제 계산/파일저장은 main_window가
worker나 export 함수를 호출해서 처리한다.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
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
    ObjectReleasingValidationResult,
    StandardVsObjectReleasingComparison,
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


def _format_object_releasing_result(
    result: CalibrationResult | None,
    validation: ObjectReleasingValidationResult | None = None,
) -> str:
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

    lines.append("")
    lines.append("Object-Releasing Hold-out Validation")
    if validation is None:
        lines.append("Not available (not computed).")
    elif not validation.success:
        lines.append(f"Not available: {validation.error_message}")
    else:
        lines.append(f"Train Frames      {len(validation.train_frame_ids)}")
        lines.append(f"Test Frames       {len(validation.test_frame_ids)}")
        lines.append(f"RMSE              {_fmt(validation.test_rms)} px")
        stats = validation.test_residual_stats
        if stats:
            lines.append(f"Median            {_fmt(stats.median)} px")
            lines.append(f"P95               {_fmt(stats.p95)} px")
            lines.append(f"P99               {_fmt(stats.p99)} px")
            lines.append(f"Max               {_fmt(stats.max)} px")
        if validation.failed_test_frame_ids:
            lines.append(f"Rejected test frames: {len(validation.failed_test_frame_ids)}")

    geom = result.target_geometry_refinement or {}
    if geom:
        lines.append("")
        lines.append(
            "Geometry refinement: "
            f"mean={_fmt(geom.get('mean_displacement'))}, "
            f"p95={_fmt(geom.get('p95_displacement'))}, "
            f"max={_fmt(geom.get('max_displacement'))}"
        )
    return "\n".join(lines)


_RO_COMPARISON_ROW_LABELS = [
    "Train RMS", "Hold-out RMSE", "Median", "P95", "P99", "Max",
    "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3",
]


def _ro_comparison_cell_values(
    comparison: StandardVsObjectReleasingComparison,
) -> tuple[list[str], list[str]]:
    """StandardVsObjectReleasingComparison -> (Standard 열, Object-Releasing 열),
    _RO_COMPARISON_ROW_LABELS와 같은 순서.
    """
    sr, sv = comparison.standard_result, comparison.standard_validation
    rr, rv = comparison.object_releasing_result, comparison.object_releasing_validation

    def intrinsic(res: CalibrationResult | None, key: str) -> float | None:
        if res is None or res.camera_matrix is None:
            return None
        m = res.camera_matrix.reshape(3, 3)
        return {"fx": m[0, 0], "fy": m[1, 1], "cx": m[0, 2], "cy": m[1, 2]}[key]

    def distortion(res: CalibrationResult | None, idx: int) -> float | None:
        if res is None or res.distortion is None:
            return None
        arr = res.distortion.reshape(-1)
        return float(arr[idx]) if idx < len(arr) else None

    def column(cal: CalibrationResult | None, val, ro_val: ObjectReleasingValidationResult | None) -> list[str]:
        stats = val.test_residual_stats if val else (ro_val.test_residual_stats if ro_val else None)
        test_rms = val.test_rms if val else (ro_val.test_rms if ro_val else None)
        return [
            _fmt(cal.rms_error if cal else None),
            _fmt(test_rms),
            _fmt(stats.median if stats else None),
            _fmt(stats.p95 if stats else None),
            _fmt(stats.p99 if stats else None),
            _fmt(stats.max if stats else None),
            _fmt(intrinsic(cal, "fx")), _fmt(intrinsic(cal, "fy")),
            _fmt(intrinsic(cal, "cx")), _fmt(intrinsic(cal, "cy")),
            _fmt(distortion(cal, 0)), _fmt(distortion(cal, 1)),
            _fmt(distortion(cal, 2)), _fmt(distortion(cal, 3)), _fmt(distortion(cal, 4)),
        ]

    standard_values = column(sr, sv, None)
    ro_values = column(rr, None, rv)
    return standard_values, ro_values


def _format_ro_comparison_summary(comparison: StandardVsObjectReleasingComparison | None) -> str:
    if comparison is None:
        return "Standard vs Object-Releasing comparison: Not run."
    if not comparison.success:
        return f"Standard vs Object-Releasing comparison unavailable: {comparison.error_message}"
    lines = [
        f"Same full-board dataset, same train/test split "
        f"(train={len(comparison.train_frame_ids)}, test={len(comparison.test_frame_ids)})."
    ]
    geom = (comparison.object_releasing_result.target_geometry_refinement if comparison.object_releasing_result else None) or {}
    if geom:
        lines.append(
            "Target geometry refinement (Object-Releasing): "
            f"mean={_fmt(geom.get('mean_displacement'))}, "
            f"p95={_fmt(geom.get('p95_displacement'))}, "
            f"max={_fmt(geom.get('max_displacement'))}"
        )
    for w in comparison.warnings:
        lines.append(f"⚠ {w}")
    return "\n".join(lines)


class _PageScrollTableWidget(QTableWidget):
    """마우스 휠을 항상 상위 페이지 스크롤로 넘기는 QTableWidget.

    Model Comparison 표는 모든 행을 펼쳐서(_fit_table_to_rows) 표 자체는
    스크롤할 필요가 없게 만들었지만, 프레임/스타일 여백 오차로 내부에
    아주 약간의 스크롤 범위가 남으면 표 위에서 휠을 몇 번 더 굴려야
    페이지가 움직이는 것처럼 느껴져 불편했다. 휠 이벤트를 항상 무시해서
    표가 스스로 스크롤하지 않고 곧바로 부모 QScrollArea(페이지 전체)로
    넘어가게 한다.
    """

    def wheelEvent(self, event) -> None:
        event.ignore()


class ResultView(QWidget):
    export_opencv_requested = Signal(object)  # CameraModelType
    cross_dataset_requested = Signal()

    def __init__(self, parent: QWidget | None = None, *, standalone: bool = True):
        super().__init__(parent)
        self._calibration_results: dict[CameraModelType, CalibrationResult] = {}
        self._object_releasing_result: CalibrationResult | None = None

        # DatasetView와 동일하게 탭 전체를 하나의 세로 scroll area로 감싼다.
        # 아래 비교표가 자체적으로 세로 스크롤되면 wheel focus를 표 안으로
        # 옮겨야 해서 불편하므로, 표는 모든 행을 펼치고 페이지 전체를 스크롤한다.
        self.model_comparison_widget = QWidget()
        model_page_layout = QVBoxLayout(self.model_comparison_widget)
        self.model_comparison_scroll_area = QScrollArea()
        self.model_comparison_scroll_area.setWidgetResizable(True)
        self.model_comparison_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.model_comparison_content = QWidget()
        model_layout = QVBoxLayout(self.model_comparison_content)
        self.model_comparison_scroll_area.setWidget(self.model_comparison_content)
        model_page_layout.addWidget(self.model_comparison_scroll_area)

        # --- 비교/추천 테이블 ---
        compare_group = QGroupBox("Model Comparison & Validation & Score")
        compare_layout = QVBoxLayout(compare_group)
        self._row_labels = [
            "Train RMS", "Test RMS", "Test P95", "Edge RMS", "Straightness",
            "Radial Edge", "AIC", "BIC", "Stability", "Observability",
            "Undistortion", "Model Score", "Selection Conf.", "Recommend",
        ]
        self.table = _PageScrollTableWidget(len(self._row_labels), len(_MODEL_ORDER))
        self.table.setHorizontalHeaderLabels([_MODEL_LABELS[m] for m in _MODEL_ORDER])
        self.table.setVerticalHeaderLabels(self._row_labels)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        compare_layout.addWidget(self.table)
        self.recommendation_label = QLabel("아직 계산되지 않았습니다.")
        self.recommendation_label.setWordWrap(True)
        compare_layout.addWidget(self.recommendation_label)
        model_layout.addWidget(compare_group)
        self.advanced_group = QGroupBox("▶ Advanced Calibration")
        self.advanced_group.setObjectName("advancedCalibrationPanel")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        self.advanced_group.setToolTip(
            "Object-Releasing calibration을 선택한 경우에만 펼칠 수 있습니다."
        )
        advanced_group_layout = QVBoxLayout(self.advanced_group)
        self.advanced_content = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_content)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_group_layout.addWidget(self.advanced_content)
        self.object_releasing_label = QLabel("Object-Releasing: Not run.")
        self.object_releasing_label.setWordWrap(True)
        advanced_layout.addWidget(self.object_releasing_label)

        advanced_layout.addWidget(QLabel("Standard Brown-Conrady vs Object-Releasing"))
        self.ro_comparison_table = _PageScrollTableWidget(len(_RO_COMPARISON_ROW_LABELS), 2)
        self.ro_comparison_table.setHorizontalHeaderLabels(["Standard Brown", "Object-Releasing"])
        self.ro_comparison_table.setVerticalHeaderLabels(_RO_COMPARISON_ROW_LABELS)
        self.ro_comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ro_comparison_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.ro_comparison_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        advanced_layout.addWidget(self.ro_comparison_table)
        self.ro_comparison_summary_label = QLabel("Standard vs Object-Releasing comparison: Not run.")
        self.ro_comparison_summary_label.setWordWrap(True)
        advanced_layout.addWidget(self.ro_comparison_summary_label)

        model_layout.addWidget(self.advanced_group)
        self.advanced_group.toggled.connect(self._on_advanced_group_toggled)
        self.set_advanced_calibration_available(False)
        self._fit_table_to_rows(self.table)
        self._fit_table_to_rows(self.ro_comparison_table)

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

    @staticmethod
    def _fit_table_to_rows(table: QTableWidget) -> None:
        """모든 행을 펼쳐 페이지 scroll만 사용하도록 table 높이를 맞춘다."""
        table.resizeRowsToContents()
        header_height = table.horizontalHeader().sizeHint().height()
        rows_height = sum(table.rowHeight(row) for row in range(table.rowCount()))
        table.setFixedHeight(header_height + rows_height + table.frameWidth() * 2 + 4)

    def _on_advanced_group_toggled(self, expanded: bool) -> None:
        """Advanced 내용은 Object-Releasing 사용 중일 때만 펼친다."""
        expanded = bool(expanded and self._advanced_calibration_available)
        if self.advanced_group.isChecked() != expanded:
            self.advanced_group.blockSignals(True)
            self.advanced_group.setChecked(expanded)
            self.advanced_group.blockSignals(False)
        self.advanced_content.setVisible(expanded)
        self.advanced_group.setTitle(
            "▼ Advanced Calibration" if expanded else "▶ Advanced Calibration"
        )

    def set_advanced_calibration_available(self, available: bool) -> None:
        """Object-Releasing 선택 상태에서만 Advanced 토글을 활성화한다."""
        self._advanced_calibration_available = bool(available)
        self.advanced_group.setEnabled(self._advanced_calibration_available)
        if not self._advanced_calibration_available:
            self.advanced_group.setChecked(False)
            self._on_advanced_group_toggled(False)

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
        object_releasing_validation: ObjectReleasingValidationResult | None = None,
        standard_vs_object_releasing: StandardVsObjectReleasingComparison | None = None,
    ) -> None:
        self._calibration_results = calibration_results
        self._object_releasing_result = object_releasing_result
        self.object_releasing_label.setText(
            _format_object_releasing_result(object_releasing_result, object_releasing_validation)
        )
        self._set_ro_comparison(standard_vs_object_releasing)
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
        self._fit_table_to_rows(self.table)

    def _set_ro_comparison(self, comparison: StandardVsObjectReleasingComparison | None) -> None:
        self.ro_comparison_summary_label.setText(_format_ro_comparison_summary(comparison))
        if comparison is None or not comparison.success:
            for row in range(len(_RO_COMPARISON_ROW_LABELS)):
                self.ro_comparison_table.setItem(row, 0, QTableWidgetItem("N/A"))
                self.ro_comparison_table.setItem(row, 1, QTableWidgetItem("N/A"))
            return
        standard_values, ro_values = _ro_comparison_cell_values(comparison)
        for row, (sv, rv) in enumerate(zip(standard_values, ro_values)):
            self.ro_comparison_table.setItem(row, 0, QTableWidgetItem(sv))
            self.ro_comparison_table.setItem(row, 1, QTableWidgetItem(rv))

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
