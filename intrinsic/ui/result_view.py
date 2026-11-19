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
    ValidationResult,
)
from calibration.recommender import model_selection_status
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
    register_lens_requested = Signal(object)  # CameraModelType - CAMERA–LiDAR 렌즈 프리셋으로
    cross_dataset_requested = Signal()

    def __init__(self, parent: QWidget | None = None, *, standalone: bool = True):
        super().__init__(parent)
        self._calibration_results: dict[CameraModelType, CalibrationResult] = {}

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
            "Validation Status", "Train RMS", "Test Frames", "Successful Test Frames",
            "Failed Test Frames", "Failure Reason", "Test RMS", "Test P95",
            "Test Edge RMS", "Straightness",
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

        self._fit_table_to_rows(self.table)

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
        self.register_lens_button = QPushButton("CAMERA–LiDAR 렌즈로 등록")
        self.register_lens_button.setToolTip(
            "이 모델의 K·D 를 CAMERA–LiDAR 캘리브레이션의 렌즈 프리셋으로 등록합니다 "
            "(3단계에서 카메라마다 고르는 목록)."
        )
        self.register_lens_button.clicked.connect(
            lambda: self.register_lens_requested.emit(self.model_combo.currentData())
        )
        export_layout.addWidget(self.register_lens_button)
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
    ) -> None:
        self._calibration_results = calibration_results
        score_by_model = {s.model_name: s for s in scores}
        for col, m in enumerate(_MODEL_ORDER):
            cal = calibration_results.get(m)
            val = validation_results.get(m)
            score = score_by_model.get(m)

            _eligible, status, ineligible_reason = model_selection_status(cal, val)
            validation_status = score.selection_status if score else status
            failure_reason_text = (
                (score.selection_ineligibility_reason if score else None)
                or ineligible_reason
                or (val.error_message if val and val.error_message else None)
                or ""
            )
            test_frame_count = str(len(val.test_frame_ids)) if val else "N/A"
            successful_test_frame_count = str(len(val.per_frame_error)) if val else "N/A"
            failed_test_frame_count = str(len(val.failed_test_frame_ids)) if val else "N/A"
            # frame별 pose 추정 실패 사유(Fisheye fallback 단계 등) - 전체 요약은
            # Failure Reason 셀 텍스트에 짧게 붙이고, frame별 detail은 tooltip으로.
            failed_reason_tooltip = ""
            if val and val.failed_test_frame_ids:
                per_frame_reasons = getattr(val, "failed_test_frame_reasons", {}) or {}
                failed_reason_tooltip = "\n".join(
                    f"{fid}: {per_frame_reasons.get(fid, '(사유 미기록)')}"
                    for fid in val.failed_test_frame_ids
                )
                if not failure_reason_text:
                    failure_reason_text = f"{len(val.failed_test_frame_ids)} test frame(s) pose 추정 실패"
            train_rms = _fmt(cal.rms_error) if cal and cal.success else "FAIL"
            test_rms = _fmt(val.test_rms) if val and val.success else "N/A"
            if val and val.success and val.edge_rms is not None:
                edge_rms = _fmt(val.edge_rms)
            else:
                edge_rms = "N/A"
            straightness = _fmt(val.straightness_residual) if val and val.straightness_residual is not None else "N/A"
            if straightness != "N/A" and getattr(val, "straightness_source", None):
                suffix = "TEST" if val.straightness_source == "test" else "TRAIN fallback"
                straightness = f"{straightness} ({suffix})"
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
            score_str = f"{score.score:.3f}" if score and score.is_selection_eligible else (
                "Not eligible" if score else "N/A"
            )
            selection_conf = (
                f"{score.selection_confidence:.0f}% {score.selection_confidence_level}"
                if score and score.selection_confidence is not None and score.selection_confidence_level else "N/A"
            )
            recommend = "⭐" if (score and score.is_recommended) else ""

            for row, value in enumerate(
                [
                    validation_status, train_rms, test_frame_count, successful_test_frame_count,
                    failed_test_frame_count, failure_reason_text, test_rms, p95, edge_rms, straightness,
                    radial, aic, bic, stability, observability,
                    undistortion, score_str, selection_conf, recommend,
                ]
            ):
                item = QTableWidgetItem(value)
                # RMSE/P95/Stability 등의 숫자는 기본 흰색을 유지하고 실제 추천
                # 판정 셀만 semantic GOOD 색으로 강조한다.
                if row == len(self._row_labels) - 1 and recommend:
                    item.setForeground(qcolor(Theme.GOOD))
                # "Failed Test Frames"/"Failure Reason" 행에는 frame별 실패
                # 사유(예: Fisheye fallback 단계)를 tooltip으로 붙여서, 표에는
                # 요약만 두고 detail은 hover로 볼 수 있게 한다.
                if failed_reason_tooltip and self._row_labels[row] in ("Failed Test Frames", "Failure Reason"):
                    item.setToolTip(failed_reason_tooltip)
                self.table.setItem(row, col, item)

        self._update_model_status()
        self._fit_table_to_rows(self.table)

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

    # ------------------------------------------------------------------
    # Repeated K-Fold - 고정 3행(Brown-Conrady/Rational/Fisheye) 표 +
    # 실시간 진행률(progress bar, 모델별 WAITING/RUNNING/COMPLETE/FAILED).
    # 계산/metric/fold split은 전혀 손대지 않는다 - 여기는 오직 이미 계산된
    # KFoldProgressEvent/RepeatedKFoldResult를 화면에 어떻게 보여줄지만 담당.
    # ------------------------------------------------------------------

    def select_model(self, model: CameraModelType) -> None:
        idx = self.model_combo.findData(model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self._update_model_status()

    def prompt_save_path(self, default_name: str, filter_str: str) -> str | None:
        path, _ = QFileDialog.getSaveFileName(self, "저장 위치 선택", default_name, filter_str)
        return path or None
