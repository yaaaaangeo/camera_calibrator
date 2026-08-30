"""
camera_calibrator.ui.external_compare_view
===============================================

사용자 요청 - "예전에 다른 사람/다른 툴로 구한 파라미터"와 "지금 이 툴로
구한 파라미터" 중 뭐가 더 정확한지 정량적으로, 누구나 납득할 수 있게
비교하는 화면.

계산은 전부 calibration/external_compare.py가 한다 - 이 파일은 입력(외부
파라미터를 어떻게 받을지: OpenCV YAML 파일 또는 수동 입력)과 결과 표시
(비교표, 한 줄 평, Original/Reference/Candidate undistortion visual comparison)
만 담당한다 (백엔드/UI 분리 원칙, 다른 view들과 동일).
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
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
    CameraConfig,
    CameraModelType,
    Dataset,
    PatternConfig,
    ValidationResult,
)
from calibration.recommender import compute_model_scores
from calibration.detector import summarize_dataset
from calibration.external_compare import (
    ComparisonSide,
    ExternalCameraParams,
    ExternalComparisonResult,
    compare_reference_candidate_calibrations,
    compare_with_external_params,
)
from calibration.calibration_io import StandardCalibration, load_standard_calibration
from ui.worker import BenchmarkDetectionWorker, ExternalComparisonWorker, run_worker_in_thread
from ui.theme import Theme

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

_PANEL_MAX_WIDTH = 460
_BEST_CELL_COLOR = QColor(Theme.TABLE_BEST)
_WINNER_CELL_COLOR = QColor(Theme.TABLE_WINNER)
_TIE_CELL_COLOR = QColor(Theme.TABLE_TIE)
_TAB_HEADER_COLORS = Theme.TABLE_HEADER_VARIANTS


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
        self._calibration_results: dict[CameraModelType, CalibrationResult] = {}
        self._validation_results: dict[CameraModelType, ValidationResult] = {}
        self._use_rational_model = False
        self._last_result: ExternalComparisonResult | None = None
        self._loaded_yaml_path: str | None = None
        self._loaded_calibration: StandardCalibration | None = None
        self._reference_calibration: StandardCalibration | None = None
        self._candidate_calibration: StandardCalibration | None = None
        self._benchmark_dataset: Dataset | None = None
        self._benchmark_image_paths: list[str] = []
        self._comparison_thread = None
        self._comparison_worker = None
        self._benchmark_thread = None
        self._benchmark_worker = None

        root_layout = QVBoxLayout(self)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        intro = QLabel(
            "두 calibration 파일을 직접 비교하거나, 외부 파라미터와 현재 선택한 내 모델을 "
            "Hold-out test 프레임에서 같은 조건으로 재평가해 비교합니다. "
            "어느 한쪽에 유리한 조건을 주지 않는 정량 비교입니다."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addWidget(self._build_pair_file_group())
        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_run_row())
        legend = QLabel(
            "표 색상: 짙은 녹색 = 더 낮은 오차/우세한 결과 · 회색 = 동률 또는 줄 구분 · "
            "파라미터 차이는 그 자체로 우열이 아니므로 색상 판정을 하지 않습니다."
        )
        legend.setWordWrap(True)
        legend.setStyleSheet(
            f"background: {Theme.BG_SECONDARY}; border: 1px solid {Theme.BORDER}; "
            f"padding: 6px; color: {Theme.TEXT_SECONDARY};"
        )
        layout.addWidget(legend)
        layout.addWidget(self._build_benchmark_tabs(), stretch=1)
        self.scroll_area.setWidget(content)
        root_layout.addWidget(self.scroll_area)

    # ------------------------------------------------------------------
    # 입력 영역
    # ------------------------------------------------------------------

    def _make_collapsible_group(self, title: str, body: QWidget, *, checked: bool = True) -> QWidget:
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QPushButton()
        header.setCheckable(True)
        header.setChecked(checked)
        header.setFlat(True)
        header.setCursor(Qt.PointingHandCursor)
        header.setStyleSheet(
            f"text-align: left; font-weight: 600; padding: 6px; "
            f"border: 1px solid {Theme.BORDER}; background: {Theme.BG_SECONDARY};"
        )

        def update_header(expanded: bool) -> None:
            header.setText(f"{'▼' if expanded else '▶'} {title}")
            body.setVisible(expanded)

        header.toggled.connect(update_header)
        update_header(checked)

        body.setObjectName("collapsibleBody")
        body.setStyleSheet(
            f"QWidget#collapsibleBody {{ border-left: 1px solid {Theme.BORDER}; "
            f"border-right: 1px solid {Theme.BORDER}; "
            f"border-bottom: 1px solid {Theme.BORDER}; }}"
        )
        layout.addWidget(header)
        layout.addWidget(body)
        return group

    def _build_input_group(self) -> QGroupBox:
        body = QWidget()
        outer = QVBoxLayout(body)

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
        self.load_yaml_button = QPushButton("Calibration 파일 불러오기...")
        self.load_yaml_button.clicked.connect(self._on_load_yaml)
        yaml_row.addWidget(self.load_yaml_button)
        self.yaml_status_label = QLabel(
            "불러온 파일 없음 - 아래 수동 입력을 쓰거나 OpenCV/ROS/Kalibr/JSON calibration 파일을 불러오세요."
        )
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

        return self._make_collapsible_group("비교할 외부 파라미터", body)

    def _build_pair_file_group(self) -> QGroupBox:
        body = QWidget()
        v = QVBoxLayout(body)

        file_row = QGridLayout()
        file_row.setColumnStretch(1, 1)
        file_row.setColumnStretch(3, 1)
        self.load_reference_button = QPushButton("Reference 불러오기...")
        self.load_reference_button.clicked.connect(self._on_load_reference_calibration)
        file_row.addWidget(self.load_reference_button, 0, 0)
        self.reference_status_label = QLabel("Reference 파일 없음")
        self.reference_status_label.setWordWrap(True)
        file_row.addWidget(self.reference_status_label, 0, 1)

        self.load_candidate_button = QPushButton("Candidate 불러오기...")
        self.load_candidate_button.clicked.connect(self._on_load_candidate_calibration)
        file_row.addWidget(self.load_candidate_button, 0, 2)
        self.candidate_status_label = QLabel("Candidate 파일 없음")
        self.candidate_status_label.setWordWrap(True)
        file_row.addWidget(self.candidate_status_label, 0, 3)
        v.addLayout(file_row)

        self.run_pair_button = QPushButton("Reference vs Candidate 파일 비교 실행")
        self.run_pair_button.setProperty("role", "primary")
        self.run_pair_button.clicked.connect(self._on_run_file_pair_comparison)
        v.addWidget(self._build_evaluation_dataset_widget())
        v.addWidget(self.run_pair_button)
        return self._make_collapsible_group("Reference / Candidate 파일 비교", body)

    def _build_evaluation_dataset_widget(self) -> QWidget:
        body = QWidget()
        layout = QGridLayout(body)
        layout.setColumnStretch(1, 1)

        layout.addWidget(QLabel("Evaluation Source"), 0, 0)
        self.evaluation_source_combo = QComboBox()
        self.evaluation_source_combo.addItem("Auto [Recommended]", userData="auto")
        self.evaluation_source_combo.addItem("Internal Hold-out", userData="internal_holdout")
        self.evaluation_source_combo.addItem("Independent Benchmark", userData="independent_benchmark")
        self.evaluation_source_combo.currentIndexChanged.connect(self._update_benchmark_status)
        layout.addWidget(self.evaluation_source_combo, 0, 1, 1, 2)

        self.select_benchmark_button = QPushButton("Select Benchmark Images...")
        self.select_benchmark_button.clicked.connect(self._on_select_benchmark_images)
        layout.addWidget(self.select_benchmark_button, 1, 0)

        self.clear_benchmark_button = QPushButton("Clear Benchmark Dataset")
        self.clear_benchmark_button.clicked.connect(self._on_clear_benchmark_dataset)
        layout.addWidget(self.clear_benchmark_button, 1, 1)

        self.benchmark_status_label = QLabel()
        self.benchmark_status_label.setWordWrap(True)
        self.benchmark_status_label.setProperty("tone", "muted")
        layout.addWidget(self.benchmark_status_label, 2, 0, 1, 3)
        self._update_benchmark_status()
        return body

    @staticmethod
    def _make_double_spin(default: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-100000.0, 100000.0)
        spin.setDecimals(4)
        spin.setValue(default)
        return spin

    def _build_run_row(self) -> QWidget:
        body = QWidget()
        h = QHBoxLayout(body)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("모델:"))
        self.my_model_combo = QComboBox()
        for m in _MODEL_ORDER:
            self.my_model_combo.addItem(_MODEL_LABELS[m], userData=m)
        self.my_model_combo.currentIndexChanged.connect(self._update_benchmark_status)
        h.addWidget(self.my_model_combo)

        self.run_button = QPushButton("비교 실행")
        self.run_button.setProperty("role", "primary")
        # 실제 버튼 경로는 무거운 비교를 별도 프로세스에서 실행한다. 기존
        # _on_run_comparison()은 계산 로직 단위 테스트용 동기 진입점으로 유지.
        self.run_button.clicked.connect(self._on_run_comparison_async)
        h.addWidget(self.run_button)
        self.comparison_status_label = QLabel("비교 대기 중")
        self.comparison_status_label.setWordWrap(True)
        h.addWidget(self.comparison_status_label, stretch=1)
        h.addStretch(1)
        return self._make_collapsible_group("비교할 내모델", body)

    # ------------------------------------------------------------------
    # 결과 표시 영역
    # ------------------------------------------------------------------

    def _build_benchmark_tabs(self) -> QTabWidget:
        self.benchmark_tabs = QTabWidget()

        self.benchmark_tabs.addTab(
            self._tab_page([self._build_result_group()]),
            "Overview",
        )
        self.benchmark_tabs.addTab(
            self._tab_page([
                self._build_spatial_error_group(),
                self._build_residual_heatmap_group(),
                self._build_radial_error_group(),
                self._build_worst_case_group(),
                self._build_error_distribution_group(),
            ]),
            "Error Analysis",
        )
        self.benchmark_tabs.addTab(
            self._tab_page([self._build_visual_group()]),
            "Visual Comparison",
        )
        self.benchmark_tabs.addTab(
            self._tab_page([
                self._build_benchmark_validation_group(),
                self._build_statistical_tests_group(),
                self._build_bootstrap_comparison_group(),
            ]),
            "Statistical Validation",
        )
        self.benchmark_tabs.addTab(
            self._tab_page([self._build_model_comparison_group()]),
            "Model Analysis",
        )
        self.benchmark_tabs.addTab(
            self._tab_page([self._build_parameter_diff_group(), self._build_parameter_diagnostics_group()]),
            "Parameter Analysis",
        )
        self.benchmark_tabs.addTab(
            self._tab_page([self._build_final_benchmark_group()]),
            "Final Report",
        )
        self.benchmark_tabs.setMinimumHeight(520)
        self.benchmark_tabs.currentChanged.connect(self._on_benchmark_tab_changed)
        self._style_result_tables()
        return self.benchmark_tabs

    def _style_result_tables(self) -> None:
        for tab_index in range(self.benchmark_tabs.count()):
            page = self.benchmark_tabs.widget(tab_index)
            header_color = _TAB_HEADER_COLORS[tab_index % len(_TAB_HEADER_COLORS)]
            for table in page.findChildren(QTableWidget):
                if table is self.parameter_table:
                    header_color = _TAB_HEADER_COLORS[4]
                elif table is self.final_benchmark_table or table is self.final_report_evidence_table:
                    header_color = _TAB_HEADER_COLORS[5]
                table.setAlternatingRowColors(True)
                table.setStyleSheet(
                    f"QTableView {{ font-size: 10pt; color: {Theme.TEXT_PRIMARY}; "
                    f"background-color: {Theme.TABLE_ODD}; alternate-background-color: {Theme.TABLE_EVEN}; "
                    f"gridline-color: #2A2A2A; selection-background-color: {Theme.BG_SELECTED}; }}"
                    f"QHeaderView::section {{ background-color: {header_color}; "
                    f"color: {Theme.TEXT_PRIMARY}; font-size: 10pt; font-weight: 600; padding: 6px; "
                    f"border: none; border-right: 1px solid {Theme.BORDER}; "
                    f"border-bottom: 1px solid {Theme.BORDER_STRONG}; }}"
                )

                body_font = table.font()
                body_font.setPointSize(10)
                table.setFont(body_font)
                table.verticalHeader().setDefaultSectionSize(28)

                header = table.horizontalHeader()
                header.setDefaultAlignment(Qt.AlignCenter)
                header.setMinimumSectionSize(72)
                header.setMaximumSectionSize(360)
                columns = table.columnCount()
                if columns == 0:  # heatmap은 결과 렌더 시 20x20 열 폭을 별도 지정
                    continue
                if table is self.final_report_evidence_table:
                    header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
                    header.setSectionResizeMode(1, QHeaderView.Stretch)
                elif columns <= 5:
                    for col in range(columns):
                        header.setSectionResizeMode(col, QHeaderView.Stretch)
                else:
                    for col in range(columns):
                        header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
                    header.setSectionResizeMode(columns - 1, QHeaderView.Stretch)

    @staticmethod
    def _highlight_item(table: QTableWidget, row: int, col: int, color: QColor, *, bold: bool = False) -> None:
        item = table.item(row, col)
        if item is None:
            return
        item.setBackground(color)
        if bold:
            font = item.font()
            font.setBold(True)
            item.setFont(font)

    @classmethod
    def _highlight_lower_pair(
        cls,
        table: QTableWidget,
        row: int,
        reference_col: int,
        candidate_col: int,
        reference_value,
        candidate_value,
    ) -> None:
        if reference_value is None or candidate_value is None:
            return
        if abs(float(reference_value) - float(candidate_value)) < 1e-12:
            cls._highlight_item(table, row, reference_col, _TIE_CELL_COLOR)
            cls._highlight_item(table, row, candidate_col, _TIE_CELL_COLOR)
        elif float(reference_value) < float(candidate_value):
            cls._highlight_item(table, row, reference_col, _BEST_CELL_COLOR, bold=True)
        else:
            cls._highlight_item(table, row, candidate_col, _BEST_CELL_COLOR, bold=True)

    @classmethod
    def _highlight_named_winner(
        cls,
        table: QTableWidget,
        row: int,
        winner: str | None,
        reference_label: str,
        candidate_label: str,
        reference_col: int,
        candidate_col: int,
        winner_col: int,
    ) -> None:
        if winner == reference_label:
            cls._highlight_item(table, row, reference_col, _BEST_CELL_COLOR, bold=True)
            cls._highlight_item(table, row, winner_col, _WINNER_CELL_COLOR, bold=True)
        elif winner == candidate_label:
            cls._highlight_item(table, row, candidate_col, _BEST_CELL_COLOR, bold=True)
            cls._highlight_item(table, row, winner_col, _WINNER_CELL_COLOR, bold=True)
        elif winner and winner.lower() == "tie":
            cls._highlight_item(table, row, reference_col, _TIE_CELL_COLOR)
            cls._highlight_item(table, row, candidate_col, _TIE_CELL_COLOR)
            cls._highlight_item(table, row, winner_col, _TIE_CELL_COLOR, bold=True)

    def _tab_page(self, widgets: list[QWidget]) -> QScrollArea:
        """각 결과 탭은 자기 스크롤을 가진다.

        예전에는 External Compare 전체에 바깥 스크롤 하나만 있어 Overview의
        스크롤 offset이 높이가 다른 탭에도 그대로 적용됐고, 탭을 바꾸면 데이터가
        있는데도 빈 영역부터 보였다.
        """
        page = QScrollArea()
        page.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch(1)
        page.setWidget(content)
        return page

    def _on_benchmark_tab_changed(self, index: int) -> None:
        page = self.benchmark_tabs.widget(index)
        if isinstance(page, QScrollArea):
            page.verticalScrollBar().setValue(0)
        self.scroll_area.ensureWidgetVisible(self.benchmark_tabs, 0, 20)

    def _build_result_group(self) -> QGroupBox:
        group = QGroupBox("정량 비교 결과 (Hold-out test 프레임 기준, 동일 조건)")
        v = QVBoxLayout(group)

        self.table = QTableWidget(11, 4)
        self.table.setHorizontalHeaderLabels(["Reference", "Candidate", "Improvement", "Winner"])
        self.table.setVerticalHeaderLabels(
            [
                "Mean (px)",
                "Median (px)",
                "RMSE (px)",
                "Std (px)",
                "P90 (px)",
                "P95 (px)",
                "P99 (px)",
                "Max (px)",
                "Edge RMS - 외곽 (px)",
                "Straightness - 직선성 (px)",
                "프레임별 승 개수",
            ]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.table)

        self.verdict_label = QLabel("아직 비교를 실행하지 않았습니다.")
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setProperty("role", "sectionTitle")
        v.addWidget(self.verdict_label)

        self.decision_label = QLabel("Decision: N/A")
        self.decision_label.setWordWrap(True)
        self.decision_label.setProperty("tone", "info")
        v.addWidget(self.decision_label)

        self.caveats_label = QLabel("")
        self.caveats_label.setWordWrap(True)
        self.caveats_label.setProperty("tone", "muted")
        v.addWidget(self.caveats_label)

        return group

    def _build_visual_group(self) -> QGroupBox:
        group = QGroupBox("Undistortion Visual Comparison")
        v = QVBoxLayout(group)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("이미지:"))
        self.image_combo = QComboBox()
        control_row.addWidget(self.image_combo, stretch=1)
        self.refresh_visual_button = QPushButton("갱신")
        self.refresh_visual_button.clicked.connect(self._update_visual)
        control_row.addWidget(self.refresh_visual_button)
        v.addLayout(control_row)

        visual_grid = QGridLayout()
        visual_grid.setColumnStretch(0, 1)
        visual_grid.setColumnStretch(1, 1)
        visual_grid.setColumnStretch(2, 1)

        original_box, self.original_image_label, self.original_caption_label = self._make_visual_panel(
            "Original", "비교를 실행하면 표시됩니다."
        )
        reference_box, self.reference_image_label, self.reference_caption_label = self._make_visual_panel(
            "Reference Undistorted", "비교를 실행하면 표시됩니다."
        )
        candidate_box, self.candidate_image_label, self.candidate_caption_label = self._make_visual_panel(
            "Candidate Undistorted", "비교를 실행하면 표시됩니다."
        )
        overlay_box, self.overlay_image_label, self.overlay_caption_label = self._make_visual_panel(
            "Overlay View", "비교를 실행하면 표시됩니다."
        )
        difference_box, self.difference_image_label, self.difference_caption_label = self._make_visual_panel(
            "Difference View", "비교를 실행하면 표시됩니다."
        )

        # Backward-compatible aliases for older UI tests and integrations.
        self.external_image_label = self.reference_image_label
        self.external_caption_label = self.reference_caption_label
        self.mine_image_label = self.candidate_image_label
        self.mine_caption_label = self.candidate_caption_label

        visual_grid.addWidget(original_box, 0, 0)
        visual_grid.addWidget(reference_box, 0, 1)
        visual_grid.addWidget(candidate_box, 0, 2)
        visual_grid.addWidget(overlay_box, 1, 0, 1, 2)
        visual_grid.addWidget(difference_box, 1, 2)

        v.addLayout(visual_grid)
        return group

    def _make_visual_panel(self, title: str, initial_text: str) -> tuple[QGroupBox, QLabel, QLabel]:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        image_label = QLabel(initial_text)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumHeight(180)
        image_label.setProperty("surface", "image")
        layout.addWidget(image_label)
        caption_label = QLabel("")
        caption_label.setWordWrap(True)
        caption_label.setProperty("tone", "muted")
        layout.addWidget(caption_label)
        return box, image_label, caption_label

    def _build_final_benchmark_group(self) -> QGroupBox:
        group = QGroupBox("Calibration Benchmark Report")
        v = QVBoxLayout(group)

        self.final_report_label = QLabel("아직 비교를 실행하지 않았습니다.")
        self.final_report_label.setWordWrap(True)
        self.final_report_label.setProperty("role", "sectionTitle")
        v.addWidget(self.final_report_label)

        self.final_report_evidence_table = QTableWidget(0, 2)
        self.final_report_evidence_table.setHorizontalHeaderLabels(["Section", "Evidence Summary"])
        self.final_report_evidence_table.horizontalHeader().setStretchLastSection(True)
        self.final_report_evidence_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.final_report_evidence_table)

        final_table_label = QLabel("Final Benchmark Table")
        final_table_label.setProperty("role", "sectionTitle")
        v.addWidget(final_table_label)

        self.final_benchmark_table = QTableWidget(0, 5)
        self.final_benchmark_table.setHorizontalHeaderLabels(
            ["Metric", "Reference", "Candidate", "Improvement", "Winner"]
        )
        self.final_benchmark_table.horizontalHeader().setStretchLastSection(True)
        self.final_benchmark_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.final_benchmark_table)
        return group

    def _build_worst_case_group(self) -> QGroupBox:
        group = QGroupBox("Worst-case Analysis")
        v = QVBoxLayout(group)

        self.worst_case_table = QTableWidget(0, 6)
        self.worst_case_table.setHorizontalHeaderLabels(
            ["Category", "Reference Worst", "Reference Value", "Candidate Worst", "Candidate Value", "Winner"]
        )
        self.worst_case_table.horizontalHeader().setStretchLastSection(True)
        self.worst_case_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.worst_case_table)
        return group

    def _build_spatial_error_group(self) -> QGroupBox:
        group = QGroupBox("Spatial Error 3x3 / 5x5")
        v = QVBoxLayout(group)

        self.spatial_error_table = QTableWidget(0, 13)
        self.spatial_error_table.setHorizontalHeaderLabels(
            [
                "Grid", "Cell", "Ref N", "Cand N",
                "Ref Mean", "Cand Mean", "Mean Imp.",
                "Ref RMSE", "Cand RMSE", "RMSE Imp.",
                "Ref P95", "Cand P95", "P95 Imp.",
            ]
        )
        self.spatial_error_table.horizontalHeader().setStretchLastSection(True)
        self.spatial_error_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.spatial_error_table)
        return group

    def _build_residual_heatmap_group(self) -> QGroupBox:
        group = QGroupBox("Residual Heatmap Reference / Candidate / Difference")
        v = QVBoxLayout(group)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("Metric:"))
        self.heatmap_metric_combo = QComboBox()
        self.heatmap_metric_combo.addItem("RMSE 20x20", userData="rmse_20x20")
        self.heatmap_metric_combo.addItem("P95 20x20", userData="p95_20x20")
        self.heatmap_metric_combo.currentIndexChanged.connect(self._render_selected_residual_heatmap)
        control_row.addWidget(self.heatmap_metric_combo)
        control_row.addStretch(1)
        v.addLayout(control_row)

        self.heatmap_summary_label = QLabel("아직 비교를 실행하지 않았습니다.")
        self.heatmap_summary_label.setWordWrap(True)
        self.heatmap_summary_label.setProperty("tone", "muted")
        v.addWidget(self.heatmap_summary_label)

        grid = QGridLayout()
        self.reference_heatmap_table = self._make_heatmap_table()
        self.candidate_heatmap_table = self._make_heatmap_table()
        self.difference_heatmap_table = self._make_heatmap_table()
        grid.addWidget(QLabel("Reference"), 0, 0)
        grid.addWidget(QLabel("Candidate"), 0, 1)
        grid.addWidget(QLabel("Difference (Candidate - Reference)"), 0, 2)
        grid.addWidget(self.reference_heatmap_table, 1, 0)
        grid.addWidget(self.candidate_heatmap_table, 1, 1)
        grid.addWidget(self.difference_heatmap_table, 1, 2)
        v.addLayout(grid)
        return group

    def _make_heatmap_table(self) -> QTableWidget:
        table = QTableWidget(0, 0)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)
        table.setMinimumHeight(180)
        return table

    def _build_radial_error_group(self) -> QGroupBox:
        group = QGroupBox("Radial Error Profile")
        v = QVBoxLayout(group)

        self.radial_error_table = QTableWidget(0, 12)
        self.radial_error_table.setHorizontalHeaderLabels(
            [
                "Profile", "Band", "Radius", "Ref N", "Cand N",
                "Ref RMSE", "Cand RMSE", "RMSE Imp.",
                "Ref P95", "Cand P95", "P95 Imp.", "Max Imp.",
            ]
        )
        self.radial_error_table.horizontalHeader().setStretchLastSection(True)
        self.radial_error_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.radial_error_table)
        return group

    def _build_error_distribution_group(self) -> QGroupBox:
        group = QGroupBox("Error Distribution Comparison")
        v = QVBoxLayout(group)

        self.error_distribution_summary_label = QLabel("아직 비교를 실행하지 않았습니다.")
        self.error_distribution_summary_label.setWordWrap(True)
        self.error_distribution_summary_label.setProperty("tone", "muted")
        v.addWidget(self.error_distribution_summary_label)

        self.error_distribution_table = QTableWidget(0, 7)
        self.error_distribution_table.setHorizontalHeaderLabels(
            ["Bin", "Ref Count", "Cand Count", "Ref Density", "Cand Density", "Ref CDF", "Cand CDF"]
        )
        self.error_distribution_table.horizontalHeader().setStretchLastSection(True)
        self.error_distribution_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.error_distribution_table)
        return group

    def _build_model_comparison_group(self) -> QGroupBox:
        group = QGroupBox("Model Comparison: Ideal Pinhole / Brown-Conrady / Rational / Fisheye")
        v = QVBoxLayout(group)

        note = QLabel(
            "현재 프로젝트의 3개 calibration model을 hold-out validation과 information criteria 기준으로 비교합니다."
        )
        note.setWordWrap(True)
        note.setProperty("tone", "muted")
        v.addWidget(note)

        self.model_comparison_table = QTableWidget(0, 10)
        self.model_comparison_table.setHorizontalHeaderLabels(
            [
                "Model", "Status", "Train RMS", "Hold-out RMSE", "Hold-out P95",
                "AIC", "BIC", "Complexity", "Score", "Recommendation",
            ]
        )
        self.model_comparison_table.horizontalHeader().setStretchLastSection(True)
        self.model_comparison_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.model_comparison_table)
        return group

    def _build_parameter_diff_group(self) -> QGroupBox:
        group = QGroupBox("Parameter / FOV Difference")
        v = QVBoxLayout(group)

        self.parameter_note_label = QLabel(
            "Parameter similarity ≠ Calibration accuracy. "
            "파라미터 차이는 진단 신호이고, 최종 판단은 hold-out residual/edge/radial/straightness를 함께 봐야 합니다."
        )
        self.parameter_note_label.setWordWrap(True)
        self.parameter_note_label.setProperty("tone", "muted")
        v.addWidget(self.parameter_note_label)

        self.parameter_table = QTableWidget(0, 5)
        self.parameter_table.setHorizontalHeaderLabels(
            ["Parameter", "Reference", "Candidate", "Abs Diff", "Rel Diff"]
        )
        self.parameter_table.horizontalHeader().setStretchLastSection(True)
        self.parameter_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.parameter_table)
        return group

    def _build_benchmark_validation_group(self) -> QGroupBox:
        group = QGroupBox("Benchmark Hold-out / K-fold / Generalization")
        v = QVBoxLayout(group)

        self.benchmark_validation_note_label = QLabel(
            "Reference/Candidate 파라미터를 고정하고 각 validation subset에서 pose만 다시 추정합니다."
        )
        self.benchmark_validation_note_label.setWordWrap(True)
        self.benchmark_validation_note_label.setProperty("tone", "muted")
        v.addWidget(self.benchmark_validation_note_label)

        self.benchmark_validation_table = QTableWidget(0, 9)
        self.benchmark_validation_table.setHorizontalHeaderLabels(
            [
                "Validation",
                "Splits",
                "Ref Train",
                "Ref Val",
                "Ref Gap",
                "Cand Train",
                "Cand Val",
                "Cand Gap",
                "Improvement",
            ]
        )
        self.benchmark_validation_table.horizontalHeader().setStretchLastSection(True)
        self.benchmark_validation_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.benchmark_validation_table)
        return group

    def _build_statistical_tests_group(self) -> QGroupBox:
        group = QGroupBox("Statistical Significance")
        v = QVBoxLayout(group)

        note = QLabel(
            "같은 이미지의 Reference/Candidate per-frame RMS를 paired sample로 비교합니다. "
            "차이는 Candidate - Reference이므로 음수면 Candidate 오차가 더 낮습니다."
        )
        note.setWordWrap(True)
        note.setProperty("tone", "muted")
        v.addWidget(note)

        self.statistical_tests_table = QTableWidget(0, 8)
        self.statistical_tests_table.setHorizontalHeaderLabels(
            ["Test", "N", "Statistic", "p-value", "Effect", "Mean Diff", "Median Diff", "Interpretation"]
        )
        self.statistical_tests_table.horizontalHeader().setStretchLastSection(True)
        self.statistical_tests_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.statistical_tests_table)
        return group

    def _build_bootstrap_comparison_group(self) -> QGroupBox:
        group = QGroupBox("Bootstrap Comparison")
        v = QVBoxLayout(group)

        note = QLabel(
            "공통 프레임 pair를 bootstrap resampling해서 Candidate가 Reference보다 낮은 RMSE를 낼 확률과 CI를 추정합니다."
        )
        note.setWordWrap(True)
        note.setProperty("tone", "muted")
        v.addWidget(note)

        self.bootstrap_table = QTableWidget(0, 2)
        self.bootstrap_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.bootstrap_table.horizontalHeader().setStretchLastSection(True)
        self.bootstrap_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.bootstrap_table)
        return group

    def _build_parameter_diagnostics_group(self) -> QGroupBox:
        group = QGroupBox("Parameter Observability / Stability / Covariance / Sensitivity")
        v = QVBoxLayout(group)

        note = QLabel(
            "Reference/Candidate 각각에 대해 validation 프레임 pose를 고정하고 intrinsic/distortion Jacobian을 수치미분으로 근사합니다."
        )
        note.setWordWrap(True)
        note.setProperty("tone", "muted")
        v.addWidget(note)

        self.parameter_observability_table = QTableWidget(0, 10)
        self.parameter_observability_table.setHorizontalHeaderLabels(
            [
                "Side", "Jacobian", "Rank", "Condition",
                "Min SV", "Max SV", "Max |Corr|", "Weak Params", "Top Correlations", "Warnings",
            ]
        )
        self.parameter_observability_table.horizontalHeader().setStretchLastSection(True)
        self.parameter_observability_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.parameter_observability_table)

        self.parameter_singular_table = QTableWidget(0, 3)
        self.parameter_singular_table.setHorizontalHeaderLabels(["Side", "Index", "Singular Value"])
        self.parameter_singular_table.horizontalHeader().setStretchLastSection(True)
        self.parameter_singular_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.parameter_singular_table)

        self.parameter_stability_table = QTableWidget(0, 7)
        self.parameter_stability_table.setHorizontalHeaderLabels(
            ["Side", "Parameter", "Value", "Std", "CI Low", "CI High", "Stability"]
        )
        self.parameter_stability_table.horizontalHeader().setStretchLastSection(True)
        self.parameter_stability_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.parameter_stability_table)

        self.parameter_sensitivity_table = QTableWidget(0, 6)
        self.parameter_sensitivity_table.setHorizontalHeaderLabels(
            ["Side", "Parameter", "Value", "Perturbation", "RMSE Delta", "Sensitivity / Unit"]
        )
        self.parameter_sensitivity_table.horizontalHeader().setStretchLastSection(True)
        self.parameter_sensitivity_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.parameter_sensitivity_table)

        self.parameter_covariance_table = QTableWidget(0, 4)
        self.parameter_covariance_table.setHorizontalHeaderLabels(["Side", "Row", "Col", "Covariance"])
        self.parameter_covariance_table.horizontalHeader().setStretchLastSection(True)
        self.parameter_covariance_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.parameter_covariance_table)

        self.parameter_correlation_table = QTableWidget(0, 4)
        self.parameter_correlation_table.setHorizontalHeaderLabels(["Side", "Row", "Col", "Correlation"])
        self.parameter_correlation_table.horizontalHeader().setStretchLastSection(True)
        self.parameter_correlation_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.parameter_correlation_table)
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
        calibration_results: dict[CameraModelType, CalibrationResult] | None = None,
        use_rational_model: bool = False,
    ) -> None:
        self._dataset = dataset
        self._camera_config = camera_config
        self._pattern_config = pattern_config
        self._validation_results = validation_results
        self._calibration_results = calibration_results or {}
        self._use_rational_model = use_rational_model
        self._update_benchmark_status()

    # ------------------------------------------------------------------
    # 외부 YAML 불러오기
    # ------------------------------------------------------------------

    def _on_load_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Calibration 파일 선택",
            "",
            "Calibration Files (*.yaml *.yml *.json);;YAML (*.yaml *.yml);;JSON (*.json)",
        )
        if not path:
            return
        try:
            loaded = load_standard_calibration(path)
        except Exception as e:  # noqa: BLE001
            # 이전에 성공한 파일이 남아 있으면 이후 [비교 실행]이 눈에 보이는
            # 수동 입력 대신 그 오래된 파일을 계속 사용한다. 실패 시 파일 모드를
            # 확실히 해제해 아래 입력란이 즉시 유효해지게 한다.
            self._loaded_yaml_path = None
            self._loaded_calibration = None
            self.yaml_status_label.setText("불러오기 실패 - 아래 수동 입력을 사용합니다.")
            self._manual_group.setEnabled(True)
            QMessageBox.critical(self, "불러오기 실패", str(e))
            return

        self._loaded_yaml_path = path
        self._loaded_calibration = loaded
        if loaded.model_name is not None:
            idx = self.external_model_combo.findData(loaded.model_name)
            if idx >= 0:
                self.external_model_combo.setCurrentIndex(idx)
        self.fx_spin.setValue(float(loaded.camera_matrix[0, 0]))
        self.fy_spin.setValue(float(loaded.camera_matrix[1, 1]))
        self.cx_spin.setValue(float(loaded.camera_matrix[0, 2]))
        self.cy_spin.setValue(float(loaded.camera_matrix[1, 2]))
        self.distortion_edit.setText(", ".join(f"{float(v):.12g}" for v in loaded.distortion.reshape(-1)))

        if loaded.model_name is not None:
            self.yaml_status_label.setText(
                f"불러옴: {path}\n"
                f"포맷: {loaded.source_format}, 해상도: {loaded.width or '?'}x{loaded.height or '?'}\n"
                f"(이 파일에 저장된 모델 종류 '{_MODEL_LABELS.get(loaded.model_name, loaded.model_name)}'를 자동 선택했습니다 - "
                "실제와 다르면 위에서 바꿔주세요.)"
            )
        else:
            self.yaml_status_label.setText(
                f"불러옴: {path}\n"
                f"포맷: {loaded.source_format}, 해상도: {loaded.width or '?'}x{loaded.height or '?'}\n"
                "(이 파일에는 모델 종류 정보가 없거나 애매합니다 - 위에서 직접 선택해 주세요.)"
            )

    def _load_calibration_file(self) -> tuple[str, StandardCalibration] | None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Calibration 파일 선택",
            "",
            "Calibration Files (*.yaml *.yml *.json);;YAML (*.yaml *.yml);;JSON (*.json)",
        )
        if not path:
            return None
        try:
            return path, load_standard_calibration(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "불러오기 실패", str(e))
            return None

    @staticmethod
    def _format_loaded_calibration_status(prefix: str, path: str, cal: StandardCalibration) -> str:
        model = cal.model_name.value if cal.model_name is not None else "?"
        return (
            f"{prefix}: {path}\n"
            f"포맷: {cal.source_format}, 모델: {model}, 해상도: {cal.width or '?'}x{cal.height or '?'}"
        )

    def _on_load_reference_calibration(self) -> None:
        loaded = self._load_calibration_file()
        if loaded is None:
            return
        path, calibration = loaded
        calibration.label = calibration.label or "Reference"
        self._reference_calibration = calibration
        self.reference_status_label.setText(
            self._format_loaded_calibration_status("Reference", path, calibration)
        )

    def _on_load_candidate_calibration(self) -> None:
        loaded = self._load_calibration_file()
        if loaded is None:
            return
        path, calibration = loaded
        calibration.label = calibration.label or "Candidate"
        self._candidate_calibration = calibration
        self.candidate_status_label.setText(
            self._format_loaded_calibration_status("Candidate", path, calibration)
        )

    def _evaluation_mode(self) -> str:
        if not hasattr(self, "evaluation_source_combo"):
            return "auto"
        return self.evaluation_source_combo.currentData() or "auto"

    def _update_benchmark_status(self) -> None:
        if not hasattr(self, "benchmark_status_label"):
            return
        mode = self._evaluation_mode()
        internal_count = 0
        if self._validation_results:
            model = self.my_model_combo.currentData() if hasattr(self, "my_model_combo") else None
            validation = self._validation_results.get(model) if model is not None else None
            internal_count = len(validation.test_frame_ids) if validation is not None else 0

        if self._benchmark_dataset is None:
            selected = (
                "Current evaluation: Internal Hold-out"
                if mode != "independent_benchmark"
                else "Current evaluation: Independent Benchmark requested, but no benchmark images are loaded."
            )
            self.benchmark_status_label.setText(
                f"{selected}\n"
                f"Internal Hold-out: {internal_count} images\n"
                "Independent Benchmark: not provided (optional). "
                "Providing a separate benchmark dataset enables higher-confidence comparison."
            )
            return

        detected = self._benchmark_dataset.num_detected
        total = self._benchmark_dataset.num_total
        current = "Independent Benchmark [Recommended]" if mode in ("auto", "independent_benchmark") else "Internal Hold-out"
        self.benchmark_status_label.setText(
            f"Current evaluation: {current}\n"
            f"Dataset: benchmark\n"
            f"Images: {total}\n"
            f"Detected: {detected}\n"
            f"Usable paired frames: {detected}\n"
            "Evaluation confidence: HIGH if there is no calibration/benchmark overlap and enough usable frames."
        )

    def _on_select_benchmark_images(self) -> None:
        if self._pattern_config is None or self._camera_config is None:
            QMessageBox.warning(self, "패턴 설정 없음", "먼저 Camera Setup / Pattern을 설정하고 이미지를 불러오세요.")
            return
        if self._benchmark_thread is not None and self._benchmark_thread.isRunning():
            QMessageBox.information(self, "검출 진행 중", "Benchmark 이미지 검출이 이미 진행 중입니다.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Independent Benchmark 이미지 선택",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp)",
        )
        if not paths:
            return

        self._benchmark_image_paths = list(paths)
        # detect_dataset()은 이미지가 많으면(특히 수백 장) 몇 초~몇십 초가
        # 걸릴 수 있다. GUI 스레드에서 직접 기다리면 그동안 Qt 이벤트 루프가
        # 멈춰 OS가 "python3 is not responding"을 띄운다 - QThread 워커로 분리.
        worker = BenchmarkDetectionWorker(paths, self._pattern_config, self._camera_config)
        thread = run_worker_in_thread(worker, self)
        worker.progress.connect(self.benchmark_status_label.setText)
        worker.dataset_ready.connect(self._on_benchmark_dataset_ready)
        worker.error.connect(self._on_benchmark_worker_error)
        thread.finished.connect(self._on_benchmark_worker_finished)
        self._benchmark_thread, self._benchmark_worker = thread, worker
        self.select_benchmark_button.setEnabled(False)
        self.benchmark_status_label.setText(f"Benchmark 이미지 검출 중... ({len(paths)}장)")
        thread.start()

    def _on_benchmark_dataset_ready(self, dataset: Dataset) -> None:
        self._benchmark_dataset = dataset
        self._update_benchmark_status()
        QMessageBox.information(self, "Benchmark 불러오기 완료", summarize_dataset(dataset))

    def _on_benchmark_worker_error(self, message: str) -> None:
        self._benchmark_image_paths = []
        QMessageBox.critical(self, "Benchmark 불러오기 실패", message)

    def _on_benchmark_worker_finished(self) -> None:
        self.select_benchmark_button.setEnabled(True)
        self._benchmark_thread = None
        self._benchmark_worker = None

    def _on_clear_benchmark_dataset(self) -> None:
        self._benchmark_dataset = None
        self._benchmark_image_paths = []
        self._update_benchmark_status()

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
            width=self._camera_config.width if self._camera_config else None,
            height=self._camera_config.height if self._camera_config else None,
        )

    def _external_params_from_yaml(self) -> ExternalCameraParams:
        loaded = self._loaded_calibration or load_standard_calibration(self._loaded_yaml_path)
        model = self.external_model_combo.currentData()
        # 파일 값은 입력 위젯에 채워 넣되, 사용자가 그 값을 보정한 경우 화면에
        # 보이는 값이 실제 비교에도 쓰여야 한다. 이전 구현은 위젯 편집을 무시하고
        # 원본 YAML 배열을 다시 사용해 '직접 입력이 막힌' 것처럼 보였다.
        if self.distortion_edit.text().strip():
            K = np.array([
                [self.fx_spin.value(), 0.0, self.cx_spin.value()],
                [0.0, self.fy_spin.value(), self.cy_spin.value()],
                [0.0, 0.0, 1.0],
            ])
            D = _parse_distortion_text(self.distortion_edit.text())
        else:
            K = np.asarray(loaded.camera_matrix, dtype=np.float64).reshape(3, 3)
            D = np.asarray(loaded.distortion, dtype=np.float64).reshape(-1)
        return ExternalCameraParams(
            label=self.label_edit.text().strip() or loaded.label or "예전 결과",
            model_name=model,
            camera_matrix=K,
            distortion=D,
            source_note=(
                self.source_note_edit.text().strip()
                or f"{loaded.source_format}: {loaded.source_path or self._loaded_yaml_path}"
            ),
            width=loaded.width,
            height=loaded.height,
            distortion_model=loaded.distortion_model,
        )

    # ------------------------------------------------------------------
    # 비교 실행
    # ------------------------------------------------------------------

    def _prepare_comparison_inputs(self):
        if self._dataset is None or self._camera_config is None or self._pattern_config is None:
            QMessageBox.warning(self, "데이터 없음", "먼저 이미지를 불러오고 캘리브레이션을 실행하세요.")
            return None

        my_model = self.my_model_combo.currentData()
        my_validation = self._validation_results.get(my_model)
        if my_validation is None or not my_validation.success:
            QMessageBox.warning(
                self, "Hold-out 결과 없음",
                f"{_MODEL_LABELS.get(my_model, my_model)} 모델의 Hold-out Validation 결과가 없습니다. "
                "먼저 [캘리브레이션 실행]을 완료해 주세요.",
            )
            return None

        try:
            if self._loaded_yaml_path:
                external = self._external_params_from_yaml()
            else:
                external = self._external_params_from_manual_input()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "외부 파라미터 입력 오류", str(e))
            return None
        return my_model, my_validation, external

    def _on_run_comparison(self) -> None:
        """동기 비교 진입점. UI 버튼은 _on_run_comparison_async를 사용한다."""
        prepared = self._prepare_comparison_inputs()
        if prepared is None:
            return
        my_model, my_validation, external = prepared

        result = compare_with_external_params(
            self._dataset, self._camera_config, self._pattern_config,
            my_model, my_validation, external,
            use_rational_model=self._use_rational_model,
        )
        self._display_comparison_result(result)

    def _on_run_comparison_async(self) -> None:
        if self._comparison_thread is not None and self._comparison_thread.isRunning():
            QMessageBox.information(self, "비교 진행 중", "External Compare 계산이 이미 진행 중입니다.")
            return
        prepared = self._prepare_comparison_inputs()
        if prepared is None:
            return
        my_model, my_validation, external = prepared
        worker = ExternalComparisonWorker(
            self._dataset,
            self._camera_config,
            self._pattern_config,
            my_model,
            my_validation,
            external,
            self._use_rational_model,
        )
        thread = run_worker_in_thread(worker, self)
        worker.progress.connect(self.comparison_status_label.setText)
        worker.result_ready.connect(self._display_comparison_result)
        worker.error.connect(self._on_comparison_error)
        thread.finished.connect(self._on_comparison_finished)
        self._comparison_thread, self._comparison_worker = thread, worker
        self.run_button.setEnabled(False)
        self.comparison_status_label.setText("External Compare 계산 시작 중...")
        thread.start()

    def _display_comparison_result(self, result: ExternalComparisonResult) -> None:
        self._last_result = result
        self._render_result(result)
        self._populate_image_combo(result)
        self.benchmark_tabs.setCurrentIndex(0)
        overview_page = self.benchmark_tabs.widget(0)
        if isinstance(overview_page, QScrollArea):
            overview_page.verticalScrollBar().setValue(0)
        self.scroll_area.ensureWidgetVisible(self.benchmark_tabs, 0, 20)
        if result.mine.success and result.external.success:
            self.comparison_status_label.setText("비교 완료 - 아래 Overview에 결과를 표시했습니다.")
        else:
            self.comparison_status_label.setText(f"비교 실패: {result.verdict}")

    def _on_comparison_error(self, message: str) -> None:
        self.comparison_status_label.setText(message)
        QMessageBox.critical(self, "External Compare 실패", message)

    def _on_comparison_finished(self) -> None:
        self.run_button.setEnabled(True)
        self._comparison_thread = None
        self._comparison_worker = None

    def _on_run_file_pair_comparison(self) -> None:
        if self._dataset is None or self._camera_config is None or self._pattern_config is None:
            QMessageBox.warning(self, "데이터 없음", "먼저 이미지를 불러오고 캘리브레이션을 실행하세요.")
            return
        if self._reference_calibration is None or self._candidate_calibration is None:
            QMessageBox.warning(self, "파일 부족", "Reference와 Candidate calibration 파일을 모두 불러오세요.")
            return

        split_model = self.my_model_combo.currentData()
        validation = self._validation_results.get(split_model)
        benchmark_requested = self._benchmark_dataset is not None and self._evaluation_mode() in (
            "auto",
            "independent_benchmark",
        )
        if (validation is None or not validation.success) and not benchmark_requested:
            QMessageBox.warning(
                self, "Hold-out 결과 없음",
                "Reference/Candidate 파일 비교에도 동일한 validation split이 필요합니다. "
                "먼저 [캘리브레이션 실행]을 완료해 주세요.",
            )
            return
        test_frame_ids = validation.test_frame_ids if validation is not None and validation.success else []

        result = compare_reference_candidate_calibrations(
            self._dataset,
            self._camera_config,
            self._pattern_config,
            self._reference_calibration,
            self._candidate_calibration,
            test_frame_ids,
            independent_benchmark_dataset=self._benchmark_dataset,
            evaluation_mode=self._evaluation_mode(),
        )
        self._last_result = result
        self._render_result(result)
        self._populate_image_combo(result)
        self._update_benchmark_status()

    def _render_result(self, result: ExternalComparisonResult) -> None:
        self.table.setHorizontalHeaderLabels([result.external.label, result.mine.label, "Improvement", "Winner"])

        def _fmt_value(value, fmt="{:.3f}"):
            return fmt.format(value) if value is not None else "N/A"

        def _fmt_improvement(value):
            return f"{value:+.1f}%" if value is not None else "N/A"

        def _set_row(row: int, reference_val, candidate_val, improvement, winner, fmt="{:.3f}"):
            self.table.setItem(row, 0, QTableWidgetItem(_fmt_value(reference_val, fmt)))
            self.table.setItem(row, 1, QTableWidgetItem(_fmt_value(candidate_val, fmt)))
            self.table.setItem(row, 2, QTableWidgetItem(_fmt_improvement(improvement)))
            self.table.setItem(row, 3, QTableWidgetItem(winner or "N/A"))
            self._highlight_named_winner(
                self.table, row, winner, result.external.label, result.mine.label, 0, 1, 3
            )

        if result.mine.success and result.external.success:
            for row, metric in enumerate(result.metric_rows[:10]):
                _set_row(
                    row,
                    metric.reference_value,
                    metric.candidate_value,
                    metric.improvement_pct,
                    metric.winner,
                )
            frame_winner = (
                result.mine.label if result.mine_win_count > result.external_win_count
                else result.external.label if result.external_win_count > result.mine_win_count
                else "Tie"
            )
            _set_row(10, result.external_win_count, result.mine_win_count, None, frame_winner, fmt="{:.0f}")
        else:
            for row in range(self.table.rowCount()):
                for col in range(self.table.columnCount()):
                    self.table.setItem(row, col, QTableWidgetItem("-"))

        self.verdict_label.setText(result.verdict)
        decision = result.winner_decision
        quality = "OK" if decision.data_quality_ok else "INSUFFICIENT"
        source_label = (
            "Independent Benchmark"
            if result.evaluation_source == "independent_benchmark"
            else "Internal Hold-out"
        )
        self.decision_label.setText(
            f"Decision: {decision.status} "
            f"(Candidate {decision.candidate_score:.1f} / Reference {decision.reference_score:.1f}, "
            f"margin {decision.score_margin:.1f}, data quality {quality})\n"
            f"Evaluation Source: {source_label}, Confidence: {result.confidence.upper()}"
        )
        self.caveats_label.setText("\n".join(result.caveats))
        self._render_final_report_summary(result)
        self._render_final_benchmark_table(result)
        self._render_spatial_error_table(result)
        self._render_radial_error_table(result)
        self._render_selected_residual_heatmap()
        self._render_worst_case_table(result)
        self._render_error_distribution_table(result)
        self._render_benchmark_validation_table(result)
        self._render_statistical_tests_table(result)
        self._render_bootstrap_table(result)
        self._render_parameter_diff_table(result)
        self._render_parameter_diagnostics_tables(result)

    def _render_final_benchmark_table(self, result: ExternalComparisonResult) -> None:
        rows = result.final_benchmark_rows
        self.final_benchmark_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            self.final_benchmark_table.setItem(row_index, 0, QTableWidgetItem(row.metric))
            self.final_benchmark_table.setItem(row_index, 1, QTableWidgetItem(row.reference))
            self.final_benchmark_table.setItem(row_index, 2, QTableWidgetItem(row.candidate))
            self.final_benchmark_table.setItem(row_index, 3, QTableWidgetItem(row.improvement))
            self.final_benchmark_table.setItem(row_index, 4, QTableWidgetItem(row.winner))
            self._highlight_named_winner(
                self.final_benchmark_table,
                row_index,
                row.winner,
                result.external.label,
                result.mine.label,
                1,
                2,
                4,
            )

    def _render_final_report_summary(self, result: ExternalComparisonResult) -> None:
        decision = result.winner_decision
        source_label = (
            "Independent Benchmark"
            if result.evaluation_source == "independent_benchmark"
            else "Internal Hold-out"
        )
        confidence_label = result.confidence.upper()
        benchmark_line = ""
        if result.benchmark_image_count:
            benchmark_line = (
                f"\nBenchmark images: {result.benchmark_image_count}"
                f"\nUsable paired frames: {result.benchmark_usable_frames}"
                f"\nCalibration/Benchmark overlap: {result.benchmark_overlap_count}"
            )
        self.final_report_label.setText(
            f"FINAL VERDICT: {decision.status}\n"
            f"Evaluation Source: {source_label}\n"
            f"Confidence: {confidence_label}"
            f"{benchmark_line}\n"
            f"One-line diagnosis: {result.verdict or decision.status}"
        )

        def final_value(metric_name: str, attr: str) -> str:
            row = next((r for r in result.final_benchmark_rows if r.metric == metric_name), None)
            return getattr(row, attr) if row is not None else "N/A"

        def metric_summary(metric_name: str) -> str:
            row = next((r for r in result.final_benchmark_rows if r.metric == metric_name), None)
            if row is None:
                return f"{metric_name}: N/A"
            return (
                f"{metric_name}: Reference {row.reference}, Candidate {row.candidate}, "
                f"Improvement {row.improvement}, Winner {row.winner}"
            )

        stats = []
        for test in result.statistical_tests:
            p = "N/A" if test.p_value is None else ("<1e-12" if test.p_value < 1e-12 else f"{test.p_value:.3g}")
            effect = "N/A" if test.effect_size is None else f"{test.effect_size:.3g}"
            stats.append(f"{test.test_name} p={p}, effect={effect} {test.effect_size_name}".strip())
        if result.bootstrap_comparison and result.bootstrap_comparison.probability_candidate_better is not None:
            stats.append(
                "Bootstrap P(Candidate < Reference)="
                f"{result.bootstrap_comparison.probability_candidate_better * 100.0:.2f}%"
            )

        visual_bits = [
            metric_summary("Edge RMS"),
            metric_summary("Radial P95"),
            metric_summary("Straightness"),
            f"Residual heatmaps: {', '.join(result.residual_heatmaps) or 'N/A'}",
            f"Worst-case rows: {len(result.worst_case_rows)}",
        ]

        param_bits = []
        for key in ("reference", "candidate"):
            diag = result.parameter_diagnostics.get(key)
            if diag is None:
                continue
            weak = ", ".join(diag.weak_parameters) or "none"
            corr = "N/A" if diag.max_abs_correlation is None else f"{diag.max_abs_correlation:.3f}"
            cond = "N/A" if diag.condition_number is None else f"{diag.condition_number:.3g}"
            param_bits.append(
                f"{diag.side_label}: rank {diag.rank}/{diag.jacobian_cols}, condition {cond}, "
                f"max |corr| {corr}, weak {weak}"
            )

        sections = [
            ("Evaluation Source", f"{source_label} / Confidence {confidence_label}"),
            ("Performance Comparison", " / ".join([
                metric_summary("RMSE"),
                metric_summary("P95"),
                metric_summary("Frame wins"),
            ])),
            ("Statistical Evidence", "; ".join(stats) or "N/A"),
            ("Visual Evidence", "; ".join(visual_bits)),
            ("Parameter Analysis", "; ".join(param_bits) or "N/A"),
            ("FINAL VERDICT", decision.status),
            ("One-line diagnosis", result.verdict or "N/A"),
        ]

        self.final_report_evidence_table.setRowCount(len(sections))
        for row_index, (section, summary) in enumerate(sections):
            self.final_report_evidence_table.setItem(row_index, 0, QTableWidgetItem(section))
            self.final_report_evidence_table.setItem(row_index, 1, QTableWidgetItem(summary))

    def _model_analysis_summary(self) -> str:
        calibration_results = self._calibration_results or {}
        if not calibration_results:
            return "N/A"
        try:
            scores = compute_model_scores(
                calibration_results,
                self._validation_results or {},
                use_rational_model=self._use_rational_model,
            )
        except Exception:  # noqa: BLE001
            return "Model score unavailable."
        if not scores:
            return "N/A"
        recommended = next((s for s in scores if s.is_recommended), min(scores, key=lambda s: s.score))
        return (
            f"Recommended {_MODEL_LABELS.get(recommended.model_name, recommended.model_name.value)} "
            f"(score {recommended.score:.3f}, AIC "
            f"{'N/A' if recommended.aic is None else f'{recommended.aic:.3f}'}, BIC "
            f"{'N/A' if recommended.bic is None else f'{recommended.bic:.3f}'})"
        )

    def _render_model_comparison_table(self) -> None:
        calibration_results = self._calibration_results or {}
        validation_results = self._validation_results or {}
        if not calibration_results:
            self.model_comparison_table.setRowCount(0)
            return

        try:
            scores = compute_model_scores(
                calibration_results,
                validation_results,
                use_rational_model=self._use_rational_model,
            )
        except Exception:  # noqa: BLE001 - UI should still show raw model status if scoring fails.
            scores = []
        score_by_model = {s.model_name: s for s in scores}

        rows = [m for m in _MODEL_ORDER if m in calibration_results]
        self.model_comparison_table.setRowCount(len(rows))

        def _fmt_px(value) -> str:
            return "N/A" if value is None else f"{value:.3f} px"

        def _fmt_plain(value) -> str:
            return "N/A" if value is None else f"{value:.3f}"

        for row_index, model in enumerate(rows):
            cal = calibration_results.get(model)
            val = validation_results.get(model)
            score = score_by_model.get(model)
            test_p95 = None
            if val and val.test_residual_stats:
                test_p95 = val.test_residual_stats.p95
            status = "OK" if cal and cal.success and val and val.success else "Unavailable"
            if cal and not cal.success:
                status = "Calibration failed"
            elif val and not val.success:
                status = "Hold-out failed"
            recommendation = "Recommended" if score and score.is_recommended else ""
            if score and score.selection_confidence_level:
                recommendation = (
                    f"{recommendation} ({score.selection_confidence_level})"
                    if recommendation else score.selection_confidence_level
                )
            values = [
                _MODEL_LABELS.get(model, model.value),
                status,
                _fmt_px(cal.rms_error if cal else None),
                _fmt_px(val.test_rms if val else None),
                _fmt_px(test_p95),
                _fmt_plain(score.aic if score else None),
                _fmt_plain(score.bic if score else None),
                str(score.parameter_count) if score else "N/A",
                _fmt_plain(score.score if score else None),
                recommendation or "N/A",
            ]
            for col, value in enumerate(values):
                self.model_comparison_table.setItem(row_index, col, QTableWidgetItem(value))

    def _render_worst_case_table(self, result: ExternalComparisonResult) -> None:
        rows = result.worst_case_rows
        self.worst_case_table.setRowCount(len(rows))

        def _fmt(value) -> str:
            return "N/A" if value is None else f"{value:.3f} px"

        for row_index, row in enumerate(rows):
            self.worst_case_table.setItem(row_index, 0, QTableWidgetItem(row.category))
            self.worst_case_table.setItem(row_index, 1, QTableWidgetItem(row.reference_location))
            self.worst_case_table.setItem(row_index, 2, QTableWidgetItem(_fmt(row.reference_value)))
            self.worst_case_table.setItem(row_index, 3, QTableWidgetItem(row.candidate_location))
            self.worst_case_table.setItem(row_index, 4, QTableWidgetItem(_fmt(row.candidate_value)))
            winner = row.winner
            if row.improvement_pct is not None:
                winner = f"{winner} ({row.improvement_pct:+.1f}%)"
            self.worst_case_table.setItem(row_index, 5, QTableWidgetItem(winner))
            self._highlight_named_winner(
                self.worst_case_table,
                row_index,
                row.winner,
                result.external.label,
                result.mine.label,
                2,
                4,
                5,
            )

    def _render_spatial_error_table(self, result: ExternalComparisonResult) -> None:
        rows = []
        for grid_name in ("3x3", "5x5"):
            grid = result.spatial_comparisons.get(grid_name)
            if grid is None:
                continue
            for cell in grid.cells:
                rows.append((grid_name, cell))

        self.spatial_error_table.setRowCount(len(rows))

        def _fmt(value, suffix: str = "") -> str:
            return "N/A" if value is None else f"{value:.3f}{suffix}"

        for row_index, (grid_name, cell) in enumerate(rows):
            values = [
                grid_name,
                f"R{cell.row + 1} C{cell.col + 1}",
                str(cell.num_reference_points),
                str(cell.num_candidate_points),
                _fmt(cell.reference_mean, " px"),
                _fmt(cell.candidate_mean, " px"),
                _fmt(cell.improvement_mean_pct, "%"),
                _fmt(cell.reference_rmse, " px"),
                _fmt(cell.candidate_rmse, " px"),
                _fmt(cell.improvement_rmse_pct, "%"),
                _fmt(cell.reference_p95, " px"),
                _fmt(cell.candidate_p95, " px"),
                _fmt(cell.improvement_p95_pct, "%"),
            ]
            for col, value in enumerate(values):
                self.spatial_error_table.setItem(row_index, col, QTableWidgetItem(value))
            self._highlight_lower_pair(
                self.spatial_error_table, row_index, 4, 5, cell.reference_mean, cell.candidate_mean
            )
            self._highlight_lower_pair(
                self.spatial_error_table, row_index, 7, 8, cell.reference_rmse, cell.candidate_rmse
            )
            self._highlight_lower_pair(
                self.spatial_error_table, row_index, 10, 11, cell.reference_p95, cell.candidate_p95
            )

    def _render_radial_error_table(self, result: ExternalComparisonResult) -> None:
        rows = []
        for profile_name in ("quartiles", "bands"):
            profile = result.radial_comparisons.get(profile_name)
            if profile is None:
                continue
            for band in profile.bands:
                rows.append((profile_name, band))

        self.radial_error_table.setRowCount(len(rows))

        def _fmt(value, suffix: str = "") -> str:
            return "N/A" if value is None else f"{value:.3f}{suffix}"

        for row_index, (profile_name, band) in enumerate(rows):
            values = [
                profile_name,
                band.label,
                f"{band.radius_min_norm:.2f}-{band.radius_max_norm:.2f}",
                str(band.num_reference_points),
                str(band.num_candidate_points),
                _fmt(band.reference_rmse, " px"),
                _fmt(band.candidate_rmse, " px"),
                _fmt(band.improvement_rmse_pct, "%"),
                _fmt(band.reference_p95, " px"),
                _fmt(band.candidate_p95, " px"),
                _fmt(band.improvement_p95_pct, "%"),
                _fmt(band.improvement_max_pct, "%"),
            ]
            for col, value in enumerate(values):
                self.radial_error_table.setItem(row_index, col, QTableWidgetItem(value))
            self._highlight_lower_pair(
                self.radial_error_table, row_index, 5, 6, band.reference_rmse, band.candidate_rmse
            )
            self._highlight_lower_pair(
                self.radial_error_table, row_index, 8, 9, band.reference_p95, band.candidate_p95
            )

    def _render_selected_residual_heatmap(self, *_args) -> None:
        result = self._last_result
        if result is None:
            return
        key = self.heatmap_metric_combo.currentData() or "rmse_20x20"
        heatmap = result.residual_heatmaps.get(key)
        if heatmap is None:
            self.heatmap_summary_label.setText("Residual heatmap is not available.")
            for table in (self.reference_heatmap_table, self.candidate_heatmap_table, self.difference_heatmap_table):
                table.setRowCount(0)
                table.setColumnCount(0)
            return

        self.heatmap_summary_label.setText(
            f"{heatmap.metric.upper()} heatmap {heatmap.rows}x{heatmap.cols} / "
            f"Reference max {self._fmt_heatmap_value(heatmap.reference_max)} px / "
            f"Candidate max {self._fmt_heatmap_value(heatmap.candidate_max)} px / "
            f"|Difference| max {self._fmt_heatmap_value(heatmap.difference_abs_max)} px"
        )
        self._fill_heatmap_table(
            self.reference_heatmap_table,
            heatmap.rows,
            heatmap.cols,
            {(c.row, c.col): c.reference_value for c in heatmap.cells},
            max_abs=heatmap.reference_max,
            mode="single",
        )
        self._fill_heatmap_table(
            self.candidate_heatmap_table,
            heatmap.rows,
            heatmap.cols,
            {(c.row, c.col): c.candidate_value for c in heatmap.cells},
            max_abs=heatmap.candidate_max,
            mode="single",
        )
        self._fill_heatmap_table(
            self.difference_heatmap_table,
            heatmap.rows,
            heatmap.cols,
            {(c.row, c.col): c.difference_value for c in heatmap.cells},
            max_abs=heatmap.difference_abs_max,
            mode="difference",
        )

    @staticmethod
    def _fmt_heatmap_value(value) -> str:
        return "N/A" if value is None else f"{value:.3f}"

    def _fill_heatmap_table(
        self,
        table: QTableWidget,
        rows: int,
        cols: int,
        values: dict[tuple[int, int], float | None],
        *,
        max_abs: float | None,
        mode: str,
    ) -> None:
        table.setRowCount(rows)
        table.setColumnCount(cols)
        scale = float(max_abs) if max_abs and max_abs > 0 else 1.0
        for r in range(rows):
            table.setRowHeight(r, 16)
            for c in range(cols):
                table.setColumnWidth(c, 22)
                value = values.get((r, c))
                text = "" if value is None else f"{value:.1f}"
                item = QTableWidgetItem(text)
                item.setToolTip("N/A" if value is None else f"{value:.4f} px")
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(self._heatmap_color(value, scale, mode))
                table.setItem(r, c, item)

    @staticmethod
    def _heatmap_color(value: float | None, scale: float, mode: str) -> QColor:
        if value is None:
            return QColor(Theme.BG_SECONDARY)
        if mode == "difference":
            ratio = min(abs(float(value)) / scale, 1.0)
            if value > 0:
                return QColor(55 + int(70 * ratio), 27, 27)
            if value < 0:
                return QColor(29, 48 + int(38 * ratio), 18)
            return QColor(Theme.BG_TERTIARY)
        ratio = min(max(float(value) / scale, 0.0), 1.0)
        low = QColor(Theme.COVERAGE_HIGH)
        high = QColor(Theme.HEATMAP_HIGH)
        return QColor(
            int(low.red() + (high.red() - low.red()) * ratio),
            int(low.green() + (high.green() - low.green()) * ratio),
            int(low.blue() + (high.blue() - low.blue()) * ratio),
        )

    def _render_error_distribution_table(self, result: ExternalComparisonResult) -> None:
        distribution = result.error_distribution
        if distribution is None:
            self.error_distribution_summary_label.setText("Error distribution is not available.")
            self.error_distribution_table.setRowCount(0)
            return

        def _fmt(value) -> str:
            return "N/A" if value is None else f"{value:.3f}"

        self.error_distribution_summary_label.setText(
            "Points: "
            f"Reference {distribution.num_reference_points}, Candidate {distribution.num_candidate_points} / "
            f"Median {_fmt(distribution.reference_median)} vs {_fmt(distribution.candidate_median)} px / "
            f"P95 {_fmt(distribution.reference_p95)} vs {_fmt(distribution.candidate_p95)} px"
        )
        self.error_distribution_table.setRowCount(len(distribution.bins))
        for row_index, row in enumerate(distribution.bins):
            self.error_distribution_table.setItem(
                row_index, 0, QTableWidgetItem(f"{row.bin_start:.3f}-{row.bin_end:.3f}")
            )
            self.error_distribution_table.setItem(row_index, 1, QTableWidgetItem(str(row.reference_count)))
            self.error_distribution_table.setItem(row_index, 2, QTableWidgetItem(str(row.candidate_count)))
            self.error_distribution_table.setItem(row_index, 3, QTableWidgetItem(f"{row.reference_density:.3f}"))
            self.error_distribution_table.setItem(row_index, 4, QTableWidgetItem(f"{row.candidate_density:.3f}"))
            self.error_distribution_table.setItem(row_index, 5, QTableWidgetItem(f"{row.reference_cdf:.3f}"))
            self.error_distribution_table.setItem(row_index, 6, QTableWidgetItem(f"{row.candidate_cdf:.3f}"))

    def _render_benchmark_validation_table(self, result: ExternalComparisonResult) -> None:
        rows = result.benchmark_validation_rows
        self.benchmark_validation_table.setRowCount(len(rows))

        def _fmt(value, suffix: str = "") -> str:
            if value is None:
                return "N/A"
            return f"{value:.3f}{suffix}"

        for row_index, row in enumerate(rows):
            self.benchmark_validation_table.setItem(row_index, 0, QTableWidgetItem(row.name))
            self.benchmark_validation_table.setItem(row_index, 1, QTableWidgetItem(str(row.num_splits)))
            self.benchmark_validation_table.setItem(
                row_index, 2, QTableWidgetItem(_fmt(row.reference_train_rms_mean))
            )
            self.benchmark_validation_table.setItem(
                row_index,
                3,
                QTableWidgetItem(
                    f"{_fmt(row.reference_validation_rms_mean)} ± {_fmt(row.reference_validation_rms_std)}"
                ),
            )
            self.benchmark_validation_table.setItem(
                row_index, 4, QTableWidgetItem(_fmt(row.reference_train_validation_gap))
            )
            self.benchmark_validation_table.setItem(
                row_index, 5, QTableWidgetItem(_fmt(row.candidate_train_rms_mean))
            )
            self.benchmark_validation_table.setItem(
                row_index,
                6,
                QTableWidgetItem(
                    f"{_fmt(row.candidate_validation_rms_mean)} ± {_fmt(row.candidate_validation_rms_std)}"
                ),
            )
            self.benchmark_validation_table.setItem(
                row_index, 7, QTableWidgetItem(_fmt(row.candidate_train_validation_gap))
            )
            improvement = "N/A" if row.improvement_pct is None else f"{row.improvement_pct:+.1f}%"
            self.benchmark_validation_table.setItem(row_index, 8, QTableWidgetItem(improvement))
            self._highlight_lower_pair(
                self.benchmark_validation_table,
                row_index,
                3,
                6,
                row.reference_validation_rms_mean,
                row.candidate_validation_rms_mean,
            )

    def _render_statistical_tests_table(self, result: ExternalComparisonResult) -> None:
        rows = result.statistical_tests
        self.statistical_tests_table.setRowCount(len(rows))

        def _fmt(value) -> str:
            return "N/A" if value is None else f"{value:.6g}"

        def _fmt_p(value) -> str:
            if value is None:
                return "N/A"
            return "<1e-12" if value < 1e-12 else f"{value:.6g}"

        for row_index, row in enumerate(rows):
            effect = _fmt(row.effect_size)
            if row.effect_size_name and row.effect_size is not None:
                effect = f"{effect} ({row.effect_size_name})"
            self.statistical_tests_table.setItem(row_index, 0, QTableWidgetItem(row.test_name))
            self.statistical_tests_table.setItem(row_index, 1, QTableWidgetItem(str(row.n_pairs)))
            self.statistical_tests_table.setItem(row_index, 2, QTableWidgetItem(_fmt(row.statistic)))
            self.statistical_tests_table.setItem(row_index, 3, QTableWidgetItem(_fmt_p(row.p_value)))
            self.statistical_tests_table.setItem(row_index, 4, QTableWidgetItem(effect))
            self.statistical_tests_table.setItem(row_index, 5, QTableWidgetItem(_fmt(row.mean_diff)))
            self.statistical_tests_table.setItem(row_index, 6, QTableWidgetItem(_fmt(row.median_diff)))
            self.statistical_tests_table.setItem(row_index, 7, QTableWidgetItem(row.interpretation))

    def _render_bootstrap_table(self, result: ExternalComparisonResult) -> None:
        bootstrap = result.bootstrap_comparison
        if bootstrap is None:
            self.bootstrap_table.setRowCount(0)
            return

        def _fmt(value, suffix: str = "") -> str:
            return "N/A" if value is None else f"{value:.6g}{suffix}"

        def _fmt_pct(value) -> str:
            return "N/A" if value is None else f"{value * 100.0:.2f}%"

        def _fmt_ci(low, high, suffix: str = "") -> str:
            if low is None or high is None:
                return "N/A"
            return f"[{low:.6g}, {high:.6g}]{suffix}"

        ci_label = f"{bootstrap.confidence_level * 100:.0f}%"
        rows = [
            ("Pairs", str(bootstrap.n_pairs)),
            ("Bootstrap samples", str(bootstrap.n_bootstrap)),
            ("P(Candidate Error < Reference Error)", _fmt_pct(bootstrap.probability_candidate_better)),
            ("Reference RMSE", _fmt(bootstrap.reference_rmse, " px")),
            (f"Reference RMSE {ci_label} CI", _fmt_ci(bootstrap.reference_rmse_ci_low, bootstrap.reference_rmse_ci_high, " px")),
            ("Candidate RMSE", _fmt(bootstrap.candidate_rmse, " px")),
            (f"Candidate RMSE {ci_label} CI", _fmt_ci(bootstrap.candidate_rmse_ci_low, bootstrap.candidate_rmse_ci_high, " px")),
            ("Improvement", _fmt(bootstrap.improvement_pct, "%")),
            (f"Improvement {ci_label} CI", _fmt_ci(bootstrap.improvement_ci_low, bootstrap.improvement_ci_high, "%")),
        ]
        self.bootstrap_table.setRowCount(len(rows))
        for row_index, (name, value) in enumerate(rows):
            self.bootstrap_table.setItem(row_index, 0, QTableWidgetItem(name))
            self.bootstrap_table.setItem(row_index, 1, QTableWidgetItem(value))

    def _render_parameter_diff_table(self, result: ExternalComparisonResult) -> None:
        rows = list(result.parameter_diff_rows)
        if result.fov_diff_rows:
            rows.append(None)
            rows.extend(result.fov_diff_rows)

        self.parameter_table.setRowCount(len(rows))
        self.parameter_table.setHorizontalHeaderLabels(
            [f"Parameter", result.external.label, result.mine.label, "Abs Diff", "Rel Diff"]
        )

        def _fmt_value(value, unit: str = "") -> str:
            if value is None:
                return "N/A"
            suffix = f" {unit}" if unit else ""
            return f"{value:.6g}{suffix}"

        def _fmt_rel(value) -> str:
            return f"{value:.3f}%" if value is not None else "N/A"

        for row_index, row in enumerate(rows):
            if row is None:
                self.parameter_table.setItem(row_index, 0, QTableWidgetItem("FOV"))
                for col in range(1, self.parameter_table.columnCount()):
                    self.parameter_table.setItem(row_index, col, QTableWidgetItem(""))
                continue
            self.parameter_table.setItem(row_index, 0, QTableWidgetItem(row.name))
            self.parameter_table.setItem(row_index, 1, QTableWidgetItem(_fmt_value(row.reference_value, row.unit)))
            self.parameter_table.setItem(row_index, 2, QTableWidgetItem(_fmt_value(row.candidate_value, row.unit)))
            self.parameter_table.setItem(row_index, 3, QTableWidgetItem(_fmt_value(row.absolute_diff, row.unit)))
            self.parameter_table.setItem(row_index, 4, QTableWidgetItem(_fmt_rel(row.relative_diff_pct)))

    def _render_parameter_diagnostics_tables(self, result: ExternalComparisonResult) -> None:
        diagnostics = result.parameter_diagnostics or {}

        def _fmt(value) -> str:
            return "N/A" if value is None else f"{value:.6g}"

        stability_rows = []
        sensitivity_rows = []
        covariance_rows = []
        observability_rows = []
        singular_rows = []
        correlation_rows = []
        for key in ("reference", "candidate"):
            diag = diagnostics.get(key)
            if diag is None:
                continue
            observability_rows.append(diag)
            for i, value in enumerate(diag.singular_values):
                singular_rows.append((diag.side_label, i + 1, value))
            for row in diag.stability_rows:
                stability_rows.append((diag.side_label, row))
            for row in diag.sensitivity_rows:
                sensitivity_rows.append((diag.side_label, row))
            labels = diag.parameter_labels
            for r, values in enumerate(diag.covariance_matrix):
                for c, value in enumerate(values):
                    covariance_rows.append((
                        diag.side_label,
                        labels[r] if r < len(labels) else str(r),
                        labels[c] if c < len(labels) else str(c),
                        value,
                    ))
            for r, values in enumerate(diag.correlation_matrix):
                for c, value in enumerate(values):
                    correlation_rows.append((
                        diag.side_label,
                        labels[r] if r < len(labels) else str(r),
                        labels[c] if c < len(labels) else str(c),
                        value,
                    ))

        self.parameter_observability_table.setRowCount(len(observability_rows))
        for row_index, diag in enumerate(observability_rows):
            top_corr = ", ".join(
                f"{a}-{b}:{value:.3f}" for a, b, value in diag.top_correlations[:3]
            ) or "N/A"
            weak = ", ".join(diag.weak_parameters) or "N/A"
            warnings = "; ".join(diag.warnings) or "N/A"
            values = [
                diag.side_label,
                f"{diag.jacobian_rows}x{diag.jacobian_cols}",
                "N/A" if diag.rank is None else f"{diag.rank}/{diag.jacobian_cols}",
                _fmt(diag.condition_number),
                _fmt(diag.min_singular_value),
                _fmt(diag.max_singular_value),
                _fmt(diag.max_abs_correlation),
                weak,
                top_corr,
                warnings,
            ]
            for col, value in enumerate(values):
                self.parameter_observability_table.setItem(row_index, col, QTableWidgetItem(value))

        self.parameter_singular_table.setRowCount(len(singular_rows))
        for row_index, (side_label, index, value) in enumerate(singular_rows):
            self.parameter_singular_table.setItem(row_index, 0, QTableWidgetItem(side_label))
            self.parameter_singular_table.setItem(row_index, 1, QTableWidgetItem(str(index)))
            self.parameter_singular_table.setItem(row_index, 2, QTableWidgetItem(_fmt(value)))

        self.parameter_stability_table.setRowCount(len(stability_rows))
        for row_index, (side_label, row) in enumerate(stability_rows):
            self.parameter_stability_table.setItem(row_index, 0, QTableWidgetItem(side_label))
            self.parameter_stability_table.setItem(row_index, 1, QTableWidgetItem(row.parameter))
            self.parameter_stability_table.setItem(row_index, 2, QTableWidgetItem(_fmt(row.value)))
            self.parameter_stability_table.setItem(row_index, 3, QTableWidgetItem(_fmt(row.std)))
            self.parameter_stability_table.setItem(row_index, 4, QTableWidgetItem(_fmt(row.ci_low)))
            self.parameter_stability_table.setItem(row_index, 5, QTableWidgetItem(_fmt(row.ci_high)))
            stability = "N/A" if row.stability_score is None else f"{row.stability_score:.1f}/100"
            self.parameter_stability_table.setItem(row_index, 6, QTableWidgetItem(stability))

        self.parameter_sensitivity_table.setRowCount(len(sensitivity_rows))
        for row_index, (side_label, row) in enumerate(sensitivity_rows):
            self.parameter_sensitivity_table.setItem(row_index, 0, QTableWidgetItem(side_label))
            self.parameter_sensitivity_table.setItem(row_index, 1, QTableWidgetItem(row.parameter))
            self.parameter_sensitivity_table.setItem(row_index, 2, QTableWidgetItem(_fmt(row.value)))
            self.parameter_sensitivity_table.setItem(row_index, 3, QTableWidgetItem(_fmt(row.perturbation)))
            self.parameter_sensitivity_table.setItem(row_index, 4, QTableWidgetItem(_fmt(row.rmse_delta)))
            self.parameter_sensitivity_table.setItem(row_index, 5, QTableWidgetItem(_fmt(row.sensitivity_per_unit)))

        self.parameter_covariance_table.setRowCount(len(covariance_rows))
        for row_index, (side_label, row_label, col_label, value) in enumerate(covariance_rows):
            self.parameter_covariance_table.setItem(row_index, 0, QTableWidgetItem(side_label))
            self.parameter_covariance_table.setItem(row_index, 1, QTableWidgetItem(row_label))
            self.parameter_covariance_table.setItem(row_index, 2, QTableWidgetItem(col_label))
            self.parameter_covariance_table.setItem(row_index, 3, QTableWidgetItem(_fmt(value)))

        self.parameter_correlation_table.setRowCount(len(correlation_rows))
        for row_index, (side_label, row_label, col_label, value) in enumerate(correlation_rows):
            self.parameter_correlation_table.setItem(row_index, 0, QTableWidgetItem(side_label))
            self.parameter_correlation_table.setItem(row_index, 1, QTableWidgetItem(row_label))
            self.parameter_correlation_table.setItem(row_index, 2, QTableWidgetItem(col_label))
            self.parameter_correlation_table.setItem(row_index, 3, QTableWidgetItem(_fmt(value)))

    def _populate_image_combo(self, result: ExternalComparisonResult) -> None:
        self.image_combo.clear()
        if not (result.mine.success and result.external.success):
            return
        common_ids = sorted(set(result.mine.per_frame_error) & set(result.external.per_frame_error))
        for fid in common_ids:
            candidate_error = result.mine.per_frame_error[fid]
            reference_error = result.external.per_frame_error[fid]
            self.image_combo.addItem(
                f"{fid}  (Reference {reference_error:.2f}px / Candidate {candidate_error:.2f}px)",
                userData=fid,
            )
        if common_ids:
            self._update_visual()

    # ------------------------------------------------------------------
    # 이미지 비교 - Original / Reference / Candidate / Overlay / Difference
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
            message = f"이미지를 읽을 수 없습니다: {frame.image_info.path}"
            for label in (
                self.original_caption_label,
                self.reference_caption_label,
                self.candidate_caption_label,
                self.overlay_caption_label,
                self.difference_caption_label,
            ):
                label.setText(message)
            return

        reference = self._undistort_side(img, result.external, self.reference_caption_label)
        candidate = self._undistort_side(img, result.mine, self.candidate_caption_label)
        if reference is None or candidate is None:
            return

        reference, candidate = self._align_visual_pair(reference, candidate)
        overlay = cv2.addWeighted(reference, 0.5, candidate, 0.5, 0.0)
        difference, mean_diff, max_diff = self._build_difference_view(reference, candidate)

        frame_id = frame.image_info.image_id
        self.original_image_label.setPixmap(_cv_to_qpixmap(img))
        self.reference_image_label.setPixmap(_cv_to_qpixmap(reference))
        self.candidate_image_label.setPixmap(_cv_to_qpixmap(candidate))
        self.overlay_image_label.setPixmap(_cv_to_qpixmap(overlay))
        self.difference_image_label.setPixmap(_cv_to_qpixmap(difference))

        ref_error = result.external.per_frame_error.get(frame_id)
        cand_error = result.mine.per_frame_error.get(frame_id)
        self.original_caption_label.setText(f"원본 프레임: {frame_id}")
        self.reference_caption_label.setText(
            self._side_caption(result.external, ref_error, prefix="Reference")
        )
        self.candidate_caption_label.setText(
            self._side_caption(result.mine, cand_error, prefix="Candidate")
        )
        self.overlay_caption_label.setText("Reference/Candidate undistorted image 50:50 overlay")
        self.difference_caption_label.setText(
            f"절대 차분 heatmap - 평균 {mean_diff:.2f}, 최대 {max_diff:.1f} intensity"
        )

    def _undistort_side(
        self, img: np.ndarray, side: ComparisonSide, caption_label: QLabel,
    ) -> np.ndarray | None:
        from calibration.models.common import undistort_image
        from calibration.types import CalibrationResult

        fake_result = CalibrationResult(
            model_name=side.model_name, camera_matrix=side.camera_matrix,
            distortion=side.distortion, success=True,
        )
        try:
            return undistort_image(img, fake_result, self._camera_config)
        except ValueError as e:
            caption_label.setText(str(e))
            return None

    def _align_visual_pair(self, reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if reference.shape[:2] == candidate.shape[:2]:
            return reference, candidate
        h, w = reference.shape[:2]
        return reference, cv2.resize(candidate, (w, h), interpolation=cv2.INTER_LINEAR)

    def _build_difference_view(
        self, reference: np.ndarray, candidate: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        diff = cv2.absdiff(reference, candidate)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        max_diff = float(np.max(gray)) if gray.size else 0.0
        mean_diff = float(np.mean(gray)) if gray.size else 0.0
        if max_diff > 0:
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        else:
            normalized = np.zeros_like(gray)
        heatmap_code = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        heatmap = cv2.applyColorMap(normalized.astype(np.uint8), heatmap_code)
        return heatmap, mean_diff, max_diff

    def _side_caption(self, side: ComparisonSide, per_frame_error: float | None, prefix: str) -> str:
        if per_frame_error is None:
            return f"{prefix}: {side.label}"
        return f"{prefix}: {side.label} - 이 프레임 재투영 오차 {per_frame_error:.3f}px"
