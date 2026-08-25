"""
camera_calibrator.ui.model_refitting_view
=========================================

UI for approximating an 8-coefficient OpenCV rational pinhole calibration as a
standard 5-coefficient pinhole model.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLayout,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.model_refitting import ModelRefitResult, RefitErrorStats
from calibration.types import CalibrationResult, CameraConfig, CameraModelType
from ui.theme import set_tone


def _fmt(value: float | None, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else "N/A"


def _matrix_text(arr: np.ndarray | None) -> str:
    if arr is None:
        return "N/A"
    return np.array2string(np.asarray(arr, dtype=np.float64), precision=8, suppress_small=False)


class ModelRefittingView(QWidget):
    refit_requested = Signal(object)  # dict options

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._calibration_results: dict[CameraModelType, CalibrationResult] = {}
        self._camera_config: CameraConfig | None = None
        self._last_result: ModelRefitResult | None = None

        # 이 탭은 표와 파라미터 편집기를 함께 보여 주므로 필요한 세로 길이가
        # 비교적 크다. 최상위 레이아웃에 모두 직접 넣으면 창이 낮아졌을 때
        # Qt가 자식 위젯을 최소 높이 아래로 압축해 내용이 겹쳐 보일 수 있다.
        # 내용의 자연스러운 높이는 유지하고, 부족한 공간은 탭 자체의 세로
        # 스크롤로 처리한다.
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.NoFrame)
        outer_layout.addWidget(self.scroll_area)

        content = QWidget()
        self.scroll_area.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setSizeConstraint(QLayout.SetMinimumSize)

        intro = QLabel(
            "Rational/Extended Pinhole 8계수 모델을 reference로 삼아, 이미지 전체 샘플에서 "
            "projection 차이가 최소가 되도록 OpenCV 5계수 Pinhole 모델을 최적화합니다. "
            "D8[:5] 절단이 아니라 model approximation입니다."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        settings = QGroupBox("Refitting 설정")
        form = QFormLayout(settings)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Full Intrinsic Refitting (fx, fy, cx, cy + D5)", userData="full")
        self.mode_combo.addItem("Distortion Only (K 고정 + D5)", userData="distortion_only")
        form.addRow("Mode", self.mode_combo)

        self.grid_x_spin = QSpinBox()
        self.grid_x_spin.setRange(10, 200)
        self.grid_x_spin.setValue(80)
        self.grid_y_spin = QSpinBox()
        self.grid_y_spin.setRange(10, 160)
        self.grid_y_spin.setValue(50)
        grid_box = QWidget()
        grid_layout = QGridLayout(grid_box)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.addWidget(self.grid_x_spin, 0, 0)
        grid_layout.addWidget(QLabel("x"), 0, 1)
        grid_layout.addWidget(self.grid_y_spin, 0, 2)
        form.addRow("Grid", grid_box)

        self.edge_weight_check = QCheckBox("Edge weighting 사용")
        form.addRow("", self.edge_weight_check)

        self.loss_combo = QComboBox()
        for loss in ("linear", "soft_l1", "huber"):
            self.loss_combo.addItem(loss, userData=loss)
        form.addRow("Loss", self.loss_combo)

        self.run_button = QPushButton("Model Refitting 실행")
        self.run_button.setProperty("role", "primary")
        self.run_button.clicked.connect(self._emit_refit_requested)
        form.addRow("", self.run_button)
        layout.addWidget(settings)

        self.status_label = QLabel("Rational Extended Pinhole 결과가 있으면 refitting을 실행할 수 있습니다.")
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("tone", "muted")
        layout.addWidget(self.status_label)

        self.summary_table = QTableWidget(2, 6)
        self.summary_table.setHorizontalHeaderLabels(["Method", "RMSE", "P95", "P99", "Max", "Edge RMSE"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.summary_table)

        self.region_table = QTableWidget(3, 7)
        self.region_table.setHorizontalHeaderLabels(["Region", "RMSE", "Mean", "Median", "P95", "P99", "Max"])
        self.region_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.region_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.region_table)

        params = QGroupBox("Refitted Parameters")
        params_layout = QGridLayout(params)
        self.k_text = QPlainTextEdit()
        self.k_text.setReadOnly(True)
        self.d_text = QPlainTextEdit()
        self.d_text.setReadOnly(True)
        params_layout.addWidget(QLabel("K5"), 0, 0)
        params_layout.addWidget(QLabel("D5 [k1, k2, p1, p2, k3]"), 0, 1)
        params_layout.addWidget(self.k_text, 1, 0)
        params_layout.addWidget(self.d_text, 1, 1)
        layout.addWidget(params)

        layout.addStretch(1)
        self._clear_tables()

    def set_context(
        self,
        calibration_results: dict[CameraModelType, CalibrationResult],
        camera_config: CameraConfig | None,
    ) -> None:
        self._calibration_results = calibration_results
        self._camera_config = camera_config
        result = calibration_results.get(CameraModelType.EXTENDED_PINHOLE)
        usable = bool(
            result
            and result.success
            and result.camera_matrix is not None
            and result.distortion is not None
            and np.asarray(result.distortion).size >= 8
            and camera_config is not None
        )
        self.run_button.setEnabled(usable)
        if usable:
            self.status_label.setText("8계수 Extended/Rational 결과를 찾았습니다. 5계수 근사를 실행할 수 있습니다.")
            set_tone(self.status_label, "good")
        else:
            self.status_label.setText(
                "Rational model 사용(k4~k6 포함)으로 Extended Pinhole을 계산해야 refitting을 실행할 수 있습니다."
            )
            set_tone(self.status_label, "warning")

    def _emit_refit_requested(self) -> None:
        self.refit_requested.emit({
            "mode": self.mode_combo.currentData(),
            "grid_size": (self.grid_x_spin.value(), self.grid_y_spin.value()),
            "edge_weighting": self.edge_weight_check.isChecked(),
            "loss": self.loss_combo.currentData(),
        })

    def set_running(self) -> None:
        self.run_button.setEnabled(False)
        self.status_label.setText("Model Refitting 계산 중...")
        set_tone(self.status_label, "muted")

    def set_error(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.status_label.setText(f"Model Refitting 실패: {message}")
        set_tone(self.status_label, "bad")

    def set_result(self, result: ModelRefitResult) -> None:
        self._last_result = result
        self.run_button.setEnabled(True)
        improvement = result.improvement_rmse_pct
        suffix = f" · Naive 대비 RMSE {improvement:.1f}% 개선" if improvement is not None else ""
        self.status_label.setText(
            f"완료: success={result.optimization.success}, iterations={result.optimization.iterations}, "
            f"cost={result.optimization.cost:.6g}{suffix}"
        )
        set_tone(self.status_label, "good" if result.optimization.success else "warning")
        self._fill_summary(result)
        self._fill_regions(result)
        self.k_text.setPlainText(_matrix_text(result.K_refitted))
        self.d_text.setPlainText(_matrix_text(result.D_refitted.reshape(-1)))

    def _clear_tables(self) -> None:
        for row, name in enumerate(("Naive truncation", "Optimized refitting")):
            self.summary_table.setItem(row, 0, QTableWidgetItem(name))
            for col in range(1, 6):
                self.summary_table.setItem(row, col, QTableWidgetItem("N/A"))
        for row, region in enumerate(("center", "middle", "edge")):
            self.region_table.setItem(row, 0, QTableWidgetItem(region))
            for col in range(1, 7):
                self.region_table.setItem(row, col, QTableWidgetItem("N/A"))
        self.k_text.setPlainText("")
        self.d_text.setPlainText("")

    def _fill_summary(self, result: ModelRefitResult) -> None:
        rows = [
            ("Naive truncation", result.naive_error, result.naive_region_error.get("edge")),
            ("Optimized refitting", result.error, result.region_error.get("edge")),
        ]
        for row, (name, stats, edge) in enumerate(rows):
            values = [
                name,
                _fmt(stats.rmse_px),
                _fmt(stats.p95_px),
                _fmt(stats.p99_px),
                _fmt(stats.max_px),
                _fmt(edge.rmse_px if edge else None),
            ]
            for col, value in enumerate(values):
                self.summary_table.setItem(row, col, QTableWidgetItem(value))

    def _fill_regions(self, result: ModelRefitResult) -> None:
        for row, region in enumerate(("center", "middle", "edge")):
            stats = result.region_error.get(region, RefitErrorStats())
            values = [
                region,
                _fmt(stats.rmse_px),
                _fmt(stats.mean_px),
                _fmt(stats.median_px),
                _fmt(stats.p95_px),
                _fmt(stats.p99_px),
                _fmt(stats.max_px),
            ]
            for col, value in enumerate(values):
                self.region_table.setItem(row, col, QTableWidgetItem(value))
