"""
camera_calibrator.ui.windshield_workspace
==============================================

Windshield Refraction Calibration 전용 Workspace(사용자 스펙 4/5/24번).

Camera Intrinsic Calibration(ui/intrinsic_workspace.py, ui/main_window.py)과
완전히 분리된 화면이다 - Base Camera Model(K,D)은 여기서 절대 다시 계산하지
않고, 이미 확정된 값을 "고정"으로 불러와 표시만 한다("🔒 Base K,D fixed").

ui/intrinsic_workspace.py::IntrinsicWorkspace(얇은 shell, 실제 상태는
main_window.py의 레거시 상태 머신에 있음)와 달리, 이 Workspace는 자기 상태와
오케스트레이션 로직을 전부 이 클래스 안에 둔다 - 82KB짜리 main_window.py를
더 키우지 않기 위한 의도적인 선택이다.

지금은 Phase 1(Baseline)만 실제로 계산할 수 있다. Spherical/Residual Ray/
Spline은 라디오 버튼은 보이되 비활성화("Coming soon")로 표시한다(사용자 스펙
11/21번 - Neural Network/고급 모델은 이번 라운드에 구현하지 않음).

사용자 스펙 5/6번 UI 목업의 6단계(Base Camera/Dataset/Baseline/Windshield
Model/Validation/Comparison)를 4개 탭으로 압축했다 - Baseline 결과 표시와
Windshield Model 선택은 사실 "같은 계산의 두 측면"이라 별도 탭으로 나누면
빈 화면 전환만 늘어난다고 판단했고, Validation(Train/Test)도 같은 결과 화면의
컬럼 두 개(Train/Test)로 이미 나란히 보여준다. 기능은 전부 존재하되 탭 개수만
줄인 조직화 상의 단순화다.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.detector import detect_dataset
from calibration.library import list_cameras, list_runs, load_run_project
from calibration.models.common import regional_edge_average
from calibration.project_io import load_project
from calibration.types import (
    CalibrationProject,
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    PatternConfig,
)
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldConfig, WindshieldModelType
from calibration.windshield.validation import run_windshield_calibration
from export.opencv import (
    detect_model_hint_from_opencv_yaml,
    load_camera_matrix_and_distortion_from_opencv_yaml,
)
from export.windshield import export_windshield_yaml
from ui.radial_profile_view import RadialProfileChartWidget
from ui.theme import Theme
from ui.windshield_vector_field_view import VectorFieldChartWidget

_MODEL_LABELS = {
    CameraModelType.PINHOLE: "Ideal Pinhole",
    CameraModelType.BROWN_CONRADY: "Brown-Conrady",
    CameraModelType.EXTENDED_PINHOLE: "Rational",
    CameraModelType.FISHEYE: "Fisheye",
}

_WINDSHIELD_MODEL_LABELS = {
    WindshieldModelType.BASELINE: "Baseline",
    WindshieldModelType.SPHERICAL: "Spherical",
    WindshieldModelType.RESIDUAL_RAY: "Residual Ray",
    WindshieldModelType.SPLINE: "Spline [Advanced]",
}

_STATS_ROWS = [
    ("RMS", "rmse"),
    ("Median", "median"),
    ("P95", "p95"),
    ("P99", "p99"),
    ("Max", "max"),
]

_REGIONAL_ROWS = ["center", "left", "right", "top", "bottom", "corner"]

# spinbox의 "설정 안 함" sentinel - 사용자가 값을 만지지 않으면 config에
# 아무것도 쓰지 않는다(Baseline 등 굴절률/sphere 개념이 없는 모델을 실행할
# 때 의미 없는 값이 끼어들지 않게).
_UNSET_SPINBOX_VALUE = 0.0


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"


def _fmt_deg(v) -> str:
    return f"{v:.3f}°" if isinstance(v, (int, float)) else "N/A"


class _ScrollTable(QTableWidget):
    """마우스 휠을 항상 페이지 스크롤로 넘기는 QTableWidget.
    ui/result_view.py::_PageScrollTableWidget과 동일한 패턴(표 자체가
    스크롤하지 않고 곧바로 부모 페이지 스크롤로 넘어가게 함)."""

    def wheelEvent(self, event) -> None:
        event.ignore()


def _fit_table_to_rows(table: QTableWidget) -> None:
    """모든 행을 펼쳐 페이지 스크롤만 쓰도록 table 높이를 맞춘다.
    ui/result_view.py::_fit_table_to_rows와 동일한 패턴."""
    table.resizeRowsToContents()
    header_height = table.horizontalHeader().sizeHint().height()
    rows_height = sum(table.rowHeight(row) for row in range(table.rowCount()))
    table.setFixedHeight(header_height + rows_height + table.frameWidth() * 2 + 4)


class WindshieldWorkspace(QWidget):
    back_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._camera_config: CameraConfig | None = None
        self._pattern_config: PatternConfig | None = None
        self._windshield_dataset: Dataset | None = None
        self._windshield_config: WindshieldConfig | None = None
        self._windshield_results: dict[WindshieldModelType, WindshieldCalibrationResult] = {}
        # 마지막으로 화면에 표시된(=Export 대상) 모델 - export_button과
        # _on_export_windshield_yaml이 특정 모델(예: Baseline)에 고정되지
        # 않고 "방금 실행/표시한 결과"를 export하도록 추적한다.
        self._current_displayed_model: WindshieldModelType | None = None

        # MainWindow가 load_base_from_calibration_results()로 넘겨주는,
        # 현재 세션에서 이미 계산된 Standard 4모델 결과 (Base Camera 탭의
        # "Load from current session" 버튼이 여기서 고른다).
        self._session_calibration_results: dict[CameraModelType, CalibrationResult] = {}
        self._session_camera_config: CameraConfig | None = None
        self._session_pattern_config: PatternConfig | None = None

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        home_button = QPushButton("← Calibration Home")
        home_button.clicked.connect(self.back_requested.emit)
        header.addWidget(home_button)
        header.addStretch(1)
        layout.addLayout(header)

        title = QLabel("WINDSHIELD REFRACTION CALIBRATION")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(title)
        subtitle = QLabel(
            "앞유리 굴절로 생기는 기하학적(geometric) 픽셀 변위만 측정/보정합니다. "
            "Reflection(글레어/고스트)은 다루지 않습니다."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color: {Theme.TEXT_SECONDARY};")
        layout.addWidget(subtitle)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_base_camera_tab(), "① Base Camera")
        self.tabs.addTab(self._build_dataset_tab(), "② Dataset")
        self.tabs.addTab(self._build_model_tab(), "③ Windshield Model")
        self.tabs.addTab(self._build_comparison_tab(), "④ Comparison")
        layout.addWidget(self.tabs, stretch=1)

    # ------------------------------------------------------------------
    # MainWindow 연동 API
    # ------------------------------------------------------------------
    def load_base_from_calibration_results(
        self,
        calibration_results: dict[CameraModelType, CalibrationResult],
        camera_config: CameraConfig | None,
        pattern_config: PatternConfig | None,
    ) -> None:
        """MainWindow가 Home -> Windshield Refraction 진입 시 호출.
        현재 세션에서 이미 계산된 결과를 "Load from current session" 버튼으로
        바로 쓸 수 있게 후보로만 등록한다 - 여기서 자동으로 Base를 확정하지는
        않는다(사용자가 명시적으로 모델을 선택해야 함)."""
        self._session_calibration_results = calibration_results or {}
        self._session_camera_config = camera_config
        self._session_pattern_config = pattern_config

    def import_state(self, project: CalibrationProject) -> None:
        """프로젝트 로드 시 Windshield 상태를 복원한다."""
        self._windshield_config = project.windshield_config
        self._windshield_dataset = project.windshield_dataset
        self._windshield_results = dict(project.windshield_results or {})
        if self._windshield_config is not None:
            self._camera_config = project.camera_config
            self._pattern_config = project.pattern_config
            self._refresh_base_label()
        if self._windshield_dataset is not None:
            self._refresh_dataset_label()
        baseline_result = self._windshield_results.get(WindshieldModelType.BASELINE)
        if baseline_result is not None:
            self._display_result(baseline_result)
        self._refresh_comparison_table()

    def export_state(
        self,
    ) -> tuple[WindshieldConfig | None, Dataset | None, dict[WindshieldModelType, WindshieldCalibrationResult]]:
        return self._windshield_config, self._windshield_dataset, self._windshield_results

    # ------------------------------------------------------------------
    # ① Base Camera
    # ------------------------------------------------------------------
    def _build_base_camera_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        button_row = QHBoxLayout()
        for text, handler in (
            ("Load from current session", self._on_load_from_session),
            ("Load from Library...", self._on_load_from_library),
            ("Load OpenCV YAML...", self._on_load_from_yaml),
            ("Load .ccproj...", self._on_load_from_ccproj),
        ):
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            button_row.addWidget(btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        group = QGroupBox("Base Camera")
        form = QVBoxLayout(group)
        self.base_info_label = QLabel("아직 Base Camera를 불러오지 않았습니다.")
        self.base_info_label.setWordWrap(True)
        form.addWidget(self.base_info_label)
        self.base_lock_label = QLabel("")
        self.base_lock_label.setStyleSheet(f"color: {Theme.WARNING}; font-weight: 700; font-size: 14px;")
        form.addWidget(self.base_lock_label)
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _pick_calibration_result(
        self, calibration_results: dict[CameraModelType, CalibrationResult]
    ) -> CalibrationResult | None:
        candidates = {
            m: r for m, r in calibration_results.items()
            if r and r.success and r.camera_matrix is not None and r.distortion is not None
        }
        if not candidates:
            QMessageBox.warning(self, "Base Camera", "사용 가능한 (성공한) Calibration 결과가 없습니다.")
            return None
        labels = [_MODEL_LABELS.get(m, m.value) for m in candidates]
        label_to_model = {_MODEL_LABELS.get(m, m.value): m for m in candidates}
        choice, ok = QInputDialog.getItem(self, "Base Camera Model 선택", "Model:", labels, 0, False)
        if not ok or not choice:
            return None
        return candidates[label_to_model[choice]]

    def _apply_base(
        self,
        calibration_result: CalibrationResult,
        camera_config: CameraConfig | None,
        pattern_config: PatternConfig | None,
    ) -> None:
        self._windshield_config = WindshieldConfig(
            base_model_name=calibration_result.model_name,
            base_camera_matrix=calibration_result.camera_matrix.copy(),
            base_distortion=calibration_result.distortion.copy(),
        )
        self._camera_config = camera_config
        self._pattern_config = pattern_config
        self._refresh_base_label()

    def _refresh_base_label(self) -> None:
        cfg = self._windshield_config
        if cfg is None:
            self.base_info_label.setText("아직 Base Camera를 불러오지 않았습니다.")
            self.base_lock_label.setText("")
            return
        K, D = cfg.base_camera_matrix, cfg.base_distortion
        res = (
            f"{self._camera_config.width}x{self._camera_config.height}"
            if self._camera_config else "알 수 없음"
        )
        pattern_note = (
            "" if self._pattern_config is not None
            else "\n(패턴 정보 없음 - Dataset 검출을 하려면 '현재 세션' 또는 '.ccproj'로 불러오세요)"
        )
        self.base_info_label.setText(
            f"Camera Model : {_MODEL_LABELS.get(cfg.base_model_name, cfg.base_model_name.value)}\n"
            f"fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}\n"
            f"Distortion   : [{', '.join(f'{v:.5f}' for v in D.ravel())}]\n"
            f"Image Size   : {res}"
            f"{pattern_note}"
        )
        self.base_lock_label.setText("🔒 Base K,D fixed during windshield calibration")

    def _on_load_from_session(self) -> None:
        result = self._pick_calibration_result(self._session_calibration_results)
        if result is None:
            return
        self._apply_base(result, self._session_camera_config, self._session_pattern_config)

    def _on_load_from_library(self) -> None:
        cameras = list_cameras()
        if not cameras:
            QMessageBox.information(self, "Library", "Library에 저장된 카메라가 없습니다.")
            return
        camera, ok = QInputDialog.getItem(self, "Library", "Camera:", cameras, 0, False)
        if not ok or not camera:
            return
        runs = list_runs(camera)
        if not runs:
            QMessageBox.information(self, "Library", "이 카메라에는 저장된 run이 없습니다.")
            return
        run_labels = [f"{r.created_at}  ({r.num_images}장)" for r in runs]
        label_to_run = dict(zip(run_labels, runs))
        run_label, ok = QInputDialog.getItem(self, "Library", "Run:", run_labels, 0, False)
        if not ok or not run_label:
            return
        run = label_to_run[run_label]
        try:
            project, _missing = load_run_project(run.run_dir)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Library", f"불러오기 실패: {e}")
            return
        result = self._pick_calibration_result(project.calibration_results)
        if result is None:
            return
        self._apply_base(result, project.camera_config, project.pattern_config)

    def _on_load_from_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "OpenCV YAML 불러오기", "", "YAML (*.yml *.yaml)")
        if not path:
            return
        try:
            camera_matrix, distortion = load_camera_matrix_and_distortion_from_opencv_yaml(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "OpenCV YAML", f"불러오기 실패: {e}")
            return
        model = detect_model_hint_from_opencv_yaml(path)
        if model is None:
            labels = [_MODEL_LABELS[m] for m in _MODEL_LABELS]
            label_to_model = {v: k for k, v in _MODEL_LABELS.items()}
            choice, ok = QInputDialog.getItem(self, "Camera Model", "이 YAML의 Camera Model:", labels, 0, False)
            if not ok or not choice:
                return
            model = label_to_model[choice]
        fake_result = CalibrationResult(
            model_name=model, camera_matrix=camera_matrix, distortion=distortion, success=True,
        )
        # YAML만으로는 image 해상도/패턴 정보를 신뢰성 있게 복원할 수 없다 -
        # 이미 세션/프로젝트에서 로드된 값이 있으면 그대로 유지한다.
        self._apply_base(fake_result, self._camera_config, self._pattern_config)

    def _on_load_from_ccproj(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, ".ccproj 불러오기", "", "Camera Calibrator Project (*.ccproj)")
        if not path:
            return
        try:
            project, _missing = load_project(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, ".ccproj", f"불러오기 실패: {e}")
            return
        result = self._pick_calibration_result(project.calibration_results)
        if result is None:
            return
        self._apply_base(result, project.camera_config, project.pattern_config)

    # ------------------------------------------------------------------
    # ② Dataset
    # ------------------------------------------------------------------
    def _build_dataset_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        button_row = QHBoxLayout()
        load_btn = QPushButton("Load windshield images...")
        load_btn.clicked.connect(self._on_load_dataset)
        button_row.addWidget(load_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        group = QGroupBox("Dataset")
        form = QVBoxLayout(group)
        self.dataset_info_label = QLabel("아직 Windshield Dataset을 불러오지 않았습니다.")
        self.dataset_info_label.setWordWrap(True)
        form.addWidget(self.dataset_info_label)
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _refresh_dataset_label(self) -> None:
        ds = self._windshield_dataset
        if ds is None:
            self.dataset_info_label.setText("아직 Windshield Dataset을 불러오지 않았습니다.")
            return
        coverage = (ds.num_detected / ds.num_total * 100.0) if ds.num_total else 0.0
        self.dataset_info_label.setText(
            f"Images : {ds.num_total}\n"
            f"Valid  : {ds.num_detected}\n"
            f"Coverage : {coverage:.0f}%"
        )

    def _on_load_dataset(self) -> None:
        if self._pattern_config is None:
            QMessageBox.warning(
                self, "Dataset",
                "패턴 정보가 없습니다. 먼저 Base Camera 탭에서 '현재 세션' 또는 '.ccproj'로 "
                "Base Camera를 불러오세요 (패턴 정보가 함께 옵니다).",
            )
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Windshield 이미지 선택", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not paths:
            return
        try:
            dataset = detect_dataset(paths, self._pattern_config)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Dataset", f"검출 실패: {e}")
            return
        self._windshield_dataset = dataset
        self._refresh_dataset_label()

    # ------------------------------------------------------------------
    # ③ Windshield Model
    # ------------------------------------------------------------------
    def _build_model_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        model_group = QGroupBox("Windshield Models")
        model_layout = QVBoxLayout(model_group)
        self._model_button_group = QButtonGroup(self)
        # Spherical(Phase 2)은 STEP 2에서 실제로 구현됐으므로 활성화한다 -
        # Residual Ray/Spline(Phase 3/4)은 여전히 미구현이라 비활성화 상태로 둔다.
        _ENABLED_MODELS = (WindshieldModelType.BASELINE, WindshieldModelType.SPHERICAL)
        for model in (
            WindshieldModelType.BASELINE,
            WindshieldModelType.SPHERICAL,
            WindshieldModelType.RESIDUAL_RAY,
            WindshieldModelType.SPLINE,
        ):
            radio = QRadioButton(_WINDSHIELD_MODEL_LABELS[model])
            radio.setProperty("windshield_model", model.value)
            if model == WindshieldModelType.BASELINE:
                radio.setChecked(True)
            if model not in _ENABLED_MODELS:
                radio.setEnabled(False)
                radio.setToolTip("Coming soon - 아직 구현되지 않았습니다.")
            if model == WindshieldModelType.SPHERICAL:
                radio.toggled.connect(self._on_spherical_radio_toggled)
            self._model_button_group.addButton(radio)
            model_layout.addWidget(radio)
        layout.addWidget(model_group)

        self.spherical_advanced_group = QGroupBox("Advanced (Spherical)")
        self.spherical_advanced_group.setVisible(False)
        advanced_form = QFormLayout(self.spherical_advanced_group)
        self.glass_index_spin = QDoubleSpinBox()
        self.glass_index_spin.setRange(0.0, 3.0)
        self.glass_index_spin.setDecimals(3)
        self.glass_index_spin.setSpecialValueText("(default ~1.52)")
        self.glass_index_spin.setValue(_UNSET_SPINBOX_VALUE)
        advanced_form.addRow("Glass refractive index:", self.glass_index_spin)
        self.sphere_radius_spin = QDoubleSpinBox()
        self.sphere_radius_spin.setRange(0.0, 100.0)
        self.sphere_radius_spin.setDecimals(2)
        self.sphere_radius_spin.setSpecialValueText("(auto)")
        self.sphere_radius_spin.setValue(_UNSET_SPINBOX_VALUE)
        advanced_form.addRow("Initial sphere radius (m):", self.sphere_radius_spin)
        self.standoff_spin = QDoubleSpinBox()
        self.standoff_spin.setRange(0.0, 10.0)
        self.standoff_spin.setDecimals(2)
        self.standoff_spin.setSpecialValueText("(auto)")
        self.standoff_spin.setValue(_UNSET_SPINBOX_VALUE)
        advanced_form.addRow("Initial standoff distance (m):", self.standoff_spin)
        layout.addWidget(self.spherical_advanced_group)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._on_run_windshield_calibration)
        run_row.addWidget(self.run_button)
        self.export_button = QPushButton("Export Windshield YAML...")
        self.export_button.clicked.connect(self._on_export_windshield_yaml)
        self.export_button.setEnabled(False)
        run_row.addWidget(self.export_button)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        self.run_summary_label = QLabel("")
        self.run_summary_label.setWordWrap(True)
        layout.addWidget(self.run_summary_label)

        result_group = QGroupBox("RESULT (Train / Test)")
        result_layout = QVBoxLayout(result_group)
        self.stats_table = _ScrollTable(len(_STATS_ROWS) + len(_REGIONAL_ROWS) + 4, 2)
        self.stats_table.setHorizontalHeaderLabels(["Train", "Test"])
        row_labels = (
            [label for label, _ in _STATS_ROWS]
            + ["Mean dx", "Mean dy"]
            + [r.capitalize() for r in _REGIONAL_ROWS]
            + ["Edge Avg", "Ray Angular Error"]
        )
        self.stats_table.setVerticalHeaderLabels(row_labels)
        result_layout.addWidget(self.stats_table)

        charts_row = QHBoxLayout()
        self.radial_chart = RadialProfileChartWidget()
        charts_row.addWidget(self.radial_chart, stretch=1)
        self.vector_field_chart = VectorFieldChartWidget()
        charts_row.addWidget(self.vector_field_chart, stretch=1)
        result_layout.addLayout(charts_row)

        layout.addWidget(result_group, stretch=1)
        return page

    def _selected_windshield_model(self) -> WindshieldModelType:
        for button in self._model_button_group.buttons():
            if button.isChecked():
                return WindshieldModelType(button.property("windshield_model"))
        return WindshieldModelType.BASELINE

    def _on_spherical_radio_toggled(self, checked: bool) -> None:
        self.spherical_advanced_group.setVisible(checked)

    def _apply_spherical_advanced_settings(self) -> None:
        """Advanced (Spherical) spinbox 값을 self._windshield_config에 반영한다.
        호출부(_on_run_windshield_calibration)가 이미 self._windshield_config가
        None이 아님을 확인한 뒤에만 부르므로, 여기서 다시 None 체크할 필요는
        없다 - live valueChanged 핸들러로 연결하지 않는 이유이기도 하다(탭
        ①을 거치지 않고 이 스핀박스를 먼저 만지면 config가 아직 없을 수 있음)."""
        assert self._windshield_config is not None
        if self.glass_index_spin.value() > _UNSET_SPINBOX_VALUE:
            self._windshield_config.glass_refractive_index = self.glass_index_spin.value()
        radius = self.sphere_radius_spin.value()
        standoff = self.standoff_spin.value()
        hint: dict[str, float] = {}
        if radius > _UNSET_SPINBOX_VALUE:
            hint["sphere_radius"] = radius
        if standoff > _UNSET_SPINBOX_VALUE:
            hint["standoff_m"] = standoff
        if hint:
            self._windshield_config.windshield_position_hint = hint

    def _on_run_windshield_calibration(self) -> None:
        if self._windshield_config is None:
            QMessageBox.warning(self, "Windshield Calibration", "먼저 Base Camera를 불러오세요.")
            return
        if self._windshield_dataset is None or self._windshield_dataset.num_detected == 0:
            QMessageBox.warning(self, "Windshield Calibration", "먼저 Windshield Dataset을 불러오세요.")
            return
        if self._camera_config is None:
            QMessageBox.warning(self, "Windshield Calibration", "Camera 해상도 정보(CameraConfig)가 없습니다.")
            return

        selected_model = self._selected_windshield_model()
        self._windshield_config.windshield_model = selected_model
        if selected_model == WindshieldModelType.SPHERICAL:
            self._apply_spherical_advanced_settings()

        try:
            result = run_windshield_calibration(self._windshield_dataset, self._windshield_config, self._camera_config)
        except NotImplementedError as e:
            QMessageBox.information(self, "Windshield Calibration", str(e))
            return
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Windshield Calibration", f"계산 실패: {e}")
            return

        self._windshield_results[result.windshield_model] = result
        self._display_result(result)
        self._refresh_comparison_table()

    def _display_result(self, result: WindshieldCalibrationResult) -> None:
        self._current_displayed_model = result.windshield_model
        if not result.success:
            self.run_summary_label.setText(f"실패: {result.error_message}")
            self.export_button.setEnabled(False)
            return

        note = result.warning_message or ""
        self.run_summary_label.setText(
            f"Train: {len(result.train_frame_ids)} frames · Test: {len(result.test_frame_ids)} frames · "
            f"Failed: {len(result.failed_frame_ids)} · 🔒 Base K,D fixed (never re-optimized). "
            f"Test frames are evaluated only, never used to fit windshield parameters. {note}"
        )
        # 어떤 모델이든 성공한 결과는 export할 수 있다 - Baseline 전용으로
        # 잠가두면 Spherical(및 향후 다른 모델) 결과를 export할 방법이 없어진다.
        self.export_button.setEnabled(result.success)

        train_stats, test_stats = result.residual_stats, result.test_residual_stats
        row = 0
        for _, attr in _STATS_ROWS:
            train_v = getattr(train_stats, attr) if train_stats else None
            test_v = getattr(test_stats, attr) if test_stats else None
            self.stats_table.setItem(row, 0, QTableWidgetItem(_fmt(train_v)))
            self.stats_table.setItem(row, 1, QTableWidgetItem(_fmt(test_v)))
            row += 1
        self.stats_table.setItem(row, 0, QTableWidgetItem(_fmt(result.mean_dx)))
        self.stats_table.setItem(row, 1, QTableWidgetItem(_fmt(result.test_mean_dx)))
        row += 1
        self.stats_table.setItem(row, 0, QTableWidgetItem(_fmt(result.mean_dy)))
        self.stats_table.setItem(row, 1, QTableWidgetItem(_fmt(result.test_mean_dy)))
        row += 1
        for region in _REGIONAL_ROWS:
            train_v = getattr(result.regional_error, region) if result.regional_error else None
            test_v = getattr(result.test_regional_error, region) if result.test_regional_error else None
            self.stats_table.setItem(row, 0, QTableWidgetItem(_fmt(train_v)))
            self.stats_table.setItem(row, 1, QTableWidgetItem(_fmt(test_v)))
            row += 1
        train_edge = regional_edge_average(result.regional_error) if result.regional_error else None
        test_edge = regional_edge_average(result.test_regional_error) if result.test_regional_error else None
        self.stats_table.setItem(row, 0, QTableWidgetItem(_fmt(train_edge)))
        self.stats_table.setItem(row, 1, QTableWidgetItem(_fmt(test_edge)))
        row += 1
        # Ray Angular Error - Baseline은 굴절 exit point 개념이 없어 항상 None
        # (표에는 N/A로 표시) - Spherical만 실제 값을 채운다.
        self.stats_table.setItem(row, 0, QTableWidgetItem(_fmt_deg(result.ray_angular_error_deg)))
        self.stats_table.setItem(row, 1, QTableWidgetItem(_fmt_deg(result.test_ray_angular_error_deg)))
        _fit_table_to_rows(self.stats_table)

        self.radial_chart.set_profile(result.radial_profile)
        self.vector_field_chart.set_spatial_error_map(result.spatial_error_map)

    def _on_export_windshield_yaml(self) -> None:
        result = self._windshield_results.get(self._current_displayed_model)
        if result is None or not result.success:
            QMessageBox.warning(self, "Export", "Export할 결과가 없습니다.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Windshield YAML 저장", "windshield.yml", "YAML (*.yml *.yaml)")
        if not path:
            return
        try:
            export_windshield_yaml(result, self._camera_config, path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Export", f"저장 실패: {e}")
            return
        QMessageBox.information(self, "Export", f"저장했습니다:\n{path}")

    # ------------------------------------------------------------------
    # ④ Comparison
    # ------------------------------------------------------------------
    def _build_comparison_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "WINDSHIELD MODEL COMPARISON - Camera Model(Pinhole/Brown/Rational/Fisheye) "
            "비교와는 완전히 별개의 표입니다."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.comparison_table = _ScrollTable(5, 0)
        self.comparison_table.setVerticalHeaderLabels(
            ["Hold-out RMS", "P95", "Edge RMS", "Ray Angular Error", "Improvement %"]
        )
        layout.addWidget(self.comparison_table)
        layout.addStretch(1)
        return page

    def _refresh_comparison_table(self) -> None:
        order = [
            WindshieldModelType.BASELINE,
            WindshieldModelType.SPHERICAL,
            WindshieldModelType.RESIDUAL_RAY,
            WindshieldModelType.SPLINE,
        ]
        present = [m for m in order if m in self._windshield_results]
        self.comparison_table.setColumnCount(len(present))
        self.comparison_table.setHorizontalHeaderLabels([_WINDSHIELD_MODEL_LABELS[m] for m in present])

        baseline = self._windshield_results.get(WindshieldModelType.BASELINE)
        baseline_rms = (
            baseline.test_residual_stats.rmse
            if baseline and baseline.test_residual_stats and baseline.test_residual_stats.rmse
            else None
        )

        for col, model in enumerate(present):
            result = self._windshield_results[model]
            test_stats = result.test_residual_stats
            rms = test_stats.rmse if test_stats else None
            p95 = test_stats.p95 if test_stats else None
            # Hold-out(Test) 기준으로 일관되게 비교한다 - Train 쪽 regional_error를
            # 쓰면 "Hold-out RMS/P95"와 기준이 달라져 비교표 의미가 흐려진다.
            edge = regional_edge_average(result.test_regional_error) if result.test_regional_error else None
            if model == WindshieldModelType.BASELINE or baseline_rms is None or rms is None:
                improvement = None
            else:
                improvement = (baseline_rms - rms) / baseline_rms * 100.0

            self.comparison_table.setItem(0, col, QTableWidgetItem(_fmt(rms)))
            self.comparison_table.setItem(1, col, QTableWidgetItem(_fmt(p95)))
            self.comparison_table.setItem(2, col, QTableWidgetItem(_fmt(edge)))
            self.comparison_table.setItem(3, col, QTableWidgetItem(_fmt_deg(result.test_ray_angular_error_deg)))
            self.comparison_table.setItem(
                4, col, QTableWidgetItem(f"{improvement:.0f}%" if improvement is not None else "-")
            )
        _fit_table_to_rows(self.comparison_table)
