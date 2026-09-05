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

Baseline/Spherical/Residual Ray(Grid+RBF)/Spline(Phase 4) 전부 실제로
계산할 수 있다 - 더 이상 "Coming soon" 비활성 모델은 없다(Neural Residual/
Reflection은 이 Workspace의 범위 밖이라 아예 라디오 버튼 자체가 없다).

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
    QComboBox,
    QRadioButton,
    QSpinBox,
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
from calibration.windshield.base import (
    WindshieldCalibrationResult,
    WindshieldConfig,
    WindshieldModelType,
    WindshieldResultKey,
    windshield_result_key,
    windshield_result_key_for_result,
    windshield_result_key_label,
)
from calibration.windshield.reflection import ReflectionDatasetResult, ReflectionEvaluationConfig, ReflectionImagePair
from calibration.windshield.residual_ray import DEFAULT_GRID_COLS, DEFAULT_GRID_ROWS, DEFAULT_LAMBDA_MAG, DEFAULT_LAMBDA_SMOOTH
from calibration.windshield.residual_rbf import DEFAULT_RBF_NUM_CENTERS, DEFAULT_RBF_SMOOTHING
# UI는 neural_residual.py를 절대 import하지 않는다(STEP 5 안정화 라운드
# 항목 1) - neural_config.py는 PyTorch를 전혀 import하지 않는 순수 Python
# 상수 모듈이라, "Neural 기본값을 UI에 표시하는 것만으로 PyTorch가 로드"
# 되는 문제 없이 이 값들을 쓸 수 있다.
from calibration.windshield.neural_config import (
    DEFAULT_NEURAL_ACTIVATION,
    DEFAULT_NEURAL_BATCH_SIZE,
    DEFAULT_NEURAL_HIDDEN_DIMS,
    DEFAULT_NEURAL_LEARNING_RATE,
    DEFAULT_NEURAL_MAX_EPOCHS,
    DEFAULT_NEURAL_PATIENCE,
    DEFAULT_NEURAL_SEED,
    DEFAULT_NEURAL_WEIGHT_DECAY,
)
from calibration.windshield.spline import (
    DEFAULT_LAMBDA_CURVE as SPLINE_DEFAULT_LAMBDA_CURVE,
    DEFAULT_LAMBDA_MAG as SPLINE_DEFAULT_LAMBDA_MAG,
    DEFAULT_LAMBDA_SMOOTH as SPLINE_DEFAULT_LAMBDA_SMOOTH,
    DEFAULT_MAX_DISPLACEMENT_M,
    DEFAULT_SPLINE_COLS,
    DEFAULT_SPLINE_ROWS,
)
from export.opencv import (
    detect_model_hint_from_opencv_yaml,
    load_camera_matrix_and_distortion_from_opencv_yaml,
)
from export.windshield import export_windshield_yaml
from export.reflection import export_reflection_yaml
from ui.reflection_worker import ReflectionEvaluationWorker
from ui.radial_profile_view import RadialProfileChartWidget
from ui.theme import Theme
from ui.windshield_vector_field_view import VectorFieldChartWidget
from ui.windshield_worker import WindshieldCalibrationWorker
from ui.worker import run_worker_in_thread

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
        self._windshield_results: dict[WindshieldResultKey, WindshieldCalibrationResult] = {}
        self._reflection_pairs: list[ReflectionImagePair] = []
        self._reflection_results: dict[str, ReflectionDatasetResult] = {}
        self._reflection_result: ReflectionDatasetResult | None = None
        self._reflection_normal_path = ""
        self._reflection_reference_path = ""
        # 마지막으로 화면에 표시된(=Export 대상) 모델 - export_button과
        # _on_export_windshield_yaml이 특정 모델(예: Baseline)에 고정되지
        # 않고 "방금 실행/표시한 결과"를 export하도록 추적한다.
        self._current_displayed_model: WindshieldResultKey | None = None

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
        self.tabs.addTab(self._build_reflection_tab(), "⑤ Reflection")
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
        self._reflection_results = dict(getattr(project, "reflection_results", {}) or {})
        self._reflection_result = next(iter(self._reflection_results.values()), None)
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
        if self._reflection_result is not None:
            self._display_reflection_result(self._reflection_result)

    def export_state(
        self,
    ) -> tuple[WindshieldConfig | None, Dataset | None, dict[WindshieldResultKey, WindshieldCalibrationResult], dict[str, ReflectionDatasetResult]]:
        return self._windshield_config, self._windshield_dataset, self._windshield_results, self._reflection_results

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
        # Baseline/Spherical/Residual Ray(Grid+RBF)/Spline(Phase 4) 전부
        # 실제로 구현됐으므로 활성화한다 - 더 이상 미구현 모델이 없다.
        _ENABLED_MODELS = (
            WindshieldModelType.BASELINE,
            WindshieldModelType.SPHERICAL,
            WindshieldModelType.RESIDUAL_RAY,
            WindshieldModelType.SPLINE,
        )
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
            if model == WindshieldModelType.SPLINE:
                radio.toggled.connect(self._on_spline_radio_toggled)
            if model == WindshieldModelType.RESIDUAL_RAY:
                radio.toggled.connect(self._on_residual_ray_radio_toggled)
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

        self.residual_ray_advanced_group = QGroupBox("Advanced (Residual Ray)")
        self.residual_ray_advanced_group.setVisible(False)
        rr_layout = QVBoxLayout(self.residual_ray_advanced_group)

        method_row = QHBoxLayout()
        self._residual_ray_method_button_group = QButtonGroup(self)
        self.residual_ray_method_grid_radio = QRadioButton("Grid")
        self.residual_ray_method_grid_radio.setChecked(True)
        self.residual_ray_method_rbf_radio = QRadioButton("RBF")
        self.residual_ray_method_neural_radio = QRadioButton("Neural")
        self._residual_ray_method_button_group.addButton(self.residual_ray_method_grid_radio)
        self._residual_ray_method_button_group.addButton(self.residual_ray_method_rbf_radio)
        self._residual_ray_method_button_group.addButton(self.residual_ray_method_neural_radio)
        method_row.addWidget(QLabel("Method"))
        method_row.addWidget(self.residual_ray_method_grid_radio)
        method_row.addWidget(self.residual_ray_method_rbf_radio)
        method_row.addWidget(self.residual_ray_method_neural_radio)
        method_row.addStretch(1)
        rr_layout.addLayout(method_row)

        self.residual_grid_settings_group = QGroupBox("Grid Settings")
        grid_settings_layout = QVBoxLayout(self.residual_grid_settings_group)

        mode_row = QHBoxLayout()
        self._grid_mode_button_group = QButtonGroup(self)
        self.grid_mode_auto_radio = QRadioButton("AUTO (권장)")
        self.grid_mode_auto_radio.setChecked(True)
        self.grid_mode_manual_radio = QRadioButton("Manual")
        self._grid_mode_button_group.addButton(self.grid_mode_auto_radio)
        self._grid_mode_button_group.addButton(self.grid_mode_manual_radio)
        mode_row.addWidget(self.grid_mode_auto_radio)
        mode_row.addWidget(self.grid_mode_manual_radio)
        mode_row.addStretch(1)
        grid_settings_layout.addLayout(mode_row)

        rr_form = QFormLayout()
        self.grid_rows_spin = QSpinBox()
        self.grid_rows_spin.setRange(2, 20)
        self.grid_rows_spin.setValue(DEFAULT_GRID_ROWS)
        self.grid_rows_spin.setEnabled(False)
        rr_form.addRow("Grid Rows:", self.grid_rows_spin)
        self.grid_cols_spin = QSpinBox()
        self.grid_cols_spin.setRange(2, 30)
        self.grid_cols_spin.setValue(DEFAULT_GRID_COLS)
        self.grid_cols_spin.setEnabled(False)
        rr_form.addRow("Grid Cols:", self.grid_cols_spin)
        self.lambda_mag_spin = QDoubleSpinBox()
        self.lambda_mag_spin.setRange(0.0, 10.0)
        self.lambda_mag_spin.setDecimals(6)
        self.lambda_mag_spin.setSingleStep(0.0001)
        self.lambda_mag_spin.setValue(DEFAULT_LAMBDA_MAG)
        rr_form.addRow("Magnitude λ:", self.lambda_mag_spin)
        self.lambda_smooth_spin = QDoubleSpinBox()
        self.lambda_smooth_spin.setRange(0.0, 10.0)
        self.lambda_smooth_spin.setDecimals(6)
        self.lambda_smooth_spin.setSingleStep(0.001)
        self.lambda_smooth_spin.setValue(DEFAULT_LAMBDA_SMOOTH)
        rr_form.addRow("Smoothness λ:", self.lambda_smooth_spin)
        grid_settings_layout.addLayout(rr_form)
        rr_layout.addWidget(self.residual_grid_settings_group)

        self.residual_rbf_settings_group = QGroupBox("RBF Settings")
        self.residual_rbf_settings_group.setVisible(False)
        rbf_form = QFormLayout(self.residual_rbf_settings_group)
        self.rbf_kernel_combo = QComboBox()
        self.rbf_kernel_combo.addItem("Thin Plate Spline", "thin_plate_spline")
        rbf_form.addRow("Kernel:", self.rbf_kernel_combo)
        self.rbf_mode_auto_radio = QRadioButton("AUTO")
        self.rbf_mode_auto_radio.setChecked(True)
        self.rbf_mode_manual_radio = QRadioButton("Manual")
        self._rbf_mode_button_group = QButtonGroup(self)
        self._rbf_mode_button_group.addButton(self.rbf_mode_auto_radio)
        self._rbf_mode_button_group.addButton(self.rbf_mode_manual_radio)
        rbf_mode_row = QHBoxLayout()
        rbf_mode_row.addWidget(self.rbf_mode_auto_radio)
        rbf_mode_row.addWidget(self.rbf_mode_manual_radio)
        rbf_mode_row.addStretch(1)
        rbf_form.addRow("Centers/Smoothing:", rbf_mode_row)
        self.rbf_centers_spin = QSpinBox()
        self.rbf_centers_spin.setRange(3, 512)
        self.rbf_centers_spin.setValue(DEFAULT_RBF_NUM_CENTERS)
        self.rbf_centers_spin.setEnabled(False)
        rbf_form.addRow("Centers:", self.rbf_centers_spin)
        self.rbf_smoothing_spin = QDoubleSpinBox()
        self.rbf_smoothing_spin.setRange(0.0, 1.0)
        self.rbf_smoothing_spin.setDecimals(6)
        self.rbf_smoothing_spin.setSingleStep(0.0001)
        self.rbf_smoothing_spin.setValue(DEFAULT_RBF_SMOOTHING)
        self.rbf_smoothing_spin.setEnabled(False)
        rbf_form.addRow("Smoothing:", self.rbf_smoothing_spin)
        rr_layout.addWidget(self.residual_rbf_settings_group)

        # NEURAL SETTINGS - Advanced 항목은 QFormLayout 하나에 전부 넣는다
        # (RBF/Grid보다 옵션이 많지만, "너무 많은 옵션은 Advanced에 숨겨도
        # 된다"는 요구사항에 따라 별도 collapsible 위젯은 만들지 않는다 -
        # 이 그룹 자체가 이미 method==Neural일 때만 보인다).
        self.residual_neural_settings_group = QGroupBox("NEURAL SETTINGS")
        self.residual_neural_settings_group.setVisible(False)
        neural_form = QFormLayout(self.residual_neural_settings_group)
        neural_form.addRow("Architecture:", QLabel("Tiny MLP (2 → 32 → 64 → 32 → 3, SiLU)"))
        self.neural_epochs_spin = QSpinBox()
        self.neural_epochs_spin.setRange(10, 5000)
        self.neural_epochs_spin.setValue(DEFAULT_NEURAL_MAX_EPOCHS)
        neural_form.addRow("Epochs (max):", self.neural_epochs_spin)
        self.neural_lr_spin = QDoubleSpinBox()
        self.neural_lr_spin.setRange(1e-6, 1.0)
        self.neural_lr_spin.setDecimals(6)
        self.neural_lr_spin.setSingleStep(0.0001)
        self.neural_lr_spin.setValue(DEFAULT_NEURAL_LEARNING_RATE)
        neural_form.addRow("Learning Rate:", self.neural_lr_spin)
        self.neural_weight_decay_spin = QDoubleSpinBox()
        self.neural_weight_decay_spin.setRange(0.0, 1.0)
        self.neural_weight_decay_spin.setDecimals(6)
        self.neural_weight_decay_spin.setSingleStep(0.0001)
        self.neural_weight_decay_spin.setValue(DEFAULT_NEURAL_WEIGHT_DECAY)
        neural_form.addRow("Weight Decay:", self.neural_weight_decay_spin)
        self.neural_lambda_mag_spin = QDoubleSpinBox()
        self.neural_lambda_mag_spin.setRange(0.0, 10.0)
        self.neural_lambda_mag_spin.setDecimals(6)
        self.neural_lambda_mag_spin.setSingleStep(0.001)
        self.neural_lambda_mag_spin.setValue(0.01)
        neural_form.addRow("Magnitude λ:", self.neural_lambda_mag_spin)
        self.neural_lambda_smooth_spin = QDoubleSpinBox()
        self.neural_lambda_smooth_spin.setRange(0.0, 10.0)
        self.neural_lambda_smooth_spin.setDecimals(6)
        self.neural_lambda_smooth_spin.setSingleStep(0.001)
        self.neural_lambda_smooth_spin.setValue(0.01)
        neural_form.addRow("Smoothness λ:", self.neural_lambda_smooth_spin)
        self.neural_early_stop_check = QRadioButton("ON")
        self.neural_early_stop_check.setChecked(True)
        self.neural_early_stop_check.setEnabled(False)  # 이번 라운드는 항상 ON(요구사항)
        neural_form.addRow("Early Stop:", self.neural_early_stop_check)
        self.neural_patience_spin = QSpinBox()
        self.neural_patience_spin.setRange(1, 500)
        self.neural_patience_spin.setValue(DEFAULT_NEURAL_PATIENCE)
        neural_form.addRow("Patience:", self.neural_patience_spin)
        self.neural_seed_spin = QSpinBox()
        self.neural_seed_spin.setRange(0, 2**31 - 1)
        self.neural_seed_spin.setValue(DEFAULT_NEURAL_SEED)
        neural_form.addRow("Seed:", self.neural_seed_spin)
        self.neural_batch_size_spin = QSpinBox()
        self.neural_batch_size_spin.setRange(16, 4096)
        self.neural_batch_size_spin.setValue(DEFAULT_NEURAL_BATCH_SIZE)
        neural_form.addRow("Batch Size:", self.neural_batch_size_spin)
        rr_layout.addWidget(self.residual_neural_settings_group)

        self.grid_mode_manual_radio.toggled.connect(self.grid_rows_spin.setEnabled)
        self.grid_mode_manual_radio.toggled.connect(self.grid_cols_spin.setEnabled)
        self.rbf_mode_manual_radio.toggled.connect(self.rbf_centers_spin.setEnabled)
        self.rbf_mode_manual_radio.toggled.connect(self.rbf_smoothing_spin.setEnabled)
        self.residual_ray_method_grid_radio.toggled.connect(self.residual_grid_settings_group.setVisible)
        self.residual_ray_method_rbf_radio.toggled.connect(self.residual_rbf_settings_group.setVisible)
        self.residual_ray_method_neural_radio.toggled.connect(self.residual_neural_settings_group.setVisible)
        layout.addWidget(self.residual_ray_advanced_group)

        self.spline_advanced_group = QGroupBox("Advanced (Spline)")
        self.spline_advanced_group.setVisible(False)
        spline_layout = QVBoxLayout(self.spline_advanced_group)
        spline_form = QFormLayout()
        spline_form.addRow("Base Surface:", QLabel("Spherical (frozen)"))

        spline_mode_row = QHBoxLayout()
        self._spline_mode_button_group = QButtonGroup(self)
        self.spline_mode_auto_radio = QRadioButton("AUTO (권장)")
        self.spline_mode_auto_radio.setChecked(True)
        self.spline_mode_manual_radio = QRadioButton("Manual")
        self._spline_mode_button_group.addButton(self.spline_mode_auto_radio)
        self._spline_mode_button_group.addButton(self.spline_mode_manual_radio)
        spline_mode_row.addWidget(self.spline_mode_auto_radio)
        spline_mode_row.addWidget(self.spline_mode_manual_radio)
        spline_mode_row.addStretch(1)
        spline_form.addRow("Control Grid:", spline_mode_row)

        self.spline_rows_spin = QSpinBox()
        self.spline_rows_spin.setRange(4, 12)
        self.spline_rows_spin.setValue(DEFAULT_SPLINE_ROWS)
        self.spline_rows_spin.setEnabled(False)
        spline_form.addRow("Manual Rows:", self.spline_rows_spin)
        self.spline_cols_spin = QSpinBox()
        self.spline_cols_spin.setRange(4, 16)
        self.spline_cols_spin.setValue(DEFAULT_SPLINE_COLS)
        self.spline_cols_spin.setEnabled(False)
        spline_form.addRow("Manual Cols:", self.spline_cols_spin)

        self.spline_lambda_mag_spin = QDoubleSpinBox()
        self.spline_lambda_mag_spin.setRange(0.0, 10.0)
        self.spline_lambda_mag_spin.setDecimals(6)
        self.spline_lambda_mag_spin.setSingleStep(0.001)
        self.spline_lambda_mag_spin.setValue(SPLINE_DEFAULT_LAMBDA_MAG)
        spline_form.addRow("Magnitude λ:", self.spline_lambda_mag_spin)
        self.spline_lambda_smooth_spin = QDoubleSpinBox()
        self.spline_lambda_smooth_spin.setRange(0.0, 10.0)
        self.spline_lambda_smooth_spin.setDecimals(6)
        self.spline_lambda_smooth_spin.setSingleStep(0.01)
        self.spline_lambda_smooth_spin.setValue(SPLINE_DEFAULT_LAMBDA_SMOOTH)
        spline_form.addRow("Smoothness λ:", self.spline_lambda_smooth_spin)
        self.spline_lambda_curve_spin = QDoubleSpinBox()
        self.spline_lambda_curve_spin.setRange(0.0, 10.0)
        self.spline_lambda_curve_spin.setDecimals(6)
        self.spline_lambda_curve_spin.setSingleStep(0.01)
        self.spline_lambda_curve_spin.setValue(SPLINE_DEFAULT_LAMBDA_CURVE)
        spline_form.addRow("Curvature λ:", self.spline_lambda_curve_spin)

        self.spline_max_displacement_spin = QDoubleSpinBox()
        self.spline_max_displacement_spin.setRange(1.0, 50.0)
        self.spline_max_displacement_spin.setDecimals(2)
        self.spline_max_displacement_spin.setSuffix(" mm")
        self.spline_max_displacement_spin.setValue(DEFAULT_MAX_DISPLACEMENT_M * 1000.0)
        spline_form.addRow("Max deformation:", self.spline_max_displacement_spin)
        spline_layout.addLayout(spline_form)

        self.spline_mode_manual_radio.toggled.connect(self.spline_rows_spin.setEnabled)
        self.spline_mode_manual_radio.toggled.connect(self.spline_cols_spin.setEnabled)
        layout.addWidget(self.spline_advanced_group)

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

        layout.addWidget(result_group)

        self.residual_ray_diagnostics_group = QGroupBox("RESIDUAL RAY DIAGNOSTICS")
        self.residual_ray_diagnostics_group.setVisible(False)
        diag_form = QFormLayout(self.residual_ray_diagnostics_group)
        self.diag_residual_method_label = QLabel("N/A")
        diag_form.addRow("Method:", self.diag_residual_method_label)
        self.diag_selected_grid_label = QLabel("N/A")
        diag_form.addRow("Selected Grid:", self.diag_selected_grid_label)
        self.diag_rbf_settings_label = QLabel("N/A")
        diag_form.addRow("RBF Settings:", self.diag_rbf_settings_label)
        self.diag_neural_architecture_label = QLabel("N/A")
        diag_form.addRow("Neural Architecture:", self.diag_neural_architecture_label)
        self.diag_neural_training_label = QLabel("N/A")
        diag_form.addRow("Neural Training:", self.diag_neural_training_label)
        self.diag_neural_total_loss_label = QLabel("N/A")
        diag_form.addRow("Neural Total Loss:", self.diag_neural_total_loss_label)
        self.diag_neural_seed_stability_label = QLabel("N/A")
        diag_form.addRow("Seed Stability:", self.diag_neural_seed_stability_label)
        self.diag_selection_mode_label = QLabel("N/A")
        diag_form.addRow("Selection Mode:", self.diag_selection_mode_label)
        self.diag_runtime_params_label = QLabel("N/A")
        diag_form.addRow("Residual Value Params:", self.diag_runtime_params_label)
        self.diag_storage_params_label = QLabel("N/A")
        diag_form.addRow("Stored Numeric Values:", self.diag_storage_params_label)
        self.diag_pose_params_label = QLabel("N/A")
        diag_form.addRow("Train Pose Param Count:", self.diag_pose_params_label)
        self.diag_holdout_label = QLabel("N/A")
        diag_form.addRow("Repeated Hold-out:", self.diag_holdout_label)
        self.diag_ray_stability_label = QLabel("N/A")
        diag_form.addRow("Ray Stability:", self.diag_ray_stability_label)
        self.diag_pose_movement_label = QLabel("N/A")
        diag_form.addRow("Pose Movement (STAGE B):", self.diag_pose_movement_label)
        layout.addWidget(self.residual_ray_diagnostics_group)

        self.spline_diagnostics_group = QGroupBox("SPLINE DIAGNOSTICS")
        self.spline_diagnostics_group.setVisible(False)
        spline_diag_form = QFormLayout(self.spline_diagnostics_group)
        self.diag_spline_sphere_label = QLabel("N/A")
        spline_diag_form.addRow("Base Sphere:", self.diag_spline_sphere_label)
        self.diag_spline_grid_label = QLabel("N/A")
        spline_diag_form.addRow("Spline Grid:", self.diag_spline_grid_label)
        self.diag_spline_params_label = QLabel("N/A")
        spline_diag_form.addRow("Surface Params:", self.diag_spline_params_label)
        self.diag_spline_selection_mode_label = QLabel("N/A")
        spline_diag_form.addRow("Selection Mode:", self.diag_spline_selection_mode_label)
        self.diag_spline_deformation_label = QLabel("N/A")
        spline_diag_form.addRow("Deformation |Δs|:", self.diag_spline_deformation_label)
        self.diag_spline_holdout_label = QLabel("N/A")
        spline_diag_form.addRow("Repeated Hold-out:", self.diag_spline_holdout_label)
        self.diag_spline_ray_stability_label = QLabel("N/A")
        spline_diag_form.addRow("Ray Stability:", self.diag_spline_ray_stability_label)
        self.diag_spline_surface_stability_label = QLabel("N/A")
        spline_diag_form.addRow("Surface Stability:", self.diag_spline_surface_stability_label)
        self.diag_spline_pose_movement_label = QLabel("N/A")
        spline_diag_form.addRow("Pose Movement (STAGE B):", self.diag_spline_pose_movement_label)
        layout.addWidget(self.spline_diagnostics_group)

        layout.addStretch(1)
        return page

    def _selected_windshield_model(self) -> WindshieldModelType:
        for button in self._model_button_group.buttons():
            if button.isChecked():
                return WindshieldModelType(button.property("windshield_model"))
        return WindshieldModelType.BASELINE

    def _on_spherical_radio_toggled(self, checked: bool) -> None:
        self.spherical_advanced_group.setVisible(checked)

    def _on_residual_ray_radio_toggled(self, checked: bool) -> None:
        self.residual_ray_advanced_group.setVisible(checked)

    def _on_spline_radio_toggled(self, checked: bool) -> None:
        self.spline_advanced_group.setVisible(checked)

    def _apply_spline_advanced_settings(self) -> None:
        """Advanced (Spline) 위젯 값을 self._windshield_config.spline_hint에
        반영한다 - Residual Ray의 helper와 완전히 별도로 둔다(사용자 스펙
        1번과 동일한 원칙, 서로 다른 모델의 advanced 설정을 섞지 않는다).

        AUTO면 auto_spline=1.0만 쓰고 spline_rows/cols는 넣지 않는다
        (calibrate_spline이 auto_spline>0이면 select_best_spline_grid_
        resolution으로 스스로 해상도를 고른다). Manual이면 auto_spline=0.0 +
        사용자가 고른 rows/cols를 명시적으로 넣는다. λ/max displacement는
        AUTO/Manual 여부와 무관하게 항상 포함한다(mm -> m 변환)."""
        assert self._windshield_config is not None
        hint: dict[str, object] = {
            "lambda_mag": self.spline_lambda_mag_spin.value(),
            "lambda_smooth": self.spline_lambda_smooth_spin.value(),
            "lambda_curve": self.spline_lambda_curve_spin.value(),
            "max_displacement_m": self.spline_max_displacement_spin.value() / 1000.0,
        }
        if self.spline_mode_auto_radio.isChecked():
            hint["auto_spline"] = 1.0
        else:
            hint["auto_spline"] = 0.0
            hint["spline_rows"] = float(self.spline_rows_spin.value())
            hint["spline_cols"] = float(self.spline_cols_spin.value())
        self._windshield_config.spline_hint = hint

    def _apply_residual_ray_advanced_settings(self) -> None:
        """Advanced (Residual Ray) 위젯 값을 self._windshield_config.residual_ray_hint에
        반영한다 - _apply_spherical_advanced_settings()와 별도 helper로 둔다
        (사용자 스펙 1번, 두 모델의 advanced 설정을 섞지 않는다).

        AUTO가 선택돼 있으면 auto_grid=1.0만 쓰고 grid_rows/cols는 아예
        넣지 않는다(calibrate_residual_ray가 auto_grid>0이면 select_best_
        grid_resolution으로 스스로 해상도를 고르고, 그 결과로 config를
        새로 만들어 쓴다 - 여기서 미리 grid_rows/cols를 채워 넣으면 오히려
        혼란을 준다). Manual이면 auto_grid=0.0 + 사용자가 고른 rows/cols를
        명시적으로 넣는다. λ 값은 AUTO/Manual 여부와 무관하게 항상 포함한다.
        """
        assert self._windshield_config is not None
        if self.residual_ray_method_rbf_radio.isChecked():
            hint: dict[str, object] = {
                "method": "rbf",
                "rbf_kernel": self.rbf_kernel_combo.currentData(),
            }
            if self.rbf_mode_auto_radio.isChecked():
                hint["auto_rbf"] = 1.0
            else:
                hint["auto_rbf"] = 0.0
                hint["rbf_num_centers"] = float(self.rbf_centers_spin.value())
                hint["rbf_smoothing"] = float(self.rbf_smoothing_spin.value())
        elif self.residual_ray_method_neural_radio.isChecked():
            # Neural은 이번 라운드에 AUTO architecture search가 없다(사용자
            # 스펙 27/28번) - Grid/RBF의 AUTO/Manual 라디오와 달리 위젯 값이
            # 항상 곧바로 적용되는 "Manual"뿐이다.
            hint = {
                "method": "neural",
                "neural_hidden_dims": list(DEFAULT_NEURAL_HIDDEN_DIMS),
                "neural_activation": DEFAULT_NEURAL_ACTIVATION,
                "neural_max_epochs": float(self.neural_epochs_spin.value()),
                "neural_learning_rate": float(self.neural_lr_spin.value()),
                "neural_weight_decay": float(self.neural_weight_decay_spin.value()),
                "neural_lambda_mag": float(self.neural_lambda_mag_spin.value()),
                "neural_lambda_smooth": float(self.neural_lambda_smooth_spin.value()),
                "neural_patience": float(self.neural_patience_spin.value()),
                "neural_seed": float(self.neural_seed_spin.value()),
                "neural_batch_size": float(self.neural_batch_size_spin.value()),
            }
        else:
            hint = {
                "method": "grid",
                "lambda_mag": self.lambda_mag_spin.value(),
                "lambda_smooth": self.lambda_smooth_spin.value(),
            }
            if self.grid_mode_auto_radio.isChecked():
                hint["auto_grid"] = 1.0
            else:
                hint["auto_grid"] = 0.0
                hint["grid_rows"] = float(self.grid_rows_spin.value())
                hint["grid_cols"] = float(self.grid_cols_spin.value())
        self._windshield_config.residual_ray_hint = hint

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
        if selected_model == WindshieldModelType.RESIDUAL_RAY:
            self._apply_residual_ray_advanced_settings()
        if selected_model == WindshieldModelType.SPLINE:
            self._apply_spline_advanced_settings()

        # Residual Ray의 STAGE A/B + Repeated Hold-out은 수 초~수십 초가 걸릴
        # 수 있어 GUI 스레드에서 직접 돌리면 그동안 창 이동/크기 조절/다른 탭
        # 렌더링이 전부 멈춘다 - ui/worker.py의 기존 QObject worker +
        # run_worker_in_thread() 패턴을 그대로 재사용해 백그라운드로 옮긴다
        # (ui/windshield_worker.py 참고). 여기서 워커에 넘기는 건 순수 data
        # object(Dataset/WindshieldConfig/CameraConfig)뿐이고, Qt 위젯은 결과
        # signal을 받는 아래 핸들러들(항상 main thread에서 실행됨) 안에서만
        # 만진다.
        self.run_button.setEnabled(False)
        self.run_summary_label.setText("Running...")

        worker = WindshieldCalibrationWorker(self._windshield_dataset, self._windshield_config, self._camera_config)
        thread = run_worker_in_thread(worker, self)
        worker.result_ready.connect(self._on_windshield_calibration_finished)
        worker.not_implemented.connect(self._on_windshield_calibration_not_implemented)
        worker.error.connect(self._on_windshield_calibration_error)
        self._windshield_thread, self._windshield_worker = thread, worker
        thread.finished.connect(lambda: self.run_button.setEnabled(True))
        thread.start()

    def _on_windshield_calibration_finished(self, result: WindshieldCalibrationResult) -> None:
        self._windshield_results[windshield_result_key_for_result(result)] = result
        self._display_result(result)
        self._refresh_comparison_table()

    def _on_windshield_calibration_not_implemented(self, message: str) -> None:
        self.run_summary_label.setText("")
        QMessageBox.information(self, "Windshield Calibration", message)

    def _on_windshield_calibration_error(self, message: str) -> None:
        self.run_summary_label.setText(f"실패: {message}")
        QMessageBox.critical(self, "Windshield Calibration", message)

    def _display_result(self, result: WindshieldCalibrationResult) -> None:
        self._current_displayed_model = windshield_result_key_for_result(result)
        if not result.success:
            self.run_summary_label.setText(f"실패: {result.error_message}")
            self.export_button.setEnabled(False)
            self.residual_ray_diagnostics_group.setVisible(False)
            self.spline_diagnostics_group.setVisible(False)
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

        self._update_residual_ray_diagnostics(result)
        self._update_spline_diagnostics(result)

    def _update_spline_diagnostics(self, result: WindshieldCalibrationResult) -> None:
        """Spline 결과의 fitted_params에 이미 backend(calibration.windshield.
        spline.calibrate_spline/run_spline_calibration_with_diagnostics)가
        계산해 둔 값만 그대로 읽어 표시한다 - UI에서 새로 계산하지 않는다
        (Residual Ray 진단 패널과 동일한 원칙). Baseline/Spherical/Residual
        Ray 결과에서는 패널을 숨긴다."""
        if result.windshield_model != WindshieldModelType.SPLINE:
            self.spline_diagnostics_group.setVisible(False)
            return
        self.spline_diagnostics_group.setVisible(True)

        fp = result.fitted_params
        radius = fp.get("sphere_radius")
        cx, cy, cz = fp.get("sphere_center_x"), fp.get("sphere_center_y"), fp.get("sphere_center_z")
        if radius is not None and cx is not None:
            self.diag_spline_sphere_label.setText(
                f"Radius {_fmt(radius)}m · Center ({_fmt(cx)}, {_fmt(cy)}, {_fmt(cz)})"
            )
        else:
            self.diag_spline_sphere_label.setText("N/A")

        rows, cols = fp.get("spline_rows"), fp.get("spline_cols")
        self.diag_spline_grid_label.setText(f"{int(rows)} x {int(cols)}" if rows is not None and cols is not None else "N/A")
        self.diag_spline_params_label.setText(_fmt(fp.get("runtime_param_count")))

        is_auto = fp.get("diag_selection_mode_is_auto")
        self.diag_spline_selection_mode_label.setText(
            "AUTO" if is_auto == 1.0 else "Manual" if is_auto == 0.0 else "N/A"
        )

        mean_abs, max_abs = fp.get("diag_deformation_mean_abs_m"), fp.get("diag_deformation_max_abs_m")
        if mean_abs is not None and max_abs is not None:
            self.diag_spline_deformation_label.setText(f"Mean {mean_abs*1000:.3f}mm · Max {max_abs*1000:.3f}mm")
        else:
            self.diag_spline_deformation_label.setText("N/A")

        n_req, n_ok = fp.get("diag_repeated_n_requested"), fp.get("diag_repeated_n_successful")
        if n_req is not None and n_ok is not None:
            self.diag_spline_holdout_label.setText(
                f"{int(n_ok)}/{int(n_req)} successful · "
                f"Mean Test RMS {_fmt(fp.get('diag_repeated_mean_test_rmse'))} "
                f"(±{_fmt(fp.get('diag_repeated_std_test_rmse'))}) · "
                f"Mean P95 {_fmt(fp.get('diag_repeated_mean_test_p95'))} · "
                f"Mean Edge RMS {_fmt(fp.get('diag_repeated_mean_edge_rms'))}"
            )
        else:
            self.diag_spline_holdout_label.setText("N/A")

        ray_mean, ray_p95 = fp.get("diag_ray_stability_mean_deg"), fp.get("diag_ray_stability_p95_deg")
        if ray_mean is not None or ray_p95 is not None:
            self.diag_spline_ray_stability_label.setText(f"Mean {_fmt_deg(ray_mean)} · P95 {_fmt_deg(ray_p95)}")
        else:
            self.diag_spline_ray_stability_label.setText("N/A")

        surf_mean, surf_p95 = fp.get("diag_surface_stability_mean_mm"), fp.get("diag_surface_stability_p95_mm")
        if surf_mean is not None or surf_p95 is not None:
            self.diag_spline_surface_stability_label.setText(
                f"Mean {_fmt(surf_mean)}mm · P95 {_fmt(surf_p95)}mm"
            )
        else:
            self.diag_spline_surface_stability_label.setText("N/A")

        pose_r_med, pose_r_p95 = fp.get("diag_pose_delta_r_median_deg"), fp.get("diag_pose_delta_r_p95_deg")
        pose_t_med, pose_t_p95 = fp.get("diag_pose_delta_t_median_mm"), fp.get("diag_pose_delta_t_p95_mm")
        if pose_r_med is not None and pose_t_med is not None:
            self.diag_spline_pose_movement_label.setText(
                f"ΔR median {_fmt_deg(pose_r_med)} / P95 {_fmt_deg(pose_r_p95)} · "
                f"Δt median {_fmt(pose_t_med)}mm / P95 {_fmt(pose_t_p95)}mm"
            )
        else:
            self.diag_spline_pose_movement_label.setText("N/A")

    def _update_residual_ray_diagnostics(self, result: WindshieldCalibrationResult) -> None:
        """Residual Ray 결과의 fitted_params에 이미 backend(calibration.windshield.
        residual_ray.calibrate_residual_ray/run_residual_ray_calibration_with_
        diagnostics)가 계산해 둔 값만 그대로 읽어 표시한다 - 여기서 새로
        계산/추정하는 값은 하나도 없다(사용자 스펙 2번 하드 요구사항).
        Baseline/Spherical 결과에서는 이 개념 자체가 없으므로 패널을 숨긴다."""
        if result.windshield_model != WindshieldModelType.RESIDUAL_RAY:
            self.residual_ray_diagnostics_group.setVisible(False)
            return
        self.residual_ray_diagnostics_group.setVisible(True)

        fp = result.fitted_params
        method_code = fp.get("residual_ray_method", 0.0)
        is_rbf = method_code == 1.0
        is_neural = method_code == 2.0
        self.diag_residual_method_label.setText("Neural" if is_neural else "RBF" if is_rbf else "Grid")
        rows, cols = fp.get("grid_rows"), fp.get("grid_cols")
        self.diag_selected_grid_label.setText(
            f"{int(rows)} x {int(cols)}" if not is_rbf and not is_neural and rows is not None and cols is not None else "N/A"
        )
        if is_rbf:
            centers = fp.get("rbf_num_centers")
            smoothing = fp.get("rbf_smoothing")
            self.diag_rbf_settings_label.setText(
                f"Thin Plate Spline, centers {int(centers)}, smoothing {_fmt(smoothing)}"
                if centers is not None and smoothing is not None else "N/A"
            )
        else:
            self.diag_rbf_settings_label.setText("N/A")

        if is_neural:
            n_hidden = fp.get("neural_num_hidden_layers")
            if n_hidden is not None:
                hidden = " → ".join(str(int(fp.get(f"neural_hidden_dim_{i}", 0))) for i in range(int(n_hidden)))
                self.diag_neural_architecture_label.setText(f"2 → {hidden} → 3 · Params {_fmt(fp.get('neural_param_count'))}")
            else:
                self.diag_neural_architecture_label.setText("N/A")
            # STEP 5 안정화 라운드 항목 4/5 - Train/Val 둘 다 "같은 best
            # checkpoint epoch에서, 같은 정의(Ray Loss)로" 계산된 값이다
            # (neural_best_train_ray_loss/neural_best_val_ray_loss). 예전
            # neural_final_train_loss(마지막 epoch total)/neural_final_val_loss
            # (best epoch ray)처럼 서로 다른 시점/정의를 섞지 않는다.
            best_epoch = fp.get("neural_best_epoch")
            train_ray_loss = fp.get("neural_best_train_ray_loss")
            val_ray_loss = fp.get("neural_best_val_ray_loss")
            batch_size = fp.get("neural_batch_size")
            if best_epoch is not None:
                self.diag_neural_training_label.setText(
                    f"Best Epoch {int(best_epoch)} · Train Ray Loss {_fmt(train_ray_loss)} · "
                    f"Val Ray Loss {_fmt(val_ray_loss)}"
                    + (f" · Batch Size {int(batch_size)}" if batch_size is not None else "")
                )
            else:
                self.diag_neural_training_label.setText("N/A")
            train_total_loss = fp.get("neural_best_train_total_loss")
            val_total_loss = fp.get("neural_best_val_total_loss")
            if train_total_loss is not None or val_total_loss is not None:
                self.diag_neural_total_loss_label.setText(
                    f"Train Total Loss {_fmt(train_total_loss)} · Val Total Loss {_fmt(val_total_loss)}"
                )
            else:
                self.diag_neural_total_loss_label.setText("N/A")
            seed_mean, seed_p95 = fp.get("diag_seed_stability_mean_deg"), fp.get("diag_seed_stability_p95_deg")
            if seed_mean is not None or seed_p95 is not None:
                self.diag_neural_seed_stability_label.setText(f"Mean {_fmt_deg(seed_mean)} · P95 {_fmt_deg(seed_p95)}")
            else:
                self.diag_neural_seed_stability_label.setText("N/A")
        else:
            self.diag_neural_architecture_label.setText("N/A")
            self.diag_neural_training_label.setText("N/A")
            self.diag_neural_total_loss_label.setText("N/A")
            self.diag_neural_seed_stability_label.setText("N/A")

        is_auto = fp.get("diag_selection_mode_is_auto")
        self.diag_selection_mode_label.setText(
            "AUTO" if is_auto == 1.0 else "Manual" if is_auto == 0.0 else "N/A"
        )

        if is_rbf:
            residual_values = fp.get("residual_value_param_count", fp.get("runtime_param_count"))
            centers = fp.get("rbf_center_count", fp.get("rbf_num_centers"))
            stored_values = fp.get("serialized_numeric_value_count")
            self.diag_runtime_params_label.setText(
                f"{_fmt(residual_values)} (3 x {int(centers)})" if centers is not None else _fmt(residual_values)
            )
            self.diag_storage_params_label.setText(
                f"{_fmt(stored_values)} (5 x {int(centers)})" if stored_values is not None and centers is not None else "N/A"
            )
        else:
            self.diag_runtime_params_label.setText(_fmt(fp.get("runtime_param_count")))
            self.diag_storage_params_label.setText(_fmt(fp.get("runtime_param_count")))
        self.diag_pose_params_label.setText(_fmt(fp.get("pose_param_count_train")))

        n_req, n_ok = fp.get("diag_repeated_n_requested"), fp.get("diag_repeated_n_successful")
        if n_req is not None and n_ok is not None:
            self.diag_holdout_label.setText(
                f"{int(n_ok)}/{int(n_req)} successful · "
                f"Mean Test RMS {_fmt(fp.get('diag_repeated_mean_test_rmse'))} "
                f"(±{_fmt(fp.get('diag_repeated_std_test_rmse'))}) · "
                f"Mean P95 {_fmt(fp.get('diag_repeated_mean_test_p95'))} · "
                f"Mean Edge RMS {_fmt(fp.get('diag_repeated_mean_edge_rms'))}"
            )
        else:
            self.diag_holdout_label.setText("N/A")

        ray_mean, ray_p95 = fp.get("diag_ray_stability_mean_deg"), fp.get("diag_ray_stability_p95_deg")
        if ray_mean is not None or ray_p95 is not None:
            self.diag_ray_stability_label.setText(f"Mean {_fmt_deg(ray_mean)} · P95 {_fmt_deg(ray_p95)}")
        else:
            self.diag_ray_stability_label.setText("N/A")

        pose_r_med, pose_r_p95 = fp.get("diag_pose_delta_r_median_deg"), fp.get("diag_pose_delta_r_p95_deg")
        pose_t_med, pose_t_p95 = fp.get("diag_pose_delta_t_median_mm"), fp.get("diag_pose_delta_t_p95_mm")
        if pose_r_med is not None and pose_t_med is not None:
            self.diag_pose_movement_label.setText(
                f"ΔR median {_fmt_deg(pose_r_med)} / P95 {_fmt_deg(pose_r_p95)} · "
                f"Δt median {_fmt(pose_t_med)}mm / P95 {_fmt(pose_t_p95)}mm"
            )
        else:
            self.diag_pose_movement_label.setText("N/A")

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

    def _build_reflection_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        mode_group = QGroupBox("PHOTOMETRIC QUALITY")
        mode_layout = QVBoxLayout(mode_group)
        mode_row = QHBoxLayout()
        self.reflection_reference_radio = QRadioButton("Reference Pair")
        self.reflection_reference_radio.setChecked(True)
        self.reflection_no_reference_radio = QRadioButton("No-Reference")
        mode_row.addWidget(self.reflection_reference_radio)
        mode_row.addWidget(self.reflection_no_reference_radio)
        mode_row.addStretch(1)
        mode_layout.addLayout(mode_row)

        path_form = QFormLayout()
        self.reflection_normal_path_label = QLabel("N/A")
        self.reflection_reference_path_label = QLabel("N/A")
        normal_btn = QPushButton("Load Normal Image...")
        normal_btn.clicked.connect(self._on_load_reflection_normal_image)
        reference_btn = QPushButton("Load Reference Image...")
        reference_btn.clicked.connect(self._on_load_reflection_reference_image)
        normal_row = QHBoxLayout()
        normal_row.addWidget(normal_btn)
        normal_row.addWidget(self.reflection_normal_path_label, stretch=1)
        reference_row = QHBoxLayout()
        reference_row.addWidget(reference_btn)
        reference_row.addWidget(self.reflection_reference_path_label, stretch=1)
        path_form.addRow("Normal:", normal_row)
        path_form.addRow("Reference:", reference_row)

        self.reflection_threshold_spin = QDoubleSpinBox()
        self.reflection_threshold_spin.setRange(0.001, 1.0)
        self.reflection_threshold_spin.setDecimals(3)
        self.reflection_threshold_spin.setSingleStep(0.01)
        self.reflection_threshold_spin.setValue(0.08)
        path_form.addRow("Coverage Threshold:", self.reflection_threshold_spin)
        mode_layout.addLayout(path_form)

        action_row = QHBoxLayout()
        self.reflection_run_button = QPushButton("Run Evaluation")
        self.reflection_run_button.clicked.connect(self._on_run_reflection_evaluation)
        self.reflection_export_button = QPushButton("Export YAML...")
        self.reflection_export_button.setEnabled(False)
        self.reflection_export_button.clicked.connect(self._on_export_reflection_yaml)
        action_row.addWidget(self.reflection_run_button)
        action_row.addWidget(self.reflection_export_button)
        action_row.addStretch(1)
        mode_layout.addLayout(action_row)
        self.reflection_status_label = QLabel("Raw image-domain photometric evaluation. Geometry calibration is not modified.")
        mode_layout.addWidget(self.reflection_status_label)
        layout.addWidget(mode_group)

        self.reflection_metrics_table = _ScrollTable(15, 1)
        self.reflection_metrics_table.setHorizontalHeaderLabels(["Value"])
        self.reflection_metrics_table.setVerticalHeaderLabels([
            "Mode", "Alignment", "Reflection Mean", "Reflection Median", "Reflection P95", "Reflection P99", "Reflection Coverage",
            "Reflection Likelihood",
            "Bottom Mean", "Bottom Coverage", "Contrast Retention",
            "Edge Retention", "Saturation Coverage", "Glare Coverage", "Glare Strength",
        ])
        layout.addWidget(self.reflection_metrics_table)

        self.reflection_spatial_table = _ScrollTable(4, 6)
        self.reflection_spatial_table.setHorizontalHeaderLabels([f"C{c+1}" for c in range(6)])
        self.reflection_spatial_table.setVerticalHeaderLabels([f"R{r+1}" for r in range(4)])
        layout.addWidget(self.reflection_spatial_table)
        layout.addStretch(1)
        return page

    def _on_load_reflection_normal_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Normal Reflection Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self._reflection_normal_path = path
        self.reflection_normal_path_label.setText(path)

    def _on_load_reflection_reference_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Reflection-Reduced Reference Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self._reflection_reference_path = path
        self.reflection_reference_path_label.setText(path)

    def _on_run_reflection_evaluation(self) -> None:
        normal_path = getattr(self, "_reflection_normal_path", "")
        reference_path = getattr(self, "_reflection_reference_path", "")
        if not normal_path:
            QMessageBox.warning(self, "Reflection Evaluation", "Normal image is required.")
            return
        mode = "no_reference" if self.reflection_no_reference_radio.isChecked() else "reference"
        if mode == "reference" and not reference_path:
            QMessageBox.warning(self, "Reflection Evaluation", "Reference mode requires a reflection-reduced reference image.")
            return
        cfg = ReflectionEvaluationConfig(mode=mode, coverage_threshold=float(self.reflection_threshold_spin.value()))
        pair = ReflectionImagePair(normal_image_path=normal_path, reference_image_path=reference_path or None, pair_id="scene_001")
        worker = ReflectionEvaluationWorker([pair], cfg)
        thread = run_worker_in_thread(worker, self)
        self.reflection_run_button.setEnabled(False)
        self.reflection_status_label.setText("Reflection evaluation running...")
        worker.result_ready.connect(self._on_reflection_evaluation_finished)
        worker.error.connect(self._on_reflection_evaluation_error)
        worker.progress.connect(self.reflection_status_label.setText)
        self._reflection_thread, self._reflection_worker = thread, worker
        thread.start()

    def _on_reflection_evaluation_finished(self, result: ReflectionDatasetResult) -> None:
        self._reflection_result = result
        self._reflection_results["latest"] = result
        self._display_reflection_result(result)
        self.reflection_run_button.setEnabled(True)
        self.reflection_export_button.setEnabled(result.success)

    def _on_reflection_evaluation_error(self, message: str) -> None:
        self.reflection_status_label.setText(message)
        self.reflection_run_button.setEnabled(True)
        QMessageBox.critical(self, "Reflection Evaluation", message)

    def _display_reflection_result(self, dataset_result: ReflectionDatasetResult) -> None:
        result = dataset_result.pair_results[0] if dataset_result.pair_results else None
        if result is None:
            self.reflection_status_label.setText(dataset_result.error_message or "No reflection result.")
            return
        status_suffix = (
            "Reflection Likelihood: no-reference heuristic, not ground truth."
            if result.mode == "no_reference"
            else "Reference reflection evaluation after alignment and photometric normalization."
        )
        self.reflection_status_label.setText(f"Metric v{result.metric_version} | {status_suffix}")
        values = [
            result.mode,
            f"{result.alignment_status} ({_fmt(result.alignment_score)})",
            f"{result.reflection_mean * 100.0:.2f}%" if result.reflection_mean is not None else "N/A",
            f"{result.reflection_median * 100.0:.2f}%" if result.reflection_median is not None else "N/A",
            f"{result.reflection_p95 * 100.0:.2f}%" if result.reflection_p95 is not None else "N/A",
            f"{result.reflection_p99 * 100.0:.2f}%" if result.reflection_p99 is not None else "N/A",
            f"{result.reflection_coverage * 100.0:.2f}%" if result.reflection_coverage is not None else "N/A",
            f"{result.reflection_likelihood * 100.0:.2f}%" if result.reflection_likelihood is not None else "N/A",
            f"{(result.bottom_roi_mean_strength or 0.0) * 100.0:.2f}%",
            f"{(result.bottom_roi_coverage or 0.0) * 100.0:.2f}%",
            f"{result.contrast_retention * 100.0:.2f}%" if result.contrast_retention is not None else "N/A",
            f"{result.edge_retention * 100.0:.2f}%" if result.edge_retention is not None else "N/A",
            f"{result.saturation_coverage * 100.0:.2f}%",
            f"{(result.glare_coverage or 0.0) * 100.0:.2f}%",
            f"{(result.glare_strength or 0.0) * 100.0:.2f}%",
        ]
        for row, value in enumerate(values):
            self.reflection_metrics_table.setItem(row, 0, QTableWidgetItem(value))
        _fit_table_to_rows(self.reflection_metrics_table)

        for cell in result.spatial_map:
            if cell.row < self.reflection_spatial_table.rowCount() and cell.col < self.reflection_spatial_table.columnCount():
                self.reflection_spatial_table.setItem(cell.row, cell.col, QTableWidgetItem(f"{cell.mean_strength * 100.0:.1f}%"))
        _fit_table_to_rows(self.reflection_spatial_table)

    def _on_export_reflection_yaml(self) -> None:
        if self._reflection_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Reflection YAML 저장", "reflection_evaluation.yml", "YAML (*.yml *.yaml)")
        if not path:
            return
        try:
            export_reflection_yaml(self._reflection_result, path)
            self.reflection_status_label.setText(f"Reflection YAML saved: {path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Reflection Export", str(e))

    def _refresh_comparison_table(self) -> None:
        order: list[WindshieldResultKey] = [
            WindshieldModelType.BASELINE,
            WindshieldModelType.SPHERICAL,
            windshield_result_key(WindshieldModelType.RESIDUAL_RAY, "grid"),
            windshield_result_key(WindshieldModelType.RESIDUAL_RAY, "rbf"),
            windshield_result_key(WindshieldModelType.RESIDUAL_RAY, "neural"),
            WindshieldModelType.SPLINE,
        ]
        present = [m for m in order if m in self._windshield_results]
        self.comparison_table.setColumnCount(len(present))
        self.comparison_table.setHorizontalHeaderLabels([windshield_result_key_label(m) for m in present])

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
