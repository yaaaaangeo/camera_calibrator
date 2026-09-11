"""
camera_calibrator.ui.windshield_workspace
==============================================

`WindshieldWorkspace` = Windshield Calibration 전용 최상위 UI
orchestration(사용자 스펙 4/5/24번). Camera Intrinsic Calibration
(ui/intrinsic_workspace.py, ui/main_window.py)과 완전히 분리된 화면이다 -
Base Camera Model(K,D)은 여기서 절대 재보정하지 않는다. 이미 확정된 값을
"고정"으로 불러와 표시만 한다("🔒 Base K,D fixed").

이 Workspace가 다루는 두 축(절대 하나로 합치지 않는다 - 서로 다른 문제다):

    Geometry(기하 보정 - Windshield Model 탭)
        Baseline / Spherical / Residual Grid / Residual RBF /
        Neural Residual / Spline

    Photometric(광학적 아티팩트 평가/억제 - 별도 탭)
        Reflection Evaluation / Reflection Suppression
        Ghost Evaluation / Ghost Suppression

실제 계산 알고리즘(굴절 모델 피팅, Reflection/Ghost 지표 계산, Suppression
재구성 등)은 전부 `calibration/windshield/*`에 있다 - 이 파일과 아래 패널
파일들은 계산을 하지 않는다. Workspace/패널이 하는 일은:

    입력을 모은다(파일 선택, 파라미터 spinbox/radio) →
    Worker(QThread)를 만들어 계산을 위임한다 →
    Worker가 돌려준 Result 객체를 받아 표/차트/이미지로 그린다.

탭 구성(사용자 스펙 5/6번 UI 목업의 6단계를 6개 탭으로 그대로 반영):

    ① Base Camera   - 이미 확정된 K,D를 불러와 고정 표시
    ② Dataset       - Windshield 캘리브레이션용 이미지 검출
    ③ Windshield Model - Geometry 모델 선택/실행/결과(Train/Test 나란히 표시)
    ④ Comparison    - Geometry 모델들끼리의 Hold-out 비교표
    ⑤ Reflection    - Evaluation / Suppression sub-tab
    ⑥ Ghost         - Evaluation / Suppression sub-tab

Priority 7 안정화 - God Object 분리:
이 파일은 원래 ~2430줄짜리 단일 클래스(`WindshieldWorkspace`)에 위 6개 탭
UI가 전부 들어있었다. 계산 로직은 하나도 바꾸지 않고, 각 탭의 UI 코드만
아래처럼 mixin 클래스 파일로 옮겼다:

    ui/windshield_common.py                       - 공유 상수/헬퍼(순환 import 방지용)
    ui/windshield_geometry_panel.py                - ③ Windshield Model 탭(Geometry)
    ui/windshield_comparison_panel.py              - ④ Comparison 탭(Geometry)
    ui/windshield_reflection_panel.py              - ⑤ Reflection Evaluation sub-tab(Photometric)
    ui/windshield_reflection_suppression_panel.py  - ⑤ Reflection Suppression sub-tab(Photometric)
    ui/windshield_ghost_panel.py                   - ⑥ Ghost Evaluation sub-tab(Photometric)
    ui/windshield_ghost_suppression_panel.py       - ⑥ Ghost Suppression sub-tab(Photometric)

`WindshieldWorkspace`는 이 mixin들을 전부 다중 상속해서 하나의 `self`
(같은 QWidget 인스턴스)를 공유한다 - 어느 mixin에 정의된 메서드/속성이든
`self.xxx`로 그대로 접근되므로(Python MRO), 파일이 나뉘어도 기존 동작이
전혀 바뀌지 않는다. 이 클래스 자체는 orchestration(상태 초기화, 탭 구성,
MainWindow 연동 API: `load_base_from_calibration_results`/`import_state`/
`export_state`)과 Base Camera/Dataset(①/②) 탭만 직접 담당한다 - 이 두
탭은 다른 모든 탭이 의존하는 "입력을 불러오는" 역할이라 orchestration에
가깝다고 보고 별도 파일로 빼지 않았다.

`ui/main_window.py`가 하던 `from ui.windshield_workspace import
WindshieldWorkspace` 임포트는 전혀 바뀌지 않는다 - 파일 경로, 클래스 이름,
공개 메서드 시그니처 모두 그대로다.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from calibration.detector import detect_dataset
from calibration.library import list_cameras, list_runs, load_run_project
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
)
from calibration.windshield.reflection import ReflectionDatasetResult, ReflectionImagePair
from calibration.windshield.ghost import GhostDatasetResult, GhostField
from export.opencv import (
    detect_model_hint_from_opencv_yaml,
    load_camera_matrix_and_distortion_from_opencv_yaml,
)
from ui.theme import Theme
from ui.windshield_common import ResponsiveRow, _MODEL_LABELS, make_scrollable_page
from ui.windshield_comparison_panel import ComparisonPanelMixin
from ui.windshield_geometry_panel import GeometryPanelMixin
from ui.windshield_ghost_panel import GhostPanelMixin
from ui.windshield_ghost_suppression_panel import GhostSuppressionPanelMixin
from ui.windshield_reflection_panel import ReflectionPanelMixin
from ui.windshield_reflection_suppression_panel import ReflectionSuppressionPanelMixin


def _camera_model_label(model: CameraModelType | str) -> str:
    try:
        enum_model = CameraModelType(model)
    except ValueError:
        return str(model)
    return _MODEL_LABELS.get(enum_model, enum_model.value)


def _camera_model_from_key_or_result(
    model: CameraModelType | str,
    result: CalibrationResult | None,
) -> CameraModelType | str:
    try:
        return CameraModelType(model)
    except ValueError:
        if result is not None:
            try:
                return CameraModelType(result.model_name)
            except ValueError:
                pass
        return model


class WindshieldWorkspace(
    GeometryPanelMixin,
    ComparisonPanelMixin,
    ReflectionPanelMixin,
    ReflectionSuppressionPanelMixin,
    GhostPanelMixin,
    GhostSuppressionPanelMixin,
    QWidget,
):
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
        # Ghost / Double Image(STEP 8) - Reflection과 완전히 별도 상태다
        # (사용자 스펙 1번, "Ghost는 기존 Reflection 기능 안에 넣지 않는다").
        self._ghost_results: dict[str, GhostDatasetResult] = {}
        self._ghost_result: GhostDatasetResult | None = None
        self._ghost_image_path = ""
        self._ghost_dataset_dir = ""
        # ghost_models(STEP 8 stabilization 4번) - Fit/Load된 GhostField를
        # 여기 등록해야 project save 시 `.ccproj`에 실제로 저장된다. YAML
        # export("Save Model...")와는 별개다(4-F번) - YAML 저장을 누르지
        # 않아도 fit/load된 모델은 project 안에 남아 있어야 한다.
        self._ghost_models: dict[str, GhostField] = {}
        self._ghost_suppression_model_path = ""
        self._ghost_suppression_input_path = ""
        self._ghost_suppression_result = None
        # 마지막으로 화면에 표시된(=Export 대상) 모델 - export_button과
        # _on_export_windshield_yaml이 특정 모델(예: Baseline)에 고정되지
        # 않고 "방금 실행/표시한 결과"를 export하도록 추적한다.
        self._current_displayed_model: WindshieldResultKey | None = None

        # MainWindow가 load_base_from_calibration_results()로 넘겨주는,
        # 현재 세션에서 이미 계산된 Standard 4모델 결과 (Base Camera 탭의
        # "Load from current session" 버튼이 여기서 고른다).
        self._session_calibration_results: dict[CameraModelType | str, CalibrationResult] = {}
        self._session_camera_config: CameraConfig | None = None
        self._session_pattern_config: PatternConfig | None = None

        layout = QVBoxLayout(self)
        header = ResponsiveRow(breakpoint=520)
        self.home_button = QPushButton("← Calibration Home")
        self.home_button.clicked.connect(self.back_requested.emit)
        header.addWidget(self.home_button)
        self.windshield_guide_button = QPushButton("? Windshield Guide")
        self.windshield_guide_button.setToolTip("Windshield Refraction 초보자 작업 매뉴얼을 엽니다.")
        self.windshield_guide_button.clicked.connect(self._show_windshield_guide)
        header.addWidget(self.windshield_guide_button)
        header.addStretch(1)
        layout.addWidget(header)

        title = QLabel("WINDSHIELD REFRACTION CALIBRATION")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(title)
        subtitle = QLabel(
            "앞유리 굴절로 생기는 기하학적(geometric) 픽셀 변위는 Windshield Model 탭에서, "
            "Reflection(글레어)과 Ghost(이중상)는 각각 별도의 Photometric 탭에서 다룹니다."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color: {Theme.TEXT_SECONDARY};")
        layout.addWidget(subtitle)

        self.tabs = QTabWidget()
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.tabs.tabBar().setExpanding(False)
        self.tabs.setElideMode(Qt.ElideNone)
        self.tabs.addTab(make_scrollable_page(self._build_base_camera_tab()), "① Base Camera")
        self.tabs.addTab(make_scrollable_page(self._build_dataset_tab()), "② Dataset")
        self.tabs.addTab(make_scrollable_page(self._build_model_tab()), "③ Windshield Model")
        self.tabs.addTab(make_scrollable_page(self._build_comparison_tab()), "④ Comparison")
        self.tabs.addTab(self._build_reflection_tab(), "⑤ Reflection")
        self.tabs.addTab(self._build_ghost_tab(), "⑥ Ghost")
        layout.addWidget(self.tabs, stretch=1)

    def _show_windshield_guide(self) -> None:
        # Keep help rendering in ui.help_view instead of creating a second help
        # system inside the workspace.
        from ui.help_view import WindshieldGuideDialog

        dialog = WindshieldGuideDialog(self)
        dialog.exec()

    # ------------------------------------------------------------------
    # MainWindow 연동 API
    # ------------------------------------------------------------------
    def load_base_from_calibration_results(
        self,
        calibration_results: dict[CameraModelType | str, CalibrationResult],
        camera_config: CameraConfig | None,
        pattern_config: PatternConfig | None,
    ) -> None:
        """MainWindow가 Home -> Windshield Refraction 진입 시 호출.
        현재 세션에서 이미 계산된 결과를 "Load from current session" 버튼으로
        바로 쓸 수 있게 후보로만 등록한다 - 여기서 자동으로 Base를 확정하지는
        않는다(사용자가 명시적으로 모델을 선택해야 함)."""
        self._session_calibration_results = {
            _camera_model_from_key_or_result(model, result): result
            for model, result in (calibration_results or {}).items()
        }
        self._session_camera_config = camera_config
        self._session_pattern_config = pattern_config

    def import_state(self, project: CalibrationProject) -> None:
        """프로젝트 로드 시 Windshield 상태를 복원한다."""
        self._windshield_config = project.windshield_config
        self._windshield_dataset = project.windshield_dataset
        self._windshield_results = dict(project.windshield_results or {})
        self._reflection_results = dict(getattr(project, "reflection_results", {}) or {})
        self._reflection_result = next(iter(self._reflection_results.values()), None)
        self._ghost_results = dict(getattr(project, "ghost_results", {}) or {})
        self._ghost_result = next(iter(self._ghost_results.values()), None)
        self._ghost_models = dict(getattr(project, "ghost_models", {}) or {})
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
        if self._ghost_result is not None:
            self._display_ghost_result(self._ghost_result)

    def export_state(
        self,
    ) -> tuple[
        WindshieldConfig | None,
        Dataset | None,
        dict[WindshieldResultKey, WindshieldCalibrationResult],
        dict[str, ReflectionDatasetResult],
        dict[str, GhostDatasetResult],
        dict[str, GhostField],
    ]:
        return self._windshield_config, self._windshield_dataset, self._windshield_results, self._reflection_results, self._ghost_results, self._ghost_models

    # ------------------------------------------------------------------
    # ① Base Camera
    # ------------------------------------------------------------------
    def _build_base_camera_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        button_row = ResponsiveRow(breakpoint=820)
        self.base_load_buttons = []
        for text, handler in (
            ("Load from current session", self._on_load_from_session),
            ("Load from Library...", self._on_load_from_library),
            ("Load OpenCV YAML...", self._on_load_from_yaml),
            ("Load .ccproj...", self._on_load_from_ccproj),
        ):
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            button_row.addWidget(btn)
            self.base_load_buttons.append(btn)
        button_row.addStretch(1)
        layout.addWidget(button_row)

        group = QGroupBox("Base Camera")
        form = QVBoxLayout(group)
        self.base_info_label = QLabel("아직 Base Camera를 불러오지 않았습니다.")
        self.base_info_label.setWordWrap(True)
        form.addWidget(self.base_info_label)
        self.base_lock_label = QLabel("")
        self.base_lock_label.setWordWrap(True)
        self.base_lock_label.setStyleSheet(f"color: {Theme.WARNING}; font-weight: 700; font-size: 14px;")
        form.addWidget(self.base_lock_label)
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _pick_calibration_result(
        self, calibration_results: dict[CameraModelType | str, CalibrationResult]
    ) -> CalibrationResult | None:
        candidates = {
            m: r for m, r in calibration_results.items()
            if r and r.success and r.camera_matrix is not None and r.distortion is not None
        }
        if not candidates:
            QMessageBox.warning(self, "Base Camera", "사용 가능한 (성공한) Calibration 결과가 없습니다.")
            return None
        labels = [_camera_model_label(m) for m in candidates]
        label_to_model = {_camera_model_label(m): m for m in candidates}
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
            base_model_name=CameraModelType(calibration_result.model_name),
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
            f"Camera Model : {_camera_model_label(cfg.base_model_name)}\n"
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

        button_row = ResponsiveRow(breakpoint=520)
        self.load_dataset_button = QPushButton("Load windshield images...")
        self.load_dataset_button.clicked.connect(self._on_load_dataset)
        button_row.addWidget(self.load_dataset_button)
        button_row.addStretch(1)
        layout.addWidget(button_row)

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
