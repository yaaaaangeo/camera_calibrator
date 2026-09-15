"""
camera_calibrator.ui.main_window
====================================

설계 문서 14번 UI 구성안 + 16번 폴더 구조를 따른다.
이 파일은 "조립"만 한다 - 검출/캘리브레이션/추천/이상치 계산은 전부
calibration/*.py에 있고, 여기서는 그 함수들을 worker.py를 통해 호출하고
결과를 각 view 위젯에 그대로 전달할 뿐이다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from PySide6.QtCore import QThread, QTimer, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.types import (
    AprilGridVariant,
    CalibrationMethod,
    CalibrationResult,
    CalibrationProject,
    CameraConfig,
    CameraModelType,
    CircleGridType,
    CrossDatasetValidationResult,
    Dataset,
    ModelScore,
    PatternConfig,
    PatternType,
    ValidationResult,
    OptimizerResult,
)
from calibration.sanity_check import run_sanity_checks
from calibration.optimizer import apply_optimized_calibration, restore_original_calibration
from calibration.recommender import compute_final_result
from calibration.ros_live import ROS_LIVE_BACKEND
from calibration.project_io import load_project, save_project, PROJECT_EXTENSION
import calibration.paper_evidence as paper_evidence
from export.opencv import export_opencv_yaml
from export.output_manager import DEFAULT_OUTPUT_ROOT, OutputManager, model_filename
from export.csv_export import export_csv
from export.json_export import export_json
from export.reflection import export_reflection_yaml
from export.report import export_html_report
from export.ros import export_ros_camera_info
from export.windshield import export_windshield_yaml
from calibration.image_selection import write_selection_manifest
from calibration.windshield.base import windshield_result_key_for_result
from calibration.windshield.ghost import save_ghost_model

from ui.calibration_home_view import CalibrationHomeView
from ui.help_view import HelpView
from ui.intrinsic_workspace import IntrinsicWorkspace
from ui.live_capture_dialog import LiveCaptureDialog
from ui.windshield_common import ResponsiveRow, configure_form_layout
from ui.wheel_guard import WheelChangeGuard
from ui.worker import (
    PipelineWorker,
    ImageSelectionWorker,
    CrossDatasetValidationWorker,
    SelfCheckWorker,
    BagTopicDiscoveryWorker,
    BagExtractionWorker,
    LibrarySaveWorker,
    SceneSubsetCalibrationWorker,
    SubsetComparisonWorker,
    OptimizerWorker,
    run_worker_in_thread,
)
from ui.kfold_worker import RepeatedKFoldWorker
from calibration.scene_quality import add_original_comparison_warnings, compute_scene_quality_analysis
from calibration.subset_comparison import ComparisonTolerance
from ui.library_view import LibraryView
from ui.windshield_workspace import WindshieldWorkspace

logger = logging.getLogger(__name__)

# 자동 저장 파일 경로 - 프로젝트 폴더가 아니라 홈 디렉터리 밑 고정 위치에 둔다.
# 앱이 응답 없음/강제 종료로 죽어도 다음 실행에서 항상 같은 경로를 확인해
# 복구를 제안할 수 있어야 하기 때문 (사용자가 저장 위치를 고를 필요 없음).
_AUTOSAVE_DIR = Path.home() / ".camera_calibrator"
_AUTOSAVE_PATH = _AUTOSAVE_DIR / "autosave.ccproj"

_SHEEP_TRACK_LEN = 14  # 진행률 불명 구간에서 양이 걸어가는 트랙의 칸 수

# ChArUco에서 흔히 쓰이는 사전 목록 (cv2.aruco.DICT_* 속성명 그대로)
_ARUCO_DICTIONARIES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
]
_IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


class MainWindow(QMainWindow):
    @property
    def image_paths(self):
        return self.intrinsic_state.image_paths

    @image_paths.setter
    def image_paths(self, value):
        self.intrinsic_state.image_paths = value

    @property
    def dataset(self):
        return self.intrinsic_state.dataset

    @dataset.setter
    def dataset(self, value):
        self.intrinsic_state.dataset = value

    @property
    def camera_config(self):
        return self.intrinsic_state.camera_config

    @camera_config.setter
    def camera_config(self, value):
        self.intrinsic_state.camera_config = value

    @property
    def pattern_config(self):
        return self.intrinsic_state.pattern_config

    @pattern_config.setter
    def pattern_config(self, value):
        self.intrinsic_state.pattern_config = value

    @property
    def calibration_results(self):
        return self.intrinsic_state.calibration_results

    @calibration_results.setter
    def calibration_results(self, value):
        self.intrinsic_state.calibration_results = value

    @property
    def object_releasing_result(self):
        return self.intrinsic_state.object_releasing_result

    @object_releasing_result.setter
    def object_releasing_result(self, value):
        self.intrinsic_state.object_releasing_result = value

    @property
    def object_releasing_validation_result(self):
        return self.intrinsic_state.object_releasing_validation_result

    @object_releasing_validation_result.setter
    def object_releasing_validation_result(self, value):
        self.intrinsic_state.object_releasing_validation_result = value

    @property
    def standard_vs_object_releasing_comparison(self):
        return self.intrinsic_state.standard_vs_object_releasing_comparison

    @standard_vs_object_releasing_comparison.setter
    def standard_vs_object_releasing_comparison(self, value):
        self.intrinsic_state.standard_vs_object_releasing_comparison = value

    @property
    def validation_results(self):
        return self.intrinsic_state.validation_results

    @validation_results.setter
    def validation_results(self, value):
        self.intrinsic_state.validation_results = value

    @property
    def cross_dataset_results(self):
        return self.intrinsic_state.cross_dataset_results

    @cross_dataset_results.setter
    def cross_dataset_results(self, value):
        self.intrinsic_state.cross_dataset_results = value

    @property
    def scores(self):
        return self.intrinsic_state.scores

    @scores.setter
    def scores(self, value):
        self.intrinsic_state.scores = value

    @property
    def outlier_result(self):
        return self.intrinsic_state.outlier_result

    @outlier_result.setter
    def outlier_result(self, value):
        self.intrinsic_state.outlier_result = value

    @property
    def repeated_kfold_results(self):
        return self.intrinsic_state.repeated_kfold_results

    @repeated_kfold_results.setter
    def repeated_kfold_results(self, value):
        self.intrinsic_state.repeated_kfold_results = value

    @property
    def calibration_method(self):
        return self.intrinsic_state.calibration_method

    @calibration_method.setter
    def calibration_method(self, value):
        self.intrinsic_state.calibration_method = value

    @property
    def scene_quality_analysis(self):
        return self.intrinsic_state.scene_quality_analysis

    @scene_quality_analysis.setter
    def scene_quality_analysis(self, value):
        self.intrinsic_state.scene_quality_analysis = value

    @property
    def subset_calibration_result(self):
        return self.intrinsic_state.subset_calibration_result

    @subset_calibration_result.setter
    def subset_calibration_result(self, value):
        self.intrinsic_state.subset_calibration_result = value

    @property
    def optimizer_results(self):
        return self.intrinsic_state.optimizer_results

    @optimizer_results.setter
    def optimizer_results(self, value):
        self.intrinsic_state.optimizer_results = value

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calibration Tool")
        # Qt 기본 동작은 마우스 포인터만 spinbox/combo/tab 위에 있어도 휠로
        # 값/선택 탭을 바꾼다. 앱 전역 필터로 우발 변경을 차단한다.
        self._wheel_change_guard = WheelChangeGuard(self)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self._wheel_change_guard)
        # 예전엔 화면 크기와 무관하게 무조건 1280x860으로 고정 리사이즈했다 -
        # 실사용자 버그: 화면(또는 사용 가능 영역, 예를 들어 작업표시줄/독을 뺀
        # 영역)이 860px보다 낮으면 창 아래쪽(탭 내용, 버튼 등)이 화면 밖으로
        # 잘려 나갔다. 항상 화면의 "사용 가능한 영역"(available geometry -
        # 작업표시줄 등을 제외한 실제로 창을 놓을 수 있는 크기)을 기준으로
        # 최대 1280x860, 최소한 화면의 90%까지는 채우도록 계산한다.
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(1280, available.width())
            height = min(860, available.height())
            # 화면이 작아도 UI가 너무 쪼그라들지 않게 최소 크기는 유지하되,
            # 그 최소 크기가 사용 가능 영역보다 크면(아주 작은 화면) 영역에 맞춘다.
            width = max(width, min(960, available.width()))
            height = max(height, min(640, available.height()))
            self.resize(width, height)
            # 창이 화면 밖으로 나가지 않도록 사용 가능 영역 안쪽에 위치시킨다.
            self.move(
                available.x() + max(0, (available.width() - width) // 2),
                available.y() + max(0, (available.height() - height) // 2),
            )
        else:
            self.resize(1280, 860)
        self.setMinimumSize(800, 600)

        # --- 상태 ---
        IntrinsicWorkspace.initialize_owner_state(self)
        self._thread: QThread | None = None
        self._worker = None  # QThread가 살아있는 동안 GC 방지용 강한 참조
        self._self_check_thread: QThread | None = None
        self._selection_thread: QThread | None = None
        self._selection_worker = None
        self._bag_thread: QThread | None = None
        self._bag_worker = None  # QThread가 살아있는 동안 GC 방지용 강한 참조
        self._bag_progress_dialog: QProgressDialog | None = None
        self._bag_topic_thread: QThread | None = None
        self._bag_topic_worker = None
        self._bag_topic_progress_dialog: QProgressDialog | None = None
        self._self_check_worker = None  # 위와 동일한 이유로 별도 워커도 강한 참조 보관
        self._library_thread: QThread | None = None
        self._library_worker = None
        self._optimizer_thread: QThread | None = None
        self._optimizer_worker = None
        self._export_dialog: QDialog | None = None  # result_view가 생긴 뒤 지연 생성
        # 사용자 결과물은 이 관리자 아래 한 세션으로 모인다. 앱 크래시 복구용
        # ~/.camera_calibrator/autosave.ccproj와 Library 내부 저장은 의도적으로
        # 별도 수명주기이므로 여기로 합치지 않는다.
        self.output_manager = OutputManager()

        self._build_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.workspace_stack = QStackedWidget()
        self.home_view = CalibrationHomeView()
        settings_panel = self._build_settings_panel()

        self.intrinsic_workspace = IntrinsicWorkspace.create_for_main_window(self, settings_panel)
        self.intrinsic_workspace.back_requested.connect(self._show_home)
        self.library_view = LibraryView()
        self.windshield_workspace = WindshieldWorkspace()
        self.windshield_workspace.set_output_manager(self.output_manager)
        self.windshield_workspace.back_requested.connect(self._show_home)
        self.workspace_stack.addWidget(self.home_view)
        self.workspace_stack.addWidget(self.intrinsic_workspace)
        self.workspace_stack.addWidget(self.library_view)
        self.workspace_stack.addWidget(self.windshield_workspace)
        layout.addWidget(self.workspace_stack, stretch=1)

        self.status_label = QLabel("이미지를 불러온 뒤 [캘리브레이션 실행]을 누르세요.")
        self.statusBar().addWidget(self.status_label, stretch=1)
        self.pipeline_progress_bar = QProgressBar()
        self.pipeline_progress_bar.setMinimumWidth(180)
        self.pipeline_progress_bar.setTextVisible(True)
        self.pipeline_progress_bar.hide()
        self.statusBar().addPermanentWidget(self.pipeline_progress_bar)
        # 진행률을 알 수 없는 구간(Standard 4모델 + Hold-out 계산 중)에서 기본 Qt
        # 인디케이터 대신 양이 진행 바를 가로질러 걸어가는 애니메이션을 보여준다.
        self._sheep_pos = 0
        self._sheep_timer = QTimer(self)
        self._sheep_timer.setInterval(140)
        self._sheep_timer.timeout.connect(self._advance_busy_sheep)

        self.home_view.intrinsic_requested.connect(self._show_intrinsic_workspace)
        self.home_view.library_requested.connect(self._show_library_workspace)
        self.home_view.windshield_requested.connect(self._show_windshield_workspace)
        self.library_view.back_requested.connect(self._show_home)

        # 앱이 응답 없음/강제 종료 등으로 꺼져도 마지막으로 완료된 계산
        # 결과는 자동 저장본에서 복구할 수 있게, 창이 뜨자마자 한 번 확인한다.
        # (실제 사용자 버그: 큰 rosbag을 불러오다 응답 없음이 떠서 강제 종료한
        # 뒤 다시 켜면 방금까지 보이던 계산 결과가 전부 사라져 있었음 -
        # 저장하지 않은 결과는 메모리에만 있어서 프로세스가 죽으면 없어지는
        # 게 원인이었다. 매 실행 완료 시 자동 저장해두면 이런 경우에도
        # Export만큼은 다시 할 수 있다.)
        QTimer.singleShot(0, self._offer_autosave_recovery)

    def _show_home(self) -> None:
        self.workspace_stack.setCurrentWidget(self.home_view)
        self.status_label.setText("Calibration Type을 선택하세요.")

    def _show_intrinsic_workspace(self) -> None:
        self.workspace_stack.setCurrentWidget(self.intrinsic_workspace)
        self.status_label.setText("Camera Intrinsic Workspace")

    def _show_library_workspace(self) -> None:
        self.workspace_stack.setCurrentWidget(self.library_view)
        self.status_label.setText("Library")

    def _show_windshield_workspace(self) -> None:
        self.windshield_workspace.load_base_from_calibration_results(
            self.calibration_results, self.camera_config, self.pattern_config
        )
        self.workspace_stack.setCurrentWidget(self.windshield_workspace)
        self.status_label.setText("Windshield Refraction Workspace")

    # ------------------------------------------------------------------
    # 설정 패널 (설계 문서 14번 ① Camera Setup, ③ Calibration Pattern)
    # ------------------------------------------------------------------

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("파일")

        save_action = QAction("프로젝트 저장", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save_project_session)
        file_menu.addAction(save_action)

        save_as_action = QAction("프로젝트 다른 이름으로 저장...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._on_save_project_as)
        file_menu.addAction(save_as_action)

        load_action = QAction("프로젝트 불러오기...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._on_load_project)
        file_menu.addAction(load_action)

        help_menu = menu_bar.addMenu("설명")
        guide_action = QAction("사용 설명서 열기", self)
        guide_action.setShortcut("F1")
        guide_action.triggered.connect(self._on_show_help)
        help_menu.addAction(guide_action)

        tools_menu = menu_bar.addMenu("도구")
        self.self_check_action = QAction("자체 진단 (합성 데이터로 정확도 확인)...", self)
        self.self_check_action.setToolTip(
            "정답을 미리 아는 가짜(합성) ChArUco 데이터로 Ideal Pinhole/Brown-Conrady/\n"
            "Rational/Fisheye 모델을 돌려서 복원된 fx/fy/cx/cy가 정답에 가까운지 확인합니다.\n"
            "현재 불러온 이미지/캘리브레이션 결과와는 무관하며, 몇 초~수십 초 걸립니다."
        )
        self.self_check_action.triggered.connect(self._on_run_self_check)
        tools_menu.addAction(self.self_check_action)

    def _on_show_help(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Camera Calibration Tool 사용 설명서")
        dialog.resize(980, 720)
        layout = QVBoxLayout(dialog)
        layout.addWidget(HelpView(dialog))
        dialog.exec()

    def _build_output_session_bar(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        self.output_session_label = QLineEdit(f"Output Session: {DEFAULT_OUTPUT_ROOT}")
        self.output_session_label.setReadOnly(True)
        row.addWidget(self.output_session_label, stretch=1)
        change_button = QPushButton("폴더 변경…")
        change_button.clicked.connect(self._on_change_output_root)
        row.addWidget(change_button)
        open_button = QPushButton("폴더 열기")
        open_button.clicked.connect(self._on_open_output_folder)
        row.addWidget(open_button)
        export_all_button = QPushButton("Export All")
        export_all_button.clicked.connect(self._on_export_all)
        row.addWidget(export_all_button)
        return bar

    def _session_camera_name(self) -> str:
        config = self.camera_config
        if config is not None and config.sensor_name:
            return config.sensor_name
        edit = getattr(self, "sensor_name_edit", None)
        return edit.text().strip() if edit is not None else ""

    def _ensure_output_session(self) -> Path:
        session = self.output_manager.ensure_session(self._session_camera_name())
        self._update_output_manifest_context()
        self.output_session_label.setText(f"Output Session: {session}")
        return session

    def _update_output_manifest_context(self) -> None:
        if not self.output_manager.active:
            return
        camera = self.camera_config
        pattern = self.pattern_config
        successful = [
            getattr(getattr(result, "model_name", model), "value", str(model))
            for model, result in self.calibration_results.items()
            if getattr(result, "success", False)
        ]
        recommended = next(
            (
                getattr(score.model_name, "value", str(score.model_name))
                for score in self.scores
                if getattr(score, "is_recommended", False)
            ),
            None,
        )
        selected = recommended
        if hasattr(self, "result_view"):
            combo = getattr(self.result_view, "export_model_combo", None)
            if combo is not None:
                data = combo.currentData()
                selected = getattr(data, "value", data) or selected
        self.output_manager.update_context(
            camera={
                "name": getattr(camera, "sensor_name", "") if camera else "",
                "resolution": [getattr(camera, "width", 0), getattr(camera, "height", 0)] if camera else None,
            },
            pattern={
                "type": getattr(getattr(pattern, "type", None), "value", None),
                "squares": [getattr(pattern, "squares_x", 0), getattr(pattern, "squares_y", 0)] if pattern else None,
                "square_size_m": getattr(pattern, "square_size", None),
                "marker_size_m": getattr(pattern, "marker_size", None),
                "dictionary": getattr(pattern, "dictionary", None),
            },
            calibration={
                "available_models": successful,
                "selected_model": selected,
                "recommended_model": recommended,
            },
            paper_metrics_available=bool(self.repeated_kfold_results),
        )

    def _on_change_output_root(self) -> None:
        root = QFileDialog.getExistingDirectory(
            self, "Output Root 선택", str(self.output_manager.output_root)
        )
        if not root:
            return
        self.output_manager = OutputManager(root)
        self.windshield_workspace.set_output_manager(self.output_manager)
        session = self._ensure_output_session()
        self.status_label.setText(f"Output Session 생성: {session}")

    def _on_open_output_folder(self) -> None:
        folder = self._ensure_output_session()
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            QMessageBox.warning(self, "폴더 열기 실패", str(folder))

    def _on_export_all(self) -> None:
        """Export every currently available artifact, isolating each failure."""
        session = self._ensure_output_session()
        exported: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []

        def attempt(label: str, available: bool, operation) -> None:
            outcome = self.output_manager.execute_export(label, operation if available else None)
            if outcome["status"] == "skipped":
                skipped.append(label)
                return
            if outcome["status"] == "exported":
                exported.append(label)
                return
            logger.error("Export All: %s failed: %s", label, outcome["error"])
            failed.append(f"{label}: {outcome['error']}")

        configs_ready = self.camera_config is not None and self.pattern_config is not None
        successful = {
            model: result for model, result in self.calibration_results.items()
            if getattr(result, "success", False)
        }
        for model, result in successful.items():
            def export_intrinsic(model=model, result=result):
                path = self.output_manager.intrinsic_path(model)
                source = "optimized" if (
                    self.optimizer_results.get(model) and self.optimizer_results[model].applied
                ) else "original_opencv"
                return export_opencv_yaml(
                    result, self.camera_config, self.pattern_config, str(path),
                    calibration_source=source,
                )
            attempt(f"intrinsic.{model_filename(model)}", configs_ready, export_intrinsic)

        subset = self.subset_calibration_result
        subset_result = getattr(subset, "calibration_result", None)
        def export_subset():
            path = self.output_manager.intrinsic_path(subset_result.model_name, subset=True)
            return export_opencv_yaml(
                subset_result, self.camera_config, self.pattern_config, str(path),
                calibration_source="best_subset",
                selected_frame_ids=subset.selected_frame_ids,
            )
        attempt(
            "intrinsic.subset",
            configs_ready and subset_result is not None and getattr(subset_result, "success", False),
            export_subset,
        )

        attempt(
            "project",
            self.dataset is not None and configs_ready,
            lambda: self._save_project_to_path(
                str(self.output_manager.project_path()), record=False, notify=False
            ),
        )

        chosen = next(
            (score.model_name for score in self.scores if getattr(score, "is_recommended", False)),
            next(iter(successful), None),
        )
        if chosen is not None and self.dataset is not None and configs_ready:
            final = compute_final_result(
                CameraModelType(chosen), self.calibration_results, self.validation_results,
                outlier_result=self.outlier_result, scores=self.scores,
            )
            attempt(
                "reports.html", True,
                lambda: export_html_report(
                    self._session_camera_name() or "camera_calibrator",
                    self.camera_config, self.pattern_config, self.dataset,
                    self.calibration_results, self.validation_results, final,
                    str(self.output_manager.report_path("report.html")),
                    cross_dataset_results=self.cross_dataset_results,
                ),
            )
            attempt(
                "reports.json", True,
                lambda: export_json(
                    self.camera_config, self.pattern_config, self.dataset,
                    self.calibration_results, self.validation_results, CameraModelType(chosen),
                    str(self.output_manager.report_path("calibration.json")),
                    final_result=final, model_scores=self.scores,
                    cross_dataset_results=self.cross_dataset_results,
                ),
            )
            attempt(
                "reports.dataset_csv", True,
                lambda: export_csv(
                    self.dataset,
                    str(self.output_manager.report_path("dataset.csv")),
                    successful[chosen],
                    (self.camera_config.width, self.camera_config.height),
                ),
            )
            attempt(
                "intrinsic.ros", True,
                lambda: export_ros_camera_info(
                    successful[chosen], self.camera_config, str(self.output_manager.ros_path())
                ),
            )
        else:
            skipped.extend(["reports.html", "reports.json", "reports.dataset_csv", "intrinsic.ros"])

        ws_config, _ws_dataset, ws_results, reflection_results, _ghost_results, ghost_models = (
            self.windshield_workspace.export_state()
        )
        for key, result in ws_results.items():
            variant_key = windshield_result_key_for_result(result)
            variant = (
                f"{variant_key[0].value}_{variant_key[1]}"
                if isinstance(variant_key, tuple) else result.windshield_model.value
            )
            def export_geometry(result=result, variant=variant):
                path = self.output_manager.windshield_geometry_path(result.base_model_name, variant)
                export_windshield_yaml(result, self.camera_config, str(path))
                paths = [path]
                neural = path.with_name(path.stem + "_neural.pt")
                if neural.exists():
                    paths.append(neural)
                return paths
            attempt(f"windshield.geometry.{variant}", getattr(result, "success", False) and self.camera_config is not None, export_geometry)

        for key, result in reflection_results.items():
            safe = model_filename(key)
            attempt(
                f"windshield.reflection.{safe}", result is not None,
                lambda result=result, safe=safe: export_reflection_yaml(
                    result, str(self.output_manager.reflection_path(f"reflection_{safe}.yaml"))
                ),
            )
        for key, field in ghost_models.items():
            safe = model_filename(key)
            attempt(
                f"windshield.ghost.{safe}", field is not None,
                lambda field=field, safe=safe: save_ghost_model(
                    field, str(self.output_manager.ghost_path(f"ghost_{safe}.yaml"))
                ),
            )

        if self.validation_results and self.repeated_kfold_results and configs_ready and self.dataset is not None:
            def export_paper():
                stability = {
                    model: (cal.param_uncertainty_bootstrap or cal.param_uncertainty)
                    for model, cal in self.calibration_results.items()
                }
                repeated = next(iter(self.repeated_kfold_results.values()))
                metadata = paper_evidence.build_paper_metadata(
                    self.camera_config, self.pattern_config, self.dataset,
                    k=repeated.k, n_repeats=repeated.n_repeats, base_seed=repeated.base_seed,
                )
                written = paper_evidence.export_paper_metrics(
                    str(self.output_manager.paper_directory()),
                    single_holdout=self.validation_results,
                    repeated=self.repeated_kfold_results,
                    stability_by_model=stability,
                    metadata=metadata,
                )
                return list(written.values())
            attempt("paper.metrics", True, export_paper)
        else:
            skipped.append("paper.metrics")

        details = [
            f"Session: {session}",
            f"완료 {len(exported)}개: {', '.join(exported) or '-'}",
            f"건너뜀 {len(skipped)}개: {', '.join(skipped) or '-'}",
            f"실패 {len(failed)}개: {', '.join(failed) or '-'}",
        ]
        QMessageBox.information(self, "Export All 결과", "\n\n".join(details))
        self.status_label.setText(f"Export All 완료: {session} (완료 {len(exported)}, 실패 {len(failed)})")

    def _build_settings_panel(self) -> QWidget:
        group = QGroupBox("▼ Camera Setup / Pattern")
        group.setObjectName("settingsPanel")
        group.setCheckable(True)
        group.setChecked(True)
        # checkable QGroupBox의 동작은 유지하되 theme에서 indicator를 0px로 숨긴다.
        # 사용자는 제목의 화살표/문구를 클릭해서만 접고 펼치므로 별도 체크박스가
        # 보이지 않는다.
        group.setToolTip("제목의 화살표를 클릭해 Camera Setup 영역을 접거나 펼칩니다.")
        group_layout = QVBoxLayout(group)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        outer = ResponsiveRow(breakpoint=1050)
        outer.box_layout.setSpacing(20)
        content_layout.addWidget(outer)
        settings_scroll = QScrollArea()
        settings_scroll.setObjectName("cameraSettingsScrollArea")
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.NoFrame)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        settings_scroll.setMaximumHeight(390)
        settings_scroll.setWidget(content)
        self.settings_scroll_area = settings_scroll
        group_layout.addWidget(settings_scroll)
        self.settings_group = group
        self.settings_content = content
        group.toggled.connect(self._on_settings_panel_toggled)

        camera_form = QFormLayout()
        camera_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        # QFormLayout에 세로 간격을 따로 정해주지 않으면 Qt가 부모 레이아웃인
        # 상위 responsive row의 spacing 값을 그대로 물려받는다 - 그래서
        # 상위 간격으로 열 사이만 넓혀도 이 폼의 행간(세로
        # 간격)까지 같이 넓어지는 부작용이 있었다. 원래 기본값(6px)으로
        # 고정해서 가로/세로 간격을 서로 독립적으로 만든다.
        camera_form.setVerticalSpacing(6)
        self.sensor_name_edit = QLineEdit()
        self.sensor_name_edit.setPlaceholderText("예: econ120")
        self.sensor_name_edit.setToolTip(
            "이 카메라를 구분하는 이름입니다. Library 탭은 이 이름으로 결과를 "
            "분류합니다 - 비워두면 서로 다른 카메라의 계산 결과가 전부 같은 "
            "'camera' 항목 하나에 섞입니다."
        )
        camera_form.addRow("Camera Name", self.sensor_name_edit)

        # 중앙 열 순서: Camera Name -> INPUT(실시간|rosbag|이미지 한 줄) ->
        # 해상도 확인 -> Width/Height(한 줄). 해상도 확인이 Width/Height
        # 바로 위에 있어야 "확인해서 자동으로 채운다"는 흐름이 자연스럽다.
        input_row = QHBoxLayout()
        input_row.setSpacing(14)  # 버튼 3개(실시간/rosbag/이미지)가 붙어 보이지 않게 여유를 둔다
        self.load_live_button = QPushButton("실시간")
        self.load_live_button.setToolTip("ROS1/ROS2 이미지 토픽을 실시간 구독해서 직접 캡처합니다.")
        self.load_live_button.clicked.connect(self._on_load_from_live)
        self.load_bag_button = QPushButton("rosbag")
        self.load_bag_button.setToolTip("ROS1(.bag)/ROS2(.db3, .mcap) 로그에서 이미지 토픽을 뽑아 불러옵니다.")
        self.load_bag_button.clicked.connect(self._on_load_from_bag)
        self.load_button = QPushButton("이미지")
        self.load_button.setToolTip("jpg/jpeg/png/bmp 이미지 파일을 직접 선택해서 불러옵니다.")
        self.load_button.clicked.connect(self._on_load_images)
        input_row.addWidget(self.load_live_button)
        input_row.addWidget(self.load_bag_button)
        input_row.addWidget(self.load_button)
        camera_form.addRow("INPUT", input_row)

        self.loaded_label = QLabel("불러온 이미지: 0장")
        camera_form.addRow(self.loaded_label)

        selection_row = QHBoxLayout()
        self.auto_select_count_spin = QSpinBox()
        self.auto_select_count_spin.setRange(1, 35)
        self.auto_select_count_spin.setValue(35)
        self.auto_select_count_spin.setEnabled(False)
        self.auto_select_button = QPushButton("이미지 걸러내기")
        self.auto_select_button.setEnabled(False)
        self.auto_select_button.clicked.connect(self._on_auto_select_images)
        selection_row.addWidget(QLabel("최종 선택 장수"))
        selection_row.addWidget(self.auto_select_count_spin)
        selection_row.addWidget(self.auto_select_button)
        camera_form.addRow(selection_row)

        self.auto_select_status_label = QLabel("이미지를 불러오면 자동 선별을 사용할 수 있습니다.")
        self.auto_select_status_label.setWordWrap(True)
        camera_form.addRow(self.auto_select_status_label)

        self.check_resolution_button = QPushButton("해상도 확인")
        self.check_resolution_button.setToolTip(
            "JPEG/PNG 이미지를 한 장 골라 실제로 디코딩해서 크기를 확인하고,\n"
            "Width/Height 칸에 자동으로 채워 넣습니다."
        )
        self.check_resolution_button.clicked.connect(self._on_check_resolution)
        camera_form.addRow(self.check_resolution_button)

        size_row = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 20000)
        self.width_spin.setValue(1920)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 20000)
        self.height_spin.setValue(1536)
        size_row.addWidget(QLabel("Width"))
        size_row.addWidget(self.width_spin)
        size_row.addSpacing(16)  # Width 값과 Height 라벨이 붙어 보이지 않게 여유를 둔다
        size_row.addWidget(QLabel("Height"))
        size_row.addWidget(self.height_spin)
        camera_form.addRow(size_row)
        camera_column = QWidget()
        camera_column.setObjectName("cameraSetupColumn")
        camera_column.setLayout(camera_form)
        configure_form_layout(camera_form)

        pattern_form = QFormLayout()
        pattern_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        pattern_form.setVerticalSpacing(6)  # camera_form과 같은 이유(outer 간격 상속 방지)
        self.squares_x_spin = QSpinBox()
        self.squares_x_spin.setRange(3, 30)
        self.squares_x_spin.setValue(7)
        self.squares_y_spin = QSpinBox()
        self.squares_y_spin.setRange(3, 30)
        self.squares_y_spin.setValue(5)
        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(0.1, 1000.0)
        self.square_size_spin.setDecimals(2)
        self.square_size_spin.setSingleStep(0.5)
        self.square_size_spin.setSuffix(" mm")
        self.square_size_spin.setValue(40.0)
        self.marker_size_spin = QDoubleSpinBox()
        self.marker_size_spin.setRange(0.1, 1000.0)
        self.marker_size_spin.setDecimals(2)
        self.marker_size_spin.setSingleStep(0.5)
        self.marker_size_spin.setSuffix(" mm")
        self.marker_size_spin.setValue(30.0)
        self.dictionary_combo = QComboBox()
        self.dictionary_combo.addItems(_ARUCO_DICTIONARIES)
        self.dictionary_combo.setCurrentText("DICT_5X5_100")
        self.circle_grid_type_combo = QComboBox()
        self.circle_grid_type_combo.addItem("Symmetric", userData=CircleGridType.SYMMETRIC)
        self.circle_grid_type_combo.addItem("Asymmetric", userData=CircleGridType.ASYMMETRIC)
        self.aprilgrid_variant_combo = QComboBox()
        self.aprilgrid_variant_combo.addItem("OpenCV / AprilTag3 style", userData=AprilGridVariant.OPENCV_APRILTAG3)
        self.aprilgrid_variant_combo.addItem("Kalibr style (experimental)", userData=AprilGridVariant.KALIBR)

        self.calibration_method_combo = QComboBox()
        self.calibration_method_combo.addItem("Standard", userData=CalibrationMethod.STANDARD)
        self.calibration_method_combo.addItem("Object-Releasing", userData=CalibrationMethod.OBJECT_RELEASING)
        self.calibration_method_combo.setToolTip(
            "Object-Releasing uses OpenCV calibrateCameraRO.\n"
            "Every accepted view must contain the full board with identical point IDs and ordering.\n"
            "Object-Releasing is available for Checkerboard and Circle Grid only."
        )
        self.calibration_method_combo.currentIndexChanged.connect(self._on_calibration_method_changed)
        self.method_policy_label = QLabel("")
        self.method_policy_label.setWordWrap(True)

        self.pattern_type_combo = QComboBox()
        # userData로 PatternType을 직접 들고 있어서 _current_pattern_config()가
        # 문자열 비교 없이 바로 꺼내 쓸 수 있다.
        self.pattern_type_combo.addItem("ChArUco (권장)", userData=PatternType.CHARUCO)
        self.pattern_type_combo.addItem("Chessboard (일반 체스보드)", userData=PatternType.CHESSBOARD)
        self.pattern_type_combo.addItem("Circle Grid", userData=PatternType.CIRCLE_GRID)
        self.pattern_type_combo.addItem("AprilGrid (AprilTag grid)", userData=PatternType.APRILGRID)
        self.pattern_type_combo.currentIndexChanged.connect(self._on_pattern_type_changed)

        pattern_form.addRow("Calibration method", self.calibration_method_combo)
        pattern_form.addRow("", self.method_policy_label)
        pattern_form.addRow("Pattern type", self.pattern_type_combo)
        pattern_form.addRow("Squares X", self.squares_x_spin)
        pattern_form.addRow("Squares Y", self.squares_y_spin)
        pattern_form.addRow("Square size", self.square_size_spin)
        pattern_form.addRow("Marker size", self.marker_size_spin)
        pattern_form.addRow("Dictionary", self.dictionary_combo)
        pattern_form.addRow("Grid type", self.circle_grid_type_combo)
        pattern_form.addRow("AprilGrid variant", self.aprilgrid_variant_combo)
        self._pattern_form = pattern_form  # setRowVisible로 마커/딕셔너리 행을 토글하기 위해 보관
        self._on_pattern_type_changed()
        pattern_column = QWidget()
        pattern_column.setObjectName("calibrationPatternColumn")
        pattern_column.setLayout(pattern_form)
        configure_form_layout(pattern_form)
        # 넓은 화면에서는 Pattern을 가장 왼쪽, Camera를 중앙에 배치한다.
        # ResponsiveRow가 세로로 접힐 때도 이 순서가 그대로 유지된다.
        outer.addWidget(pattern_column, stretch=1)
        outer.addWidget(camera_column, stretch=1)

        # 오른쪽 열 순서: 캘리브레이션 실행 -> Export -> 취소. Rational
        # on/off 체크박스는 제거됐다 - Standard 계산은 항상 Ideal Pinhole/
        # Brown-Conrady/Rational(8계수 k1~k6,p1,p2 고정)/Fisheye 네 모델을
        # 함께 계산하고, ③ Model Comparison에서 모델을 고른다(모델 의미
        # 고정 정책 - README 4번 섹션 참고). Export/취소는 각각 옛
        # "⑦ Export" 탭과, 실행 중인 계산(코너 검출/모델 계산)을 중단하는
        # 기능을 대체한다.
        action_layout = QVBoxLayout()
        self.run_button = QPushButton("캘리브레이션 실행")
        self.run_button.setProperty("role", "primary")
        self.run_button.clicked.connect(self._on_run_pipeline)
        self.run_button.setEnabled(False)
        self.export_button = QPushButton("Export")
        self.export_button.setToolTip("계산된 모델을 골라 OpenCV YAML로 저장합니다 (옛 '⑦ Export' 탭과 동일한 기능).")
        self.export_button.clicked.connect(self._on_export_button_clicked)
        self.cancel_button = QPushButton("취소")
        self.cancel_button.setToolTip(
            "코너 검출/캘리브레이션 계산이 진행 중일 때 즉시 중단합니다.\n"
            "중단 후에는 이미지를 다시 고르거나 설정을 바꿔서 원하는 대로 다시 실행할 수 있습니다."
        )
        self.cancel_button.clicked.connect(self._on_cancel_pipeline)
        self.cancel_button.setEnabled(False)
        self._on_calibration_method_changed()
        action_layout.addWidget(self.run_button)
        action_layout.addWidget(self._build_output_session_bar())
        action_layout.addWidget(self.export_button)
        action_layout.addWidget(self.cancel_button)
        action_layout.addStretch(1)
        action_column = QWidget()
        action_column.setLayout(action_layout)
        outer.addWidget(action_column, stretch=1)

        return group

    def _on_settings_panel_toggled(self, expanded: bool) -> None:
        self.settings_group.setMaximumHeight(16777215 if expanded else 42)
        self.settings_group.setTitle(
            "▼ Camera Setup / Pattern" if expanded else "▶ Camera Setup / Pattern"
        )
        self.settings_content.setVisible(expanded)

    # ------------------------------------------------------------------
    # 이미지 로드 / 파이프라인 실행
    # ------------------------------------------------------------------

    def _sync_auto_select_controls(self, *, reset_value: bool = False) -> None:
        count = len(self.image_paths)
        enabled = count > 0
        self.auto_select_count_spin.setEnabled(enabled)
        self.auto_select_button.setEnabled(enabled)
        self.auto_select_count_spin.setMaximum(max(1, count))
        if reset_value or self.auto_select_count_spin.value() > count:
            self.auto_select_count_spin.setValue(min(35, max(1, count)))
        if not enabled:
            self.auto_select_status_label.setText("이미지를 불러오면 자동 선별을 사용할 수 있습니다.")

    def _on_load_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "캘리브레이션 이미지 선택", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not paths:
            return
        self.image_paths = paths
        self.loaded_label.setText(f"불러온 이미지: {len(paths)}장")
        self.auto_select_status_label.setText("자동 선별 전: 전체 이미지가 캘리브레이션 입력입니다.")
        self._sync_auto_select_controls(reset_value=True)
        self.run_button.setEnabled(True)

    def _on_auto_select_images(self) -> None:
        if not self.image_paths:
            QMessageBox.warning(self, "이미지 없음", "먼저 이미지를 불러오세요.")
            return

        self.pattern_config = self._current_pattern_config()
        self.camera_config = self._current_camera_config()
        target_count = min(self.auto_select_count_spin.value(), len(self.image_paths))
        worker = ImageSelectionWorker(
            self.image_paths,
            self.pattern_config,
            self.camera_config,
            target_count,
        )
        thread = run_worker_in_thread(worker, self)

        worker.progress.connect(self.status_label.setText)
        worker.progress.connect(self.auto_select_status_label.setText)
        worker.progress_value.connect(self._on_pipeline_progress_value)
        worker.dataset_ready.connect(self._on_dataset_ready)
        worker.selection_ready.connect(self._on_image_selection_ready)
        worker.error.connect(self._on_error)

        self._selection_thread, self._selection_worker = thread, worker
        self.auto_select_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.pipeline_progress_bar.setRange(0, max(1, len(self.image_paths)))
        self.pipeline_progress_bar.setValue(0)
        self.pipeline_progress_bar.show()
        thread.finished.connect(lambda: self._sync_auto_select_controls())
        thread.finished.connect(lambda: self.run_button.setEnabled(bool(self.image_paths)))
        thread.finished.connect(lambda: self.load_button.setEnabled(True))
        thread.finished.connect(self.pipeline_progress_bar.hide)
        thread.finished.connect(self._stop_busy_sheep)
        thread.finished.connect(lambda: setattr(self, "_selection_worker", None))
        thread.start()

    def _on_image_selection_ready(self, result) -> None:
        self.image_paths = list(result.selected_paths)
        self.loaded_label.setText(
            f"불러온 이미지: {len(self.image_paths)}장 (자동 선별: {result.selected_count}/{result.requested_count})"
        )
        parts = [result.summary_text()]
        if result.warnings:
            parts.extend(result.warnings[:3])
        self.auto_select_status_label.setText(" | ".join(parts))
        self._sync_auto_select_controls()

        manifest_path = write_selection_manifest(
            result,
            self.output_manager.report_path("image_selection_manifest.json"),
        )
        self.output_manager.record_export(
            "image_selection.manifest",
            manifest_path,
            metadata={
                "selected_count": result.selected_count,
                "requested_count": result.requested_count,
            },
        )
        self._ensure_output_session()

    def _on_check_resolution(self) -> None:
        """JPEG/PNG 이미지 한 장을 실제로 디코딩해 크기를 확인하고
        Width/Height에 자동 반영한다. 카메라 스펙 문서를 못 믿거나(오타/구형
        렌즈 교체 등) 잘 모를 때, 실제 데이터에서 바로 정확한 값을 얻기 위함.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "해상도 확인용 이미지 선택", "", "Images (*.jpg *.jpeg *.png)"
        )
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            QMessageBox.warning(
                self, "디코딩 실패", "선택한 파일을 JPEG/PNG 이미지로 디코딩하지 못했습니다."
            )
            return
        height, width = image.shape[:2]
        self.width_spin.setValue(width)
        self.height_spin.setValue(height)
        self.status_label.setText(f"해상도 확인 완료: {width}×{height} (Width/Height에 자동 반영됨)")

    def _on_load_from_bag(self) -> None:
        """ROS1(.bag)/ROS2(.db3, .mcap) 로그 파일에서 이미지를 뽑아 불러온다.
        rospy/rclpy 없이 순수 Python(rosbags)으로 읽으므로 ROS 설치가 필요 없다.
        """
        bag_path, _ = QFileDialog.getOpenFileName(
            self, "rosbag 파일 선택", "", "ROS bag (*.bag *.db3 *.mcap);;All files (*)"
        )
        if not bag_path:
            return

        # 큰 ROS1 bag은 토픽 목록을 얻기 위해 파일 인덱스를 여는 단계부터
        # 수 초 이상 걸린다. 추출뿐 아니라 이 검색도 반드시 GUI 밖에서 한다.
        worker = BagTopicDiscoveryWorker(bag_path)
        thread = run_worker_in_thread(worker, self)
        progress = QProgressDialog("bag 인덱스에서 이미지 토픽을 검색 중...", "", 0, 0, self)
        progress.setWindowTitle("bag 여는 중")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)

        worker.progress.connect(progress.setLabelText)
        worker.progress.connect(self.status_label.setText)
        # 반드시 MainWindow bound method에 직접 연결해야 Qt가 GUI thread로
        # queued delivery한다. lambda 안에서 dialog를 만들면 worker thread에서
        # 실행될 수 있어 QObject::setParent 오류와 segfault가 발생한다.
        worker.topics_ready.connect(self._on_bag_topics_ready)
        worker.error.connect(self._on_error)
        worker.finished.connect(progress.close)
        worker.finished.connect(self._on_bag_topic_worker_finished)

        self._bag_topic_thread, self._bag_topic_worker = thread, worker
        self._bag_topic_progress_dialog = progress
        self.load_button.setEnabled(False)
        self.load_bag_button.setEnabled(False)
        thread.finished.connect(self._on_bag_thread_finished)
        thread.start()
        progress.show()

    def _on_bag_topics_ready(self, topics: list, bag_path: str) -> None:
        """백그라운드 검색 결과를 받은 뒤에만 사용자 선택 UI를 연다."""

        if not topics:
            QMessageBox.warning(self, "이미지 토픽 없음", "이 bag 안에서 이미지 토픽을 찾지 못했습니다.")
            return

        labels = [f"{t.name}  ({t.msg_type.split('/')[-1]}, {t.count}개)" for t in topics]
        label, ok = QInputDialog.getItem(
            self, "이미지 토픽 선택", "추출할 토픽을 고르세요:", labels, 0, False
        )
        if not ok:
            return
        topic = topics[labels.index(label)].name

        interval, ok = QInputDialog.getDouble(
            self, "샘플링 간격",
            "이미지 추출 최소 간격(초)\n"
            "(bag은 보통 15~60fps라 그대로 다 뽑으면 거의 똑같은 프레임이 수백 장 나옵니다.\n"
            " 간격을 두면 자세 다양성 있는 데이터셋에 더 가까워집니다.)",
            0.5, 0.05, 30.0, 2,
        )
        if not ok:
            return

        out_dir = str(Path(bag_path).with_suffix("").as_posix()) + "_extracted"
        self._start_bag_extraction(bag_path, topic, out_dir, interval)

    def _on_bag_topic_worker_finished(self) -> None:
        self._bag_topic_progress_dialog = None

    def _on_bag_thread_finished(self) -> None:
        self.load_button.setEnabled(True)
        self.load_bag_button.setEnabled(True)

    def _start_bag_extraction(
        self, bag_path: str, topic: str, out_dir: str, interval: float
    ) -> None:
        # 이미지 추출 자체(메시지 디코딩 + 디스크 기록)는 큰 bag에서 수십 초~
        # 몇 분까지 걸릴 수 있어 QThread로 분리한다. 예전엔 여기서 바로
        # extract_images_from_bag()을 동기 호출해서, 큰 bag을 불러올 때
        # GUI 스레드가 그대로 멈춰 OS가 "python3 is not responding"을
        # 띄우는 원인이었다.
        worker = BagExtractionWorker(bag_path, topic, out_dir, interval)
        thread = run_worker_in_thread(worker, self)

        progress = QProgressDialog("bag에서 이미지 추출 준비 중...", "취소", 0, 0, self)
        progress.setWindowTitle("이미지 추출 중")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        # Lambda executes in the GUI thread and flips the worker's atomic-ish
        # boolean immediately; a queued slot could not run while worker.run()
        # owns the QThread event loop.
        progress.canceled.connect(lambda: worker.request_cancel())

        worker.progress.connect(progress.setLabelText)
        worker.progress.connect(self.status_label.setText)
        # 실사용자 버그 수정: total을 알게 되는 즉시 막대를 고정폭 퍼센트
        # 바로 전환하고(setRange(0,0)이면 계속 무한 반복 바), 그 안에서
        # 색이 실제 진행률만큼 차오르게 한다. total이 아직 0(파악 전)이면
        # 무한 반복 바를 유지한다.
        worker.progress_value.connect(self._on_bag_progress_value)
        self._bag_progress_dialog = progress
        worker.finished_extraction.connect(self._on_bag_extraction_finished)
        worker.error.connect(self._on_error)
        worker.finished.connect(progress.close)
        worker.finished.connect(self._on_bag_extraction_worker_finished)

        self._bag_thread, self._bag_worker = thread, worker
        self.load_button.setEnabled(False)
        self.load_bag_button.setEnabled(False)
        thread.finished.connect(self._on_bag_thread_finished)
        thread.start()
        progress.show()

    def _on_bag_progress_value(self, done: int, total: int) -> None:
        dialog = getattr(self, "_bag_progress_dialog", None)
        if dialog is None:
            return
        if total > 0:
            if dialog.maximum() != total:
                dialog.setMaximum(total)
            dialog.setValue(min(done, total))
        # total == 0(아직 메시지 개수를 모르는 상태)이면 setRange(0,0) 그대로
        # 두어 무한 반복 바를 유지한다 - 값을 알 수 없는데 억지로 고정폭으로
        # 바꾸면 항상 0%로 멈춰 있는 것처럼 보여 오히려 더 헷갈린다.

    def _on_bag_extraction_worker_finished(self) -> None:
        self._bag_progress_dialog = None

    def _on_bag_extraction_finished(self, extracted: list[str], bag_path: str) -> None:
        if not extracted:
            QMessageBox.warning(
                self, "추출된 이미지 없음",
                "선택한 토픽/간격으로 추출된 이미지가 없습니다 (취소했거나 해당 구간에 이미지가 없음).",
            )
            return

        self.image_paths = extracted
        self.loaded_label.setText(f"불러온 이미지: {len(extracted)}장 (rosbag: {Path(bag_path).name})")
        self.auto_select_status_label.setText("자동 선별 전: 전체 이미지가 캘리브레이션 입력입니다.")
        self._sync_auto_select_controls(reset_value=True)
        self.run_button.setEnabled(True)

    def _on_load_from_live(self) -> None:
        """실시간 ROS 토픽을 구독해서 사용자가 직접 캡처한 이미지를 불러온다."""
        if ROS_LIVE_BACKEND is None:
            QMessageBox.warning(
                self, "ROS 미설치",
                "실시간 구독을 쓰려면 이 컴퓨터에 ROS1 또는 ROS2가 설치되어 있고 "
                "환경이 source 되어 있어야 합니다 (rospy/rclpy는 pip로 설치되지 않습니다).\n\n"
                "이미 녹화된 bag 파일만 있다면 [rosbag에서 불러오기]를 대신 쓰세요.",
            )
            return

        out_dir = str(Path.cwd() / "live_captures")
        dialog = LiveCaptureDialog(
            out_dir,
            pattern_config=self._current_pattern_config(),
            camera_config=self._current_camera_config(),
            parent=self,
        )
        if dialog.exec() != LiveCaptureDialog.Accepted:
            return
        if not dialog.captured_paths:
            return

        self.image_paths = dialog.captured_paths
        if dialog.captured_image_size is not None:
            width, height = dialog.captured_image_size
            self.width_spin.setValue(width)
            self.height_spin.setValue(height)
            size_text = f", {width}×{height} 자동 반영"
        else:
            size_text = ""
        self.loaded_label.setText(
            f"불러온 이미지: {len(dialog.captured_paths)}장 (실시간 캡처{size_text})"
        )
        self.auto_select_status_label.setText("자동 선별 전: 전체 이미지가 캘리브레이션 입력입니다.")
        self._sync_auto_select_controls(reset_value=True)
        self.run_button.setEnabled(True)

    def _set_pattern_type_options_for_method(self, method: CalibrationMethod) -> None:
        if not hasattr(self, "pattern_type_combo"):
            return
        method = method if isinstance(method, CalibrationMethod) else CalibrationMethod(str(method))
        current = self.pattern_type_combo.currentData()
        all_options = [
            ("ChArUco (권장)", PatternType.CHARUCO),
            ("Chessboard (일반 체스보드)", PatternType.CHESSBOARD),
            ("Circle Grid", PatternType.CIRCLE_GRID),
            ("AprilGrid (AprilTag grid)", PatternType.APRILGRID),
        ]
        if method == CalibrationMethod.OBJECT_RELEASING:
            options = [
                ("Checkerboard (Object-Releasing recommended)", PatternType.CHESSBOARD),
                ("Circle Grid (Object-Releasing recommended)", PatternType.CIRCLE_GRID),
            ]
        else:
            options = all_options

        existing = [self.pattern_type_combo.itemData(i) for i in range(self.pattern_type_combo.count())]
        desired_items = [value for _label, value in options]
        desired = current if current in desired_items else desired_items[0]
        if existing != desired_items:
            self.pattern_type_combo.blockSignals(True)
            self.pattern_type_combo.clear()
            for label, value in options:
                self.pattern_type_combo.addItem(label, userData=value)
            self.pattern_type_combo.blockSignals(False)
        idx = self.pattern_type_combo.findData(desired)
        if idx >= 0 and idx != self.pattern_type_combo.currentIndex():
            self.pattern_type_combo.setCurrentIndex(idx)
        self._on_pattern_type_changed(update_method_policy=False)

    def _on_pattern_type_changed(self, update_method_policy: bool = True) -> None:
        """Chessboard는 marker_size/dictionary가 필요 없으니 해당 입력 행을
        숨긴다. ChArUco/AprilGrid는 마커 기반 패턴이라 둘 다 유지한다.
        """
        if not isinstance(update_method_policy, bool):
            update_method_policy = True
        pattern_type = self.pattern_type_combo.currentData()
        self._set_pattern_field_labels(pattern_type)
        uses_marker_dictionary = pattern_type in (PatternType.CHARUCO, PatternType.APRILGRID)
        # 행 인덱스: 0=Pattern type, 1=Squares X, 2=Squares Y, 3=Square size,
        # 4=Marker size, 5=Dictionary, 6=Grid type, 7=AprilGrid variant.
        self._pattern_form.setRowVisible(6, uses_marker_dictionary)
        self._pattern_form.setRowVisible(7, uses_marker_dictionary)
        self._pattern_form.setRowVisible(8, pattern_type == PatternType.CIRCLE_GRID)
        self._pattern_form.setRowVisible(9, pattern_type == PatternType.APRILGRID)
        current_dictionary = self.dictionary_combo.currentText()
        if pattern_type == PatternType.APRILGRID and not current_dictionary.startswith("DICT_APRILTAG_"):
            idx = self.dictionary_combo.findText("DICT_APRILTAG_36h11")
            if idx >= 0:
                self.dictionary_combo.setCurrentIndex(idx)
        elif pattern_type == PatternType.CHARUCO and current_dictionary.startswith("DICT_APRILTAG_"):
            idx = self.dictionary_combo.findText("DICT_5X5_100")
            if idx >= 0:
                self.dictionary_combo.setCurrentIndex(idx)
        if update_method_policy and hasattr(self, "method_policy_label"):
            self._on_calibration_method_changed()

    def _on_calibration_method_changed(self) -> None:
        method = self.calibration_method_combo.currentData()
        method = method if isinstance(method, CalibrationMethod) else CalibrationMethod(str(method))
        if hasattr(self, "result_view"):
            self.result_view.set_advanced_calibration_available(
                method == CalibrationMethod.OBJECT_RELEASING
            )
        self._set_pattern_type_options_for_method(method)
        pattern_type = self.pattern_type_combo.currentData()
        if method != CalibrationMethod.OBJECT_RELEASING:
            self._pattern_form.setRowVisible(1, False)
            self.method_policy_label.setText("")
            return
        self._pattern_form.setRowVisible(1, True)
        policy = {
            PatternType.CHESSBOARD: "Checkerboard: Object-Releasing supported and recommended. Full board required.",
            PatternType.CIRCLE_GRID: "Circle Grid: Object-Releasing supported and recommended. Full board required.",
        }.get(pattern_type, "Object-Releasing supports Checkerboard and Circle Grid only.")
        self.method_policy_label.setText(policy)

    def _set_pattern_field_labels(self, pattern_type: PatternType) -> None:
        pattern_type = pattern_type if isinstance(pattern_type, PatternType) else PatternType(str(pattern_type))
        labels = {
            PatternType.CHESSBOARD: {
                self.squares_x_spin: "Columns",
                self.squares_y_spin: "Rows",
                self.square_size_spin: "Square size",
                self.marker_size_spin: "Marker size",
                self.dictionary_combo: "Dictionary",
            },
            PatternType.CHARUCO: {
                self.squares_x_spin: "Squares X",
                self.squares_y_spin: "Squares Y",
                self.square_size_spin: "Square size",
                self.marker_size_spin: "Marker size",
                self.dictionary_combo: "Dictionary",
            },
            PatternType.CIRCLE_GRID: {
                self.squares_x_spin: "Columns",
                self.squares_y_spin: "Rows",
                self.square_size_spin: "Center spacing",
                self.marker_size_spin: "Marker size",
                self.dictionary_combo: "Dictionary",
            },
            PatternType.APRILGRID: {
                self.squares_x_spin: "Tag columns",
                self.squares_y_spin: "Tag rows",
                self.square_size_spin: "Tag pitch",
                self.marker_size_spin: "Tag size",
                self.dictionary_combo: "Tag family",
            },
        }.get(pattern_type, {})
        labels[self.circle_grid_type_combo] = "Grid type"
        labels[self.aprilgrid_variant_combo] = "Variant"
        labels[self.pattern_type_combo] = "Pattern type"
        labels[self.calibration_method_combo] = "Calibration method"
        for field, text in labels.items():
            item = self._pattern_form.labelForField(field)
            if item is not None:
                item.setText(text)

    def _current_pattern_config(self) -> PatternConfig:
        # UI는 mm로 입력받지만(보드 인쇄 스펙이 보통 mm 단위), 내부 계산/export는
        # 전부 미터(m) 기준이라 여기서 한 번만 변환한다 - 이후 파이프라인은
        # 이 값이 mm에서 왔는지 몰라도 된다.
        pattern_type = self.pattern_type_combo.currentData()
        uses_marker_dictionary = pattern_type in (PatternType.CHARUCO, PatternType.APRILGRID)
        return PatternConfig(
            type=pattern_type,
            squares_x=self.squares_x_spin.value(),
            squares_y=self.squares_y_spin.value(),
            square_size=self.square_size_spin.value() / 1000.0,
            marker_size=(self.marker_size_spin.value() / 1000.0) if uses_marker_dictionary else None,
            dictionary=self.dictionary_combo.currentText() if uses_marker_dictionary else None,
            circle_grid_type=self.circle_grid_type_combo.currentData(),
            aprilgrid_variant=self.aprilgrid_variant_combo.currentData(),
        )

    def _current_camera_config(self) -> CameraConfig:
        return CameraConfig(
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            sensor_name=self.sensor_name_edit.text().strip() or None,
        )

    def _on_run_pipeline(self) -> None:
        if not self.image_paths:
            QMessageBox.warning(self, "이미지 없음", "먼저 이미지를 불러오세요.")
            return

        self.pattern_config = self._current_pattern_config()
        self.camera_config = self._current_camera_config()
        selected_method = self.calibration_method_combo.currentData()
        selected_method = (
            selected_method
            if isinstance(selected_method, CalibrationMethod)
            else CalibrationMethod(str(selected_method))
        )
        if selected_method == CalibrationMethod.OBJECT_RELEASING and self.pattern_config.type not in (
            PatternType.CHESSBOARD,
            PatternType.CIRCLE_GRID,
        ):
            QMessageBox.warning(
                self,
                "Object-Releasing unavailable",
                "Object-Releasing supports Checkerboard and Circle Grid only. "
                "Use Standard calibration for ChArUco/AprilGrid.",
            )
            return
        self.calibration_results = {}
        self.validation_results = {}
        self.optimizer_results = {}
        self.cross_dataset_results = []
        self.scores = []
        # 새 calibration 실행은 이전 Repeated K-Fold 결과를 무효화한다 -
        # dataset/calibration이 바뀌었는데 옛 fold 결과를 Paper Metrics로
        # export하면 안 되므로(stale result 방지), 여기서 항상 비운다.
        self.repeated_kfold_results = {}
        self.result_view.set_repeated_kfold_results({})
        self.object_releasing_result = None
        self.scene_quality_analysis = None
        self.subset_calibration_result = None
        if hasattr(self, "scene_quality_view"):
            self.scene_quality_view.set_context(
                self.dataset, self.camera_config, {}, None, None
            )
        if hasattr(self, "optimizer_view"):
            self.optimizer_view.set_context({}, {}, {}, self.camera_config)

        self.calibration_method = selected_method
        worker = PipelineWorker(
            self.image_paths, self.pattern_config, self.camera_config,
            calibration_method=self.calibration_method,
        )
        thread = run_worker_in_thread(worker, self)

        worker.progress.connect(self.status_label.setText)
        worker.progress_value.connect(self._on_pipeline_progress_value)
        worker.dataset_ready.connect(self._on_dataset_ready)
        worker.quality_ready.connect(self._on_quality_ready)
        worker.models_ready.connect(self._on_models_ready)
        worker.validation_ready.connect(self._on_validation_ready)
        worker.recommendation_ready.connect(self._on_recommendation_ready)
        worker.error.connect(self._on_error)
        worker.cancelled.connect(self._on_pipeline_cancelled)

        self._thread, self._worker = thread, worker
        self.run_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.auto_select_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.pipeline_progress_bar.setRange(0, max(1, len(self.image_paths)))
        self.pipeline_progress_bar.setValue(0)
        self.pipeline_progress_bar.show()
        thread.finished.connect(lambda: self.run_button.setEnabled(True))
        thread.finished.connect(lambda: self.load_button.setEnabled(True))
        thread.finished.connect(lambda: self._sync_auto_select_controls())
        thread.finished.connect(lambda: self.cancel_button.setEnabled(False))
        thread.finished.connect(self.pipeline_progress_bar.hide)
        thread.finished.connect(self._stop_busy_sheep)
        thread.start()

    def _on_cancel_pipeline(self) -> None:
        """진행 중인 코너 검출/캘리브레이션 계산을 즉시 중단한다.
        실행 중인 자식 프로세스를 강제 종료하므로 몇 초씩 기다릴 필요 없이
        바로 멈추고, 이미지를 다시 고르거나 설정을 바꿔 다시 실행할 수 있는
        상태로 돌아간다(불러온 이미지 목록 자체는 그대로 남는다).
        """
        if self._worker is None:
            return
        self.status_label.setText("취소 요청됨 - 계산을 중단하는 중...")
        self.cancel_button.setEnabled(False)
        self._worker.request_cancel()

    def _on_pipeline_cancelled(self) -> None:
        self.status_label.setText("계산이 취소되었습니다. 원하는 데이터/설정으로 다시 실행할 수 있습니다.")

    def _on_pipeline_progress_value(self, done: int, total: int) -> None:
        if total <= 0:
            self._start_busy_sheep()
        else:
            self._stop_busy_sheep()
            self.pipeline_progress_bar.setRange(0, total)
            self.pipeline_progress_bar.setValue(min(done, total))
            self.pipeline_progress_bar.setFormat("%p%")
        self.pipeline_progress_bar.show()

    def _start_busy_sheep(self) -> None:
        if self._sheep_timer.isActive():
            return
        self._sheep_pos = 0
        self.pipeline_progress_bar.setRange(0, _SHEEP_TRACK_LEN)
        self._render_busy_sheep()
        self._sheep_timer.start()

    def _stop_busy_sheep(self) -> None:
        self._sheep_timer.stop()

    def _advance_busy_sheep(self) -> None:
        self._sheep_pos = (self._sheep_pos + 1) % (_SHEEP_TRACK_LEN + 1)
        self._render_busy_sheep()

    def _render_busy_sheep(self) -> None:
        track = "·" * self._sheep_pos + "🐑" + "·" * (_SHEEP_TRACK_LEN - self._sheep_pos)
        self.pipeline_progress_bar.setValue(self._sheep_pos)
        self.pipeline_progress_bar.setFormat(track)

    # --- PipelineWorker 콜백 ---

    def _on_dataset_ready(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.dataset_view.set_dataset(dataset)
        self.dataset_view.set_dataset_quality_score(dataset.quality_score)

    def _on_quality_ready(self, warnings: list[str]) -> None:
        if self.dataset is None:
            return
        self.dataset_view.set_quality(self.dataset.coverage_grid, self.dataset.diversity, warnings)

    def _on_models_ready(
        self,
        results: dict[CameraModelType, CalibrationResult],
        object_releasing_result: CalibrationResult | None = None,
        object_releasing_validation_result=None,
        standard_vs_object_releasing_comparison=None,
    ) -> None:
        self.calibration_results = results
        self.object_releasing_result = object_releasing_result
        self.object_releasing_validation_result = object_releasing_validation_result
        self.standard_vs_object_releasing_comparison = standard_vs_object_releasing_comparison
        # Ranking은 sanity check와 독립적이다. 자체 진단에서 예외가 나더라도
        # Initial Calibration 결과는 즉시 Ranking 탭에 반영되어야 한다.
        self._update_scene_quality_analysis()
        if object_releasing_result is not None:
            self.status_label.setText(
                object_releasing_result.warning_message
                or object_releasing_result.error_message
                or "Object-Releasing completed."
            )
        if self.dataset is not None and self.camera_config is not None:
            self.preview_view.set_context(self.dataset, self.camera_config, results, self.pattern_config)
            self.dataset_view.set_dataset(self.dataset)  # per_frame_error 채워졌으니 갱신
            # 설계 문서 8번 - Standard 4모델 계산이 끝날 때마다 sanity check도 함께 갱신한다
            # (RMS가 낮아 보여도 결과가 물리적으로 이상할 수 있으므로 항상 확인).
            try:
                checks = run_sanity_checks(list(results.values()), self.camera_config)
                self.result_view.set_sanity_checks(checks)
            except Exception:  # noqa: BLE001 - 진단 실패가 결과 UI 전체를 막지 않게 한다.
                logger.exception("Sanity check failed after calibration; keeping result views available")
        self._refresh_result_view()
        self.optimizer_view.set_context(
            self.calibration_results, self.validation_results, self.optimizer_results, self.camera_config
        )

    def _on_validation_ready(self, results: dict[CameraModelType, ValidationResult]) -> None:
        self.validation_results = results
        self._refresh_result_view()
        self.optimizer_view.set_context(
            self.calibration_results, self.validation_results, self.optimizer_results, self.camera_config
        )

    # --- Post-OpenCV optimizer -------------------------------------------------

    def _on_optimizer_run(self, model: CameraModelType, settings) -> None:
        model = CameraModelType(model)
        validation = self.validation_results.get(model)
        calibration = self.calibration_results.get(model)
        if (
            self.dataset is None or self.camera_config is None or self.pattern_config is None
            or validation is None or not validation.train_frame_ids
            or calibration is None or not calibration.success
        ):
            QMessageBox.warning(
                self, "Optimizer unavailable",
                "Complete OpenCV calibration and frozen hold-out validation first.",
            )
            return
        worker = OptimizerWorker(
            self.dataset, self.camera_config, self.pattern_config, model,
            validation.train_frame_ids, validation.test_frame_ids, settings,
        )
        thread = run_worker_in_thread(worker, self)
        worker.progress.connect(self.status_label.setText)
        worker.result_ready.connect(self._on_optimizer_result)
        worker.error.connect(self._on_optimizer_error)
        worker.cancelled.connect(self._on_optimizer_cancelled)
        self._optimizer_thread, self._optimizer_worker = thread, worker
        self.optimizer_view.set_running(True)
        thread.finished.connect(lambda: self.optimizer_view.set_running(False))
        thread.start()

    def _on_optimizer_cancel(self) -> None:
        if self._optimizer_worker is not None:
            self.status_label.setText("Optimizer cancellation requested...")
            self._optimizer_worker.request_cancel()

    def _on_optimizer_cancelled(self) -> None:
        self.status_label.setText("Optimizer cancelled; original calibration was preserved.")
        self.optimizer_view.set_cancelled()

    def _on_optimizer_error(self, message: str) -> None:
        self.status_label.setText(message)
        self.optimizer_view.set_failed(message)
        QMessageBox.warning(self, "Optimizer failed", message)

    def _on_optimizer_result(self, result: OptimizerResult) -> None:
        self.optimizer_results[result.model_name] = result
        self.optimizer_view.set_context(
            self.calibration_results, self.validation_results, self.optimizer_results, self.camera_config
        )
        self.optimizer_view.select_model(result.model_name)
        self.status_label.setText(f"Optimizer completed: {result.recommendation}")
        self._autosave()

    def _on_optimizer_apply(self, model: CameraModelType) -> None:
        model = CameraModelType(model)
        result = self.optimizer_results.get(model)
        if not result or not result.success or result.optimized_calibration is None:
            return
        self.calibration_results[model] = apply_optimized_calibration(
            self.calibration_results[model], result
        )
        self._refresh_after_optimizer_change(model)
        self.status_label.setText(f"Applied optimized {model.value} calibration.")
        self._autosave()

    def _on_optimizer_restore(self, model: CameraModelType) -> None:
        model = CameraModelType(model)
        result = self.optimizer_results.get(model)
        if not result or result.pre_apply_calibration is None:
            return
        self.calibration_results[model] = restore_original_calibration(result)
        self._refresh_after_optimizer_change(model)
        self.status_label.setText(f"Restored original OpenCV {model.value} calibration.")
        self._autosave()

    def _refresh_after_optimizer_change(self, model: CameraModelType) -> None:
        self.optimizer_view.set_context(
            self.calibration_results, self.validation_results, self.optimizer_results, self.camera_config
        )
        self.optimizer_view.select_model(model)
        self._refresh_result_view()
        if self.dataset is not None and self.camera_config is not None:
            self.preview_view.set_context(
                self.dataset, self.camera_config, self.calibration_results, self.pattern_config
            )

    def _on_recommendation_ready(self, scores: list[ModelScore], message: str) -> None:
        self.scores = scores
        self.result_view.set_recommendation_message(message)
        recommended = next((s.model_name for s in scores if s.is_recommended), None)
        if recommended is not None:
            self.result_view.select_model(recommended)
            self.preview_view.select_model(recommended)
            self._update_scene_quality_analysis(recommended)
        else:
            self._update_scene_quality_analysis()
        self._refresh_result_view()
        # 계산이 완전히 끝난 시점(추천까지 나온 시점)이라 여기서 조용히
        # 자동 저장한다 - 다음에 앱이 비정상 종료돼도 이 결과는 남는다.
        self._autosave()
        self._auto_save_calibration_outputs()
        self._save_to_library()

    def _autosave(self) -> None:
        """마지막으로 완료된 계산 결과를 홈 디렉터리의 고정 파일에 저장한다.

        실제 사용자 버그: 계산 결과가 메모리에만 있고 프로젝트로 저장하지
        않은 상태에서 앱이 응답 없음 -> 강제 종료로 죽으면, 다시 켰을 때
        방금까지 화면에 있던 Model Comparison/Export 결과가 통째로
        사라져서 Export를 다시 할 수 없었다. 이 자동 저장은 사용자가
        직접 [파일 -> 프로젝트 저장]을 누르지 않아도 매 계산 완료 시점마다
        복구 지점을 하나 남겨둔다. 사용자 명시적 저장(_on_save_project)을
        대체하지 않는다 - 그건 여전히 사용자가 원하는 위치/이름으로 남긴다.
        """
        if self.dataset is None or self.camera_config is None or self.pattern_config is None:
            return
        IntrinsicWorkspace.sync_owner_state(self)
        try:
            _AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
            project = CalibrationProject(
                project_name=self.camera_config.sensor_name or "autosave",
                camera_config=self.camera_config,
                pattern_config=self.pattern_config,
                dataset=self.dataset,
                calibration_results=self.calibration_results,
                object_releasing_result=self.object_releasing_result,
                object_releasing_validation_result=self.object_releasing_validation_result,
                standard_vs_object_releasing_comparison=self.standard_vs_object_releasing_comparison,
                validation_results=self.validation_results,
                cross_dataset_results=self.cross_dataset_results,
                model_scores=self.scores,
                outlier_result=self.outlier_result,
                scene_quality_analysis=self.scene_quality_analysis,
                subset_calibration_result=self.subset_calibration_result,
                optimizer_results=self.optimizer_results,
            )
            save_project(project, str(_AUTOSAVE_PATH))
            logger.debug("자동 저장 완료: %s", _AUTOSAVE_PATH)
        except Exception:  # noqa: BLE001 - 자동 저장 실패로 사용자 작업을 막으면 안 됨
            logger.exception("자동 저장 실패 (무시하고 계속 진행)")

    def _auto_save_calibration_outputs(self) -> None:
        """성공한 intrinsic 모델을 현재 Output Session에 자동 저장한다."""
        if self.camera_config is None or self.pattern_config is None:
            return
        saved: list[str] = []
        self._ensure_output_session()
        for model, result in self.calibration_results.items():
            if not result.success:
                continue
            try:
                path = self.output_manager.intrinsic_path(model)
                source = "optimized" if (
                    self.optimizer_results.get(model) and self.optimizer_results[model].applied
                ) else "original_opencv"
                export_opencv_yaml(
                    result, self.camera_config, self.pattern_config, str(path),
                    calibration_source=source,
                )
                key = f"intrinsic.{model_filename(model)}"
                self.output_manager.record_export(key, path, metadata={"automatic": True})
                saved.append(path.name)
            except Exception:  # noqa: BLE001 - 모델 하나의 실패가 나머지를 막지 않게 한다.
                logger.exception("%s 파라미터 자동 저장 실패 (계속 진행)", model)
        if saved:
            self.status_label.setText(
                self.status_label.text() + f"  ·  Output Session 저장: {', '.join(saved)}"
            )

    def _save_to_library(self) -> None:
        """이 실행의 이미지+결과 전체를 Library 탭이 읽는 library/ 폴더로 복사한다.

        이미지 수백 장을 복사하는 파일 I/O라 GUI 스레드를 막지 않도록 QThread로
        분리한다 (Independent Benchmark 검출을 분리한 것과 같은 이유).
        """
        if self.dataset is None or self.camera_config is None or self.pattern_config is None:
            return
        IntrinsicWorkspace.sync_owner_state(self)
        worker = LibrarySaveWorker(
            self.dataset, self.camera_config, self.pattern_config,
            self.calibration_results, self.validation_results, self.scores,
        )
        thread = run_worker_in_thread(worker, self)
        worker.saved.connect(self._on_library_saved)
        worker.error.connect(self._on_library_save_error)
        self._library_thread, self._library_worker = thread, worker
        # 이미지가 많으면 복사에 몇 초~수십 초 걸린다 - 저장이 끝나기 전에
        # 폴더를 열어보고 "왜 없지?"하고 오해하지 않도록, 시작하자마자 바로
        # 진행 중임을 상태표시줄에 남긴다 (완료/실패는 각각의 콜백이 이어서 갱신).
        self.status_label.setText(
            self.status_label.text() + "  ·  Library에 저장 중... (이미지 복사라 시간이 걸릴 수 있습니다)"
        )
        thread.start()

    def _on_library_saved(self, run_dir: str) -> None:
        self.status_label.setText(self.status_label.text() + f"  ·  Library에 저장됨: {run_dir}")
        if hasattr(self, "library_view"):
            self.library_view.mark_dirty()

    def _on_library_save_error(self, message: str) -> None:
        # Library 저장은 부가 기능이라 QMessageBox로 계산 흐름을 막지 않는다 -
        # 다만 콘솔 로그(전체 traceback)와 상태표시줄(요약 한 줄) 양쪽에 남겨서
        # 조용히 사라지지 않게 한다.
        logger.warning(message)
        first_line = message.strip().splitlines()[0] if message.strip() else message
        self.status_label.setText(self.status_label.text() + f"  ·  ⚠ Library 저장 실패: {first_line}")

    def _offer_autosave_recovery(self) -> None:
        """창이 뜨자마자 한 번, 이전 실행의 자동 저장본이 있으면 복구를 제안한다."""
        if not _AUTOSAVE_PATH.exists():
            return
        reply = QMessageBox.question(
            self, "이전 계산 결과 복구",
            "이전 실행에서 자동 저장된 계산 결과가 있습니다.\n"
            "(응답 없음/강제 종료 등으로 예기치 않게 꺼졌을 때를 대비한 백업입니다.)\n\n"
            "지금 불러올까요? ([아니오]를 눌러도 파일은 삭제되지 않습니다.)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            self._load_project_from_path(str(_AUTOSAVE_PATH))

    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "오류", message)
        self.status_label.setText(message)

    # ------------------------------------------------------------------
    # 자체 진단 (합성 데이터 기반 정확도 확인) - "도구" 메뉴
    # ------------------------------------------------------------------

    def _on_run_self_check(self) -> None:
        self.self_check_action.setEnabled(False)
        worker = SelfCheckWorker()
        thread = run_worker_in_thread(worker, self)

        worker.progress.connect(self.status_label.setText)
        worker.result_ready.connect(self._on_self_check_result)
        worker.error.connect(self._on_self_check_error)

        self._self_check_thread, self._self_check_worker = thread, worker
        thread.finished.connect(lambda: self.self_check_action.setEnabled(True))
        thread.start()

    def _on_self_check_result(self, results: list) -> None:
        all_passed = all(r.passed for r in results)
        header = "✅ 모든 항목 통과" if all_passed else "⚠ 일부 항목 실패"
        lines = [f"<b>{header}</b><br><br>"]
        for r in results:
            if not r.success:
                lines.append(f"✕ <b>{r.label}</b>: 계산 실패 - {r.message}<br><br>")
                continue
            mark = "✅" if r.passed else "✕"
            lines.append(f"{mark} <b>{r.label}</b><br>{r.message}<br><br>")

        box = QMessageBox(self)
        box.setWindowTitle("자체 진단 결과")
        box.setTextFormat(Qt.RichText)
        box.setIcon(QMessageBox.Information if all_passed else QMessageBox.Warning)
        box.setText("".join(lines))
        box.setInformativeText(
            "정답을 미리 아는 합성 데이터로 검증한 결과입니다. 실제 카메라로 찍은\n"
            "데이터셋의 정확도를 보장하지는 않으며, 계산 엔진 자체가 정상 동작하는지\n"
            "확인하는 용도입니다."
        )
        box.exec()
        self.status_label.setText(header)

    def _on_self_check_error(self, message: str) -> None:
        QMessageBox.critical(self, "자체 진단 실패", message)
        self.status_label.setText(message)

    def _refresh_result_view(self) -> None:
        method = self.calibration_method_combo.currentData()
        self.result_view.set_advanced_calibration_available(
            method == CalibrationMethod.OBJECT_RELEASING
        )
        self.result_view.set_comparison(
            self.calibration_results,
            self.validation_results,
            self.scores,
            self.object_releasing_result,
            object_releasing_validation=self.object_releasing_validation_result,
            standard_vs_object_releasing=self.standard_vs_object_releasing_comparison,
            image_size=(self.camera_config.width, self.camera_config.height)
            if self.camera_config is not None else None,
        )
        self.result_view.set_cross_dataset_results(self.cross_dataset_results)

    def _update_scene_quality_analysis(self, model: CameraModelType | None = None) -> None:
        if self.dataset is None or self.pattern_config is None:
            return
        if model is None:
            result = next((r for r in self.calibration_results.values() if r.success), None)
            model = result.model_name if result is not None else None
        else:
            try:
                model = model if isinstance(model, CameraModelType) else CameraModelType(str(model))
            except (TypeError, ValueError):
                model = None
            result = next(
                (
                    value for key, value in self.calibration_results.items()
                    if value.success and (
                        value.model_name == model
                        or key == model
                        or str(key) == (model.value if model is not None else "")
                    )
                ),
                None,
            )
        if result is not None and result.success:
            self.scene_quality_analysis = compute_scene_quality_analysis(
                self.dataset, result, self.pattern_config
            )
        else:
            self.scene_quality_analysis = None
        validation = next(
            (
                value for key, value in self.validation_results.items()
                if key == model or str(key) == getattr(model, "value", None)
            ),
            None,
        )
        self.scene_quality_view.set_context(
            self.dataset, self.camera_config, self.calibration_results,
            self.scene_quality_analysis, self.subset_calibration_result,
            frozen_holdout_ids=(validation.test_frame_ids if validation else []),
        )

    def _on_scene_quality_model_changed(self, model: CameraModelType) -> None:
        self._update_scene_quality_analysis(model)

    def _on_subset_recalibrate_requested(self, frame_ids: list[str], model: CameraModelType) -> None:
        if self.dataset is None or self.camera_config is None or self.pattern_config is None:
            QMessageBox.warning(self, "Subset Calibration 불가", "먼저 Initial Calibration을 실행하세요.")
            return
        if model is None and self.scene_quality_analysis is not None:
            model = self.scene_quality_analysis.model_name
        try:
            model = model if isinstance(model, CameraModelType) else CameraModelType(str(model))
        except (TypeError, ValueError):
            QMessageBox.warning(
                self, "Subset Calibration 불가",
                "선택된 Camera Model이 없습니다. Initial Calibration을 다시 실행하세요.",
            )
            return
        original = next(
            (
                value for key, value in self.calibration_results.items()
                if value.model_name == model or key == model or str(key) == model.value
            ),
            None,
        )
        if original is None or not original.success:
            QMessageBox.warning(
                self, "Subset Calibration 불가",
                f"{model.value} Initial Calibration 결과가 없거나 실패했습니다.",
            )
            return
        original_validation = next(
            (
                value for key, value in self.validation_results.items()
                if key == model or str(key) == model.value
            ),
            None,
        )
        frozen_holdout_ids = list(original_validation.test_frame_ids) if original_validation else []
        leaked = sorted(set(frame_ids) & set(frozen_holdout_ids))
        if leaked:
            QMessageBox.critical(
                self, "Subset Calibration 차단",
                "선택 장면에 Frozen Hold-out이 포함되어 데이터 누수가 발생합니다:\n"
                + ", ".join(leaked),
            )
            return
        worker = SceneSubsetCalibrationWorker(
            self.dataset, frame_ids, self.camera_config, self.pattern_config, model,
            frozen_holdout_ids=frozen_holdout_ids,
        )
        thread = run_worker_in_thread(worker, self)
        worker.progress.connect(self.status_label.setText)
        worker.result_ready.connect(self._on_subset_calibration_ready)
        worker.error.connect(self._on_error)
        self.scene_quality_view.recalibrate_button.setEnabled(False)
        thread.finished.connect(lambda: self.scene_quality_view.recalibrate_button.setEnabled(True))
        self._subset_thread, self._subset_worker = thread, worker
        thread.start()

    def _on_subset_calibration_ready(self, result) -> None:
        result.original_validation_result = next(
            (
                value for key, value in self.validation_results.items()
                if key == result.model_name or str(key) == result.model_name.value
            ),
            None,
        )
        add_original_comparison_warnings(
            result,
            next(
                (
                    value for key, value in self.calibration_results.items()
                    if value.model_name == result.model_name
                    or key == result.model_name
                    or str(key) == result.model_name.value
                ),
                None,
            ),
            result.original_validation_result,
        )
        self.subset_calibration_result = result
        if result.calibration_result and result.calibration_result.success:
            self.status_label.setText(
                f"Subset Calibration 완료: {len(result.selected_frame_ids)} scenes, "
                f"RMS {result.calibration_result.rms_error:.3f}px"
            )
        elif result.calibration_result:
            self.status_label.setText(f"Subset Calibration 실패: {result.calibration_result.error_message}")
        self.scene_quality_view.set_context(
            self.dataset, self.camera_config, self.calibration_results,
            self.scene_quality_analysis, self.subset_calibration_result,
            frozen_holdout_ids=(result.original_validation_result.test_frame_ids if result.original_validation_result else []),
        )
        self._autosave()

    def _image_paths_from_directory(self, directory: str) -> list[str]:
        paths: list[str] = []
        p = Path(directory)
        for ext in _IMAGE_EXTENSIONS:
            paths.extend(sorted(str(x) for x in p.glob(ext)))
        return paths

    def _on_cross_dataset_requested(self) -> None:
        if (
            self.dataset is None or self.camera_config is None or self.pattern_config is None
            or not any(r.success for r in self.calibration_results.values())
        ):
            QMessageBox.warning(self, "Cross-Dataset 불가", "먼저 캘리브레이션을 실행하세요.")
            return

        directory = QFileDialog.getExistingDirectory(self, "Dataset B/C 이미지 폴더 선택")
        if not directory:
            return
        paths = self._image_paths_from_directory(directory)
        if not paths:
            QMessageBox.warning(self, "이미지 없음", "선택한 폴더에서 jpg/jpeg/png/bmp 이미지를 찾지 못했습니다.")
            return

        default_label = Path(directory).name or f"Dataset {len(self.cross_dataset_results) + 1}"
        dataset_id, ok = QInputDialog.getText(
            self,
            "Target Dataset Label",
            "Report/JSON에 표시할 target dataset 이름:",
            text=default_label,
        )
        if not ok:
            return
        dataset_id = dataset_id.strip() or default_label

        source_dataset_id = self.camera_config.sensor_name or "Dataset A"
        worker = CrossDatasetValidationWorker(
            {dataset_id: paths},
            self.calibration_results,
            self.camera_config,
            self.pattern_config,
            source_dataset_id=source_dataset_id,
        )
        thread = run_worker_in_thread(worker, self)
        worker.progress.connect(self.status_label.setText)
        worker.results_ready.connect(self._on_cross_dataset_results_ready)
        worker.error.connect(self._on_error)

        self._thread, self._worker = thread, worker
        self.result_view.cross_dataset_button.setEnabled(False)
        thread.finished.connect(lambda: self.result_view.cross_dataset_button.setEnabled(True))
        thread.start()

    def _on_cross_dataset_results_ready(self, results: list[CrossDatasetValidationResult]) -> None:
        self.cross_dataset_results.extend(results)
        self.result_view.set_cross_dataset_results(self.cross_dataset_results)
        ok = sum(1 for r in results if r.success)
        self.status_label.setText(f"Cross-dataset validation 완료: {ok}/{len(results)} 성공")
        self._autosave()

    def _on_repeated_kfold_requested(self, k: int, n_repeats: int) -> None:
        """논문용 Repeated K-Fold(Brown-Conrady/Rational/Fisheye 동일 fold
        partition 비교) 실행. calibration/kfold.py::run_repeated_kfold_all_models
        가 실제 계산을 전부 담당하고, 여기서는 QThread 배선만 한다."""
        if self.dataset is None or self.camera_config is None or self.pattern_config is None:
            QMessageBox.warning(self, "Repeated K-Fold 불가", "먼저 데이터셋을 불러오세요.")
            return

        # ResultView.start_repeated_kfold_progress()는 이미 버튼 클릭 시점에
        # ResultView._on_kfold_run_clicked()에서 호출됐다(progress bar/상태
        # 표를 즉시 0/total, WAITING으로 되돌려 "실행 중" 신호를 바로 보여줌) -
        # 여기서는 실제 백그라운드 계산을 시작하는 QThread 배선만 한다.
        worker = RepeatedKFoldWorker(
            self.dataset, self.camera_config, self.pattern_config, k=k, n_repeats=n_repeats,
        )
        thread = run_worker_in_thread(worker, self)
        worker.progress.connect(self.status_label.setText)
        worker.progress_event.connect(self.result_view.update_repeated_kfold_progress)
        worker.results_ready.connect(self._on_repeated_kfold_results_ready)
        worker.error.connect(self._on_repeated_kfold_error)

        self._kfold_thread, self._kfold_worker = thread, worker
        self.result_view.kfold_run_button.setEnabled(False)
        # 계산 중에는 Export Paper Metrics도 막는다 - 계산이 끝나기 전에는
        # (아직 self.repeated_kfold_results가 새 값으로 갱신되지 않았으므로)
        # export할 게 없거나, 있어도 이번 실행이 끝나길 기다리는 게 맞다.
        self.result_view.kfold_export_button.setEnabled(False)
        thread.finished.connect(lambda: self.result_view.kfold_run_button.setEnabled(True))
        thread.finished.connect(lambda: self.result_view.kfold_export_button.setEnabled(True))
        thread.start()

    def _on_repeated_kfold_results_ready(self, results: dict) -> None:
        # Export Paper Metrics는 이 저장된 결과를 그대로 쓴다 - 버튼을 누를
        # 때 다시 계산하지 않는다. 새 calibration 실행/project 로드 시에는
        # _on_run_pipeline()/project 로드 핸들러가 이 값을 비워 stale export를 막는다.
        self.repeated_kfold_results = results
        self.result_view.set_repeated_kfold_results(results)
        total = next(iter(results.values())).total_folds if results else 0
        ok = sum(r.n_successful_runs for r in results.values())
        self.status_label.setText(f"Repeated K-Fold 완료: {ok}/{total * len(results)} fold 성공 (모델별 표 참고)")

    def _on_repeated_kfold_error(self, message: str) -> None:
        """일반 _on_error와 달리, progress bar는 그대로 두고(어디까지
        진행됐었는지 보여줌) RUNNING이었던 모델 행만 FAILED로 표시한다 -
        Run 버튼 재활성화는 기존과 동일하게 thread.finished가 처리한다."""
        self.result_view.set_repeated_kfold_error(message)
        self._on_error(message)

    def _on_export_paper_metrics_requested(self) -> None:
        """"Export Paper Metrics" - calibration/paper_evidence.py가 이미
        계산해 둔 export_paper_metrics()를 그대로 호출한다. 이 핸들러는
        계산을 전혀 하지 않는다 - self.repeated_kfold_results(Run Repeated
        K-Fold가 채워 둔 저장된 결과)를 그대로 쓰고, 버튼을 누를 때
        K-Fold를 다시 계산하지 않는다.

        기존 OpenCV YAML Export(_on_export_opencv, deployment용 K/D 저장)와
        완전히 별개다 - 여기는 논문 분석용 raw evidence(CSV/JSON)만 만든다.
        """
        if not self.calibration_results:
            QMessageBox.warning(self, "Export Paper Metrics 불가", "먼저 Calibration을 실행하세요.")
            return
        if not self.validation_results:
            QMessageBox.warning(self, "Export Paper Metrics 불가", "먼저 Validation을 실행하세요.")
            return
        if not self.repeated_kfold_results:
            QMessageBox.warning(
                self, "Export Paper Metrics 불가",
                "논문용 Repeated K-Fold 결과가 없습니다. 먼저 Run Repeated K-Fold를 실행하세요.",
            )
            return

        try:
            self._ensure_output_session()
            directory = self.output_manager.paper_directory()
            # Paper Intrinsic Stability(fx/fy/cx/cy)와 All-Parameter
            # Stability(overall_stability, distortion 포함 가능)는
            # ParameterUncertainty 안에 이미 별도 필드로 분리되어 있다 -
            # ui/stability_view.py와 동일하게 bootstrap 결과를 우선한다.
            stability_by_model = {
                model: (cal.param_uncertainty_bootstrap or cal.param_uncertainty)
                for model, cal in self.calibration_results.items()
            }
            any_repeated = next(iter(self.repeated_kfold_results.values()))
            metadata = paper_evidence.build_paper_metadata(
                self.camera_config, self.pattern_config, self.dataset,
                k=any_repeated.k, n_repeats=any_repeated.n_repeats, base_seed=any_repeated.base_seed,
            )
            written = paper_evidence.export_paper_metrics(
                str(directory),
                single_holdout=self.validation_results,
                repeated=self.repeated_kfold_results,
                stability_by_model=stability_by_model,
                metadata=metadata,
            )
        except Exception as e:  # noqa: BLE001 - export 실패가 GUI를 죽이면 안 됨
            QMessageBox.critical(self, "Export Paper Metrics 실패", f"Paper Metrics export 중 오류가 발생했습니다:\n{e}")
            return

        paths = [directory / name for name in written.keys()]
        self.output_manager.record_export("paper.metrics", paths)
        file_list = "\n".join(f"- {name}" for name in sorted(written.keys()))
        message = f"Paper Metrics export 완료\n{directory}\n\nGenerated:\n{file_list}"
        QMessageBox.information(self, "Export Paper Metrics", message)
        self.status_label.setText(f"Paper Metrics export 완료: {directory} ({len(written)}개 파일)")

    # ------------------------------------------------------------------
    # Export (OpenCV YAML - deployment용, Paper Metrics와 완전히 별개)
    # ------------------------------------------------------------------

    def _on_export_button_clicked(self) -> None:
        """Camera Setup 오른쪽 열의 Export 버튼. 옛 "⑦ Export" 탭에 있던
        모델 선택 콤보/상태 표시/Export 버튼(ResultView.export_widget)을
        그대로 다이얼로그에 담아 보여준다 - 계산 로직은 하나도 바뀌지 않고
        탭 자리만 다이얼로그로 옮겼다.
        """
        if self._export_dialog is None:
            self._export_dialog = self._build_export_dialog()
        self._export_dialog.show()
        self._export_dialog.raise_()
        self._export_dialog.activateWindow()

    def _build_export_dialog(self) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle("Export")
        layout = QVBoxLayout(dialog)
        layout.addWidget(self.result_view.export_widget)
        close_button = QPushButton("닫기")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        return dialog

    def _on_export_opencv(self, model: CameraModelType) -> None:
        result = self.calibration_results.get(model)
        if not result or not result.success or self.camera_config is None or self.pattern_config is None:
            QMessageBox.warning(self, "Export 불가", f"{CameraModelType(model).value} 모델의 캘리브레이션 결과가 없습니다.")
            return
        try:
            self._ensure_output_session()
            path = self.output_manager.intrinsic_path(model)
            source = "optimized" if (
                self.optimizer_results.get(model) and self.optimizer_results[model].applied
            ) else "original_opencv"
            export_opencv_yaml(
                result, self.camera_config, self.pattern_config, str(path),
                calibration_source=source,
            )
            self.output_manager.record_export(f"intrinsic.{model_filename(model)}", path)
            self.status_label.setText(f"OpenCV YAML 저장 완료: {path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Export 실패", str(e))

    def _on_export_subset_calibration(self) -> None:
        subset = self.subset_calibration_result
        result = subset.calibration_result if subset is not None else None
        if (
            subset is None or result is None or not result.success
            or self.camera_config is None or self.pattern_config is None
        ):
            QMessageBox.warning(
                self, "Subset Export 불가",
                "먼저 Scene을 선택해 Re-Calibration을 완료하세요.",
            )
            return
        try:
            self._ensure_output_session()
            path = self.output_manager.intrinsic_path(result.model_name, subset=True)
            export_opencv_yaml(
                result,
                self.camera_config,
                self.pattern_config,
                str(path),
                calibration_source="best_subset",
                selected_frame_ids=subset.selected_frame_ids,
            )
            self.output_manager.record_export(
                f"intrinsic.subset.{model_filename(result.model_name)}",
                path,
                metadata={"selected_frame_ids": subset.selected_frame_ids},
            )
            self.status_label.setText(
                f"Subset OpenCV YAML 저장 완료: {path} "
                f"({len(subset.selected_frame_ids)} scenes)"
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Subset Export 실패", str(exc))

    def _on_validate_best_subset(self) -> None:
        """Collect paths in the GUI, then run the same evaluator used by CLI."""
        if self.dataset is None or self.camera_config is None or self.pattern_config is None:
            QMessageBox.warning(self, "Validation 불가", "먼저 hold-out 이미지가 포함된 Dataset을 불러오세요.")
            return
        baseline, _ = QFileDialog.getOpenFileName(
            self, "Baseline calibration YAML", "", "Calibration (*.yaml *.yml *.json)"
        )
        if not baseline:
            return
        candidate, _ = QFileDialog.getOpenFileName(
            self, "Best Subset calibration YAML", "", "Calibration (*.yaml *.yml *.json)"
        )
        if not candidate:
            return
        manifest, _ = QFileDialog.getOpenFileName(
            self, "Frozen hold-out split manifest", "", "Manifest (*.json *.yaml *.yml)"
        )
        if not manifest:
            return
        self._ensure_output_session()
        output_dir = str(self.output_manager.validation_directory())
        relative_pct, ok = QInputDialog.getDouble(
            self, "판정 tolerance", "Relative tolerance (%):", 5.0, 0.0, 100.0, 1
        )
        if not ok:
            return
        absolute_px, ok = QInputDialog.getDouble(
            self, "판정 tolerance", "Absolute tolerance (px):", 0.10, 0.0, 100.0, 3
        )
        if not ok:
            return
        worker = SubsetComparisonWorker(
            self.dataset, self.camera_config, self.pattern_config,
            baseline, candidate, manifest, output_dir,
            ComparisonTolerance(relative=relative_pct / 100.0, absolute_px=absolute_px),
        )
        thread = run_worker_in_thread(worker, self)
        progress = QProgressDialog("Frozen hold-out 비교 준비 중...", "취소", 0, 0, self)
        progress.setWindowTitle("Validate Best Subset")
        progress.setWindowModality(Qt.WindowModal)
        progress.canceled.connect(worker.request_cancel)
        worker.progress.connect(progress.setLabelText)
        worker.error.connect(self._on_error)
        worker.result_ready.connect(self._on_subset_comparison_ready)
        worker.finished.connect(progress.close)
        self._subset_comparison_thread = thread
        self._subset_comparison_worker = worker
        self._subset_comparison_progress = progress
        progress.show()
        thread.start()

    def _on_subset_comparison_ready(self, result, paths) -> None:
        if getattr(self, "output_manager", None) is not None:
            self.output_manager.record_export(
                "validation.subset_comparison", list(paths.values()), metadata={"verdict": result.get("verdict")}
            )
        def metric(name: str) -> str:
            row = result["comparison"][name]
            left = "N/A" if row["baseline"] is None else f"{row['baseline']:.3f}"
            right = "N/A" if row["candidate"] is None else f"{row['candidate']:.3f}"
            return f"{name}: {left} → {right} ({row['status']})"

        summary = "\n".join([
            f"{result['verdict']} — {' '.join(result['verdict_reasons'])}",
            "Data leakage check: PASS",
            metric("train_rms"), metric("holdout_rms"), metric("p95"),
            metric("p99"), metric("edge_rms"), metric("straightness"),
            f"Success: {result['baseline']['success_rate'] * 100:.1f}% → "
            f"{result['candidate']['success_rate'] * 100:.1f}%",
            f"Common frames: {result['paired']['common_frame_count']}",
            f"Report: {paths['report']}",
        ])
        dialog = QDialog(self)
        dialog.setWindowTitle("Best Subset Validation 결과")
        dialog.resize(900, 620)
        layout = QVBoxLayout(dialog)
        summary_label = QLabel(summary)
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)
        table = QTableWidget(len(result["per_frame"]), 7)
        table.setHorizontalHeaderLabels([
            "Frame", "Common", "Baseline RMS", "Subset RMS", "Delta",
            "Baseline Edge", "Subset Edge",
        ])
        for row_index, row in enumerate(result["per_frame"]):
            values = [
                row["frame_id"], "Yes" if row["common_success"] else "No",
                row.get("baseline_rms"), row.get("candidate_rms"), row.get("rms_delta"),
                row.get("baseline_edge_rms"), row.get("candidate_edge_rms"),
            ]
            for column, value in enumerate(values):
                text = f"{value:.3f}" if isinstance(value, float) else ("N/A" if value is None else str(value))
                table.setItem(row_index, column, QTableWidgetItem(text))
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.resizeColumnsToContents()
        layout.addWidget(table, stretch=1)
        close_button = QPushButton("닫기")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        dialog.exec()
        self.status_label.setText(f"Best Subset Validation {result['verdict']}: {paths['report']}")

    # ------------------------------------------------------------------
    # 프로젝트 저장/불러오기 (.ccproj)
    # ------------------------------------------------------------------

    def _on_save_project(self) -> None:
        """Backward-compatible Save As entry point used by older integrations."""
        self._on_save_project_as()

    def _on_save_project_session(self) -> None:
        if self.dataset is None or self.camera_config is None or self.pattern_config is None:
            QMessageBox.warning(self, "저장할 내용 없음", "먼저 이미지를 불러오고 캘리브레이션을 실행하세요.")
            return
        IntrinsicWorkspace.sync_owner_state(self)
        self._ensure_output_session()
        path = str(self.output_manager.project_path())
        self._save_project_to_path(path)

    def _on_save_project_as(self) -> None:
        if self.dataset is None or self.camera_config is None or self.pattern_config is None:
            QMessageBox.warning(self, "저장할 내용 없음", "먼저 이미지를 불러오고 캘리브레이션을 실행하세요.")
            return
        IntrinsicWorkspace.sync_owner_state(self)
        path, _ = QFileDialog.getSaveFileName(
            self, "프로젝트 저장", f"project{PROJECT_EXTENSION}", f"Camera Calibrator Project (*{PROJECT_EXTENSION})"
        )
        if not path:
            return
        if not path.endswith(PROJECT_EXTENSION):
            path += PROJECT_EXTENSION

        self._save_project_to_path(path, record=False)

    def _save_project_to_path(
        self, path: str, *, record: bool = True, notify: bool = True
    ) -> str | None:
        windshield_config, windshield_dataset, windshield_results, reflection_results, ghost_results, ghost_models = self.windshield_workspace.export_state()
        project = CalibrationProject(
            project_name=self.camera_config.sensor_name or Path(path).stem,
            camera_config=self.camera_config,
            pattern_config=self.pattern_config,
            dataset=self.dataset,
            calibration_results=self.calibration_results,
            object_releasing_result=self.object_releasing_result,
            object_releasing_validation_result=self.object_releasing_validation_result,
            standard_vs_object_releasing_comparison=self.standard_vs_object_releasing_comparison,
            validation_results=self.validation_results,
            cross_dataset_results=self.cross_dataset_results,
            model_scores=self.scores,
            outlier_result=self.outlier_result,
            scene_quality_analysis=self.scene_quality_analysis,
            subset_calibration_result=self.subset_calibration_result,
            optimizer_results=self.optimizer_results,
            windshield_config=windshield_config,
            windshield_dataset=windshield_dataset,
            windshield_results=windshield_results,
            reflection_results=reflection_results,
            ghost_results=ghost_results,
            ghost_models=ghost_models,
        )
        try:
            saved_path = save_project(project, path)
            if record:
                self.output_manager.record_export("project", saved_path)
            if notify:
                self.status_label.setText(f"프로젝트 저장 완료: {saved_path}")
            return str(saved_path)
        except Exception as e:  # noqa: BLE001
            if notify:
                QMessageBox.critical(self, "저장 실패", str(e))
            return None

    def _on_load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "프로젝트 불러오기", "", f"Camera Calibrator Project (*{PROJECT_EXTENSION})"
        )
        if not path:
            return
        self._load_project_from_path(path)

    def _load_project_from_path(self, path: str) -> None:
        try:
            project, missing = load_project(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "불러오기 실패", str(e))
            return

        # --- 상태 복원 ---
        self.dataset = project.dataset
        self.camera_config = project.camera_config
        self.pattern_config = project.pattern_config
        self.calibration_results = project.calibration_results
        self.object_releasing_result = project.object_releasing_result
        self.object_releasing_validation_result = project.object_releasing_validation_result
        self.standard_vs_object_releasing_comparison = project.standard_vs_object_releasing_comparison
        self.validation_results = project.validation_results
        self.cross_dataset_results = project.cross_dataset_results
        self.scores = project.model_scores
        self.outlier_result = project.outlier_result
        self.scene_quality_analysis = project.scene_quality_analysis
        self.subset_calibration_result = project.subset_calibration_result
        self.optimizer_results = project.optimizer_results
        # Repeated K-Fold 결과는 .ccproj에 저장되지 않는다(계산 비용이 크고,
        # 이 프로젝트가 로드된 dataset/calibration과 실제로 짝이 맞는
        # 결과인지 보장할 방법이 없다) - 프로젝트를 불러오면 항상 비우고,
        # 필요하면 사용자가 다시 "Run Repeated K-Fold"를 눌러야 한다.
        self.repeated_kfold_results = {}
        self.result_view.set_repeated_kfold_results({})
        self.image_paths = [f.image_info.path for f in project.dataset.frames]
        self.windshield_workspace.import_state(project)
        IntrinsicWorkspace.sync_owner_state(self)

        # --- 설정 패널 위젯도 불러온 값으로 맞춰준다 (재계산/이어서 작업 시 일관성) ---
        self.sensor_name_edit.setText(self.camera_config.sensor_name or "")
        self.width_spin.setValue(self.camera_config.width)
        self.height_spin.setValue(self.camera_config.height)
        self.squares_x_spin.setValue(self.pattern_config.squares_x)
        self.squares_y_spin.setValue(self.pattern_config.squares_y)
        # pattern_config는 항상 미터(m) 단위로 저장돼 있으니, mm 입력 위젯에는 변환해서 넣는다.
        self.square_size_spin.setValue(self.pattern_config.square_size * 1000.0)
        loaded_method = (
            CalibrationMethod.OBJECT_RELEASING
            if any((
                self.object_releasing_result is not None,
                self.object_releasing_validation_result is not None,
                self.standard_vs_object_releasing_comparison is not None,
            ))
            else CalibrationMethod.STANDARD
        )
        idx = self.calibration_method_combo.findData(loaded_method)
        if idx >= 0:
            self.calibration_method_combo.setCurrentIndex(idx)
        self.calibration_method = loaded_method
        idx = self.pattern_type_combo.findData(self.pattern_config.type)
        if idx >= 0:
            self.pattern_type_combo.setCurrentIndex(idx)  # _on_pattern_type_changed가 자동으로 행 토글
        if self.pattern_config.marker_size is not None:
            self.marker_size_spin.setValue(self.pattern_config.marker_size * 1000.0)
        if self.pattern_config.dictionary:
            self.dictionary_combo.setCurrentText(self.pattern_config.dictionary)
        idx = self.circle_grid_type_combo.findData(self.pattern_config.circle_grid_type)
        if idx >= 0:
            self.circle_grid_type_combo.setCurrentIndex(idx)
        idx = self.aprilgrid_variant_combo.findData(self.pattern_config.aprilgrid_variant)
        if idx >= 0:
            self.aprilgrid_variant_combo.setCurrentIndex(idx)

        # --- 각 탭 새로고침 ---
        self.dataset_view.set_dataset(self.dataset)
        self.dataset_view.set_quality(self.dataset.coverage_grid, self.dataset.diversity, [])
        if self.calibration_results:
            self.preview_view.set_context(self.dataset, self.camera_config, self.calibration_results, self.pattern_config)
        self._refresh_result_view()
        self.optimizer_view.set_context(
            self.calibration_results, self.validation_results, self.optimizer_results, self.camera_config
        )
        self._update_scene_quality_analysis(
            self.scene_quality_analysis.model_name if self.scene_quality_analysis else None
        )
        if self.scores:
            recommended = next((s.model_name for s in self.scores if s.is_recommended), None)
            if recommended is not None:
                self.result_view.select_model(recommended)
                self.preview_view.select_model(recommended)

        self.loaded_label.setText(f"불러온 이미지: {len(self.image_paths)}장 (프로젝트: {Path(path).name})")
        self.auto_select_status_label.setText("프로젝트 데이터셋을 불러왔습니다. 필요하면 다시 자동 선별할 수 있습니다.")
        self._sync_auto_select_controls(reset_value=True)
        self.run_button.setEnabled(bool(self.image_paths))

        msg = f"프로젝트 불러옴: {project.project_name} ({self.dataset.num_total}장, 검출 {self.dataset.num_detected}장)"
        self.status_label.setText(msg)

        if missing:
            preview = "\n".join(missing[:10]) + (f"\n... 외 {len(missing) - 10}개" if len(missing) > 10 else "")
            QMessageBox.warning(
                self, "이미지 파일 없음",
                f"원본 이미지 {len(missing)}개를 찾을 수 없습니다 (경로가 바뀌었거나 삭제됨).\n"
                f"캘리브레이션 결과 확인/Export/이상치 재계산은 이미지 없이도 가능하지만, "
                f"'② Preview' 탭에서는 해당 이미지가 표시되지 않습니다.\n\n{preview}",
            )
