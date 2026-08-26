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
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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
    QStackedWidget,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from calibration.types import (
    CalibrationResult,
    CalibrationProject,
    CameraConfig,
    CameraModelType,
    CrossDatasetValidationResult,
    Dataset,
    ModelScore,
    OutlierResult,
    PatternConfig,
    PatternType,
    ValidationResult,
)
from calibration.calibration_io import StandardCalibration
from calibration.recommender import compute_final_result
from calibration.sanity_check import run_sanity_checks
from calibration.quality import coverage_percentage
from calibration.ros_live import ROS_LIVE_BACKEND
from calibration.project_io import load_project, save_project, PROJECT_EXTENSION
from export.opencv import export_opencv_yaml
from export.ros import export_ros_camera_info
from export.report import export_html_report
from export.json_export import export_json
from export.csv_export import export_csv
from export.stereo import stereo_pairs_to_dict, stereo_result_to_dict

from ui.calibration_home_view import CalibrationHomeView
from ui.help_view import HelpView
from ui.intrinsic_workspace import IntrinsicWorkspace
from ui.stereo_workspace import StereoWorkspace
from ui.live_capture_dialog import LiveCaptureDialog
from ui.wheel_guard import WheelChangeGuard
from ui.worker import (
    PipelineWorker,
    OutlierPruneWorker,
    CrossDatasetValidationWorker,
    ModelRefittingWorker,
    SelfCheckWorker,
    BagTopicDiscoveryWorker,
    BagExtractionWorker,
    LibrarySaveWorker,
    run_worker_in_thread,
)
from ui.library_view import LibraryView

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
    def use_rational_model(self):
        return self.intrinsic_state.use_rational_model

    @use_rational_model.setter
    def use_rational_model(self, value):
        self.intrinsic_state.use_rational_model = value

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Calibration Tool")
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
        self._bag_thread: QThread | None = None
        self._bag_worker = None  # QThread가 살아있는 동안 GC 방지용 강한 참조
        self._bag_progress_dialog: QProgressDialog | None = None
        self._bag_topic_thread: QThread | None = None
        self._bag_topic_worker = None
        self._bag_topic_progress_dialog: QProgressDialog | None = None
        self._self_check_worker = None  # 위와 동일한 이유로 별도 워커도 강한 참조 보관
        self._model_refit_thread: QThread | None = None
        self._model_refit_worker = None
        self._library_thread: QThread | None = None
        self._library_worker = None
        self._pending_stereo_intrinsic_slot: str | None = None

        self._build_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.workspace_stack = QStackedWidget()
        self.home_view = CalibrationHomeView()
        settings_panel = self._build_settings_panel()

        self.intrinsic_workspace = IntrinsicWorkspace.create_for_main_window(self, settings_panel)
        self.intrinsic_workspace.back_requested.connect(self._show_home)
        self.stereo_workspace = StereoWorkspace()
        self.library_view = LibraryView()
        self.workspace_stack.addWidget(self.home_view)
        self.workspace_stack.addWidget(self.intrinsic_workspace)
        self.workspace_stack.addWidget(self.stereo_workspace)
        self.workspace_stack.addWidget(self.library_view)
        layout.addWidget(self.workspace_stack, stretch=1)

        self.status_label = QLabel("이미지를 불러온 뒤 [캘리브레이션 실행]을 누르세요.")
        self.statusBar().addWidget(self.status_label, stretch=1)
        self.pipeline_progress_bar = QProgressBar()
        self.pipeline_progress_bar.setMinimumWidth(180)
        self.pipeline_progress_bar.setTextVisible(True)
        self.pipeline_progress_bar.hide()
        self.statusBar().addPermanentWidget(self.pipeline_progress_bar)
        # 진행률을 알 수 없는 구간(3모델 + Hold-out 계산 중)에서 기본 Qt
        # 인디케이터 대신 양이 진행 바를 가로질러 걸어가는 애니메이션을 보여준다.
        self._sheep_pos = 0
        self._sheep_timer = QTimer(self)
        self._sheep_timer.setInterval(140)
        self._sheep_timer.timeout.connect(self._advance_busy_sheep)

        self.home_view.intrinsic_requested.connect(self._show_intrinsic_workspace)
        self.home_view.stereo_requested.connect(self._show_stereo_workspace)
        self.home_view.library_requested.connect(self._show_library_workspace)
        self.library_view.back_requested.connect(self._show_home)
        self.stereo_workspace.back_requested.connect(self._show_home)
        self.stereo_workspace.calibrate_intrinsic_requested.connect(self._on_stereo_intrinsic_requested)

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

    def _show_stereo_workspace(self) -> None:
        self.stereo_workspace.set_pattern_config(self.pattern_config or self._current_pattern_config())
        self.workspace_stack.setCurrentWidget(self.stereo_workspace)
        self.status_label.setText("Camera-to-Camera Stereo Workspace")

    def _show_library_workspace(self) -> None:
        self.workspace_stack.setCurrentWidget(self.library_view)
        self.status_label.setText("Library")

    def _on_stereo_intrinsic_requested(self, slot: str) -> None:
        self._pending_stereo_intrinsic_slot = slot
        if self.calibration_results and self.camera_config is not None:
            chosen = next((s.model_name for s in self.scores if s.is_recommended), None)
            if chosen is None:
                chosen = self.result_view.model_combo.currentData()
            result = self.calibration_results.get(chosen)
            if result is not None and result.success:
                self.stereo_workspace.set_previous_intrinsic(slot, self._standard_from_result(result))
                self._show_stereo_workspace()
                return
        self._show_intrinsic_workspace()
        self.status_label.setText(
            f"{slot} Intrinsic이 필요합니다. 기존 Intrinsic Workspace에서 캘리브레이션을 완료하면 자동으로 연결됩니다."
        )

    def _standard_from_result(self, result: CalibrationResult) -> StandardCalibration:
        return StandardCalibration(
            label=result.model_name.value,
            camera_matrix=result.camera_matrix,
            distortion=result.distortion,
            model_name=result.model_name,
            distortion_model=None,
            width=self.camera_config.width if self.camera_config else None,
            height=self.camera_config.height if self.camera_config else None,
            camera_name=self.camera_config.sensor_name if self.camera_config else None,
            source_format="camera_calibrator_result",
        )

    # ------------------------------------------------------------------
    # 설정 패널 (설계 문서 14번 ① Camera Setup, ③ Calibration Pattern)
    # ------------------------------------------------------------------

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("파일")

        save_action = QAction("프로젝트 저장...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

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
            "정답을 미리 아는 가짜(합성) ChArUco 데이터로 Pinhole/Extended Pinhole/\n"
            "Rational model을 돌려서 복원된 fx/fy/cx/cy가 정답에 가까운지 확인합니다.\n"
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
        outer = QHBoxLayout(content)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(20)  # 3등분 섹션(Camera/Pattern/Actions) 사이 가로 간격만 살짝 추가
        group_layout.addWidget(content)
        self.settings_group = group
        self.settings_content = content
        group.toggled.connect(self._on_settings_panel_toggled)

        camera_form = QFormLayout()
        camera_form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        # QFormLayout에 세로 간격을 따로 정해주지 않으면 Qt가 부모 레이아웃인
        # outer(QHBoxLayout)의 spacing 값을 그대로 물려받는다 - 그래서
        # outer.setSpacing()으로 가로 간격만 넓혔는데도 이 폼의 행간(세로
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
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 20000)
        self.width_spin.setValue(1920)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 20000)
        self.height_spin.setValue(1536)
        camera_form.addRow("Width", self.width_spin)
        camera_form.addRow("Height", self.height_spin)

        self.rational_checkbox = QCheckBox("Rational model 사용 (k4~k6 포함)")
        self.rational_checkbox.setToolTip(
            "Extended Pinhole에서 cv2.CALIB_RATIONAL_MODEL을 켭니다.\n"
            "기본값(꺼짐)은 k1,k2,p1,p2,k3 5계수만 추정합니다.\n"
            "켜면 k1~k6,p1,p2 8개를 추정합니다 (OpenCV 버전에 따라 배열 길이\n"
            "자체는 14칸으로 나올 수 있으나, 나머지 6개(s1~s4,taux,tauy)는\n"
            "항상 0으로 고정되고 실제 추정 자유도는 8개입니다).\n"
            "⚠ 파라미터가 많을수록 데이터가 충분치 않으면 k4~k6 값이 불안정하게\n"
            "튈 수 있습니다. 광각/왜곡이 매우 심한 렌즈 + 데이터가 많을 때만\n"
            "권장하며, 켠 뒤에는 Model Score/Test RMS로 실제 개선됐는지 꼭 확인하세요.\n"
            "(CLI의 --rational 플래그와 동일한 옵션입니다.)"
        )
        camera_form.addRow("", self.rational_checkbox)
        outer.addLayout(camera_form, stretch=1)

        pattern_form = QFormLayout()
        pattern_form.setRowWrapPolicy(QFormLayout.DontWrapRows)
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

        self.pattern_type_combo = QComboBox()
        # userData로 PatternType을 직접 들고 있어서 _current_pattern_config()가
        # 문자열 비교 없이 바로 꺼내 쓸 수 있다.
        self.pattern_type_combo.addItem("ChArUco (권장)", userData=PatternType.CHARUCO)
        self.pattern_type_combo.addItem("Chessboard (일반 체스보드)", userData=PatternType.CHESSBOARD)
        self.pattern_type_combo.addItem("AprilGrid (AprilTag grid)", userData=PatternType.APRILGRID)
        self.pattern_type_combo.currentIndexChanged.connect(self._on_pattern_type_changed)

        pattern_form.addRow("Pattern type", self.pattern_type_combo)
        pattern_form.addRow("Squares X", self.squares_x_spin)
        pattern_form.addRow("Squares Y", self.squares_y_spin)
        pattern_form.addRow("Square size", self.square_size_spin)
        pattern_form.addRow("Marker size", self.marker_size_spin)
        pattern_form.addRow("Dictionary", self.dictionary_combo)
        self._pattern_form = pattern_form  # setRowVisible로 마커/딕셔너리 행을 토글하기 위해 보관
        outer.addLayout(pattern_form, stretch=1)

        action_layout = QVBoxLayout()
        self.load_button = QPushButton("이미지 불러오기")
        self.load_button.clicked.connect(self._on_load_images)
        self.load_bag_button = QPushButton("rosbag에서 불러오기")
        self.load_bag_button.clicked.connect(self._on_load_from_bag)
        self.load_live_button = QPushButton("실시간 카메라 구독")
        self.load_live_button.clicked.connect(self._on_load_from_live)
        self.loaded_label = QLabel("불러온 이미지: 0장")
        self.run_button = QPushButton("캘리브레이션 실행")
        self.run_button.setProperty("role", "primary")
        self.run_button.clicked.connect(self._on_run_pipeline)
        self.run_button.setEnabled(False)
        action_layout.addWidget(self.load_button)
        action_layout.addWidget(self.load_bag_button)
        action_layout.addWidget(self.load_live_button)
        action_layout.addWidget(self.loaded_label)
        action_layout.addWidget(self.run_button)
        action_layout.addStretch(1)
        outer.addLayout(action_layout, stretch=1)

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

    def _on_load_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "캘리브레이션 이미지 선택", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not paths:
            return
        self.image_paths = paths
        self.loaded_label.setText(f"불러온 이미지: {len(paths)}장")
        self.run_button.setEnabled(True)

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
        progress.canceled.connect(worker.request_cancel)

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
        self.run_button.setEnabled(True)

    def _on_pattern_type_changed(self) -> None:
        """Chessboard는 marker_size/dictionary가 필요 없으니 해당 입력 행을
        숨긴다. ChArUco/AprilGrid는 마커 기반 패턴이라 둘 다 유지한다.
        """
        pattern_type = self.pattern_type_combo.currentData()
        uses_marker_dictionary = pattern_type in (PatternType.CHARUCO, PatternType.APRILGRID)
        # 행 인덱스: 0=Pattern type, 1=Squares X, 2=Squares Y, 3=Square size,
        # 4=Marker size, 5=Dictionary (addRow 호출 순서와 동일).
        self._pattern_form.setRowVisible(4, uses_marker_dictionary)
        self._pattern_form.setRowVisible(5, uses_marker_dictionary)
        current_dictionary = self.dictionary_combo.currentText()
        if pattern_type == PatternType.APRILGRID and not current_dictionary.startswith("DICT_APRILTAG_"):
            idx = self.dictionary_combo.findText("DICT_APRILTAG_36h11")
            if idx >= 0:
                self.dictionary_combo.setCurrentIndex(idx)
        elif pattern_type == PatternType.CHARUCO and current_dictionary.startswith("DICT_APRILTAG_"):
            idx = self.dictionary_combo.findText("DICT_5X5_100")
            if idx >= 0:
                self.dictionary_combo.setCurrentIndex(idx)

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
        self.calibration_results = {}
        self.validation_results = {}
        self.cross_dataset_results = []
        self.scores = []

        self.use_rational_model = self.rational_checkbox.isChecked()
        worker = PipelineWorker(
            self.image_paths, self.pattern_config, self.camera_config,
            use_rational_model=self.use_rational_model,
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

        self._thread, self._worker = thread, worker
        self.run_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.pipeline_progress_bar.setRange(0, max(1, len(self.image_paths)))
        self.pipeline_progress_bar.setValue(0)
        self.pipeline_progress_bar.show()
        thread.finished.connect(lambda: self.run_button.setEnabled(True))
        thread.finished.connect(lambda: self.load_button.setEnabled(True))
        thread.finished.connect(self.pipeline_progress_bar.hide)
        thread.finished.connect(self._stop_busy_sheep)
        thread.start()

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
        self.coverage_view.set_dataset_quality_score(dataset.quality_score)

    def _on_quality_ready(self, warnings: list[str]) -> None:
        if self.dataset is None:
            return
        self.coverage_view.set_quality(self.dataset.coverage_grid, self.dataset.diversity, warnings)

    def _on_models_ready(self, results: dict[CameraModelType, CalibrationResult]) -> None:
        self.calibration_results = results
        if self.dataset is not None and self.camera_config is not None:
            self.preview_view.set_context(self.dataset, self.camera_config, results, self.pattern_config)
            self.dataset_view.set_dataset(self.dataset)  # per_frame_error 채워졌으니 갱신
            if self.pattern_config is not None:
                self.straightness_view.set_context(self.dataset, self.camera_config, results, self.pattern_config)
            # 설계 문서 8번 - 3모델 계산이 끝날 때마다 sanity check도 함께 갱신한다
            # (RMS가 낮아 보여도 결과가 물리적으로 이상할 수 있으므로 항상 확인).
            checks = run_sanity_checks(list(results.values()), self.camera_config)
            self.result_view.set_sanity_checks(checks)
        self.radial_profile_view.set_results(results)
        self.model_refitting_view.set_context(results, self.camera_config)
        self._refresh_result_view()

    def _on_validation_ready(self, results: dict[CameraModelType, ValidationResult]) -> None:
        self.validation_results = results
        self._refresh_result_view()
        if self.dataset is not None and self.camera_config is not None and self.pattern_config is not None:
            self.external_compare_view.set_context(
                self.dataset, self.camera_config, self.pattern_config,
                self.validation_results,
                calibration_results=self.calibration_results,
                use_rational_model=self.use_rational_model,
            )

    def _on_recommendation_ready(self, scores: list[ModelScore], message: str) -> None:
        self.scores = scores
        self.result_view.set_recommendation_message(message)
        recommended = next((s.model_name for s in scores if s.is_recommended), None)
        if recommended is not None:
            self.result_view.select_model(recommended)
            self.preview_view.select_model(recommended)
            self.radial_profile_view.select_model(recommended)
            self.straightness_view.select_model(recommended)
        self._refresh_result_view()
        # 계산이 완전히 끝난 시점(추천까지 나온 시점)이라 여기서 조용히
        # 자동 저장한다 - 다음에 앱이 비정상 종료돼도 이 결과는 남는다.
        self._autosave()
        self._auto_save_calibration_outputs()
        self._save_to_library()
        if self._pending_stereo_intrinsic_slot is not None and recommended is not None:
            result = self.calibration_results.get(recommended)
            if result is not None and result.success:
                self.stereo_workspace.set_previous_intrinsic(
                    self._pending_stereo_intrinsic_slot,
                    self._standard_from_result(result),
                )
                self._pending_stereo_intrinsic_slot = None
                self._show_stereo_workspace()

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
                validation_results=self.validation_results,
                cross_dataset_results=self.cross_dataset_results,
                model_scores=self.scores,
                outlier_result=self.outlier_result,
                stereo_result=(
                    stereo_result_to_dict(self.stereo_workspace.result)
                    if self.stereo_workspace.result is not None else None
                ),
                stereo_pairs=stereo_pairs_to_dict(self.stereo_workspace.pairs),
            )
            save_project(project, str(_AUTOSAVE_PATH))
            logger.debug("자동 저장 완료: %s", _AUTOSAVE_PATH)
        except Exception:  # noqa: BLE001 - 자동 저장 실패로 사용자 작업을 막으면 안 됨
            logger.exception("자동 저장 실패 (무시하고 계속 진행)")

    def _auto_save_calibration_outputs(self) -> None:
        """Pinhole/Extended Pinhole/Fisheye 중 성공한 모델의 파라미터(K/D)를
        OpenCV YAML로 output/ 폴더에 자동 저장한다.

        위 _autosave()는 앱 크래시 복구용 고정 파일 하나를 계속 덮어쓰는
        반면, 이건 사용자가 나중에 실제로 꺼내 쓸 결과물이라 실행마다 타임
        스탬프가 붙은 별개 파일로 남긴다 - 재실행해도 이전 결과가 지워지지
        않는다.
        """
        if self.camera_config is None or self.pattern_config is None:
            return
        output_dir = Path.cwd() / "output"
        sensor = self.camera_config.sensor_name or "camera"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved: list[str] = []
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            for model, result in self.calibration_results.items():
                if not result.success:
                    continue
                # self.calibration_results는 QThread의 Signal(dict)을 건너온
                # 딕셔너리라, PySide6가 str-Enum 키를 평범한 str로 낮춰서
                # 넘길 때가 있다 - .value 접근 전에 다시 enum으로 정규화한다
                # (calibration/library.py의 같은 문제와 동일한 원인).
                filename = f"{sensor}_{CameraModelType(model).value}_{timestamp}.yaml"
                export_opencv_yaml(result, self.camera_config, self.pattern_config, str(output_dir / filename))
                saved.append(filename)
        except Exception:  # noqa: BLE001 - 자동 저장 실패로 사용자 작업을 막으면 안 됨
            logger.exception("파라미터 자동 저장 실패 (무시하고 계속 진행)")
            return
        if saved:
            self.status_label.setText(
                self.status_label.text() + f"  ·  output/ 폴더에 저장됨: {', '.join(saved)}"
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
        self.result_view.set_comparison(self.calibration_results, self.validation_results, self.scores)
        self.result_view.set_cross_dataset_results(self.cross_dataset_results)
        self.diagnosis_view.set_results(self.calibration_results, self.validation_results, self.dataset)
        self.stability_view.set_results(self.calibration_results, self.validation_results, self.scores, self.dataset)

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

    # ------------------------------------------------------------------
    # Model Refitting
    # ------------------------------------------------------------------

    def _on_model_refit_requested(self, options: dict) -> None:
        result = self.calibration_results.get(CameraModelType.EXTENDED_PINHOLE)
        if (
            result is None or not result.success or result.camera_matrix is None
            or result.distortion is None or self.camera_config is None
        ):
            QMessageBox.warning(
                self,
                "Model Refitting 불가",
                "먼저 Rational model 사용(k4~k6 포함)으로 Extended Pinhole 캘리브레이션을 실행하세요.",
            )
            return

        worker = ModelRefittingWorker(
            result.camera_matrix,
            result.distortion,
            (self.camera_config.width, self.camera_config.height),
            options,
        )
        thread = run_worker_in_thread(worker, self)
        worker.progress.connect(self.status_label.setText)
        worker.progress.connect(lambda _msg: self.model_refitting_view.set_running())
        worker.result_ready.connect(self._on_model_refit_ready)
        worker.error.connect(self._on_model_refit_error)
        worker.finished.connect(self._on_model_refit_finished)

        self._model_refit_thread = thread
        self._model_refit_worker = worker
        thread.start()

    def _on_model_refit_ready(self, result) -> None:
        self.model_refitting_view.set_result(result)
        self.status_label.setText(
            f"Model Refitting 완료: RMSE={result.error.rmse_px:.4f}px, "
            f"Edge RMSE={result.region_error['edge'].rmse_px:.4f}px"
        )

    def _on_model_refit_error(self, message: str) -> None:
        self.model_refitting_view.set_error(message)
        QMessageBox.critical(self, "Model Refitting 실패", message)

    def _on_model_refit_finished(self) -> None:
        self._model_refit_thread = None
        self._model_refit_worker = None

    # ------------------------------------------------------------------
    # Outlier 재계산
    # ------------------------------------------------------------------

    def _on_outlier_prune_requested(self, reference_model: CameraModelType) -> None:
        if self.dataset is None or self.camera_config is None:
            QMessageBox.warning(self, "데이터 없음", "먼저 캘리브레이션을 실행하세요.")
            return

        reply = QMessageBox.question(
            self, "이상치 제거 확인",
            f"{reference_model.value} 기준으로 이상치를 탐지하고, 제외한 뒤 "
            f"세 모델 모두 재계산합니다. 계속할까요?\n"
            f"(파일은 삭제되지 않고 비활성화만 됩니다.)",
        )
        if reply != QMessageBox.Yes:
            return

        worker = OutlierPruneWorker(
            self.dataset, self.camera_config, self.pattern_config, reference_model,
            use_rational_model=self.use_rational_model,
        )
        thread = run_worker_in_thread(worker, self)

        worker.progress.connect(self.status_label.setText)
        worker.dataset_updated.connect(self._on_outlier_dataset_updated)
        worker.quality_ready.connect(self._on_quality_ready)
        worker.outlier_ready.connect(
            lambda ref_result, outlier_result: self._on_outlier_ready(reference_model, outlier_result)
        )
        worker.models_ready.connect(self._on_models_ready)
        worker.validation_ready.connect(self._on_validation_ready)
        worker.recommendation_ready.connect(self._on_recommendation_ready)
        worker.error.connect(self._on_error)

        self._thread, self._worker = thread, worker
        self.run_button.setEnabled(False)
        thread.finished.connect(lambda: self.run_button.setEnabled(True))
        thread.start()

    def _on_outlier_ready(self, reference_model: CameraModelType, outlier_result: OutlierResult) -> None:
        self.outlier_result = outlier_result  # 리포트(export/report.py)에서 재사용
        self.result_view.set_outlier_result(reference_model, outlier_result)

    def _on_outlier_dataset_updated(self, dataset: Dataset) -> None:
        """이상치 재계산은 이제 별도 프로세스(calibration/pipeline_process.py)에서
        돌아서, 프레임 상태가 바뀐 dataset은 원본과 다른 객체(pickle로 왕복한
        복사본)로 돌아온다. 그래서 화면 갱신뿐 아니라 self.dataset 참조 자체를
        이걸로 교체해야 한다 - 안 그러면 이후 export/저장/재실행이 전부 이상치
        제외 전 상태를 보게 되는 조용한 버그가 생긴다.
        """
        self.dataset = dataset
        self.dataset_view.set_dataset(dataset)
        self.coverage_view.set_dataset_quality_score(dataset.quality_score)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export_opencv(self, model: CameraModelType) -> None:
        result = self.calibration_results.get(model)
        if not result or not result.success or self.camera_config is None or self.pattern_config is None:
            QMessageBox.warning(self, "Export 불가", f"{CameraModelType(model).value} 모델의 캘리브레이션 결과가 없습니다.")
            return
        path = self.result_view.prompt_save_path("camera.yaml", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            export_opencv_yaml(result, self.camera_config, self.pattern_config, path)
            self.status_label.setText(f"OpenCV YAML 저장 완료: {path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Export 실패", str(e))

    def _on_export_ros(self, model: CameraModelType) -> None:
        result = self.calibration_results.get(model)
        if not result or not result.success or self.camera_config is None:
            QMessageBox.warning(self, "Export 불가", f"{CameraModelType(model).value} 모델의 캘리브레이션 결과가 없습니다.")
            return
        path = self.result_view.prompt_save_path("camera_info.yaml", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            export_ros_camera_info(result, self.camera_config, path)
            self.status_label.setText(f"ROS CameraInfo YAML 저장 완료: {path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Export 실패", str(e))

    def _on_export_report(self, model: CameraModelType) -> None:
        result = self.calibration_results.get(model)
        if (
            not result or not result.success or self.dataset is None
            or self.camera_config is None or self.pattern_config is None
        ):
            QMessageBox.warning(self, "Export 불가", f"{CameraModelType(model).value} 모델의 캘리브레이션 결과가 없습니다.")
            return
        path = self.result_view.prompt_save_path("calibration_report.html", "HTML (*.html)")
        if not path:
            return
        try:
            coverage_pct = (
                coverage_percentage(self.dataset.coverage_grid) if self.dataset.coverage_grid else None
            )
            final_result = compute_final_result(
                chosen_model=model,
                calibration_results=self.calibration_results,
                validation_results=self.validation_results,
                dataset_coverage_pct=coverage_pct,
                outlier_result=self.outlier_result,
                scores=self.scores,
                coverage_grid=self.dataset.coverage_grid,
                dataset_diversity=self.dataset.diversity,
            )
            export_html_report(
                project_name=(self.camera_config.sensor_name or "camera_calibrator"),
                camera_config=self.camera_config,
                pattern_config=self.pattern_config,
                dataset=self.dataset,
                calibration_results=self.calibration_results,
                validation_results=self.validation_results,
                final_result=final_result,
                path=path,
                cross_dataset_results=self.cross_dataset_results,
            )
            self.status_label.setText(f"HTML 리포트 저장 완료: {path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Export 실패", str(e))

    def _on_export_json(self, model: CameraModelType) -> None:
        result = self.calibration_results.get(model)
        if (
            not result or not result.success or self.dataset is None
            or self.camera_config is None or self.pattern_config is None
        ):
            QMessageBox.warning(self, "Export 불가", f"{CameraModelType(model).value} 모델의 캘리브레이션 결과가 없습니다.")
            return
        path = self.result_view.prompt_save_path("calibration.json", "JSON (*.json)")
        if not path:
            return
        try:
            coverage_pct = (
                coverage_percentage(self.dataset.coverage_grid) if self.dataset.coverage_grid else None
            )
            final_result = compute_final_result(
                chosen_model=model,
                calibration_results=self.calibration_results,
                validation_results=self.validation_results,
                dataset_coverage_pct=coverage_pct,
                outlier_result=self.outlier_result,
                scores=self.scores,
                coverage_grid=self.dataset.coverage_grid,
                dataset_diversity=self.dataset.diversity,
            )
            export_json(
                camera_config=self.camera_config,
                pattern_config=self.pattern_config,
                dataset=self.dataset,
                calibration_results=self.calibration_results,
                validation_results=self.validation_results,
                chosen_model=model,
                path=path,
                final_result=final_result,
                model_scores=self.scores,
                cross_dataset_results=self.cross_dataset_results,
            )
            self.status_label.setText(f"JSON 저장 완료: {path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Export 실패", str(e))

    def _on_export_csv(self, model: CameraModelType) -> None:
        # CSV는 이미지별 데이터셋 통계라 model 인자는 참고용일 뿐, 선택된
        # 모델의 성공 여부와 무관하게 데이터셋만 있으면 export 가능하다.
        if self.dataset is None:
            QMessageBox.warning(self, "Export 불가", "먼저 이미지를 불러오고 캘리브레이션을 실행하세요.")
            return
        path = self.result_view.prompt_save_path("dataset.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            export_csv(self.dataset, path)
            self.status_label.setText(f"CSV 저장 완료: {path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Export 실패", str(e))

    # ------------------------------------------------------------------
    # 프로젝트 저장/불러오기 (.ccproj)
    # ------------------------------------------------------------------

    def _on_save_project(self) -> None:
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

        project = CalibrationProject(
            project_name=self.camera_config.sensor_name or Path(path).stem,
            camera_config=self.camera_config,
            pattern_config=self.pattern_config,
            dataset=self.dataset,
            calibration_results=self.calibration_results,
            validation_results=self.validation_results,
            cross_dataset_results=self.cross_dataset_results,
            model_scores=self.scores,
            outlier_result=self.outlier_result,
            stereo_result=(
                stereo_result_to_dict(self.stereo_workspace.result)
                if self.stereo_workspace.result is not None else None
            ),
            stereo_pairs=stereo_pairs_to_dict(self.stereo_workspace.pairs),
        )
        try:
            saved_path = save_project(project, path)
            self.status_label.setText(f"프로젝트 저장 완료: {saved_path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "저장 실패", str(e))

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
        self.validation_results = project.validation_results
        self.cross_dataset_results = project.cross_dataset_results
        self.scores = project.model_scores
        self.outlier_result = project.outlier_result
        self.image_paths = [f.image_info.path for f in project.dataset.frames]
        IntrinsicWorkspace.sync_owner_state(self)
        if project.stereo_result or project.stereo_pairs:
            try:
                self.stereo_workspace.restore_project_payload(
                    result_payload=project.stereo_result,
                    pair_payload=project.stereo_pairs,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Stereo 결과 복원 실패: %s", exc)

        # --- 설정 패널 위젯도 불러온 값으로 맞춰준다 (재계산/이어서 작업 시 일관성) ---
        self.sensor_name_edit.setText(self.camera_config.sensor_name or "")
        self.width_spin.setValue(self.camera_config.width)
        self.height_spin.setValue(self.camera_config.height)
        self.squares_x_spin.setValue(self.pattern_config.squares_x)
        self.squares_y_spin.setValue(self.pattern_config.squares_y)
        # pattern_config는 항상 미터(m) 단위로 저장돼 있으니, mm 입력 위젯에는 변환해서 넣는다.
        self.square_size_spin.setValue(self.pattern_config.square_size * 1000.0)
        idx = self.pattern_type_combo.findData(self.pattern_config.type)
        if idx >= 0:
            self.pattern_type_combo.setCurrentIndex(idx)  # _on_pattern_type_changed가 자동으로 행 토글
        if self.pattern_config.marker_size is not None:
            self.marker_size_spin.setValue(self.pattern_config.marker_size * 1000.0)
        if self.pattern_config.dictionary:
            self.dictionary_combo.setCurrentText(self.pattern_config.dictionary)

        # --- 각 탭 새로고침 ---
        self.dataset_view.set_dataset(self.dataset)
        self.coverage_view.set_quality(self.dataset.coverage_grid, self.dataset.diversity, [])
        if self.calibration_results:
            self.preview_view.set_context(self.dataset, self.camera_config, self.calibration_results, self.pattern_config)
            self.radial_profile_view.set_results(self.calibration_results)
            self.straightness_view.set_context(
                self.dataset, self.camera_config, self.calibration_results, self.pattern_config
            )
            self.model_refitting_view.set_context(self.calibration_results, self.camera_config)
        if self.validation_results:
            self.external_compare_view.set_context(
                self.dataset, self.camera_config, self.pattern_config,
                self.validation_results,
                calibration_results=self.calibration_results,
                use_rational_model=self.use_rational_model,
            )
        self._refresh_result_view()
        if self.scores:
            recommended = next((s.model_name for s in self.scores if s.is_recommended), None)
            if recommended is not None:
                self.result_view.select_model(recommended)
                self.preview_view.select_model(recommended)
                self.radial_profile_view.select_model(recommended)
                self.straightness_view.select_model(recommended)
        if self.outlier_result and project.final_result:
            self.result_view.set_outlier_result(project.final_result.chosen_model, self.outlier_result)

        self.loaded_label.setText(f"불러온 이미지: {len(self.image_paths)}장 (프로젝트: {Path(path).name})")
        self.run_button.setEnabled(bool(self.image_paths))

        msg = f"프로젝트 불러옴: {project.project_name} ({self.dataset.num_total}장, 검출 {self.dataset.num_detected}장)"
        self.status_label.setText(msg)

        if missing:
            preview = "\n".join(missing[:10]) + (f"\n... 외 {len(missing) - 10}개" if len(missing) > 10 else "")
            QMessageBox.warning(
                self, "이미지 파일 없음",
                f"원본 이미지 {len(missing)}개를 찾을 수 없습니다 (경로가 바뀌었거나 삭제됨).\n"
                f"캘리브레이션 결과 확인/Export/이상치 재계산은 이미지 없이도 가능하지만, "
                f"'④ Undistort Preview' 탭에서는 해당 이미지가 표시되지 않습니다.\n\n{preview}",
            )
