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
import os
import sys
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
)
from calibration.sanity_check import run_sanity_checks
from calibration.recommender import compute_final_result
from calibration.project_io import load_project, save_project, PROJECT_EXTENSION
from export.opencv import export_opencv_yaml
from export.output_manager import DEFAULT_OUTPUT_ROOT, OutputManager, model_filename
from export.csv_export import export_csv
from export.json_export import export_json
from export.report import export_html_report
from export.ros import export_ros_camera_info
from calibration.image_selection import write_selection_manifest

from ui.calibration_home_view import CalibrationHomeView
from ui.help_view import HelpView
from ui.theme import Theme
from ui.intrinsic_workspace import IntrinsicWorkspace
from ui.live_capture_dialog import LiveCaptureDialog
from ui.responsive_layout import ResponsiveRow, configure_form_layout
from ui.wheel_guard import WheelChangeGuard
from ui.worker import (
    PipelineWorker,
    ImageSelectionWorker,
    CrossDatasetValidationWorker,
    SelfCheckWorker,
    BagTopicDiscoveryWorker,
    BagExtractionWorker,
    run_worker_in_thread,
)
from ui.room_workspace import RoomWorkspace
from ui.git_upload_dialog import GitSettingsDialog, GitUploadDialog
from export import git_upload

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


def _model_keys(results: dict) -> dict:
    """Signal(dict) 는 Qt 의 QVariantMap 을 거치며 키를 str 로 바꾼다 ('brown_conrady').
    리포트/JSON export 가 key.value 를 쓰므로 CameraModelType 으로 되돌린다."""
    out = {}
    for key, value in (results or {}).items():
        try:
            out[CameraModelType(key)] = value
        except ValueError:
            out[key] = value
    return out


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
        self._export_dialog: QDialog | None = None  # result_view가 생긴 뒤 지연 생성
        # 사용자 결과물은 이 관리자 아래 한 세션으로 모인다. 앱 크래시 복구용
        # ~/.camera_calibrator/autosave.ccproj는 의도적으로
        # 별도 수명주기이므로 여기로 합치지 않는다.
        self.output_manager = OutputManager()
        # 홈 화면에서 고른 차량 ("" = 차량 없음, 카메라 단품). shared.vehicle 참고.
        self.vehicle = ""

        self._build_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.workspace_stack = QStackedWidget()
        self.home_view = CalibrationHomeView()
        settings_panel = self._build_settings_panel()

        self.intrinsic_workspace = IntrinsicWorkspace.create_for_main_window(self, settings_panel)
        self.intrinsic_workspace.back_requested.connect(self._show_home)
        self.workspace_stack.addWidget(self.home_view)
        self.workspace_stack.addWidget(self.intrinsic_workspace)
        # 카메라·LiDAR 캘리브레이션(extrinsic/gui)은 처음 열 때 지연 로드한다.
        self.room_workspace = RoomWorkspace()
        self.room_workspace.back_requested.connect(self._show_home)
        self.workspace_stack.addWidget(self.room_workspace)
        layout.addWidget(self.workspace_stack, stretch=1)
        self._set_vehicle(self.vehicle)

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
        self.home_view.room_requested.connect(self._show_room_workspace)

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
        self._set_vehicle(self.home_view.confirm_vehicle())
        self.workspace_stack.setCurrentWidget(self.intrinsic_workspace)
        self.status_label.setText(f"Camera Intrinsic Workspace · 차량: {self.vehicle or '없음 (카메라 단품)'}")

    def _show_room_workspace(self) -> None:
        vehicle = self.home_view.confirm_vehicle()
        if not vehicle:
            QMessageBox.information(self, "CAMERA–LiDAR", "CAMERA–LiDAR 는 차량을 골라야 합니다. 홈 화면에서 차량을 고르세요.")
            return
        self._set_vehicle(vehicle)
        if self.room_workspace.ensure_loaded(vehicle):
            window = self.room_workspace.room_window
            if not getattr(window, "_home_vehicle_hooked", False):
                window.vehicle_changed.connect(self._on_room_vehicle_changed)
                window._home_vehicle_hooked = True
        self.workspace_stack.setCurrentWidget(self.room_workspace)
        self.status_label.setText(f"Camera–LiDAR Calibration · 차량: {vehicle}")

    def _on_room_vehicle_changed(self, name: str) -> None:
        """CAMERA–LiDAR 안에서 차량을 바꾸면 홈·인트린직도 같은 차량으로."""
        if name and name != self.vehicle:
            self.home_view.set_vehicle(name)
            self._set_vehicle(name)

    # ------------------------------------------------------------------
    # 차량 (홈 화면에서 고른다, shared.vehicle)
    # ------------------------------------------------------------------

    def _set_vehicle(self, name: str) -> None:
        from shared import vehicle as shared_vehicle

        self.vehicle = name or ""
        if not hasattr(self, "vehicle_label"):
            return
        if self.vehicle:
            self.vehicle_label.setText(f"<b>{self.vehicle}</b>  <span style='color:{Theme.TEXT_SECONDARY}'>(홈 화면에서 변경)</span>")
        else:
            self.vehicle_label.setText(f"<b>차량 없음</b> (카메라 단품)  <span style='color:{Theme.TEXT_SECONDARY}'>(홈 화면에서 변경)</span>")
        text = self.sensor_name_edit.text()
        names = shared_vehicle.camera_names(self.vehicle)
        combo = self.sensor_name_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(names)
        combo.setCurrentIndex(-1)
        combo.setEditText(text if not self.vehicle or not names or text in names else "")
        combo.blockSignals(False)
        self.sensor_name_edit.setPlaceholderText(
            ("이 차량의 카메라를 고르거나 새 이름 입력" if names else "새 카메라 이름 (예: front)") if self.vehicle
            else "예: econ120")
        self._refresh_camera_lens_hint()

    def _refresh_camera_lens_hint(self, *_args) -> None:
        from shared import vehicle as shared_vehicle

        label = getattr(self, "camera_lens_label", None)
        if label is None:
            return
        camera = self.sensor_name_edit.text().strip()
        if not self.vehicle or not camera:
            label.setVisible(False)
            return
        lens = shared_vehicle.lens_of(self.vehicle, camera)
        known = camera in shared_vehicle.camera_names(self.vehicle)
        label.setText(f"CAMERA–LiDAR 렌즈: {lens}" if lens else
                      ("CAMERA–LiDAR 렌즈 미등록 - 결과 Export 창에서 등록" if known else
                       f"새 카메라 - 저장할 때 {self.vehicle} 에 추가됩니다"))
        label.setVisible(True)

    def _remember_vehicle_camera(self) -> None:
        from shared import vehicle as shared_vehicle

        camera = self._session_camera_name()
        if self.vehicle and camera:
            shared_vehicle.ensure(self.vehicle)
            if shared_vehicle.add_camera(self.vehicle, camera):
                self._set_vehicle(self.vehicle)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        # 룸 페이지가 돌리던 bag 읽기/검출 QThread 를 정리하고 닫는다.
        self.room_workspace.shutdown()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # 설정 패널 (설계 문서 14번 ① Camera Setup, ③ Calibration Pattern)
    # ------------------------------------------------------------------

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        help_menu = menu_bar.addMenu("설명")
        guide_action = QAction("사용 설명서 열기", self)
        guide_action.setShortcut("F1")
        guide_action.triggered.connect(self._on_show_help)
        help_menu.addAction(guide_action)

        view_menu = menu_bar.addMenu("보기")
        theme_menu = view_menu.addMenu("화면 테마")
        self.theme_actions = {}
        for mode, label in (("dark", "다크 (검은 바탕)"), ("light", "라이트 (흰 바탕에 검은 글씨)")):
            action = QAction(label, self, checkable=True)
            action.setChecked(Theme.MODE == mode)
            action.triggered.connect(lambda _checked=False, m=mode: self._on_theme_selected(m))
            theme_menu.addAction(action)
            self.theme_actions[mode] = action

        tools_menu = menu_bar.addMenu("도구")
        self.self_check_action = QAction("Intrinsic 계산 엔진 자체 진단 (합성 데이터)...", self)
        self.self_check_action.setToolTip(
            "정답을 미리 아는 가짜(합성) ChArUco 데이터로 Ideal Pinhole/Brown-Conrady/\n"
            "Rational/Fisheye 모델을 돌려서 복원된 fx/fy/cx/cy가 정답에 가까운지 확인합니다.\n"
            "현재 불러온 이미지/캘리브레이션 결과와는 무관하며, 몇 초~수십 초 걸립니다."
        )
        self.self_check_action.triggered.connect(self._on_run_self_check)
        tools_menu.addAction(self.self_check_action)

    def _on_theme_selected(self, mode: str) -> None:
        """테마를 저장하고, 원하면 바로 다시 시작한다 (색은 위젯을 만들 때 정해진다)."""
        from ui import theme

        for m, action in self.theme_actions.items():
            action.setChecked(m == Theme.MODE)
        if mode == Theme.MODE:
            return
        try:
            theme.save_mode(mode)
        except OSError as exc:
            QMessageBox.warning(self, "화면 테마", f"설정을 저장하지 못했습니다: {exc}")
            return
        name = "라이트" if mode == "light" else "다크"
        answer = QMessageBox.question(
            self, "화면 테마",
            f"{name} 테마로 바꿉니다. 프로그램을 다시 시작해야 적용됩니다.\n지금 다시 시작할까요?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer == QMessageBox.Yes:
            self._restart_app(mode)
        else:
            self.status_label.setText(f"다음 실행부터 {name} 테마가 적용됩니다.")

    def _restart_app(self, mode: str) -> None:
        from PySide6.QtCore import QProcess

        os.environ["CALIB_THEME"] = mode  # 실행 환경에 CALIB_THEME 이 있어도 고른 테마로
        root = str(Path(__file__).resolve().parents[1])
        if not self.close():  # 닫기를 취소했으면(작업 중 등) 다시 시작하지 않는다
            return
        QProcess.startDetached(sys.executable, ["-m", "app.main", *sys.argv[1:]], root)
        QApplication.instance().quit()

    def _on_show_help(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Calibration Tool 사용 설명서")
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
        return bar

    def _build_export_rows(self) -> QWidget:
        """Export | Export All, 그 아래 Git 업로드 | ⚙ Git 설정."""
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        export_row = QHBoxLayout()
        export_row.addWidget(self.export_button, 1)
        self.export_all_button = QPushButton("Export All")
        self.export_all_button.setToolTip("YAML·리포트·프로젝트 등 지금 만들 수 있는 결과를 세션 폴더에 모두 저장합니다.")
        self.export_all_button.clicked.connect(self._on_export_all)
        export_row.addWidget(self.export_all_button, 1)
        layout.addLayout(export_row)
        git_row = QHBoxLayout()
        self.git_upload_button = QPushButton("⬆ Git 업로드")
        self.git_upload_button.setToolTip("결과를 결과 전용 Git 저장소에 올립니다. 올릴 파일·경로·저장소는 ⚙ Git 설정에서 바꿉니다.")
        self.git_upload_button.clicked.connect(self._on_git_upload)
        git_row.addWidget(self.git_upload_button, 1)
        self.git_settings_button = QPushButton("⚙ Git 설정")
        self.git_settings_button.setToolTip("저장소 주소·브랜치·작성자, 저장소 안 경로, 올릴 파일 종류")
        self.git_settings_button.clicked.connect(self._on_git_settings)
        git_row.addWidget(self.git_settings_button)
        layout.addLayout(git_row)
        return box

    def _session_camera_name(self) -> str:
        config = self.camera_config
        if config is not None and config.sensor_name:
            return config.sensor_name
        edit = getattr(self, "sensor_name_edit", None)
        return edit.text().strip() if edit is not None else ""

    def _ensure_output_session(self) -> Path:
        self._remember_vehicle_camera()
        stem = "_".join(part for part in (self.vehicle, self._session_camera_name()) if part)
        session = self.output_manager.ensure_session(stem)
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
            vehicle=self.vehicle or None,
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
        session = self._ensure_output_session()
        self.status_label.setText(f"Output Session 생성: {session}")

    def _on_open_output_folder(self) -> None:
        folder = self._ensure_output_session()
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            QMessageBox.warning(self, "폴더 열기 실패", str(folder))

    def _on_export_all(self) -> None:
        session, exported, skipped, failed = self._export_all()
        details = [
            f"Session: {session}",
            f"완료 {len(exported)}개: {', '.join(exported) or '-'}",
            f"건너뜀 {len(skipped)}개: {', '.join(skipped) or '-'}",
            f"실패 {len(failed)}개: {', '.join(failed) or '-'}",
        ]
        QMessageBox.information(self, "Export All 결과", "\n\n".join(details))
        self.status_label.setText(f"Export All 완료: {session} (완료 {len(exported)}, 실패 {len(failed)})")

    def _export_all(self) -> tuple[Path, list[str], list[str], list[str]]:
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
                return export_opencv_yaml(result, self.camera_config, self.pattern_config, str(path))
            attempt(f"intrinsic.{model_filename(model)}", configs_ready, export_intrinsic)


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
                lambda: export_csv(self.dataset, str(self.output_manager.report_path("dataset.csv"))),
            )
            attempt(
                "intrinsic.ros", True,
                lambda: export_ros_camera_info(
                    successful[chosen], self.camera_config, str(self.output_manager.ros_path())
                ),
            )
        else:
            skipped.extend(["reports.html", "reports.json", "reports.dataset_csv", "intrinsic.ros"])

        return session, exported, skipped, failed

    def _on_git_upload(self) -> None:
        """현재 결과를 Export All 로 세션 폴더에 다시 쓴 뒤, 그 폴더를 결과 저장소에 올린다."""
        if not any(getattr(r, "success", False) for r in self.calibration_results.values()):
            QMessageBox.warning(self, "Git 업로드", "먼저 캘리브레이션을 실행하세요.")
            return
        session, _exported, _skipped, failed = self._export_all()
        if failed:
            QMessageBox.warning(self, "Git 업로드", "일부 파일을 만들지 못했습니다 (나머지는 올릴 수 있습니다):\n\n"
                                + "\n".join(failed))
        camera = self._session_camera_name()
        recommended = next((getattr(s.model_name, "value", s.model_name) for s in self.scores if s.is_recommended), None)
        vehicle = self.vehicle
        GitUploadDialog(lambda config: git_upload.plan_intrinsic(session, camera, config, recommended, vehicle),
                        "Git 업로드 - Camera Intrinsic", self).exec()

    def _on_git_settings(self) -> None:
        GitSettingsDialog(self).exec()

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
        self.vehicle_label = QLabel()
        self.vehicle_label.setToolTip("홈 화면에서 고른 차량. 결과 폴더·Git 경로·카메라 목록·렌즈 등록이 이 차량 기준입니다.")
        camera_form.addRow("Vehicle", self.vehicle_label)
        # 차량을 골랐으면 그 차량의 카메라 목록에서 고른다 (새 이름도 입력 가능 - 저장할 때 차량에 추가).
        self.sensor_name_combo = QComboBox()
        self.sensor_name_combo.setEditable(True)
        self.sensor_name_combo.setInsertPolicy(QComboBox.NoInsert)
        self.sensor_name_edit = self.sensor_name_combo.lineEdit()
        self.sensor_name_edit.setPlaceholderText("예: econ120")
        self.sensor_name_combo.setToolTip(
            "이 카메라를 구분하는 이름입니다. 자동 저장/Export 결과의 이름에 쓰입니다.\n"
            "차량을 골랐으면 그 차량의 카메라 목록이 나옵니다 (CAMERA–LiDAR 와 같은 이름)."
        )
        self.sensor_name_edit.textChanged.connect(self._refresh_camera_lens_hint)
        camera_form.addRow("Camera Name", self.sensor_name_combo)
        self.camera_lens_label = QLabel()
        self.camera_lens_label.setStyleSheet(f"color: {Theme.TEXT_SECONDARY};")
        self.camera_lens_label.setVisible(False)
        camera_form.addRow("", self.camera_lens_label)

        # 왼쪽 열 순서: Camera Name -> INPUT(실시간|rosbag|이미지 한 줄) ->
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
        camera_column.setLayout(camera_form)
        configure_form_layout(camera_form)
        outer.addWidget(camera_column, stretch=1)

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


        self.pattern_type_combo = QComboBox()
        # userData로 PatternType을 직접 들고 있어서 _current_pattern_config()가
        # 문자열 비교 없이 바로 꺼내 쓸 수 있다.
        self.pattern_type_combo.addItem("ChArUco (권장)", userData=PatternType.CHARUCO)
        self.pattern_type_combo.addItem("Chessboard (일반 체스보드)", userData=PatternType.CHESSBOARD)
        self.pattern_type_combo.addItem("Circle Grid", userData=PatternType.CIRCLE_GRID)
        self.pattern_type_combo.addItem("AprilGrid (AprilTag grid)", userData=PatternType.APRILGRID)
        self.pattern_type_combo.currentIndexChanged.connect(self._on_pattern_type_changed)

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
        pattern_column.setLayout(pattern_form)
        configure_form_layout(pattern_form)
        outer.addWidget(pattern_column, stretch=1)

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
        action_layout.addWidget(self.run_button)
        action_layout.addWidget(self._build_output_session_bar())
        action_layout.addWidget(self._build_export_rows())
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

    def _on_pattern_type_changed(self, *_args) -> None:
        """Chessboard는 marker_size/dictionary가 필요 없으니 해당 입력 행을
        숨긴다. ChArUco/AprilGrid는 마커 기반 패턴이라 둘 다 유지한다.
        """
        pattern_type = self.pattern_type_combo.currentData()
        self._set_pattern_field_labels(pattern_type)
        uses_marker_dictionary = pattern_type in (PatternType.CHARUCO, PatternType.APRILGRID)
        self._pattern_form.setRowVisible(self.marker_size_spin, uses_marker_dictionary)
        self._pattern_form.setRowVisible(self.dictionary_combo, uses_marker_dictionary)
        self._pattern_form.setRowVisible(self.circle_grid_type_combo, pattern_type == PatternType.CIRCLE_GRID)
        self._pattern_form.setRowVisible(self.aprilgrid_variant_combo, pattern_type == PatternType.APRILGRID)
        current_dictionary = self.dictionary_combo.currentText()
        if pattern_type == PatternType.APRILGRID and not current_dictionary.startswith("DICT_APRILTAG_"):
            idx = self.dictionary_combo.findText("DICT_APRILTAG_36h11")
            if idx >= 0:
                self.dictionary_combo.setCurrentIndex(idx)
        elif pattern_type == PatternType.CHARUCO and current_dictionary.startswith("DICT_APRILTAG_"):
            idx = self.dictionary_combo.findText("DICT_5X5_100")
            if idx >= 0:
                self.dictionary_combo.setCurrentIndex(idx)

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
        for field, text in labels.items():
            item = self._pattern_form.labelForField(field)
            if item is not None:
                item.setText(text)

    def _current_pattern_config(self) -> PatternConfig:
        # UI는 mm로 입력받지만(보드 인쇄 스펙이 보통 mm 단위), 내부 계산/export는
        # 전부 미터(m) 기준이라 여기서 한 번만 변환한다 - 이후 파이프라인은
        # 이 값이 mm에서 왔는지 몰라도 된다.
        # PySide6 6.x 는 콤보 userData 의 str-Enum 을 그냥 str 로 돌려준다 ('charuco').
        # 리포트/JSON export 가 .value 를 쓰므로 여기서 Enum 으로 되돌린다.
        pattern_type = PatternType(self.pattern_type_combo.currentData())
        uses_marker_dictionary = pattern_type in (PatternType.CHARUCO, PatternType.APRILGRID)
        return PatternConfig(
            type=pattern_type,
            squares_x=self.squares_x_spin.value(),
            squares_y=self.squares_y_spin.value(),
            square_size=self.square_size_spin.value() / 1000.0,
            marker_size=(self.marker_size_spin.value() / 1000.0) if uses_marker_dictionary else None,
            dictionary=self.dictionary_combo.currentText() if uses_marker_dictionary else None,
            circle_grid_type=CircleGridType(self.circle_grid_type_combo.currentData()),
            aprilgrid_variant=AprilGridVariant(self.aprilgrid_variant_combo.currentData()),
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
        self.repeated_kfold_results = {}
        self.object_releasing_result = None
        self.scene_quality_analysis = None
        self.subset_calibration_result = None

        # 캘리브레이션 방식은 Standard 고정 (Object-Releasing 은 UI 에서 제외).
        self.calibration_method = CalibrationMethod.STANDARD
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
        results = _model_keys(results)
        self.calibration_results = results
        self.object_releasing_result = object_releasing_result
        self.object_releasing_validation_result = object_releasing_validation_result
        self.standard_vs_object_releasing_comparison = standard_vs_object_releasing_comparison
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

    def _on_validation_ready(self, results: dict[CameraModelType, ValidationResult]) -> None:
        self.validation_results = _model_keys(results)
        self._refresh_result_view()

    def _on_recommendation_ready(self, scores: list[ModelScore], message: str) -> None:
        self.scores = scores
        self.result_view.set_recommendation_message(message)
        recommended = next((s.model_name for s in scores if s.is_recommended), None)
        if recommended is not None:
            self.result_view.select_model(recommended)
            self.preview_view.select_model(recommended)
        self._refresh_result_view()
        # 계산이 완전히 끝난 시점(추천까지 나온 시점)이라 여기서 조용히
        # 자동 저장한다 - 다음에 앱이 비정상 종료돼도 이 결과는 남는다.
        self._autosave()
        self._auto_save_calibration_outputs()

    def _autosave(self) -> None:
        """마지막으로 완료된 계산 결과를 홈 디렉터리의 고정 파일에 저장한다.

        실제 사용자 버그: 계산 결과가 메모리에만 있고 프로젝트로 저장하지
        않은 상태에서 앱이 응답 없음 -> 강제 종료로 죽으면, 다시 켰을 때
        방금까지 화면에 있던 Model Comparison/Export 결과가 통째로
        사라져서 Export를 다시 할 수 없었다. 이 자동 저장은 사용자가
        직접 [파일 -> 프로젝트 저장]을 누르지 않아도 매 계산 완료 시점마다
        복구 지점을 하나 남겨둔다. Export All의 프로젝트 저장(project/calibration.ccproj)을
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
                export_opencv_yaml(result, self.camera_config, self.pattern_config, str(path))
                key = f"intrinsic.{model_filename(model)}"
                self.output_manager.record_export(key, path, metadata={"automatic": True})
                saved.append(path.name)
            except Exception:  # noqa: BLE001 - 모델 하나의 실패가 나머지를 막지 않게 한다.
                logger.exception("%s 파라미터 자동 저장 실패 (계속 진행)", model)
        if saved:
            self.status_label.setText(
                self.status_label.text() + f"  ·  Output Session 저장: {', '.join(saved)}"
            )

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
        self.result_view.set_comparison(
            self.calibration_results,
            self.validation_results,
            self.scores,
        )
        self.result_view.set_cross_dataset_results(self.cross_dataset_results)

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

    def _on_register_lens(self, model) -> None:
        """선택한 모델의 K·D 를 CAMERA–LiDAR 렌즈 프리셋으로 등록한다 (shared.camera)."""
        from shared import camera as shared_camera

        result = self.calibration_results.get(model)
        if not result or not result.success or self.camera_config is None:
            QMessageBox.warning(self, "렌즈 등록 불가", "선택한 모델의 캘리브레이션 결과가 없습니다.")
            return
        from shared import vehicle as shared_vehicle

        model_text = getattr(model, "value", str(model))
        camera = self._session_camera_name()
        if self.vehicle and camera:
            default = shared_vehicle.lens_of(self.vehicle, camera) or f"{self.vehicle}_{camera}"
        else:
            default = f"{camera or 'camera'}_{model_text}"
        name, ok = QInputDialog.getText(
            self, "CAMERA–LiDAR 렌즈로 등록",
            "프리셋 이름 (CAMERA–LiDAR 3단계 목록에 이 이름으로 나옵니다.\n같은 이름이 있으면 바꿉니다):",
            text=default,
        )
        if not ok or not name.strip():
            return
        try:
            lens = shared_camera.lens_from_calibration_result(result, self.camera_config.width, self.camera_config.height)
            note = (f"Camera Intrinsic {model_text}, RMS {result.rms_error:.3f}px, "
                    f"{self.camera_config.width}x{self.camera_config.height}")
            path = shared_camera.add_lens_preset(name.strip(), lens, note)
            linked = ""
            if self.vehicle and camera:
                shared_vehicle.ensure(self.vehicle)
                shared_vehicle.set_lens(self.vehicle, camera, name.strip())
                self._set_vehicle(self.vehicle)
                linked = (f"\n\n차량 {self.vehicle} 의 카메라 '{camera}' 렌즈로 지정했습니다 - "
                          "CAMERA–LiDAR 3단계에서 자동으로 골라집니다.")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "렌즈 등록 실패", str(e))
            return
        self.status_label.setText(f"CAMERA–LiDAR 렌즈 프리셋 등록: {name.strip()} ({path})")
        QMessageBox.information(self, "렌즈 등록", f"'{name.strip()}' 을(를) CAMERA–LiDAR 렌즈 목록에 등록했습니다.\n{path}{linked}")

    def _on_export_opencv(self, model: CameraModelType) -> None:
        result = self.calibration_results.get(model)
        if not result or not result.success or self.camera_config is None or self.pattern_config is None:
            QMessageBox.warning(self, "Export 불가", f"{CameraModelType(model).value} 모델의 캘리브레이션 결과가 없습니다.")
            return
        try:
            self._ensure_output_session()
            path = self.output_manager.intrinsic_path(model)
            export_opencv_yaml(result, self.camera_config, self.pattern_config, str(path))
            self.output_manager.record_export(f"intrinsic.{model_filename(model)}", path)
            self.status_label.setText(f"OpenCV YAML 저장 완료: {path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Export 실패", str(e))

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
        self.repeated_kfold_results = {}
        self.image_paths = [f.image_info.path for f in project.dataset.frames]
        IntrinsicWorkspace.sync_owner_state(self)

        # --- 설정 패널 위젯도 불러온 값으로 맞춰준다 (재계산/이어서 작업 시 일관성) ---
        self.sensor_name_edit.setText(self.camera_config.sensor_name or "")
        self.width_spin.setValue(self.camera_config.width)
        self.height_spin.setValue(self.camera_config.height)
        self.squares_x_spin.setValue(self.pattern_config.squares_x)
        self.squares_y_spin.setValue(self.pattern_config.squares_y)
        # pattern_config는 항상 미터(m) 단위로 저장돼 있으니, mm 입력 위젯에는 변환해서 넣는다.
        self.square_size_spin.setValue(self.pattern_config.square_size * 1000.0)
        self.calibration_method = CalibrationMethod.STANDARD
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
        if self.scores:
            recommended = next((s.model_name for s in self.scores if s.is_recommended), None)
            if recommended is not None:
                self.result_view.select_model(recommended)
                self.preview_view.select_model(recommended)

        self.loaded_label.setText(f"불러온 이미지: {len(self.image_paths)}장 (프로젝트: {Path(path).name})")
        self.auto_select_status_label.setText("프로젝트 데이터셋을 불러왔습니다. 필요하면 다시 자동 선별할 수 있습니다.")
        self._sync_auto_select_controls(reset_value=True)
        self.run_button.setEnabled(bool(self.image_paths))

        # 홈 화면에서 불러와도 결과가 바로 보이도록 작업공간으로 이동한다.
        self._show_intrinsic_workspace()
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
