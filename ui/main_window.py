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

from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from calibration.types import (
    CalibrationResult,
    CalibrationProject,
    CameraConfig,
    CameraModelType,
    Dataset,
    ModelScore,
    OutlierResult,
    PatternConfig,
    PatternType,
    ValidationResult,
)
from calibration.recommender import compute_final_result
from calibration.quality import coverage_percentage
from calibration.rosbag_reader import list_image_topics
from calibration.ros_live import ROS_LIVE_BACKEND
from calibration.project_io import load_project, save_project, PROJECT_EXTENSION
from export.opencv import export_opencv_yaml
from export.ros import export_ros_camera_info
from export.report import export_html_report
from export.json_export import export_json
from export.csv_export import export_csv

from ui.dataset_view import DatasetView
from ui.coverage_view import CoverageView
from ui.result_view import ResultView
from ui.preview import PreviewView
from ui.radial_profile_view import RadialProfileView
from ui.straightness_view import StraightnessView
from ui.external_compare_view import ExternalCompareView
from ui.live_capture_dialog import LiveCaptureDialog
from ui.worker import (
    PipelineWorker,
    OutlierPruneWorker,
    SelfCheckWorker,
    BagExtractionWorker,
    run_worker_in_thread,
)

logger = logging.getLogger(__name__)

# 자동 저장 파일 경로 - 프로젝트 폴더가 아니라 홈 디렉터리 밑 고정 위치에 둔다.
# 앱이 응답 없음/강제 종료로 죽어도 다음 실행에서 항상 같은 경로를 확인해
# 복구를 제안할 수 있어야 하기 때문 (사용자가 저장 위치를 고를 필요 없음).
_AUTOSAVE_DIR = Path.home() / ".camera_calibrator"
_AUTOSAVE_PATH = _AUTOSAVE_DIR / "autosave.ccproj"

# ChArUco에서 흔히 쓰이는 사전 목록 (cv2.aruco.DICT_* 속성명 그대로)
_ARUCO_DICTIONARIES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Calibration Tool")
        self.resize(1280, 860)

        # --- 상태 ---
        self.image_paths: list[str] = []
        self.dataset: Dataset | None = None
        self.camera_config: CameraConfig | None = None
        self.pattern_config: PatternConfig | None = None
        self.calibration_results: dict[CameraModelType, CalibrationResult] = {}
        self.validation_results: dict[CameraModelType, ValidationResult] = {}
        self.scores: list[ModelScore] = []
        self.outlier_result: OutlierResult | None = None
        self.use_rational_model: bool = False
        self._thread: QThread | None = None
        self._worker = None  # QThread가 살아있는 동안 GC 방지용 강한 참조
        self._self_check_thread: QThread | None = None
        self._bag_thread: QThread | None = None
        self._bag_worker = None  # QThread가 살아있는 동안 GC 방지용 강한 참조
        self._self_check_worker = None  # 위와 동일한 이유로 별도 워커도 강한 참조 보관

        self._build_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(self._build_settings_panel())

        self.tabs = QTabWidget()
        self.dataset_view = DatasetView()
        self.coverage_view = CoverageView()
        self.result_view = ResultView()
        self.preview_view = PreviewView()
        self.radial_profile_view = RadialProfileView()
        self.straightness_view = StraightnessView()
        self.external_compare_view = ExternalCompareView()
        self.tabs.addTab(self.dataset_view, "① Dataset")
        self.tabs.addTab(self.coverage_view, "② Coverage")
        self.tabs.addTab(self.result_view, "③ Model / Validation / Export")
        self.tabs.addTab(self.preview_view, "④ Undistort Preview")
        self.tabs.addTab(self.radial_profile_view, "⑤ Edge Error Map")
        self.tabs.addTab(self.straightness_view, "⑥ Straightness Map")
        self.tabs.addTab(self.external_compare_view, "⑦ 외부 결과 비교")
        layout.addWidget(self.tabs, stretch=1)

        self.status_label = QLabel("이미지를 불러온 뒤 [캘리브레이션 실행]을 누르세요.")
        self.statusBar().addWidget(self.status_label, stretch=1)

        self.result_view.outlier_prune_requested.connect(self._on_outlier_prune_requested)
        self.result_view.export_opencv_requested.connect(self._on_export_opencv)
        self.result_view.export_ros_requested.connect(self._on_export_ros)
        self.result_view.export_report_requested.connect(self._on_export_report)
        self.result_view.export_json_requested.connect(self._on_export_json)
        self.result_view.export_csv_requested.connect(self._on_export_csv)

        # 앱이 응답 없음/강제 종료 등으로 꺼져도 마지막으로 완료된 계산
        # 결과는 자동 저장본에서 복구할 수 있게, 창이 뜨자마자 한 번 확인한다.
        # (실제 사용자 버그: 큰 rosbag을 불러오다 응답 없음이 떠서 강제 종료한
        # 뒤 다시 켜면 방금까지 보이던 계산 결과가 전부 사라져 있었음 -
        # 저장하지 않은 결과는 메모리에만 있어서 프로세스가 죽으면 없어지는
        # 게 원인이었다. 매 실행 완료 시 자동 저장해두면 이런 경우에도
        # Export만큼은 다시 할 수 있다.)
        QTimer.singleShot(0, self._offer_autosave_recovery)

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

        tools_menu = menu_bar.addMenu("도구")
        self.self_check_action = QAction("자체 진단 (합성 데이터로 정확도 확인)...", self)
        self.self_check_action.setToolTip(
            "정답을 미리 아는 가짜(합성) ChArUco 데이터로 Pinhole/Extended Pinhole/\n"
            "Rational model을 돌려서 복원된 fx/fy/cx/cy가 정답에 가까운지 확인합니다.\n"
            "현재 불러온 이미지/캘리브레이션 결과와는 무관하며, 몇 초~수십 초 걸립니다."
        )
        self.self_check_action.triggered.connect(self._on_run_self_check)
        tools_menu.addAction(self.self_check_action)

    def _build_settings_panel(self) -> QWidget:
        group = QGroupBox("Camera Setup & Pattern")
        outer = QHBoxLayout(group)

        camera_form = QFormLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 20000)
        self.width_spin.setValue(1920)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 20000)
        self.height_spin.setValue(1080)
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
        outer.addLayout(camera_form)

        pattern_form = QFormLayout()
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
        self.pattern_type_combo.currentIndexChanged.connect(self._on_pattern_type_changed)

        pattern_form.addRow("Pattern type", self.pattern_type_combo)
        pattern_form.addRow("Squares X", self.squares_x_spin)
        pattern_form.addRow("Squares Y", self.squares_y_spin)
        pattern_form.addRow("Square size", self.square_size_spin)
        pattern_form.addRow("Marker size", self.marker_size_spin)
        pattern_form.addRow("Dictionary", self.dictionary_combo)
        self._pattern_form = pattern_form  # setRowVisible로 마커/딕셔너리 행을 토글하기 위해 보관
        outer.addLayout(pattern_form)

        action_layout = QVBoxLayout()
        self.load_button = QPushButton("이미지 불러오기")
        self.load_button.clicked.connect(self._on_load_images)
        self.load_bag_button = QPushButton("rosbag에서 불러오기")
        self.load_bag_button.clicked.connect(self._on_load_from_bag)
        self.load_live_button = QPushButton("실시간 카메라 구독")
        self.load_live_button.clicked.connect(self._on_load_from_live)
        self.loaded_label = QLabel("불러온 이미지: 0장")
        self.run_button = QPushButton("캘리브레이션 실행")
        self.run_button.clicked.connect(self._on_run_pipeline)
        self.run_button.setEnabled(False)
        action_layout.addWidget(self.load_button)
        action_layout.addWidget(self.load_bag_button)
        action_layout.addWidget(self.load_live_button)
        action_layout.addWidget(self.loaded_label)
        action_layout.addWidget(self.run_button)
        action_layout.addStretch(1)
        outer.addLayout(action_layout)

        return group

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

        try:
            topics = list_image_topics(bag_path)
        except ImportError as e:
            QMessageBox.critical(self, "rosbags 미설치", str(e))
            return
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "bag 읽기 실패", f"bag 파일을 여는 데 실패했습니다:\n{e}")
            return

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
        worker.finished_extraction.connect(
            lambda extracted: self._on_bag_extraction_finished(extracted, bag_path)
        )
        worker.error.connect(self._on_error)
        worker.finished.connect(progress.close)

        self._bag_thread, self._bag_worker = thread, worker
        self.load_button.setEnabled(False)
        self.load_bag_button.setEnabled(False)
        thread.finished.connect(lambda: self.load_button.setEnabled(True))
        thread.finished.connect(lambda: self.load_bag_button.setEnabled(True))
        thread.start()
        progress.show()

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
        self.loaded_label.setText(f"불러온 이미지: {len(dialog.captured_paths)}장 (실시간 캡처)")
        self.run_button.setEnabled(True)

    def _on_pattern_type_changed(self) -> None:
        """Chessboard는 marker_size/dictionary가 필요 없으니 해당 입력 행을
        숨긴다 - 안 그러면 사용자가 "이거 왜 입력해야 하지?" 헷갈리기 쉽다.
        """
        is_charuco = self.pattern_type_combo.currentData() == PatternType.CHARUCO
        # 행 인덱스: 0=Pattern type, 1=Squares X, 2=Squares Y, 3=Square size,
        # 4=Marker size, 5=Dictionary (addRow 호출 순서와 동일).
        self._pattern_form.setRowVisible(4, is_charuco)
        self._pattern_form.setRowVisible(5, is_charuco)

    def _current_pattern_config(self) -> PatternConfig:
        # UI는 mm로 입력받지만(보드 인쇄 스펙이 보통 mm 단위), 내부 계산/export는
        # 전부 미터(m) 기준이라 여기서 한 번만 변환한다 - 이후 파이프라인은
        # 이 값이 mm에서 왔는지 몰라도 된다.
        pattern_type = self.pattern_type_combo.currentData()
        is_charuco = pattern_type == PatternType.CHARUCO
        return PatternConfig(
            type=pattern_type,
            squares_x=self.squares_x_spin.value(),
            squares_y=self.squares_y_spin.value(),
            square_size=self.square_size_spin.value() / 1000.0,
            marker_size=(self.marker_size_spin.value() / 1000.0) if is_charuco else None,
            dictionary=self.dictionary_combo.currentText() if is_charuco else None,
        )

    def _current_camera_config(self) -> CameraConfig:
        return CameraConfig(width=self.width_spin.value(), height=self.height_spin.value())

    def _on_run_pipeline(self) -> None:
        if not self.image_paths:
            QMessageBox.warning(self, "이미지 없음", "먼저 이미지를 불러오세요.")
            return

        self.pattern_config = self._current_pattern_config()
        self.camera_config = self._current_camera_config()
        self.calibration_results = {}
        self.validation_results = {}
        self.scores = []

        self.use_rational_model = self.rational_checkbox.isChecked()
        worker = PipelineWorker(
            self.image_paths, self.pattern_config, self.camera_config,
            use_rational_model=self.use_rational_model,
        )
        thread = run_worker_in_thread(worker, self)

        worker.progress.connect(self.status_label.setText)
        worker.dataset_ready.connect(self._on_dataset_ready)
        worker.quality_ready.connect(self._on_quality_ready)
        worker.models_ready.connect(self._on_models_ready)
        worker.validation_ready.connect(self._on_validation_ready)
        worker.recommendation_ready.connect(self._on_recommendation_ready)
        worker.error.connect(self._on_error)

        self._thread, self._worker = thread, worker
        self.run_button.setEnabled(False)
        self.load_button.setEnabled(False)
        thread.finished.connect(lambda: self.run_button.setEnabled(True))
        thread.finished.connect(lambda: self.load_button.setEnabled(True))
        thread.start()

    # --- PipelineWorker 콜백 ---

    def _on_dataset_ready(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.dataset_view.set_dataset(dataset)

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
        self.radial_profile_view.set_results(results)
        self._refresh_result_view()

    def _on_validation_ready(self, results: dict[CameraModelType, ValidationResult]) -> None:
        self.validation_results = results
        self._refresh_result_view()
        if self.dataset is not None and self.camera_config is not None and self.pattern_config is not None:
            self.external_compare_view.set_context(
                self.dataset, self.camera_config, self.pattern_config,
                self.validation_results, use_rational_model=self.use_rational_model,
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
        try:
            _AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
            project = CalibrationProject(
                project_name=self.camera_config.sensor_name or "autosave",
                camera_config=self.camera_config,
                pattern_config=self.pattern_config,
                dataset=self.dataset,
                calibration_results=self.calibration_results,
                validation_results=self.validation_results,
                model_scores=self.scores,
                outlier_result=self.outlier_result,
            )
            save_project(project, str(_AUTOSAVE_PATH))
            logger.debug("자동 저장 완료: %s", _AUTOSAVE_PATH)
        except Exception:  # noqa: BLE001 - 자동 저장 실패로 사용자 작업을 막으면 안 됨
            logger.exception("자동 저장 실패 (무시하고 계속 진행)")

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

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export_opencv(self, model: CameraModelType) -> None:
        result = self.calibration_results.get(model)
        if not result or not result.success or self.camera_config is None or self.pattern_config is None:
            QMessageBox.warning(self, "Export 불가", f"{model.value} 모델의 캘리브레이션 결과가 없습니다.")
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
            QMessageBox.warning(self, "Export 불가", f"{model.value} 모델의 캘리브레이션 결과가 없습니다.")
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
            QMessageBox.warning(self, "Export 불가", f"{model.value} 모델의 캘리브레이션 결과가 없습니다.")
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
            QMessageBox.warning(self, "Export 불가", f"{model.value} 모델의 캘리브레이션 결과가 없습니다.")
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
            model_scores=self.scores,
            outlier_result=self.outlier_result,
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
        self.scores = project.model_scores
        self.outlier_result = project.outlier_result
        self.image_paths = [f.image_info.path for f in project.dataset.frames]

        # --- 설정 패널 위젯도 불러온 값으로 맞춰준다 (재계산/이어서 작업 시 일관성) ---
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
        if self.validation_results:
            self.external_compare_view.set_context(
                self.dataset, self.camera_config, self.pattern_config,
                self.validation_results, use_rational_model=self.use_rational_model,
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
