"""
camera_calibrator.ui.main_window
====================================

설계 문서 14번 UI 구성안 + 16번 폴더 구조를 따른다.
이 파일은 "조립"만 한다 - 검출/캘리브레이션/추천/이상치 계산은 전부
calibration/*.py에 있고, 여기서는 그 함수들을 worker.py를 통해 호출하고
결과를 각 view 위젯에 그대로 전달할 뿐이다.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
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
from calibration.rosbag_reader import list_image_topics, extract_images_from_bag
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
from ui.live_capture_dialog import LiveCaptureDialog
from ui.worker import PipelineWorker, OutlierPruneWorker, run_worker_in_thread

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
        self._thread: QThread | None = None
        self._worker = None  # QThread가 살아있는 동안 GC 방지용 강한 참조

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
        self.tabs.addTab(self.dataset_view, "① Dataset")
        self.tabs.addTab(self.coverage_view, "② Coverage")
        self.tabs.addTab(self.result_view, "③ Model / Validation / Export")
        self.tabs.addTab(self.preview_view, "④ Undistort Preview")
        self.tabs.addTab(self.radial_profile_view, "⑤ Edge Error Map")
        self.tabs.addTab(self.straightness_view, "⑥ Straightness Map")
        layout.addWidget(self.tabs, stretch=1)

        self.status_label = QLabel("이미지를 불러온 뒤 [캘리브레이션 실행]을 누르세요.")
        self.statusBar().addWidget(self.status_label, stretch=1)

        self.result_view.outlier_prune_requested.connect(self._on_outlier_prune_requested)
        self.result_view.export_opencv_requested.connect(self._on_export_opencv)
        self.result_view.export_ros_requested.connect(self._on_export_ros)
        self.result_view.export_report_requested.connect(self._on_export_report)
        self.result_view.export_json_requested.connect(self._on_export_json)
        self.result_view.export_csv_requested.connect(self._on_export_csv)

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
        try:
            extracted = extract_images_from_bag(bag_path, topic, out_dir, min_interval_sec=interval)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "이미지 추출 실패", str(e))
            return

        if not extracted:
            QMessageBox.warning(self, "추출된 이미지 없음", "선택한 토픽/간격으로 추출된 이미지가 없습니다.")
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

        worker = PipelineWorker(self.image_paths, self.pattern_config, self.camera_config)
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
            self.preview_view.set_context(self.dataset, self.camera_config, results)
            self.dataset_view.set_dataset(self.dataset)  # per_frame_error 채워졌으니 갱신
            if self.pattern_config is not None:
                self.straightness_view.set_context(self.dataset, self.camera_config, results, self.pattern_config)
        self.radial_profile_view.set_results(results)
        self._refresh_result_view()

    def _on_validation_ready(self, results: dict[CameraModelType, ValidationResult]) -> None:
        self.validation_results = results
        self._refresh_result_view()

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

    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "오류", message)
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
            self.dataset, self.camera_config, self.pattern_config, reference_model
        )
        thread = run_worker_in_thread(worker, self)

        worker.progress.connect(self.status_label.setText)
        worker.dataset_updated.connect(lambda ds: self.dataset_view.set_dataset(ds))
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
            self.preview_view.set_context(self.dataset, self.camera_config, self.calibration_results)
            self.radial_profile_view.set_results(self.calibration_results)
            self.straightness_view.set_context(
                self.dataset, self.camera_config, self.calibration_results, self.pattern_config
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
