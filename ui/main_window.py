"""
camera_calibrator.ui.main_window
====================================

설계 문서 14번 UI 구성안 + 16번 폴더 구조를 따른다.
이 파일은 "조립"만 한다 - 검출/캘리브레이션/추천/이상치 계산은 전부
calibration/*.py에 있고, 여기서는 그 함수들을 worker.py를 통해 호출하고
결과를 각 view 위젯에 그대로 전달할 뿐이다.
"""

from __future__ import annotations

from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
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
from export.opencv import export_opencv_yaml
from export.ros import export_ros_camera_info
from export.report import export_html_report

from ui.dataset_view import DatasetView
from ui.coverage_view import CoverageView
from ui.result_view import ResultView
from ui.preview import PreviewView
from ui.radial_profile_view import RadialProfileView
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
        self.tabs.addTab(self.dataset_view, "① Dataset")
        self.tabs.addTab(self.coverage_view, "② Coverage")
        self.tabs.addTab(self.result_view, "③ Model / Validation / Export")
        self.tabs.addTab(self.preview_view, "④ Undistort Preview")
        self.tabs.addTab(self.radial_profile_view, "⑤ Edge Error Map")
        layout.addWidget(self.tabs, stretch=1)

        self.status_label = QLabel("이미지를 불러온 뒤 [캘리브레이션 실행]을 누르세요.")
        self.statusBar().addWidget(self.status_label, stretch=1)

        self.result_view.outlier_prune_requested.connect(self._on_outlier_prune_requested)
        self.result_view.export_opencv_requested.connect(self._on_export_opencv)
        self.result_view.export_ros_requested.connect(self._on_export_ros)
        self.result_view.export_report_requested.connect(self._on_export_report)

    # ------------------------------------------------------------------
    # 설정 패널 (설계 문서 14번 ① Camera Setup, ③ Calibration Pattern)
    # ------------------------------------------------------------------

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
        self.square_size_spin.setRange(0.001, 1.0)
        self.square_size_spin.setDecimals(4)
        self.square_size_spin.setSingleStep(0.001)
        self.square_size_spin.setValue(0.04)
        self.marker_size_spin = QDoubleSpinBox()
        self.marker_size_spin.setRange(0.001, 1.0)
        self.marker_size_spin.setDecimals(4)
        self.marker_size_spin.setSingleStep(0.001)
        self.marker_size_spin.setValue(0.03)
        self.dictionary_combo = QComboBox()
        self.dictionary_combo.addItems(_ARUCO_DICTIONARIES)
        self.dictionary_combo.setCurrentText("DICT_5X5_100")

        pattern_form.addRow("Squares X", self.squares_x_spin)
        pattern_form.addRow("Squares Y", self.squares_y_spin)
        pattern_form.addRow("Square size (m)", self.square_size_spin)
        pattern_form.addRow("Marker size (m)", self.marker_size_spin)
        pattern_form.addRow("Dictionary", self.dictionary_combo)
        outer.addLayout(pattern_form)

        action_layout = QVBoxLayout()
        self.load_button = QPushButton("이미지 불러오기")
        self.load_button.clicked.connect(self._on_load_images)
        self.loaded_label = QLabel("불러온 이미지: 0장")
        self.run_button = QPushButton("캘리브레이션 실행")
        self.run_button.clicked.connect(self._on_run_pipeline)
        self.run_button.setEnabled(False)
        action_layout.addWidget(self.load_button)
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

    def _current_pattern_config(self) -> PatternConfig:
        return PatternConfig(
            type=PatternType.CHARUCO,
            squares_x=self.squares_x_spin.value(),
            squares_y=self.squares_y_spin.value(),
            square_size=self.square_size_spin.value(),
            marker_size=self.marker_size_spin.value(),
            dictionary=self.dictionary_combo.currentText(),
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
