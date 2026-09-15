"""
Camera intrinsic workspace wrapper.

The existing intrinsic implementation still lives in MainWindow for now. This
wrapper gives the application an explicit workspace boundary without moving the
large legacy state machine in one risky edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QTabWidget, QVBoxLayout, QWidget

from calibration.types import (
    CalibrationMethod,
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    CrossDatasetValidationResult,
    Dataset,
    ModelScore,
    OptimizerResult,
    ObjectReleasingValidationResult,
    OutlierResult,
    PatternConfig,
    RepeatedKFoldResult,
    SceneQualityAnalysis,
    StandardVsObjectReleasingComparison,
    SubsetCalibrationResult,
    ValidationResult,
)
from ui.dataset_view import DatasetView
from ui.result_view import ResultView
from ui.scene_quality_view import SceneQualityView
from ui.optimizer_view import OptimizerView
from ui.undistort_straightness_view import UndistortStraightnessView


@dataclass
class IntrinsicState:
    image_paths: list[str] = field(default_factory=list)
    dataset: Dataset | None = None
    camera_config: CameraConfig | None = None
    pattern_config: PatternConfig | None = None
    calibration_results: dict[CameraModelType, CalibrationResult] = field(default_factory=dict)
    object_releasing_result: CalibrationResult | None = None
    object_releasing_validation_result: ObjectReleasingValidationResult | None = None
    standard_vs_object_releasing_comparison: StandardVsObjectReleasingComparison | None = None
    validation_results: dict[CameraModelType, ValidationResult] = field(default_factory=dict)
    cross_dataset_results: list[CrossDatasetValidationResult] = field(default_factory=list)
    scores: list[ModelScore] = field(default_factory=list)
    outlier_result: OutlierResult | None = None
    calibration_method: CalibrationMethod = CalibrationMethod.STANDARD
    scene_quality_analysis: SceneQualityAnalysis | None = None
    subset_calibration_result: SubsetCalibrationResult | None = None
    # Paper Evidence 단계 - Repeated K-Fold(calibration/kfold.py::
    # run_repeated_kfold_all_models) 결과. "Run Repeated K-Fold" 버튼을 누를
    # 때마다 계산되어 여기 저장되고, "Export Paper Metrics"는 이 저장된
    # 결과를 그대로 쓴다(버튼을 누를 때마다 다시 계산하지 않는다). 이
    # 결과는 .ccproj에 영구 저장되지 않는다 - 프로젝트를 다시 열거나 새
    # calibration을 실행하면 비워지고(stale 방지), 필요하면 다시 실행해야
    # 한다는 제약을 그대로 둔다.
    repeated_kfold_results: dict[CameraModelType, RepeatedKFoldResult] = field(default_factory=dict)
    optimizer_results: dict[CameraModelType, OptimizerResult] = field(default_factory=dict)


class IntrinsicWorkspace(QWidget):
    back_requested = Signal()

    def __init__(self, settings_panel: QWidget, tabs: QWidget, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        home_button = QPushButton("← Calibration Home")
        home_button.clicked.connect(self.back_requested.emit)
        header.addWidget(home_button)
        header.addStretch(1)
        layout.addLayout(header)
        layout.addWidget(settings_panel)
        layout.addWidget(tabs, stretch=1)

    @staticmethod
    def initialize_owner_state(owner) -> IntrinsicState:
        state = IntrinsicState()
        owner.intrinsic_state = state
        return state

    @staticmethod
    def sync_owner_state(owner) -> None:
        state = getattr(owner, "intrinsic_state", None)
        if state is None:
            return
        # State now lives in IntrinsicState; MainWindow exposes compatibility
        # properties so older handlers can keep their readable names.

    @classmethod
    def create_for_main_window(cls, owner, settings_panel: QWidget) -> "IntrinsicWorkspace":
        tabs = QTabWidget()

        owner.dataset_view = DatasetView()
        owner.result_view = ResultView(standalone=False)
        owner.scene_quality_view = SceneQualityView()
        owner.optimizer_view = OptimizerView()
        # Undistort Preview와 Straightness Map을 한 화면으로 합친 뷰
        # (Edge Error Map은 별도 요청으로 제거됨). preview_view라는 이름은
        # main_window.py의 기존 호출부와의 혼란을 줄이기 위해 유지한다.
        owner.preview_view = UndistortStraightnessView()

        tabs.addTab(owner.dataset_view, "① Dataset")
        tabs.addTab(owner.preview_view, "② Preview")
        tabs.addTab(owner.result_view.model_comparison_widget, "③ Model Comparison")
        tabs.addTab(owner.scene_quality_view, "④ Scene Ranking")
        tabs.addTab(owner.optimizer_view, "(5) Optimizer")
        owner.tabs = tabs
        workspace = cls(settings_panel, tabs)
        workspace.connect_owner_handlers(owner)
        return workspace

    def connect_owner_handlers(self, owner) -> None:
        owner.result_view.export_opencv_requested.connect(owner._on_export_opencv)
        owner.result_view.cross_dataset_requested.connect(owner._on_cross_dataset_requested)
        owner.result_view.repeated_kfold_requested.connect(owner._on_repeated_kfold_requested)
        owner.result_view.export_paper_metrics_requested.connect(owner._on_export_paper_metrics_requested)
        owner.scene_quality_view.recalibrate_requested.connect(owner._on_subset_recalibrate_requested)
        owner.scene_quality_view.model_changed.connect(owner._on_scene_quality_model_changed)
        owner.scene_quality_view.export_subset_requested.connect(owner._on_export_subset_calibration)
        owner.scene_quality_view.validate_subset_requested.connect(owner._on_validate_best_subset)
        owner.optimizer_view.run_requested.connect(owner._on_optimizer_run)
        owner.optimizer_view.cancel_requested.connect(owner._on_optimizer_cancel)
        owner.optimizer_view.apply_requested.connect(owner._on_optimizer_apply)
        owner.optimizer_view.restore_requested.connect(owner._on_optimizer_restore)
