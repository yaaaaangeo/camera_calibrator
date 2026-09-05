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
    ObjectReleasingValidationResult,
    OutlierResult,
    PatternConfig,
    SceneQualityAnalysis,
    StandardVsObjectReleasingComparison,
    SubsetCalibrationResult,
    ValidationResult,
)
from ui.dataset_view import DatasetView
from ui.result_view import ResultView
from ui.scene_quality_view import SceneQualityView
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
        # Undistort Preview와 Straightness Map을 한 화면으로 합친 뷰
        # (Edge Error Map은 별도 요청으로 제거됨). preview_view라는 이름은
        # main_window.py의 기존 호출부와의 혼란을 줄이기 위해 유지한다.
        owner.preview_view = UndistortStraightnessView()

        tabs.addTab(owner.dataset_view, "① Dataset")
        tabs.addTab(owner.preview_view, "② Preview")
        tabs.addTab(owner.scene_quality_view, "③ Scene Ranking")
        tabs.addTab(owner.result_view.model_comparison_widget, "④ Model Comparison")
        owner.tabs = tabs
        workspace = cls(settings_panel, tabs)
        workspace.connect_owner_handlers(owner)
        return workspace

    def connect_owner_handlers(self, owner) -> None:
        owner.result_view.export_opencv_requested.connect(owner._on_export_opencv)
        owner.result_view.cross_dataset_requested.connect(owner._on_cross_dataset_requested)
        owner.scene_quality_view.recalibrate_requested.connect(owner._on_subset_recalibrate_requested)
        owner.scene_quality_view.model_changed.connect(owner._on_scene_quality_model_changed)
        owner.scene_quality_view.export_subset_requested.connect(owner._on_export_subset_calibration)
