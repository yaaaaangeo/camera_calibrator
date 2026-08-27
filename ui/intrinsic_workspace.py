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
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    CrossDatasetValidationResult,
    Dataset,
    ModelScore,
    OutlierResult,
    PatternConfig,
    ValidationResult,
)
from ui.coverage_view import CoverageView
from ui.dataset_view import DatasetView
from ui.diagnosis_view import DiagnosisView
from ui.external_compare_view import ExternalCompareView
from ui.model_refitting_view import ModelRefittingView
from ui.preview import PreviewView
from ui.radial_profile_view import RadialProfileView
from ui.result_view import ResultView
from ui.stability_view import StabilityView
from ui.straightness_view import StraightnessView


@dataclass
class IntrinsicState:
    image_paths: list[str] = field(default_factory=list)
    dataset: Dataset | None = None
    camera_config: CameraConfig | None = None
    pattern_config: PatternConfig | None = None
    calibration_results: dict[CameraModelType, CalibrationResult] = field(default_factory=dict)
    validation_results: dict[CameraModelType, ValidationResult] = field(default_factory=dict)
    cross_dataset_results: list[CrossDatasetValidationResult] = field(default_factory=list)
    scores: list[ModelScore] = field(default_factory=list)
    outlier_result: OutlierResult | None = None
    use_rational_model: bool = False


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

        owner.dataset_view = DatasetView(group_title="Dataset")
        owner.coverage_view = CoverageView()
        owner.result_view = ResultView(standalone=False)
        owner.preview_view = PreviewView()
        owner.radial_profile_view = RadialProfileView()
        owner.straightness_view = StraightnessView()
        owner.external_compare_view = ExternalCompareView()
        owner.model_refitting_view = ModelRefittingView()
        owner.diagnosis_view = DiagnosisView()
        owner.stability_view = StabilityView()
        owner.error_analysis_tab = cls._build_error_analysis_tab(
            owner.preview_view,
            owner.radial_profile_view,
            owner.straightness_view,
        )

        tabs.addTab(owner.dataset_view, "① Dataset")
        tabs.addTab(owner.coverage_view, "② Coverage")
        tabs.addTab(owner.result_view.calibration_widget, "③ Outlier")
        tabs.addTab(owner.result_view.validation_widget, "④ Validation")
        tabs.addTab(owner.error_analysis_tab, "⑤ Error Analysis")
        tabs.addTab(owner.stability_view, "⑥ Stability")
        tabs.addTab(owner.result_view.model_comparison_widget, "⑦ Model Comparison")
        tabs.addTab(owner.diagnosis_view, "⑧ Diagnosis")
        tabs.addTab(owner.result_view.export_widget, "⑨ Export")
        tabs.addTab(owner.external_compare_view, "⑩ External Compare")
        tabs.addTab(owner.model_refitting_view, "⑪ Model Refitting")
        owner.tabs = tabs
        workspace = cls(settings_panel, tabs)
        workspace.connect_owner_handlers(owner)
        return workspace

    def connect_owner_handlers(self, owner) -> None:
        owner.result_view.outlier_prune_requested.connect(owner._on_outlier_prune_requested)
        owner.result_view.export_opencv_requested.connect(owner._on_export_opencv)
        owner.result_view.export_ros_requested.connect(owner._on_export_ros)
        owner.result_view.export_report_requested.connect(owner._on_export_report)
        owner.result_view.export_json_requested.connect(owner._on_export_json)
        owner.result_view.export_csv_requested.connect(owner._on_export_csv)
        owner.result_view.cross_dataset_requested.connect(owner._on_cross_dataset_requested)
        owner.model_refitting_view.refit_requested.connect(owner._on_model_refit_requested)

    @staticmethod
    def _build_error_analysis_tab(preview_view: QWidget, radial_profile_view: QWidget, straightness_view: QWidget) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        sub_tabs = QTabWidget()
        sub_tabs.addTab(preview_view, "Undistort Preview")
        sub_tabs.addTab(radial_profile_view, "Edge Error Map")
        sub_tabs.addTab(straightness_view, "Straightness Map")
        layout.addWidget(sub_tabs)
        return tab
