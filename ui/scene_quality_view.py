"""Scene quality ranking, manual subset selection, preview and result comparison."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from calibration.scene_quality import recommend_best_subset
from calibration.models.common import undistort_image
from calibration.types import (
    CalibrationResult, CameraConfig, CameraModelType, Dataset, SceneQualityAnalysis,
    SubsetCalibrationResult,
)

_MODEL_LABELS = {
    CameraModelType.PINHOLE: "Ideal Pinhole",
    CameraModelType.BROWN_CONRADY: "Brown-Conrady",
    CameraModelType.EXTENDED_PINHOLE: "Rational",
    CameraModelType.FISHEYE: "Fisheye",
}


def _model_label(model) -> str:
    try:
        normalized = model if isinstance(model, CameraModelType) else CameraModelType(str(model))
    except (TypeError, ValueError):
        return str(model) if model is not None else "Unknown model"
    return _MODEL_LABELS.get(normalized, normalized.value)


class NumericItem(QTableWidgetItem):
    def __init__(self, text: str, value: float):
        super().__init__(text)
        self.setData(Qt.UserRole, float(value))

    def __lt__(self, other):
        if isinstance(other, QTableWidgetItem):
            left = self.data(Qt.UserRole)
            right = other.data(Qt.UserRole)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return left < right
        return super().__lt__(other)


class SceneQualityView(QWidget):
    recalibrate_requested = Signal(list, object)  # frame ids, CameraModelType
    model_changed = Signal(object)
    export_subset_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dataset: Dataset | None = None
        self._camera_config: CameraConfig | None = None
        self._analysis: SceneQualityAnalysis | None = None
        self._original_calibration_result: CalibrationResult | None = None
        self._best_subset_result: SubsetCalibrationResult | None = None
        self._original_calibration_pixmap: QPixmap | None = None
        self._best_subset_calibration_pixmap: QPixmap | None = None

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Ranking model:"))
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._emit_model_changed)
        controls.addWidget(self.model_combo)
        self.best20_button = QPushButton("Recommend Best 20")
        self.best30_button = QPushButton("Recommend Best 30")
        self.all_button = QPushButton("Select All")
        self.clear_button = QPushButton("Clear")
        self.custom_button = QPushButton("Custom Selection")
        for button in (self.best20_button, self.best30_button, self.all_button, self.clear_button, self.custom_button):
            controls.addWidget(button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.best20_button.clicked.connect(lambda: self._recommend(20))
        self.best30_button.clicked.connect(lambda: self._recommend(30))
        self.all_button.clicked.connect(lambda: self._set_all_checked(True))
        self.clear_button.clicked.connect(lambda: self._set_all_checked(False))
        self.custom_button.clicked.connect(
            lambda: self.selection_label.setText("Custom mode: Selected column checkboxes can be edited freely.")
        )

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Rank", "Scene", "Score", "RMS (px)", "Detection", "Sharpness", "Selected"]
        )
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._update_preview)
        self.table.cellClicked.connect(lambda row, _column: self._update_preview_for_row(row))
        layout.addWidget(self.table, stretch=3)

        action_row = QHBoxLayout()
        self.selection_label = QLabel("Run Initial Calibration to calculate scene ranking.")
        action_row.addWidget(self.selection_label, stretch=1)
        self.recalibrate_button = QPushButton("Re-Calibrate Selected Scenes")
        self.recalibrate_button.setProperty("role", "primary")
        self.recalibrate_button.setEnabled(False)
        self.recalibrate_button.clicked.connect(self._request_recalibration)
        action_row.addWidget(self.recalibrate_button)
        self.export_subset_button = QPushButton("Export Subset OpenCV YAML")
        self.export_subset_button.setEnabled(False)
        self.export_subset_button.clicked.connect(self.export_subset_requested.emit)
        action_row.addWidget(self.export_subset_button)
        layout.addLayout(action_row)

        # Ranking/재계산 영역의 다음 줄: 같은 scene을 전체 데이터 K/D와 Best
        # Subset K/D로 각각 보정한 이미지를 왼쪽에서 직접 비교하고, 오른쪽에는
        # 두 calibration의 숫자 비교를 둔다.
        comparison_row = QSplitter(Qt.Horizontal)

        scene_preview_group = QGroupBox("Scene Image Comparison")
        scene_preview_layout = QVBoxLayout(scene_preview_group)
        images_row = QHBoxLayout()

        self.original_calibration_group = QGroupBox("Original Calibration (all scenes)")
        original_layout = QVBoxLayout(self.original_calibration_group)
        self.original_preview_label = QLabel("Select a scene row to preview it.")
        self.original_preview_label.setAlignment(Qt.AlignCenter)
        self.original_preview_label.setMinimumSize(260, 200)
        self.original_preview_label.setProperty("surface", "image")
        original_layout.addWidget(self.original_preview_label)
        images_row.addWidget(self.original_calibration_group)

        self.best_subset_calibration_group = QGroupBox("Best Subset Calibration")
        subset_layout = QVBoxLayout(self.best_subset_calibration_group)
        self.best_subset_preview_label = QLabel(
            "Select scenes and run Re-Calibrate Selected Scenes."
        )
        self.best_subset_preview_label.setAlignment(Qt.AlignCenter)
        self.best_subset_preview_label.setWordWrap(True)
        self.best_subset_preview_label.setMinimumSize(260, 200)
        self.best_subset_preview_label.setProperty("surface", "image")
        subset_layout.addWidget(self.best_subset_preview_label)
        images_row.addWidget(self.best_subset_calibration_group)

        scene_preview_layout.addLayout(images_row)
        self.preview_status_label = QLabel("")
        self.preview_status_label.setWordWrap(True)
        scene_preview_layout.addWidget(self.preview_status_label)
        comparison_row.addWidget(scene_preview_group)

        comparison_group = QGroupBox("Original Calibration vs Best Subset Calibration")
        comparison_layout = QVBoxLayout(comparison_group)
        self.comparison_table = QTableWidget(0, 3)
        self.comparison_table.setHorizontalHeaderLabels(
            ["Metric", "Original Calibration", "Best Subset Calibration"]
        )
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.comparison_table.setEditTriggers(QTableWidget.NoEditTriggers)
        comparison_layout.addWidget(self.comparison_table)
        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        comparison_layout.addWidget(self.warning_label)
        comparison_row.addWidget(comparison_group)
        comparison_row.setStretchFactor(0, 1)
        comparison_row.setStretchFactor(1, 1)
        layout.addWidget(comparison_row, stretch=2)

    def set_context(
        self,
        dataset: Dataset | None,
        camera_config: CameraConfig | None,
        calibration_results: dict[CameraModelType, CalibrationResult],
        analysis: SceneQualityAnalysis | None,
        subset_result: SubsetCalibrationResult | None,
    ) -> None:
        self._dataset = dataset
        self._camera_config = camera_config
        current = analysis.model_name if analysis else self.model_combo.currentData()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        added_models = set()
        for key, result in calibration_results.items():
            if result.success:
                model = result.model_name or key
                try:
                    model = model if isinstance(model, CameraModelType) else CameraModelType(str(model))
                except (TypeError, ValueError):
                    continue
                if model not in added_models:
                    self.model_combo.addItem(_model_label(model), userData=model)
                    added_models.add(model)
        if self.model_combo.count() == 0:
            self.model_combo.addItem("No successful Initial Calibration", userData=None)
            self.model_combo.setEnabled(False)
        else:
            self.model_combo.setEnabled(True)
        index = self.model_combo.findData(current)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)
        self._analysis = analysis
        active_model = analysis.model_name if analysis else self.model_combo.currentData()
        self._original_calibration_result = next(
            (
                result for key, result in calibration_results.items()
                if result.success and (
                    result.model_name == active_model
                    or key == active_model
                    or str(key) == getattr(active_model, "value", None)
                )
            ),
            None,
        )
        self._best_subset_result = (
            subset_result
            if subset_result is not None and subset_result.model_name == active_model
            else None
        )
        self._populate_table()
        if self._best_subset_result:
            self._set_checked_ids(set(self._best_subset_result.selected_frame_ids))
        original_scene_count = (
            len(self._original_calibration_result.per_frame_error)
            if self._original_calibration_result else 0
        )
        self.original_calibration_group.setTitle(
            f"Original Calibration ({original_scene_count} scenes)"
        )
        subset_scene_count = (
            len(self._best_subset_result.selected_frame_ids)
            if self._best_subset_result else 0
        )
        self.best_subset_calibration_group.setTitle(
            "Best Subset Calibration"
            + (f" ({subset_scene_count} scenes)" if subset_scene_count else "")
        )
        self._set_comparison(self._original_calibration_result, self._best_subset_result)
        self.export_subset_button.setEnabled(bool(
            self._best_subset_result
            and self._best_subset_result.calibration_result
            and self._best_subset_result.calibration_result.success
        ))

    def _populate_table(self) -> None:
        self.table.setSortingEnabled(False)
        scenes = self._analysis.scenes if self._analysis else []
        self.table.setRowCount(len(scenes))
        for row, scene in enumerate(scenes):
            rank = NumericItem(str(scene.rank), scene.rank)
            scene_item = QTableWidgetItem(scene.frame_id)
            scene_item.setData(Qt.UserRole, scene.frame_id)
            values = [
                rank, scene_item,
                NumericItem(f"{scene.quality_score:.1f}", scene.quality_score),
                NumericItem(f"{scene.reprojection_error:.3f}" if scene.reprojection_error is not None else "N/A", scene.reprojection_error or 1e12),
                NumericItem(f"{scene.detection_ratio * 100:.1f}%", scene.detection_ratio),
                NumericItem(f"{scene.sharpness:.0f}" if scene.sharpness is not None else "N/A", scene.sharpness or -1),
            ]
            for col, item in enumerate(values):
                self.table.setItem(row, col, item)
            check = QTableWidgetItem("")
            check.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            check.setCheckState(Qt.Unchecked)
            check.setData(Qt.UserRole, scene.frame_id)
            self.table.setItem(row, 6, check)
        self.table.setSortingEnabled(True)
        self.table.sortItems(2, Qt.DescendingOrder)
        self.recalibrate_button.setEnabled(bool(scenes) and self.model_combo.currentData() is not None)
        if scenes:
            self.selection_label.setText(
                f"{len(scenes)} detected scenes ranked. Choose a recommendation or edit checkboxes."
            )
        elif self.model_combo.currentData() is None:
            self.selection_label.setText(
                "No successful Initial Calibration result is available. Check Model Comparison errors."
            )
        else:
            self.selection_label.setText(
                "The selected model has no per-view reprojection results to rank."
            )
        if scenes:
            self.table.selectRow(0)
            self._update_preview_for_row(0)
        else:
            self._original_calibration_pixmap = None
            self._best_subset_calibration_pixmap = None
            self.original_preview_label.clear()
            self.original_preview_label.setText("No ranked scene is available to preview.")
            self.best_subset_preview_label.clear()
            self.best_subset_preview_label.setText("-")
            self.preview_status_label.setText("")

    def _emit_model_changed(self) -> None:
        model = self.model_combo.currentData()
        if model is not None:
            self.model_changed.emit(model)

    def _selected_ids(self) -> list[str]:
        return [
            self.table.item(row, 6).data(Qt.UserRole)
            for row in range(self.table.rowCount())
            if self.table.item(row, 6).checkState() == Qt.Checked
        ]

    def _set_checked_ids(self, ids: set[str]) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 6)
            item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in ids else Qt.Unchecked)
        self.selection_label.setText(f"Selected {len(ids)} scenes. You may adjust the checkboxes manually.")

    def _set_all_checked(self, checked: bool) -> None:
        ids = {
            self.table.item(row, 6).data(Qt.UserRole)
            for row in range(self.table.rowCount())
        } if checked else set()
        self._set_checked_ids(ids)

    def _recommend(self, count: int) -> None:
        if not self._dataset or not self._analysis or not self._camera_config:
            return
        ids = recommend_best_subset(self._dataset, self._analysis, self._camera_config, count)
        self._set_checked_ids(set(ids))
        self.selection_label.setText(
            f"Recommended {len(ids)} scenes using quality + pose diversity + image coverage."
        )

    def _request_recalibration(self) -> None:
        ids = self._selected_ids()
        if len(ids) < 3:
            self.selection_label.setText("Select at least 3 detected scenes before re-calibration.")
            return
        model = self.model_combo.currentData()
        if model is None and self._analysis is not None:
            model = self._analysis.model_name
        if not isinstance(model, CameraModelType):
            try:
                model = CameraModelType(str(model))
            except (TypeError, ValueError):
                self.selection_label.setText(
                    "No valid calibration model is selected. Re-run Initial Calibration first."
                )
                return
        self.recalibrate_requested.emit(ids, model)

    def _update_preview(self) -> None:
        if not self._dataset or not self.table.selectedItems():
            return
        self._update_preview_for_row(self.table.currentRow())

    @staticmethod
    def _read_image(path: str):
        """cv2.imread의 Unicode path 제약을 피하기 위한 바이트 기반 로더."""
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
        except OSError:
            return None
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None

    def _update_preview_for_row(self, row: int) -> None:
        if not self._dataset or row < 0 or row >= self.table.rowCount():
            return
        frame_id = self.table.item(row, 1).data(Qt.UserRole)
        frame = next((f for f in self._dataset.frames if f.image_info.image_id == frame_id), None)
        if frame is None:
            self.original_preview_label.setText(f"Scene not found in Dataset: {frame_id}")
            self.best_subset_preview_label.setText("-")
            return
        image = self._read_image(frame.image_info.path)
        if image is None:
            self._original_calibration_pixmap = None
            self._best_subset_calibration_pixmap = None
            self.original_preview_label.clear()
            self.original_preview_label.setText(f"Cannot read image: {frame.image_info.path}")
            self.best_subset_preview_label.clear()
            self.best_subset_preview_label.setText("-")
            self.preview_status_label.setText("The source image may have been moved or deleted.")
            return

        self._original_calibration_pixmap = self._undistorted_pixmap(
            image, self._original_calibration_result
        )
        subset_calibration = (
            self._best_subset_result.calibration_result
            if self._best_subset_result else None
        )
        self._best_subset_calibration_pixmap = self._undistorted_pixmap(
            image, subset_calibration
        )

        if self._original_calibration_pixmap is None:
            self.original_preview_label.clear()
            self.original_preview_label.setText("Original Calibration result is unavailable.")
        if self._best_subset_calibration_pixmap is None:
            self.best_subset_preview_label.clear()
            self.best_subset_preview_label.setText(
                "Select scenes and run Re-Calibrate Selected Scenes."
            )
        self._render_preview_pixmaps()
        subset_note = (
            f"Best Subset: {len(self._best_subset_result.selected_frame_ids)} scenes"
            if self._best_subset_result else "Best Subset: not calculated"
        )
        self.preview_status_label.setText(
            f"{frame_id}  |  {Path(frame.image_info.path).name}  |  "
            f"Same source image undistorted with each calibration  |  {subset_note}"
        )

    def _undistorted_pixmap(
        self, image: np.ndarray, result: CalibrationResult | None,
    ) -> QPixmap | None:
        if self._camera_config is None or result is None or not result.success:
            return None
        try:
            corrected = undistort_image(image, result, self._camera_config)
        except (ValueError, cv2.error):
            return None
        return self._image_to_pixmap(corrected)

    @staticmethod
    def _image_to_pixmap(image) -> QPixmap:
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        height, width, channels = rgb.shape
        qimage = QImage(
            rgb.data, width, height, channels * width, QImage.Format_RGB888
        ).copy()
        return QPixmap.fromImage(qimage)

    def _render_preview_pixmaps(self) -> None:
        pairs = (
            (self._original_calibration_pixmap, self.original_preview_label),
            (self._best_subset_calibration_pixmap, self.best_subset_preview_label),
        )
        for source, label in pairs:
            if source is None:
                continue
            target = label.contentsRect().size()
            if target.width() <= 1 or target.height() <= 1:
                continue
            label.setPixmap(source.scaled(
                target, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_preview_pixmaps()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._render_preview_pixmaps()

    def _set_comparison(self, original, subset: SubsetCalibrationResult | None) -> None:
        if not subset or not subset.calibration_result:
            self.comparison_table.setRowCount(0)
            self.warning_label.setText("")
            return
        result = subset.calibration_result
        def intrinsic(cal, row, col):
            return cal.camera_matrix[row, col] if cal and cal.camera_matrix is not None else None
        def fmt(value):
            return f"{value:.4f}" if value is not None else "N/A"
        original_p95 = original.residual_stats.p95 if original and original.residual_stats else None
        subset_p95 = result.residual_stats.p95 if result.residual_stats else None
        original_stability = (
            (original.param_uncertainty_bootstrap or original.param_uncertainty).overall_stability
            if original and (original.param_uncertainty_bootstrap or original.param_uncertainty) else None
        )
        subset_stability = (
            (result.param_uncertainty_bootstrap or result.param_uncertainty).overall_stability
            if result.param_uncertainty_bootstrap or result.param_uncertainty else None
        )
        original_dist = original.distortion.reshape(-1) if original and original.distortion is not None else []
        subset_dist = result.distortion.reshape(-1) if result.distortion is not None else []
        rows = [
            ("Scenes", len(original.per_frame_error) if original else 0, len(subset.selected_frame_ids)),
            ("RMS (px)", fmt(original.rms_error if original else None), fmt(result.rms_error)),
            ("P95 (px)", fmt(original_p95), fmt(subset_p95)),
            ("Hold-out RMSE (px)", fmt(subset.original_validation_result.test_rms if subset.original_validation_result else None), fmt(subset.validation_result.test_rms if subset.validation_result else None)),
            ("Stability", f"{original_stability:.0f}%" if original_stability is not None else "N/A", f"{subset_stability:.0f}%" if subset_stability is not None else "N/A"),
            ("fx", fmt(intrinsic(original, 0, 0)), fmt(intrinsic(result, 0, 0))),
            ("fy", fmt(intrinsic(original, 1, 1)), fmt(intrinsic(result, 1, 1))),
            ("cx", fmt(intrinsic(original, 0, 2)), fmt(intrinsic(result, 0, 2))),
            ("cy", fmt(intrinsic(original, 1, 2)), fmt(intrinsic(result, 1, 2))),
            ("Distortion", ", ".join(f"{v:.4g}" for v in original_dist), ", ".join(f"{v:.4g}" for v in subset_dist)),
            ("Coverage", f"{subset.original_coverage_percentage:.1f}%", f"{subset.coverage_percentage:.1f}%"),
            ("Pose Diversity", f"{subset.original_diversity.overall:.2f}" if subset.original_diversity else "N/A", f"{subset.diversity.overall:.2f}" if subset.diversity else "N/A"),
        ]
        self.comparison_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for col, value in enumerate(values):
                self.comparison_table.setItem(row, col, QTableWidgetItem(str(value)))
        self.warning_label.setText("\n".join(f"⚠ {warning}" for warning in subset.warnings))
