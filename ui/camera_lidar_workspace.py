"""
Camera-LiDAR FAST-Calib workspace.

Bag input (topic discovery + selection + timeline preview) and file input
both converge on the same CalibrationScene; AUTO ROI (LiDAR-only
multi-plane search) or MANUAL ROI (a caller-set box) can be picked per
capture; captured scenes are tracked in a Scene Manager; the FINAL result
is always the Multi-Scene joint solve over the included scenes -- a
single-scene result is shown too, but always labeled PROVISIONAL, never
treated as the final answer.

Live (ROS1/ROS2) topic discovery/subscription is a deferred follow-up --
see the plan doc. It will reuse this same Topic Selector/AUTO-ROI/Scene-
Manager/Multi-Scene code behind a different acquisition adapter.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.calibration_io import StandardCalibration, load_standard_calibration
from camera_lidar.camera_detector import COMMON_ARUCO_DICTIONARIES, detect_camera_target
from camera_lidar.gates import compute_target_pose, evaluate_duplicate_gate, evaluate_quality_gate
from camera_lidar.pipeline import calibrate_single_scene
from camera_lidar.target_config import CORNER_ORDER, TargetConfig, load_target_config, save_target_config
from camera_lidar.types import (
    CalibrationScene,
    CapturedScene,
    ImageFrame,
    PointCloudFrame,
    ROIConfig,
    SceneType,
)
from geometry.transform import rotation_matrix_to_quaternion, rotation_matrix_to_rpy
from input.lidar import read_pcd, read_ply_ascii
from ui.camera_lidar_bag_source import CameraLidarBagSource
from ui.camera_lidar_scene_browser import CameraLidarSceneBrowser
from ui.theme import Theme, qcolor, set_tone
from ui.worker import (
    CameraLidarCalibrationWorker,
    MultiSceneCalibrationWorker,
    PolicyComparisonWorker,
    SceneExtractionWorker,
    run_worker_in_thread,
)

_SCENE_TYPE_LABEL = {
    SceneType.VALID_FULL: "FULL",
    SceneType.VALID_PARTIAL: "PARTIAL",
    SceneType.INVALID: "INVALID",
}
_CORNER_ABBREV = {
    "top_left": "TL", "top_right": "TR", "bottom_right": "BR", "bottom_left": "BL",
}

# Sync Δt display thresholds -- purely informational (bag/file capture is a
# single fixed timestamp pair, not a live stream), matching the sync-health
# "GOOD/WARNING" convention used elsewhere in this app.
_SYNC_GOOD_MS = 50.0
_SYNC_WARNING_MS = 200.0


def _matrix_text(M: np.ndarray) -> str:
    return "\n".join("  ".join(f"{v: .6f}" for v in row) for row in M)


def _quality_percent(result) -> float:
    """Simple, explicit heuristic (not a rigorous metric): 100% at 0mm
    residual, linearly down to 0% at 10mm+. Only meaningful when
    result.success -- callers should treat a failed detection as 0%
    separately rather than relying on this function for that case."""
    if result is None or not result.success or result.residual_rmse_m is None:
        return 0.0
    return float(max(0.0, min(100.0, 100.0 - result.residual_rmse_m * 1000.0 * 10.0)))


class CameraLidarWorkspace(QWidget):
    back_requested = Signal()
    calibrate_intrinsic_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.intrinsics: StandardCalibration | None = None

        # 현재 "capture 준비된" scene의 원재료 -- Bag 또는 File 어느 쪽에서
        # 왔든 여기로 모인다 (Adapter -> Common Data Model 원칙).
        self.image: np.ndarray | None = None
        self.cloud_points: np.ndarray | None = None
        self.image_timestamp: float = 0.0
        self.cloud_timestamp: float = 0.0
        self.camera_topic: str = ""
        self.lidar_topic: str = ""
        self.source_label: str = "(none)"

        self.captured_scenes: list[CapturedScene] = []
        self._scene_counter = 0

        self._thread = None
        self._worker = None
        self._multi_thread = None
        self._multi_worker = None
        self._extraction_thread = None
        self._extraction_worker = None
        self._pending_candidate_queue: list = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)

        top = QHBoxLayout()
        back = QPushButton("← Calibration Home")
        back.setMaximumHeight(28)
        back.clicked.connect(self.back_requested.emit)
        top.addWidget(back)
        title = QLabel("FAST-Calib — Camera ↔ LiDAR")
        title.setProperty("role", "sectionTitle")
        top.addWidget(title, stretch=1)
        root.addLayout(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setSpacing(8)
        scroll.setWidget(body)
        root.addWidget(scroll, stretch=1)

        body_layout.addWidget(self._build_intrinsic_group())
        body_layout.addWidget(self._build_target_group())
        body_layout.addWidget(self._build_roi_group())
        body_layout.addWidget(self._build_input_group())
        body_layout.addWidget(self._build_scene_browser_group())
        body_layout.addWidget(self._build_capture_group())
        body_layout.addWidget(self._build_scene_manager_group())
        body_layout.addWidget(self._build_multi_scene_group())
        body_layout.addStretch(1)

        self._update_intrinsic_status()
        self._update_capture_enabled()
        self._update_multi_scene_enabled()

    # ------------------------------------------------------------------
    # ① Camera Intrinsic
    # ------------------------------------------------------------------

    def set_previous_intrinsic(self, calibration: StandardCalibration) -> None:
        self.intrinsics = calibration
        self._update_intrinsic_status()
        self._update_capture_enabled()

    def _build_intrinsic_group(self) -> QGroupBox:
        group = QGroupBox("① Camera Intrinsic")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        load_button = QPushButton("Load YAML / JSON...")
        load_button.clicked.connect(self._on_load_intrinsic_file)
        row.addWidget(load_button)
        use_previous_button = QPushButton("Use Previous Intrinsic Result")
        use_previous_button.clicked.connect(lambda: self.calibrate_intrinsic_requested.emit("camera_lidar"))
        row.addWidget(use_previous_button)
        layout.addLayout(row)

        form = QFormLayout()
        self.intrinsic_model_label = QLabel("-")
        self.intrinsic_resolution_label = QLabel("-")
        self.intrinsic_fx_label = QLabel("-")
        self.intrinsic_fy_label = QLabel("-")
        self.intrinsic_cx_label = QLabel("-")
        self.intrinsic_cy_label = QLabel("-")
        self.intrinsic_distortion_label = QLabel("-")
        self.intrinsic_distortion_label.setWordWrap(True)
        self.intrinsic_status_label = QLabel("NOT READY")
        form.addRow("Model", self.intrinsic_model_label)
        form.addRow("Resolution", self.intrinsic_resolution_label)
        form.addRow("fx", self.intrinsic_fx_label)
        form.addRow("fy", self.intrinsic_fy_label)
        form.addRow("cx", self.intrinsic_cx_label)
        form.addRow("cy", self.intrinsic_cy_label)
        form.addRow("Distortion", self.intrinsic_distortion_label)
        form.addRow("STATUS", self.intrinsic_status_label)
        layout.addLayout(form)
        return group

    def _on_load_intrinsic_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Camera intrinsic 파일 선택", "",
            "Calibration files (*.yaml *.yml *.json);;All files (*)",
        )
        if not path:
            return
        try:
            calibration = load_standard_calibration(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "불러오기 실패", f"Intrinsic 파일을 읽는 중 오류:\n{e}")
            return
        self.intrinsics = calibration
        self._update_intrinsic_status()
        self._update_capture_enabled()

    def _update_intrinsic_status(self) -> None:
        if self.intrinsics is None:
            self.intrinsic_model_label.setText("-")
            self.intrinsic_resolution_label.setText("-")
            self.intrinsic_fx_label.setText("-")
            self.intrinsic_fy_label.setText("-")
            self.intrinsic_cx_label.setText("-")
            self.intrinsic_cy_label.setText("-")
            self.intrinsic_distortion_label.setText("-")
            self.intrinsic_status_label.setText("NOT READY")
            set_tone(self.intrinsic_status_label, "bad")
            return
        cal = self.intrinsics
        self.intrinsic_model_label.setText(cal.model_name.value if cal.model_name else (cal.distortion_model or "-"))
        if cal.width and cal.height:
            self.intrinsic_resolution_label.setText(f"{cal.width} × {cal.height}")
        else:
            self.intrinsic_resolution_label.setText("-")
        self.intrinsic_fx_label.setText(f"{cal.fx:.3f}")
        self.intrinsic_fy_label.setText(f"{cal.fy:.3f}")
        self.intrinsic_cx_label.setText(f"{cal.cx:.3f}")
        self.intrinsic_cy_label.setText(f"{cal.cy:.3f}")
        self.intrinsic_distortion_label.setText(
            ", ".join(f"{v:.5f}" for v in cal.distortion.reshape(-1))
        )
        self.intrinsic_status_label.setText("READY")
        set_tone(self.intrinsic_status_label, "good")

    # ------------------------------------------------------------------
    # ② Target Geometry
    # ------------------------------------------------------------------

    def _build_target_group(self) -> QGroupBox:
        group = QGroupBox("② Target Geometry (FAST-Calib circular-hole board)")
        outer = QVBoxLayout(group)

        row = QHBoxLayout()
        load_button = QPushButton("Load Target Config...")
        load_button.clicked.connect(self._on_load_target_config)
        row.addWidget(load_button)
        save_button = QPushButton("Save Target Config...")
        save_button.clicked.connect(self._on_save_target_config)
        row.addWidget(save_button)
        outer.addLayout(row)

        form = QFormLayout()
        default = TargetConfig()

        def make_spin(value_m: float) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setRange(0.001, 5000.0)
            spin.setDecimals(3)
            spin.setSuffix(" mm")
            spin.setValue(value_m * 1000.0)
            return spin

        self.marker_size_spin = make_spin(default.marker_size)
        self.delta_width_qr_spin = make_spin(default.delta_width_qr_center)
        self.delta_height_qr_spin = make_spin(default.delta_height_qr_center)
        self.delta_width_circles_spin = make_spin(default.delta_width_circles)
        self.delta_height_circles_spin = make_spin(default.delta_height_circles)
        self.circle_radius_spin = make_spin(default.circle_radius)

        # Editable (not just a fixed dropdown) so a loaded Target Config's
        # aruco_dictionary always round-trips even if it's not one of the
        # curated COMMON_ARUCO_DICTIONARIES entries. Populated from the same
        # list camera_detector.diagnose_dictionaries uses ("TEST
        # DICTIONARIES" in the Scene Browser can APPLY BEST CANDIDATE here).
        self.target_dictionary_combo = QComboBox()
        self.target_dictionary_combo.setEditable(True)
        self.target_dictionary_combo.addItems(COMMON_ARUCO_DICTIONARIES)
        self.target_dictionary_combo.setCurrentText(default.aruco_dictionary)

        form.addRow("Marker size", self.marker_size_spin)
        form.addRow("Marker center Δwidth", self.delta_width_qr_spin)
        form.addRow("Marker center Δheight", self.delta_height_qr_spin)
        form.addRow("Circle center Δwidth", self.delta_width_circles_spin)
        form.addRow("Circle center Δheight", self.delta_height_circles_spin)
        form.addRow("Circle radius", self.circle_radius_spin)
        form.addRow("ArUco Dictionary", self.target_dictionary_combo)
        outer.addLayout(form)
        return group

    def _current_target_config(self) -> TargetConfig:
        dictionary = self.target_dictionary_combo.currentText().strip() or TargetConfig().aruco_dictionary
        return TargetConfig(
            marker_size=self.marker_size_spin.value() / 1000.0,
            delta_width_qr_center=self.delta_width_qr_spin.value() / 1000.0,
            delta_height_qr_center=self.delta_height_qr_spin.value() / 1000.0,
            delta_width_circles=self.delta_width_circles_spin.value() / 1000.0,
            delta_height_circles=self.delta_height_circles_spin.value() / 1000.0,
            circle_radius=self.circle_radius_spin.value() / 1000.0,
            aruco_dictionary=dictionary,
        )

    def _apply_target_config(self, target: TargetConfig) -> None:
        self.target_dictionary_combo.setCurrentText(target.aruco_dictionary)
        self.marker_size_spin.setValue(target.marker_size * 1000.0)
        self.delta_width_qr_spin.setValue(target.delta_width_qr_center * 1000.0)
        self.delta_height_qr_spin.setValue(target.delta_height_qr_center * 1000.0)
        self.delta_width_circles_spin.setValue(target.delta_width_circles * 1000.0)
        self.delta_height_circles_spin.setValue(target.delta_height_circles * 1000.0)
        self.circle_radius_spin.setValue(target.circle_radius * 1000.0)

    def _on_load_target_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Target config YAML 선택", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            self._apply_target_config(load_target_config(path))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "불러오기 실패", f"Target config를 읽는 중 오류:\n{e}")

    def _on_save_target_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Target config 저장", "target_config.yaml", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            save_target_config(self._current_target_config(), path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "저장 실패", f"Target config를 저장하는 중 오류:\n{e}")

    # ------------------------------------------------------------------
    # ③ ROI (AUTO default / MANUAL)
    # ------------------------------------------------------------------

    def _build_roi_group(self) -> QGroupBox:
        group = QGroupBox("③ LiDAR ROI")
        outer = QVBoxLayout(group)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("ROI Mode"))
        self.roi_mode_combo = QComboBox()
        self.roi_mode_combo.addItem("AUTO (LiDAR-only multi-plane search, no box needed)", "auto")
        self.roi_mode_combo.addItem("MANUAL (box below)", "manual")
        self.roi_mode_combo.currentIndexChanged.connect(self._on_roi_mode_changed)
        mode_row.addWidget(self.roi_mode_combo, stretch=1)
        outer.addLayout(mode_row)

        self.manual_roi_widget = QWidget()
        form = QFormLayout(self.manual_roi_widget)

        def make_spin(value: float) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setDecimals(2)
            spin.setSuffix(" m")
            spin.setValue(value)
            return spin

        default = ROIConfig()
        self.roi_x_min_spin = make_spin(default.x_min)
        self.roi_x_max_spin = make_spin(default.x_max)
        self.roi_y_min_spin = make_spin(default.y_min)
        self.roi_y_max_spin = make_spin(default.y_max)
        self.roi_z_min_spin = make_spin(default.z_min)
        self.roi_z_max_spin = make_spin(default.z_max)
        form.addRow("x_min", self.roi_x_min_spin)
        form.addRow("x_max", self.roi_x_max_spin)
        form.addRow("y_min", self.roi_y_min_spin)
        form.addRow("y_max", self.roi_y_max_spin)
        form.addRow("z_min", self.roi_z_min_spin)
        form.addRow("z_max", self.roi_z_max_spin)
        reset_button = QPushButton("RESET ROI")
        reset_button.clicked.connect(lambda: self._apply_roi(ROIConfig()))
        form.addRow(reset_button)
        outer.addWidget(self.manual_roi_widget)
        self.manual_roi_widget.setVisible(False)  # AUTO is the default mode
        return group

    def _on_roi_mode_changed(self) -> None:
        self.manual_roi_widget.setVisible(self.roi_mode_combo.currentData() == "manual")

    def _current_roi_mode(self) -> str:
        return self.roi_mode_combo.currentData()

    def _apply_roi(self, roi: ROIConfig) -> None:
        self.roi_x_min_spin.setValue(roi.x_min)
        self.roi_x_max_spin.setValue(roi.x_max)
        self.roi_y_min_spin.setValue(roi.y_min)
        self.roi_y_max_spin.setValue(roi.y_max)
        self.roi_z_min_spin.setValue(roi.z_min)
        self.roi_z_max_spin.setValue(roi.z_max)

    def _current_roi(self) -> ROIConfig:
        return ROIConfig(
            x_min=self.roi_x_min_spin.value(), x_max=self.roi_x_max_spin.value(),
            y_min=self.roi_y_min_spin.value(), y_max=self.roi_y_max_spin.value(),
            z_min=self.roi_z_min_spin.value(), z_max=self.roi_z_max_spin.value(),
        )

    # ------------------------------------------------------------------
    # ④ Input Source (Bag tab / File tab)
    # ------------------------------------------------------------------

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("④ Input Source")
        outer = QVBoxLayout(group)
        tabs = QTabWidget()

        self.bag_source = CameraLidarBagSource()
        self.bag_source.scene_loaded.connect(self._on_bag_scene_loaded)
        tabs.addTab(self.bag_source, "Bag")

        tabs.addTab(self._build_file_source_tab(), "File")
        outer.addWidget(tabs)
        return group

    def _build_file_source_tab(self) -> QWidget:
        widget = QWidget()
        outer = QVBoxLayout(widget)

        row = QHBoxLayout()
        load_image_button = QPushButton("Load Image...")
        load_image_button.clicked.connect(self._on_load_image_file)
        row.addWidget(load_image_button)
        load_cloud_button = QPushButton("Load Point Cloud (.pcd/.ply)...")
        load_cloud_button.clicked.connect(self._on_load_cloud_file)
        row.addWidget(load_cloud_button)
        outer.addLayout(row)

        preview_row = QHBoxLayout()
        self.file_image_preview_label = QLabel("(no image loaded)")
        self.file_image_preview_label.setProperty("surface", "image")
        self.file_image_preview_label.setAlignment(Qt.AlignCenter)
        self.file_image_preview_label.setMinimumHeight(160)
        preview_row.addWidget(self.file_image_preview_label, stretch=1)
        outer.addLayout(preview_row)

        self.file_image_status_label = QLabel("Image: (none)")
        self.file_cloud_status_label = QLabel("Point cloud: (none)")
        outer.addWidget(self.file_image_status_label)
        outer.addWidget(self.file_cloud_status_label)
        return widget

    def _on_load_image_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Scene image 선택", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            QMessageBox.critical(self, "불러오기 실패", f"이미지를 읽을 수 없습니다:\n{path}")
            return
        self.image = image
        self.image_timestamp = time.time()
        self.camera_topic = ""
        self.file_image_status_label.setText(
            f"Image: {os.path.basename(path)} ({image.shape[1]}×{image.shape[0]})"
        )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaledToHeight(160, Qt.SmoothTransformation)
        self.file_image_preview_label.setPixmap(pixmap)
        self._update_source_label()
        self._update_capture_enabled()

    def _on_load_cloud_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Scene point cloud 선택", "", "Point clouds (*.pcd *.ply)"
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pcd":
                points = read_pcd(path)
            elif ext == ".ply":
                points = read_ply_ascii(path)
            else:
                raise ValueError(f"Unsupported point cloud extension: {ext!r}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "불러오기 실패", f"Point cloud를 읽는 중 오류:\n{e}")
            return
        self.cloud_points = points
        self.cloud_timestamp = time.time()
        self.lidar_topic = ""
        self.file_cloud_status_label.setText(
            f"Point cloud: {os.path.basename(path)} ({points.shape[0]} points)"
        )
        self._update_source_label()
        self._update_capture_enabled()

    def _on_bag_scene_loaded(
        self, image_frame: ImageFrame, cloud_frame: PointCloudFrame, camera_topic: str, lidar_topic: str
    ) -> None:
        self.image = image_frame.image
        self.image_timestamp = image_frame.timestamp
        self.cloud_points = cloud_frame.points if cloud_frame.intensity is None else np.column_stack(
            [cloud_frame.points, cloud_frame.intensity]
        )
        self.cloud_timestamp = cloud_frame.timestamp
        self.camera_topic = camera_topic
        self.lidar_topic = lidar_topic
        self._update_source_label()
        self._update_capture_enabled()

    def _update_source_label(self) -> None:
        parts = []
        if self.image is not None:
            src = self.camera_topic or "file"
            parts.append(f"image {self.image.shape[1]}×{self.image.shape[0]} ({src})")
        if self.cloud_points is not None:
            src = self.lidar_topic or "file"
            parts.append(f"cloud {self.cloud_points.shape[0]} pts ({src})")
        self.source_label = " + ".join(parts) if parts else "(none)"
        self.capture_ready_label.setText(f"Loaded: {self.source_label}")

    # ------------------------------------------------------------------
    # ⑤ Scene Browser (MARKER EXTRACTION — auto candidate discovery, Bag pathway)
    # ------------------------------------------------------------------

    def _build_scene_browser_group(self) -> QGroupBox:
        self.scene_browser = CameraLidarSceneBrowser()
        self.scene_browser.extraction_requested.connect(self._on_marker_extraction_requested)
        self.scene_browser.add_selected_requested.connect(self._on_add_selected_candidates)
        self.scene_browser.test_current_frame_requested.connect(self._on_test_current_frame_requested)
        self.scene_browser.apply_dictionary_requested.connect(self._on_apply_dictionary_requested)
        self.scene_browser.cancel_extraction_requested.connect(self._on_cancel_marker_extraction)
        return self.scene_browser

    def _on_cancel_marker_extraction(self) -> None:
        """Both self.scene_browser and self live in the GUI thread, so this
        slot itself runs synchronously on click -- but self._extraction_worker
        lives on a separate QThread, so request_cancel() (like
        BagExtractionWorker.request_cancel) is a PLAIN direct method call
        that just flips one boolean flag, never a Qt signal/slot connection.
        A signal connected straight to a slot living on the worker's own
        thread would be auto-queued by Qt and never get delivered while
        that thread is busy inside run()'s scan loop -- which would make
        CANCEL silently do nothing until the scan finished on its own,
        defeating the entire point (this was tried and empirically confirmed
        broken before landing on this direct-call approach)."""
        if self._extraction_worker is not None:
            self._extraction_worker.request_cancel()

    def _on_apply_dictionary_requested(self, dictionary_name: str) -> None:
        """User clicked APPLY BEST CANDIDATE in the TEST DICTIONARIES dialog
        -- an explicit action, not an automatic config change (spec's "no
        auto-config-change without confirmation" is satisfied by this being
        one deliberate click)."""
        self.target_dictionary_combo.setCurrentText(dictionary_name)

    def _on_marker_extraction_requested(self) -> None:
        bag_path = self.bag_source.bag_path
        camera_topic = self.bag_source.camera_topic_combo.currentData()
        lidar_topic = self.bag_source.lidar_topic_combo.currentData()
        if not bag_path or not camera_topic or not lidar_topic:
            QMessageBox.warning(
                self, "Bag / Topic 필요",
                "MARKER EXTRACTION을 실행하려면 먼저 Input Source의 Bag 탭에서 bag을 열고 "
                "Camera/LiDAR 토픽을 선택하세요.",
            )
            return
        if self.intrinsics is None:
            QMessageBox.warning(self, "Intrinsic 필요", "MARKER EXTRACTION 전에 Camera Intrinsic을 먼저 불러오세요.")
            return

        target = self._current_target_config()
        worker = SceneExtractionWorker(bag_path, camera_topic, lidar_topic, self.intrinsics, target)
        thread = run_worker_in_thread(worker, self)
        worker.progress.connect(self.scene_browser.set_progress_text)
        worker.progress_value.connect(self.scene_browser.set_progress_value)
        worker.candidates_ready.connect(self._on_scene_candidates_ready)
        worker.summary_ready.connect(self.scene_browser.set_diagnostics_summary)
        worker.error.connect(self._on_error)
        self._extraction_thread, self._extraction_worker = thread, worker
        self.scene_browser.set_extracting(True)
        thread.finished.connect(lambda: self.scene_browser.set_extracting(False))
        thread.start()

    def _on_scene_candidates_ready(self, candidates: list) -> None:
        self.scene_browser.set_candidates(candidates)

    def _on_test_current_frame_requested(self) -> None:
        if self.image is None:
            QMessageBox.warning(self, "이미지 필요", "TEST CURRENT FRAME을 실행하려면 먼저 Input Source에서 이미지를 불러오세요.")
            return
        if self.intrinsics is None:
            QMessageBox.warning(self, "Intrinsic 필요", "TEST CURRENT FRAME 전에 Camera Intrinsic을 먼저 불러오세요.")
            return

        target = self._current_target_config()
        result = detect_camera_target(self.image, self.intrinsics, target)
        expected_ids = sorted(target.marker_ids.values())
        self.scene_browser.show_detector_test_result(self.image, result, expected_ids)

    def _on_add_selected_candidates(self, candidates: list) -> None:
        usable = [c for c in candidates if c.cloud_points is not None]
        skipped = len(candidates) - len(usable)
        if skipped:
            QMessageBox.warning(
                self, "LiDAR 페어링 없음",
                f"{skipped}개 candidate는 근처에서 LiDAR 프레임을 찾지 못해 Scene Manager에 추가하지 못합니다.",
            )
        if not usable:
            return

        # Sensor-pair-changed confirmation once for the whole batch -- all
        # selected candidates share one extraction run's camera/lidar topic
        # pair, so asking per-candidate would pop a dialog for every scene.
        self.camera_topic, self.lidar_topic = usable[0].camera_topic, usable[0].lidar_topic
        if not self._confirm_sensor_pair_if_changed():
            return

        self._pending_candidate_queue = usable
        self.scene_browser.add_selected_button.setEnabled(False)
        self._process_next_candidate_in_queue()

    def _process_next_candidate_in_queue(self) -> None:
        if not self._pending_candidate_queue:
            self.scene_browser.set_candidates(self.scene_browser.candidates)  # refresh Selected/Add-button state
            return
        candidate = self._pending_candidate_queue.pop(0)
        candidate.is_selected = False  # committing it now -- don't leave it checked for an accidental re-add

        target = self._current_target_config()
        roi = self._current_roi()
        points = candidate.cloud_points
        cloud_timestamp = candidate.cloud_timestamp_s if candidate.cloud_timestamp_s is not None else candidate.representative_timestamp_s

        self.image = candidate.image
        self.cloud_points = points
        self.image_timestamp = candidate.representative_timestamp_s
        self.cloud_timestamp = cloud_timestamp
        self.camera_topic = candidate.camera_topic
        self.lidar_topic = candidate.lidar_topic

        image_frame = ImageFrame(timestamp=candidate.representative_timestamp_s, image=candidate.image, frame_id="camera")
        cloud_frame = PointCloudFrame(
            timestamp=cloud_timestamp,
            points=points[:, :3],
            frame_id="lidar",
            intensity=points[:, 3] if points.shape[1] > 3 else None,
        )
        scene = CalibrationScene(
            image=image_frame, cloud=cloud_frame, intrinsics=self.intrinsics, target=target, roi=roi,
        )

        worker = CameraLidarCalibrationWorker(scene, roi_mode=self._current_roi_mode())
        thread = run_worker_in_thread(worker, self)
        worker.progress.connect(self._on_progress)
        worker.result_ready.connect(self._on_capture_result_ready)
        worker.error.connect(self._on_error)
        thread.finished.connect(self._process_next_candidate_in_queue)
        self._thread, self._worker = thread, worker
        thread.start()

    # ------------------------------------------------------------------
    # ⑥ Scene Capture (diagnostics + provisional single-scene result)
    # ------------------------------------------------------------------

    def _build_capture_group(self) -> QGroupBox:
        group = QGroupBox("⑥ Scene Capture")
        outer = QVBoxLayout(group)

        self.capture_ready_label = QLabel("Loaded: (none)")
        outer.addWidget(self.capture_ready_label)

        self.capture_button = QPushButton("CAPTURE SCENE")
        self.capture_button.setProperty("role", "primary")
        self.capture_button.clicked.connect(self._on_capture_scene)
        outer.addWidget(self.capture_button)

        self.diagnostics_text = QPlainTextEdit()
        self.diagnostics_text.setReadOnly(True)
        self.diagnostics_text.setMinimumHeight(160)
        self.diagnostics_text.setPlainText("Capture a scene to see camera/LiDAR processing diagnostics here.")
        outer.addWidget(self.diagnostics_text)

        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(220)
        self.result_text.setPlainText("PROVISIONAL single-scene result will appear here after a successful capture.")
        outer.addWidget(QLabel("PROVISIONAL — SINGLE-SCENE (not the final multi-scene result)"))
        outer.addWidget(self.result_text)
        return group

    def _update_capture_enabled(self) -> None:
        ready = self.intrinsics is not None and self.image is not None and self.cloud_points is not None
        self.capture_button.setEnabled(ready)

    def _on_capture_scene(self) -> None:
        if self.intrinsics is None or self.image is None or self.cloud_points is None:
            return

        if not self._confirm_sensor_pair_if_changed():
            return

        target = self._current_target_config()
        roi = self._current_roi()
        image_frame = ImageFrame(
            timestamp=self.image_timestamp, image=self.image, frame_id="camera",
        )
        cloud_frame = PointCloudFrame(
            timestamp=self.cloud_timestamp,
            points=self.cloud_points[:, :3],
            frame_id="lidar",
            intensity=self.cloud_points[:, 3] if self.cloud_points.shape[1] > 3 else None,
        )
        scene = CalibrationScene(
            image=image_frame, cloud=cloud_frame, intrinsics=self.intrinsics, target=target, roi=roi,
        )

        worker = CameraLidarCalibrationWorker(scene, roi_mode=self._current_roi_mode())
        thread = run_worker_in_thread(worker, self)
        worker.progress.connect(self._on_progress)
        worker.result_ready.connect(self._on_capture_result_ready)
        worker.error.connect(self._on_error)
        self._thread, self._worker = thread, worker
        self.capture_button.setEnabled(False)
        thread.finished.connect(self._update_capture_enabled)
        thread.start()

    def _confirm_sensor_pair_if_changed(self) -> bool:
        """Requirement: don't silently mix scenes captured from different
        Camera/LiDAR topic pairs into one Multi-Scene calibration."""
        included = [c for c in self.captured_scenes if c.included]
        if not included:
            return True
        prior_pair = (included[0].camera_topic, included[0].lidar_topic)
        current_pair = (self.camera_topic, self.lidar_topic)
        if prior_pair == current_pair:
            return True
        reply = QMessageBox.warning(
            self, "SENSOR PAIR CHANGED",
            "Existing scenes were captured using a different Camera/LiDAR topic pair:\n"
            f"  Existing: camera={prior_pair[0] or '(file)'}, lidar={prior_pair[1] or '(file)'}\n"
            f"  Current:  camera={current_pair[0] or '(file)'}, lidar={current_pair[1] or '(file)'}\n\n"
            "Mixing different sensor pairs into one Multi-Scene calibration will produce a "
            "meaningless result. Continue anyway (not recommended), or Cancel and either match "
            "the previous topics or clear existing scenes first?",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        return reply == QMessageBox.Yes

    def _on_progress(self, message: str) -> None:
        pass  # progress text is short-lived; diagnostics/result panels are the durable record

    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "FAST-Calib 오류", message)

    def _on_capture_result_ready(self, result) -> None:
        gate_lines: list[str] = []
        included = result.success
        if result.success:
            quality = evaluate_quality_gate(result)
            if not quality.passed:
                included = False
                gate_lines.append(f"QUALITY GATE: FAIL -- {quality.reason}")
            else:
                gate_lines.append("QUALITY GATE: PASS")

            existing_poses = {
                c.scene_id: compute_target_pose(c.detection.lidar_centers)
                for c in self.captured_scenes
                if c.included and c.detection is not None and c.detection.lidar_centers is not None
            }
            candidate_pose = compute_target_pose(result.lidar_centers)
            duplicate = evaluate_duplicate_gate(candidate_pose, existing_poses)
            if not duplicate.passed:
                included = False
                gate_lines.append(f"DUPLICATE POSE GATE: FAIL -- {duplicate.reason}")
            else:
                gate_lines.append("DUPLICATE POSE GATE: PASS")

            gate_lines.append(
                "STABILITY GATE: N/A (requires a continuous stream -- activates with Live capture; "
                "not evaluated for Bag/File single-shot capture)"
            )
            if not included:
                gate_lines.append(
                    "\nScene added but NOT included by default -- review above and click "
                    "\"Include\" in the Scene Manager below to use it anyway."
                )

        self.diagnostics_text.setPlainText(self._format_diagnostics(result) + "\n\n" + "\n".join(gate_lines))
        self._render_single_scene_result(result)

        self._scene_counter += 1
        scene_id = f"scene_{self._scene_counter:02d}"
        target = self._current_target_config()
        roi = self._current_roi()
        image_frame = ImageFrame(timestamp=self.image_timestamp, image=self.image, frame_id="camera")
        cloud_frame = PointCloudFrame(
            timestamp=self.cloud_timestamp,
            points=self.cloud_points[:, :3],
            frame_id="lidar",
            intensity=self.cloud_points[:, 3] if self.cloud_points.shape[1] > 3 else None,
        )
        scene = CalibrationScene(
            image=image_frame, cloud=cloud_frame, intrinsics=self.intrinsics, target=target, roi=roi,
        )
        captured = CapturedScene(
            scene_id=scene_id, scene=scene,
            included=included,  # Quality/Duplicate gate failures (or detection failure) default to excluded,
                                 # but stay visible in the Scene Manager -- "Include" lets the user force it.
            camera_topic=self.camera_topic, lidar_topic=self.lidar_topic,
            roi_mode=self._current_roi_mode(), detection=result,
        )
        self.captured_scenes.append(captured)
        self._refresh_scene_table()
        self._update_multi_scene_enabled()

    def _format_diagnostics(self, result) -> str:
        lines = ["CAMERA PROCESSING", f"  Topic: {self.camera_topic or '(file)'}"]
        cam = result.camera_detection
        if cam is not None:
            lines.append(f"  Target: {'DETECTED' if cam.success else 'FAILED'}")
            lines.append(f"  Markers: {cam.markers_detected} / {cam.markers_expected}")
            if cam.reprojection_error_px is not None:
                lines.append(f"  Reprojection error: {cam.reprojection_error_px:.2f} px")
        else:
            lines.append("  (not run)")

        lines.append("")
        lines.append("LIDAR PROCESSING")
        lines.append(f"  ROI Mode: {self._current_roi_mode().upper()}")
        lidar = result.lidar_detection
        if lidar is not None:
            lines.append(f"  ROI Points: {lidar.roi_point_count}")
            if lidar.plane_candidate_count:
                lines.append(f"  Plane candidates tried: {lidar.plane_candidate_count} "
                              f"(selected index {lidar.selected_plane_index})")
            lines.append(f"  Plane Inliers: {lidar.plane_inlier_count} ({lidar.plane_inlier_ratio * 100:.1f}%)")
            lines.append(f"  Boundary Points: {lidar.boundary_point_count}")
            lines.append(f"  Circles: {lidar.valid_circle_count} / 4 "
                         f"({lidar.circle_candidate_count} candidates)")
            if lidar.circle_fit_errors_m:
                errs = ", ".join(f"{e * 1000:.2f}mm" for e in lidar.circle_fit_errors_m)
                lines.append(f"  Circle fit errors: {errs}")
        else:
            lines.append("  (not run)")

        if self.camera_topic and self.lidar_topic:
            sync_ms = abs(self.image_timestamp - self.cloud_timestamp) * 1000.0
            tone = "GOOD" if sync_ms <= _SYNC_GOOD_MS else ("WARNING" if sync_ms <= _SYNC_WARNING_MS else "BAD")
            lines.append("")
            lines.append(f"SYNC  Δt = {sync_ms:.1f} ms  [{tone}]")

        lines.append("")
        lines.append("COMMON FEATURES (camera ∩ LiDAR, canonical id)")
        for cid in CORNER_ORDER:
            mark = "✓" if cid in result.common_ids else "✕"
            lines.append(f"  {cid:14s} {mark}")
        if result.scene_type is not None:
            lines.append(f"  Scene Type: {_SCENE_TYPE_LABEL[result.scene_type]}  "
                         f"({len(result.common_ids)} / 4 common)")
        if result.missing_from_camera:
            lines.append(f"  Missing from camera: {', '.join(sorted(result.missing_from_camera))}")
        if result.missing_from_lidar:
            lines.append(f"  Missing from LiDAR: {', '.join(sorted(result.missing_from_lidar))}")
        if result.correspondence_ambiguous:
            lines.append("  ⚠ Correspondence used the 'target held upright, not mirrored' fallback "
                         "assumption (no reference calibration yet) -- see camera_lidar/correspondence.py.")

        lines.append("")
        if result.success:
            lines.append("STATUS: SCENE READY")
        else:
            lines.append("STATUS: NOT READY")
            lines.append(f"REASON: {result.error_message}")
        return "\n".join(lines)

    def _render_single_scene_result(self, result) -> None:
        if not result.success:
            self.result_text.setPlainText("CALIBRATION FAILED\n\n" + (result.error_message or "Unknown failure."))
            return

        roll, pitch, yaw = rotation_matrix_to_rpy(result.R_camera_from_lidar, degrees=True)
        qx, qy, qz, qw = rotation_matrix_to_quaternion(result.R_camera_from_lidar)
        tx, ty, tz = result.t_camera_from_lidar

        lines = [
            "PROVISIONAL RESULT (single-scene FAST-Calib)",
            "",
            "TRANSLATION (m)",
            f"  Tx = {tx: .6f}",
            f"  Ty = {ty: .6f}",
            f"  Tz = {tz: .6f}",
            "",
            "ROTATION (deg)",
            f"  Roll  = {roll: .4f}",
            f"  Pitch = {pitch: .4f}",
            f"  Yaw   = {yaw: .4f}",
            "",
            "QUATERNION (x, y, z, w)",
            f"  {qx: .6f}  {qy: .6f}  {qz: .6f}  {qw: .6f}",
            "",
            "T_camera_from_lidar (4x4)",
            _matrix_text(result.T_camera_from_lidar),
            "",
            "T_lidar_from_camera (4x4, inverse)",
            _matrix_text(result.T_lidar_from_camera),
            "",
            "CENTER REGISTRATION RESIDUAL (mm)",
            f"  RMSE   = {result.residual_rmse_m * 1000:.3f}",
            f"  Mean   = {result.residual_mean_m * 1000:.3f}",
            f"  Median = {result.residual_median_m * 1000:.3f}",
            f"  P95    = {result.residual_p95_m * 1000:.3f}",
            f"  Max    = {result.residual_max_m * 1000:.3f}",
        ]
        self.result_text.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # ⑦ Scene Manager
    # ------------------------------------------------------------------

    def _build_scene_manager_group(self) -> QGroupBox:
        group = QGroupBox("⑦ Scene Manager")
        outer = QVBoxLayout(group)

        self.scene_table = QTableWidget(0, 10)
        self.scene_table.setHorizontalHeaderLabels(
            ["Scene", "Type", "Camera", "LiDAR", "ROI", "Common", "Missing", "Sync Δt (ms)", "Quality", "Use"]
        )
        outer.addWidget(self.scene_table)

        row = QHBoxLayout()
        view_button = QPushButton("View")
        view_button.clicked.connect(self._on_view_scene)
        row.addWidget(view_button)
        include_button = QPushButton("Include")
        include_button.clicked.connect(lambda: self._set_selected_scene_included(True))
        row.addWidget(include_button)
        exclude_button = QPushButton("Exclude")
        exclude_button.clicked.connect(lambda: self._set_selected_scene_included(False))
        row.addWidget(exclude_button)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self._on_delete_scene)
        row.addWidget(delete_button)
        recapture_button = QPushButton("Re-capture")
        recapture_button.clicked.connect(self._on_recapture_scene)
        row.addWidget(recapture_button)
        row.addStretch(1)
        outer.addLayout(row)
        return group

    def _refresh_scene_table(self) -> None:
        self.scene_table.setRowCount(len(self.captured_scenes))
        for row, captured in enumerate(self.captured_scenes):
            result = captured.detection
            type_text = _SCENE_TYPE_LABEL.get(result.scene_type, "-") if result is not None else "-"
            common_text = f"{len(result.common_ids)}/4" if (result is not None and result.common_ids) else "-"
            missing_ids = (result.missing_from_camera | result.missing_from_lidar) if result is not None else frozenset()
            missing_text = ", ".join(_CORNER_ABBREV[cid] for cid in sorted(missing_ids)) if missing_ids else "-"
            if captured.camera_topic and captured.lidar_topic:
                cloud_ts = captured.scene.cloud.timestamp
                image_ts = captured.scene.image.timestamp
                sync_text = f"{abs(image_ts - cloud_ts) * 1000:.1f}"
            else:
                sync_text = "-"
            quality_text = f"{_quality_percent(result):.0f}%" if (result and result.success) else "FAILED"

            values = [
                captured.scene_id,
                type_text,
                captured.camera_topic or "(file)",
                captured.lidar_topic or "(file)",
                captured.roi_mode.upper(),
                common_text,
                missing_text,
                sync_text,
                quality_text,
                "✓" if captured.included else "✕",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 8 and result is not None:
                    item.setForeground(qcolor(Theme.GOOD if result.success else Theme.BAD))
                self.scene_table.setItem(row, col, item)

    def _selected_captured_scene(self) -> CapturedScene | None:
        row = self.scene_table.currentRow()
        if row < 0 or row >= len(self.captured_scenes):
            return None
        return self.captured_scenes[row]

    def _on_view_scene(self) -> None:
        captured = self._selected_captured_scene()
        if captured is None or captured.detection is None:
            return
        self._render_single_scene_result(captured.detection)
        self.diagnostics_text.setPlainText(
            f"(Viewing {captured.scene_id})\n\n" + self._format_diagnostics(captured.detection)
        )

    def _set_selected_scene_included(self, included: bool) -> None:
        captured = self._selected_captured_scene()
        if captured is None:
            return
        captured.included = included
        self._refresh_scene_table()
        self._update_multi_scene_enabled()

    def _on_delete_scene(self) -> None:
        row = self.scene_table.currentRow()
        if row < 0 or row >= len(self.captured_scenes):
            return
        del self.captured_scenes[row]
        self._refresh_scene_table()
        self._update_multi_scene_enabled()

    def _on_recapture_scene(self) -> None:
        captured = self._selected_captured_scene()
        if captured is None:
            return
        result = calibrate_single_scene(captured.scene, roi_mode=captured.roi_mode)
        captured.detection = result
        captured.included = result.success
        self._refresh_scene_table()
        self._update_multi_scene_enabled()

    # ------------------------------------------------------------------
    # ⑧ Multi-Scene Calibration (the final result)
    # ------------------------------------------------------------------

    def _build_multi_scene_group(self) -> QGroupBox:
        group = QGroupBox("⑧ FINAL MULTI-SCENE CALIBRATION")
        outer = QVBoxLayout(group)

        policy_row = QHBoxLayout()
        policy_row.addWidget(QLabel("Calibration Policy"))
        self.policy_combo = QComboBox()
        self.policy_combo.addItem("STRICT — 4/4 (FULL) scenes only", "strict")
        self.policy_combo.addItem("FLEXIBLE — ≥3/4 (FULL + PARTIAL) scenes", "flexible")
        self.policy_combo.addItem("COMPARE BOTH — run STRICT and FLEXIBLE, report the difference", "compare")
        policy_row.addWidget(self.policy_combo, stretch=1)
        outer.addLayout(policy_row)

        self.multi_scene_button = QPushButton("RUN MULTI-SCENE CALIBRATION")
        self.multi_scene_button.setProperty("role", "primary")
        self.multi_scene_button.clicked.connect(self._on_run_multi_scene)
        outer.addWidget(self.multi_scene_button)

        self.multi_scene_result_text = QPlainTextEdit()
        self.multi_scene_result_text.setReadOnly(True)
        self.multi_scene_result_text.setMinimumHeight(260)
        self.multi_scene_result_text.setPlainText(
            "Capture at least 2 scenes, then run Multi-Scene calibration for the final result."
        )
        outer.addWidget(self.multi_scene_result_text)
        return group

    def _update_multi_scene_enabled(self) -> None:
        included_count = sum(1 for c in self.captured_scenes if c.included)
        self.multi_scene_button.setEnabled(included_count >= 2)

    def _on_run_multi_scene(self) -> None:
        policy = self.policy_combo.currentData()
        if policy == "compare":
            worker = PolicyComparisonWorker(list(self.captured_scenes))
            thread = run_worker_in_thread(worker, self)
            worker.progress.connect(self._on_progress)
            worker.result_ready.connect(self._on_policy_comparison_ready)
            worker.error.connect(self._on_error)
        else:
            worker = MultiSceneCalibrationWorker(list(self.captured_scenes), policy=policy)
            thread = run_worker_in_thread(worker, self)
            worker.progress.connect(self._on_progress)
            worker.result_ready.connect(self._on_multi_scene_result_ready)
            worker.error.connect(self._on_error)
        self._multi_thread, self._multi_worker = thread, worker
        self.multi_scene_button.setEnabled(False)
        thread.finished.connect(self._update_multi_scene_enabled)
        thread.start()

    def _format_multi_scene_result(self, result, heading: str = "FINAL MULTI-SCENE CALIBRATION") -> str:
        if not result.success:
            return f"{heading} FAILED\n\n" + (result.error_message or "Unknown failure.")

        roll, pitch, yaw = rotation_matrix_to_rpy(result.R_camera_from_lidar, degrees=True)
        qx, qy, qz, qw = rotation_matrix_to_quaternion(result.R_camera_from_lidar)
        tx, ty, tz = result.t_camera_from_lidar

        lines = [
            f"{heading}  [policy: {result.policy.upper()}]",
            f"Scenes used: {result.scene_count}",
            "",
            "TRANSLATION (m)",
            f"  Tx = {tx: .6f}",
            f"  Ty = {ty: .6f}",
            f"  Tz = {tz: .6f}",
            "",
            "ROTATION (deg)",
            f"  Roll  = {roll: .4f}",
            f"  Pitch = {pitch: .4f}",
            f"  Yaw   = {yaw: .4f}",
            "",
            "QUATERNION (x, y, z, w)",
            f"  {qx: .6f}  {qy: .6f}  {qz: .6f}  {qw: .6f}",
            "",
            "T_camera_from_lidar (4x4)",
            _matrix_text(result.T_camera_from_lidar),
            "",
            "T_lidar_from_camera (4x4, inverse)",
            _matrix_text(result.T_lidar_from_camera),
            "",
            "OVERALL CENTER REGISTRATION RESIDUAL (mm, all scenes pooled)",
            f"  RMSE   = {result.residual_rmse_m * 1000:.3f}",
            f"  Mean   = {result.residual_mean_m * 1000:.3f}",
            f"  Median = {result.residual_median_m * 1000:.3f}",
            f"  P95    = {result.residual_p95_m * 1000:.3f}",
            f"  Max    = {result.residual_max_m * 1000:.3f}",
            "",
            "SCENE-BY-SCENE RESIDUAL (final transform re-applied to each scene)",
        ]
        for s in result.per_scene:
            status = "OUTLIER" if s.is_outlier else "Good"
            lines.append(f"  {s.scene_id}: RMSE={s.rmse_m * 1000:.2f}mm  P95={s.p95_m * 1000:.2f}mm  [{status}]")
        if result.outlier_scene_ids:
            lines.append("")
            lines.append(f"⚠ Outlier scenes: {', '.join(result.outlier_scene_ids)} "
                         f"-- review in Scene Manager and consider excluding + re-running.")
        return "\n".join(lines)

    def _on_multi_scene_result_ready(self, result) -> None:
        self._refresh_scene_table()  # captured.detection may have been refreshed per-scene
        self.multi_scene_result_text.setPlainText(self._format_multi_scene_result(result))

    def _on_policy_comparison_ready(self, comparison) -> None:
        self._refresh_scene_table()
        strict_text = self._format_multi_scene_result(comparison.strict_result, "STRICT RESULT")
        flexible_text = self._format_multi_scene_result(comparison.flexible_result, "FLEXIBLE RESULT")

        parts = [strict_text, "\n" + "=" * 60 + "\n", flexible_text]

        if comparison.impact is not None:
            parts.append("\n" + "=" * 60 + "\n")
            parts.append("STRICT vs FLEXIBLE")
            parts.append(f"  Translation Difference = {comparison.translation_difference_m * 1000:.3f} mm")
            parts.append(f"  Rotation Difference    = {comparison.rotation_difference_deg:.4f} deg")
            parts.append(f"  Residual Difference    = {comparison.residual_difference_m * 1000:.3f} mm")
            parts.append("")
            parts.append(f"PARTIAL SCENE IMPACT: {comparison.impact}")
            if comparison.impact == "LOW":
                parts.append("Partial scenes are consistent with full scenes.")
            else:
                parts.append(
                    "Partial scenes significantly change the calibration result.\n"
                    "Recommended: use STRICT mode, or review PARTIAL scenes in the Scene Manager."
                )

        self.multi_scene_result_text.setPlainText("\n".join(parts))
