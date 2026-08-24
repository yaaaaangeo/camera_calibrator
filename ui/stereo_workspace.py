"""
Camera-to-camera stereo calibration workspace.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QCheckBox,
    QListWidget,
    QMessageBox,
    QPushButton,
    QDialog,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.calibration_io import StandardCalibration, load_standard_calibration
from calibration.stereo import (
    StereoCalibrationResult,
    StereoPairObservation,
    pair_image_paths,
    set_pair_used,
)
from calibration.stereo_auditor import compute_capture_coach, compute_sync_guard
from calibration.stereo_session import StereoSession
from export.stereo import (
    export_stereo_kalibr_camchain,
    export_stereo_html,
    export_stereo_json,
    export_stereo_yaml,
    StereoRoboticsExportOptions,
    stereo_pairs_from_dict,
    stereo_result_from_dict,
)
from ui.stereo_live_capture_dialog import StereoLiveCaptureDialog
from ui.theme import Theme
from ui.worker import StereoCalibrationWorker, StereoPairDetectionWorker, run_worker_in_thread


class StereoWorkspace(QWidget):
    back_requested = Signal()
    calibrate_intrinsic_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.camera1: StandardCalibration | None = None
        self.camera2: StandardCalibration | None = None
        self.session = StereoSession()
        self.pattern_config = None
        self._pair_thread = None
        self._pair_worker = None
        self._calibration_thread = None
        self._calibration_worker = None
        self.section_groups: dict[str, QGroupBox] = {}
        self.step_buttons: list[QPushButton] = []
        self._step_order = ["intrinsics", "pairs", "result", "validation", "rectification", "evidence"]
        self._step_labels = {
            "intrinsics": "① Intrinsics",
            "pairs": "② Pair/Coach",
            "result": "⑤ Result",
            "validation": "⑦ Validation",
            "rectification": "⑥ Rectify",
            "evidence": "⑩ Evidence/Export",
        }
        self._current_step_index = 0
        self._last_unmatched_camera1_paths: list[str] = []
        self._last_unmatched_camera2_paths: list[str] = []
        self._current_camera1_paths: list[str] = []
        self._current_camera2_paths: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(6)
        back = QPushButton("← Calibration Home")
        back.setMaximumHeight(28)
        back.clicked.connect(self.back_requested.emit)
        top.addWidget(back)
        self.step_label = QLabel("Camera-to-Camera")
        self.step_label.setProperty("tone", "muted")
        top.addWidget(self.step_label, stretch=1)
        self.step_back_button = QPushButton("← Back")
        self.step_back_button.setMaximumHeight(28)
        self.step_back_button.clicked.connect(lambda: self._move_step(-1))
        self.step_next_button = QPushButton("Next →")
        self.step_next_button.setMaximumHeight(28)
        self.step_next_button.setProperty("role", "primary")
        self.step_next_button.clicked.connect(lambda: self._move_step(1))
        top.addWidget(self.step_back_button)
        top.addWidget(self.step_next_button)
        root.addLayout(top)
        step_row = QHBoxLayout()
        step_row.setSpacing(4)
        for key in self._step_order:
            label = self._step_labels[key]
            button = QPushButton(label)
            button.setMaximumHeight(28)
            button.clicked.connect(lambda _checked=False, k=key: self._focus_section(k))
            self.step_buttons.append(button)
            step_row.addWidget(button)
        root.addLayout(step_row)

        self.section_groups["intrinsics"] = self._make_step_panel(self._build_intrinsics_group())
        self.section_groups["pairs"] = self._make_step_panel(self._build_pair_group())
        self.section_groups["result"] = self._make_step_panel(self._build_result_group())
        self.section_groups["validation"] = self._make_step_panel(self._build_validation_group())
        self.section_groups["rectification"] = self._make_step_panel(self._build_rectification_export_group())
        self.section_groups["evidence"] = self._make_step_panel(self._build_evidence_export_group())
        root.addWidget(self.section_groups["intrinsics"])
        root.addWidget(self.section_groups["pairs"])
        root.addWidget(self.section_groups["result"])
        root.addWidget(self.section_groups["validation"])
        root.addWidget(self.section_groups["rectification"])
        root.addWidget(self.section_groups["evidence"], stretch=1)
        self._focus_section("intrinsics")
        self._update_state()

    @property
    def pairs(self) -> list[StereoPairObservation]:
        return self.session.pairs

    @pairs.setter
    def pairs(self, value: list[StereoPairObservation]) -> None:
        self.session.pairs = value

    @property
    def result(self) -> StereoCalibrationResult | None:
        return self.session.result

    @result.setter
    def result(self, value: StereoCalibrationResult | None) -> None:
        self.session.result = value

    def set_pattern_config(self, pattern_config) -> None:
        self.pattern_config = pattern_config

    def set_previous_intrinsic(self, slot: str, calibration: StandardCalibration) -> None:
        if slot == "camera1":
            self.camera1 = calibration
        else:
            self.camera2 = calibration
        self._update_state()

    def restore_project_payload(self, *, result_payload: dict | None = None, pair_payload: list[dict] | None = None) -> None:
        if pair_payload:
            self.pairs = stereo_pairs_from_dict(pair_payload)
        if result_payload:
            result = stereo_result_from_dict(result_payload)
            self.camera1 = result.camera1
            self.camera2 = result.camera2
            self.result = result
        else:
            self.result = None
        self._render_pairs()
        if self.result is not None:
            self._render_result()
        self._update_state()

    def _build_intrinsics_group(self) -> QGroupBox:
        group = QGroupBox("① Intrinsics")
        layout = QGridLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(5)
        self.cam1_status = QLabel("Camera 1: Intrinsic not ready")
        self.cam2_status = QLabel("Camera 2: Intrinsic not ready")
        layout.addWidget(self.cam1_status, 0, 0, 1, 3)
        layout.addWidget(self.cam2_status, 0, 3, 1, 3)

        load1 = QPushButton("Camera 1 Load Calibration")
        load1.clicked.connect(lambda: self._load_calibration("camera1"))
        change1 = QPushButton("Camera 1 Change")
        change1.clicked.connect(lambda: self._load_calibration("camera1"))
        use1 = QPushButton("Camera 1 Use Previous Result")
        use1.clicked.connect(lambda: self.calibrate_intrinsic_requested.emit("camera1"))
        new1 = QPushButton("Camera 1 Calibrate New Camera")
        new1.clicked.connect(lambda: self.calibrate_intrinsic_requested.emit("camera1"))
        layout.addWidget(load1, 1, 0)
        layout.addWidget(use1, 1, 1)
        layout.addWidget(new1, 1, 2)
        layout.addWidget(change1, 2, 0, 1, 3)

        load2 = QPushButton("Camera 2 Load Calibration")
        load2.clicked.connect(lambda: self._load_calibration("camera2"))
        change2 = QPushButton("Camera 2 Change")
        change2.clicked.connect(lambda: self._load_calibration("camera2"))
        use2 = QPushButton("Camera 2 Use Previous Result")
        use2.clicked.connect(lambda: self.calibrate_intrinsic_requested.emit("camera2"))
        new2 = QPushButton("Camera 2 Calibrate New Camera")
        new2.clicked.connect(lambda: self.calibrate_intrinsic_requested.emit("camera2"))
        layout.addWidget(load2, 1, 3)
        layout.addWidget(use2, 1, 4)
        layout.addWidget(new2, 1, 5)
        layout.addWidget(change2, 2, 3, 1, 3)
        return group

    def _build_pair_group(self) -> QGroupBox:
        group = QGroupBox("② Pair Capture / ③ Detection / Pair Manager")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(5)
        self.load_pairs_button = QPushButton("Load Stereo Pair Image Folders...")
        self.load_pairs_button.clicked.connect(self._load_pair_folders)
        row.addWidget(self.load_pairs_button)
        self.live_capture_button = QPushButton("Live Dual Camera Capture...")
        self.live_capture_button.clicked.connect(self._open_live_dual_capture)
        row.addWidget(self.live_capture_button)
        self.pairing_mode_combo = QComboBox()
        self.pairing_mode_combo.addItem("Pairing: Sorted order", "sorted")
        self.pairing_mode_combo.addItem("Pairing: Same filename", "stem")
        self.pairing_mode_combo.addItem("Pairing: Filename timestamp", "timestamp")
        self.pairing_mode_combo.addItem("Pairing: EXIF timestamp", "exif")
        self.pairing_mode_combo.addItem("Pairing: ROS sidecar timestamp", "ros_timestamp")
        row.addWidget(self.pairing_mode_combo)
        row.addWidget(QLabel("Δt ms"))
        self.pairing_tolerance_spin = QDoubleSpinBox()
        self.pairing_tolerance_spin.setRange(1.0, 10000.0)
        self.pairing_tolerance_spin.setValue(30.0)
        self.pairing_tolerance_spin.setSingleStep(5.0)
        row.addWidget(self.pairing_tolerance_spin)
        self.unmatched_preview_button = QPushButton("Unmatched 보기")
        self.unmatched_preview_button.clicked.connect(self._show_unmatched_preview)
        row.addWidget(self.unmatched_preview_button)
        self.manual_pair_button = QPushButton("Manual Pair...")
        self.manual_pair_button.clicked.connect(self._open_manual_pair_dialog)
        row.addWidget(self.manual_pair_button)
        self.run_calibration_button = QPushButton("④ Stereo Calibration 실행")
        self.run_calibration_button.setProperty("role", "primary")
        self.run_calibration_button.clicked.connect(self._run_calibration)
        row.addWidget(self.run_calibration_button)
        layout.addLayout(row)

        self.capture_coach_label = QLabel("Pair를 불러오면 Capture Coach와 Sync Guard 요약이 표시됩니다.")
        self.capture_coach_label.setWordWrap(True)
        self.capture_coach_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.capture_coach_label)

        self.pair_table = QTableWidget(0, 8)
        self.pair_table.setHorizontalHeaderLabels([
            "Pair", "Use", "Quality", "Common Corners", "Sync Δt", "Warnings", "Cam1", "Cam2"
        ])
        self.pair_table.currentCellChanged.connect(lambda *_: self._on_pair_selection_changed())
        layout.addWidget(self.pair_table)
        preview_grid = QGridLayout()
        preview_grid.setHorizontalSpacing(6)
        preview_grid.setVerticalSpacing(5)
        self.cam1_preview_label = QLabel("Camera 1 preview")
        self.cam2_preview_label = QLabel("Camera 2 preview")
        for label in (self.cam1_preview_label, self.cam2_preview_label):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(180)
            label.setProperty("surface", "image")
        preview_grid.addWidget(self.cam1_preview_label, 0, 0)
        preview_grid.addWidget(self.cam2_preview_label, 0, 1)
        layout.addLayout(preview_grid)
        reject_row = QHBoxLayout()
        reject_row.setSpacing(5)
        reject = QPushButton("Exclude Selected Pair")
        reject.clicked.connect(lambda: self._set_selected_pair_used(False))
        include = QPushButton("Include Selected Pair")
        include.clicked.connect(lambda: self._set_selected_pair_used(True))
        delete_pair = QPushButton("Delete Selected Pair")
        delete_pair.clicked.connect(self._delete_selected_pair)
        sort_sync = QPushButton("Sort by Sync Δt")
        sort_sync.clicked.connect(self._sort_pairs_by_sync_delta)
        self.outlier_only_check = QCheckBox("Outlier 후보만 보기")
        self.outlier_only_check.toggled.connect(self._render_pairs)
        reject_outliers = QPushButton("Exclude Outliers")
        reject_outliers.clicked.connect(self._exclude_outlier_candidates)
        recalibrate = QPushButton("Recalibrate")
        recalibrate.clicked.connect(self._run_calibration)
        reject_row.addWidget(reject)
        reject_row.addWidget(include)
        reject_row.addWidget(delete_pair)
        reject_row.addWidget(sort_sync)
        reject_row.addWidget(self.outlier_only_check)
        reject_row.addWidget(reject_outliers)
        reject_row.addWidget(recalibrate)
        reject_row.addStretch(1)
        layout.addLayout(reject_row)
        return group

    def _build_result_group(self) -> QGroupBox:
        group = QGroupBox("④ Calibration Result")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.result_label = QLabel("Stereo calibration 결과가 아직 없습니다.")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
        self.evidence_label = QLabel("Evidence report will appear after stereo calibration.")
        self.evidence_label.setWordWrap(True)
        self.evidence_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.evidence_label)
        self.matrix_detail_label = QLabel("Calibration matrix details will appear after stereo calibration.")
        self.matrix_detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.matrix_detail_label.setWordWrap(True)
        layout.addWidget(self.matrix_detail_label)
        return group

    def _build_validation_group(self) -> QGroupBox:
        group = QGroupBox("⑤ Validation")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.validation_table = QTableWidget(0, 5)
        self.validation_table.setHorizontalHeaderLabels(["Pair", "Common", "Epipolar", "Vertical", "Status"])
        layout.addWidget(self.validation_table)
        return group

    def _build_rectification_export_group(self) -> QGroupBox:
        group = QGroupBox("⑥ Rectification Preview")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        rect_grid = QGridLayout()
        rect_grid.setHorizontalSpacing(6)
        rect_grid.setVerticalSpacing(5)
        self.rectified_cam1_label = QLabel("Rectified Cam1 preview")
        self.rectified_cam2_label = QLabel("Rectified Cam2 preview")
        for label in (self.rectified_cam1_label, self.rectified_cam2_label):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(180)
            label.setProperty("surface", "image")
        rect_grid.addWidget(self.rectified_cam1_label, 0, 0)
        rect_grid.addWidget(self.rectified_cam2_label, 0, 1)
        layout.addLayout(rect_grid)
        return group

    def _build_evidence_export_group(self) -> QGroupBox:
        group = QGroupBox("⑩ Evidence Report / Export")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.evidence_detail_label = QLabel("Evidence report export는 Stereo Calibration 이후 사용할 수 있습니다.")
        self.evidence_detail_label.setWordWrap(True)
        self.evidence_detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.evidence_detail_label)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Audit mode"))
        self.audit_mode_combo = QComboBox()
        self.audit_mode_combo.addItem("Fast audit", "fast")
        self.audit_mode_combo.addItem("Full audit + bootstrap", "full")
        self.audit_mode_combo.setCurrentIndex(1)
        mode_row.addWidget(self.audit_mode_combo)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)
        card_grid = QGridLayout()
        card_grid.setHorizontalSpacing(6)
        card_grid.setVerticalSpacing(6)
        self.evidence_confidence_card = self._make_evidence_card("Confidence", "N/A")
        self.evidence_dataset_card = self._make_evidence_card("Dataset", "N/A")
        self.evidence_sync_card = self._make_evidence_card("Sync", "N/A")
        self.evidence_geometry_card = self._make_evidence_card("Geometry", "N/A")
        card_grid.addWidget(self.evidence_confidence_card, 0, 0)
        card_grid.addWidget(self.evidence_dataset_card, 0, 1)
        card_grid.addWidget(self.evidence_sync_card, 1, 0)
        card_grid.addWidget(self.evidence_geometry_card, 1, 1)
        layout.addLayout(card_grid)
        self.evidence_warning_label = QLabel("Warnings and recommendations will appear here.")
        self.evidence_warning_label.setWordWrap(True)
        self.evidence_warning_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.evidence_warning_label)
        robotics_row = QHBoxLayout()
        robotics_row.setSpacing(5)
        robotics_row.addWidget(QLabel("TF parent"))
        self.parent_frame_edit = QLineEdit("camera1")
        robotics_row.addWidget(self.parent_frame_edit)
        robotics_row.addWidget(QLabel("child"))
        self.child_frame_edit = QLineEdit("camera2")
        robotics_row.addWidget(self.child_frame_edit)
        robotics_row.addWidget(QLabel("rotation"))
        self.rotation_format_combo = QComboBox()
        self.rotation_format_combo.addItem("Quaternion x y z w", "quaternion")
        self.rotation_format_combo.addItem("RPY radians", "rpy_radians")
        self.rotation_format_combo.addItem("RPY degrees", "rpy_degrees")
        robotics_row.addWidget(self.rotation_format_combo)
        robotics_row.addStretch(1)
        layout.addLayout(robotics_row)
        export_row = QHBoxLayout()
        export_row.setSpacing(5)
        self.export_yaml_button = QPushButton("Export Stereo YAML")
        self.export_yaml_button.clicked.connect(self._export_yaml)
        self.export_json_button = QPushButton("Export Stereo JSON")
        self.export_json_button.clicked.connect(self._export_json)
        self.export_html_button = QPushButton("Export Stereo HTML")
        self.export_html_button.clicked.connect(self._export_html)
        self.export_kalibr_button = QPushButton("Export Kalibr Camchain")
        self.export_kalibr_button.clicked.connect(self._export_kalibr)
        export_row.addWidget(self.export_yaml_button)
        export_row.addWidget(self.export_json_button)
        export_row.addWidget(self.export_html_button)
        export_row.addWidget(self.export_kalibr_button)
        export_row.addStretch(1)
        layout.addLayout(export_row)
        return group

    def _make_evidence_card(self, title: str, value: str) -> QLabel:
        label = QLabel(f"{title}\n{value}")
        label.setWordWrap(True)
        label.setMinimumHeight(64)
        label.setStyleSheet(
            f"background: {Theme.BG_TERTIARY}; border: 1px solid {Theme.BORDER}; "
            f"border-radius: 4px; padding: 8px; color: {Theme.TEXT_VALUE};"
        )
        return label

    def _make_step_panel(self, group: QGroupBox) -> QGroupBox:
        group.setObjectName("stereoStepGroup")
        group.setStyleSheet(
            f"""
            QGroupBox#stereoStepGroup {{
                margin-top: 8px;
                padding: 8px 7px 7px 7px;
            }}
            QGroupBox#stereoStepGroup::title {{
                color: {Theme.ACCENT};
                padding: 0 4px;
            }}
            """
        )
        group.setCheckable(False)
        return group

    def _focus_section(self, key: str) -> None:
        if key in self._step_order:
            self._current_step_index = self._step_order.index(key)
        for name, group in self.section_groups.items():
            group.setVisible(name == key)
        self._update_state()

    def _move_step(self, delta: int) -> None:
        next_index = max(0, min(len(self._step_order) - 1, self._current_step_index + delta))
        self._focus_section(self._step_order[next_index])

    def _load_calibration(self, slot: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Calibration 파일 선택",
            "",
            "Calibration Files (*.yaml *.yml *.json);;YAML (*.yaml *.yml);;JSON (*.json)",
        )
        if not path:
            return
        try:
            cal = load_standard_calibration(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "불러오기 실패", str(exc))
            return
        self.set_previous_intrinsic(slot, cal)

    def _load_pair_folders(self) -> None:
        if self.pattern_config is None:
            QMessageBox.warning(self, "패턴 설정 없음", "Intrinsic Workspace에서 ChArUco/Pattern 설정을 먼저 지정하세요.")
            return
        cam1_dir = QFileDialog.getExistingDirectory(self, "Camera 1 이미지 폴더 선택")
        if not cam1_dir:
            return
        cam2_dir = QFileDialog.getExistingDirectory(self, "Camera 2 이미지 폴더 선택")
        if not cam2_dir:
            return
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        paths1 = sorted(str(p) for ext in exts for p in Path(cam1_dir).glob(ext))
        paths2 = sorted(str(p) for ext in exts for p in Path(cam2_dir).glob(ext))
        if not paths1 or not paths2:
            QMessageBox.warning(self, "이미지 없음", "두 폴더 모두 이미지가 있어야 합니다.")
            return
        pairing = pair_image_paths(
            paths1,
            paths2,
            mode=self.pairing_mode_combo.currentData(),
            max_timestamp_delta_ms=self.pairing_tolerance_spin.value(),
        )
        if not pairing.camera1_paths or not pairing.camera2_paths:
            QMessageBox.warning(self, "Pair 없음", "선택한 pairing 방식으로 묶을 수 있는 이미지가 없습니다.")
            return
        if pairing.warnings:
            unmatched = (
                f" Unmatched cam1={len(pairing.unmatched_camera1_paths)}, "
                f"cam2={len(pairing.unmatched_camera2_paths)}."
            )
            self.result_label.setText(" ".join(pairing.warnings) + unmatched)
        self._last_unmatched_camera1_paths = pairing.unmatched_camera1_paths
        self._last_unmatched_camera2_paths = pairing.unmatched_camera2_paths
        self._current_camera1_paths = list(pairing.camera1_paths)
        self._current_camera2_paths = list(pairing.camera2_paths)
        self._start_pair_detection(pairing.camera1_paths, pairing.camera2_paths)

    def _open_live_dual_capture(self) -> None:
        if self.pattern_config is None:
            QMessageBox.warning(self, "패턴 설정 없음", "Intrinsic Workspace에서 ChArUco/Pattern 설정을 먼저 지정하세요.")
            return
        output_dir = Path.cwd() / "Output" / f"stereo_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dialog = StereoLiveCaptureDialog(str(output_dir), self.pattern_config, self)
        if dialog.exec() != QDialog.Accepted:
            return
        self._start_pair_detection(dialog.captured_paths_cam1, dialog.captured_paths_cam2)

    def _start_pair_detection(self, paths1: list[str], paths2: list[str]) -> None:
        worker = StereoPairDetectionWorker(paths1, paths2, self.pattern_config)
        thread = run_worker_in_thread(worker, self)
        worker.progress.connect(self.result_label.setText)
        worker.pairs_ready.connect(self._on_pairs_ready)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(self._on_pair_worker_finished)
        self._pair_thread, self._pair_worker = thread, worker
        self.load_pairs_button.setEnabled(False)
        self.live_capture_button.setEnabled(False)
        thread.start()

    def _on_pairs_ready(self, payload) -> None:
        self.pairs, _ds1, _ds2 = payload
        self.result = None
        self.result_label.setText(f"Stereo pair detection 완료: {len(self.pairs)} pairs")
        self._render_pairs()
        self._render_capture_coach()
        self._update_state()

    def _on_worker_error(self, message: str) -> None:
        self.result_label.setText(message)
        QMessageBox.critical(self, "Stereo 작업 실패", message)

    def _on_pair_worker_finished(self) -> None:
        self._pair_thread = None
        self._pair_worker = None
        self._update_state()

    def _run_calibration(self) -> None:
        if self.camera1 is None or self.camera2 is None:
            QMessageBox.warning(self, "Intrinsic 없음", "Camera 1/2 Intrinsic을 먼저 준비하세요.")
            return
        if not self.pairs:
            QMessageBox.warning(self, "Pair 없음", "Stereo pair 이미지를 먼저 불러오세요.")
            return
        w = self.camera1.width or self.camera2.width
        h = self.camera1.height or self.camera2.height
        if not w or not h:
            QMessageBox.warning(self, "해상도 없음", "Calibration 파일에 image resolution이 필요합니다.")
            return
        audit_mode = self.audit_mode_combo.currentData() if hasattr(self, "audit_mode_combo") else "full"
        worker = StereoCalibrationWorker(
            self.pairs,
            self.camera1,
            self.camera2,
            (int(w), int(h)),
            audit_mode=audit_mode or "full",
        )
        thread = run_worker_in_thread(worker, self)
        worker.progress.connect(self.result_label.setText)
        worker.result_ready.connect(self._on_stereo_result_ready)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(self._on_calibration_worker_finished)
        self._calibration_thread, self._calibration_worker = thread, worker
        self.run_calibration_button.setEnabled(False)
        thread.start()

    def _on_stereo_result_ready(self, result: StereoCalibrationResult) -> None:
        self.result = result
        self._render_result()
        self._update_state()

    def _on_calibration_worker_finished(self) -> None:
        self._calibration_thread = None
        self._calibration_worker = None
        self._update_state()

    def _set_selected_pair_used(self, used: bool) -> None:
        row = self.pair_table.currentRow()
        visible_pairs = self._visible_pairs()
        if row < 0 or row >= len(visible_pairs):
            return
        set_pair_used(visible_pairs[row], used)
        self._render_pairs()
        self._render_capture_coach()

    def _outlier_pair_ids(self) -> set[str]:
        return self.session.outlier_pair_ids()

    def _visible_pairs(self) -> list[StereoPairObservation]:
        return self.session.visible_pairs(outliers_only=self.outlier_only_check.isChecked())

    def _exclude_outlier_candidates(self) -> None:
        outliers = self._outlier_pair_ids()
        if not outliers:
            QMessageBox.information(self, "Outlier 없음", "현재 validation 결과에서 Outlier 후보가 없습니다.")
            return
        changed = self.session.reject_outliers()
        self.result_label.setText(f"Outlier 후보 {changed}개를 제외했습니다. Recalibrate를 눌러 다시 계산하세요.")
        self._render_pairs()
        self._render_capture_coach()
        self._update_state()

    def _delete_selected_pair(self) -> None:
        row = self.pair_table.currentRow()
        visible_pairs = self._visible_pairs()
        if row < 0 or row >= len(visible_pairs):
            return
        pair = visible_pairs[row]
        self.pairs = [item for item in self.pairs if item is not pair]
        self.result = None
        self.result_label.setText(f"{pair.pair_id} 삭제됨. 필요하면 Stereo Calibration을 다시 실행하세요.")
        self._render_pairs()
        self._render_capture_coach()
        self._update_state()

    def _sort_pairs_by_sync_delta(self) -> None:
        self.pairs = sorted(
            self.pairs,
            key=lambda pair: float("inf") if pair.sync_delta_ms is None else abs(float(pair.sync_delta_ms)),
        )
        self._render_pairs()

    def _render_pairs(self) -> None:
        visible_pairs = self._visible_pairs()
        outliers = self._outlier_pair_ids()
        self.pair_table.setRowCount(len(visible_pairs))
        for row, pair in enumerate(visible_pairs):
            values = [
                pair.pair_id,
                "Use" if pair.used else "Reject",
                self._quality_summary(pair),
                str(pair.common_count),
                "N/A" if pair.sync_delta_ms is None else f"{pair.sync_delta_ms:.1f} ms",
                "; ".join((["Validation outlier"] if pair.pair_id in outliers else []) + pair.quality_warnings),
                "Detected",
                "Detected",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 2:
                    item.setToolTip(self._quality_detail(pair))
                if col == 5 and value:
                    item.setToolTip(value)
                self.pair_table.setItem(row, col, item)
        if visible_pairs and self.pair_table.currentRow() < 0:
            self.pair_table.setCurrentCell(0, 0)
        self._render_pair_preview()
        self._render_capture_coach()

    def _current_image_size(self) -> tuple[int, int]:
        w = self.camera1.width if self.camera1 is not None and self.camera1.width else None
        h = self.camera1.height if self.camera1 is not None and self.camera1.height else None
        if w and h:
            return int(w), int(h)
        for pair in self.pairs:
            if pair.image_path_cam1:
                img = cv2.imread(pair.image_path_cam1, cv2.IMREAD_COLOR)
                if img is not None:
                    return int(img.shape[1]), int(img.shape[0])
        return 640, 480

    def _render_capture_coach(self) -> None:
        if not hasattr(self, "capture_coach_label"):
            return
        if not self.pairs:
            self.capture_coach_label.setText("Pair를 불러오면 Capture Coach와 Sync Guard 요약이 표시됩니다.")
            return
        capture = compute_capture_coach(self.pairs, self._current_image_size())
        sync = compute_sync_guard(self.pairs, threshold_ms=self.pairing_tolerance_spin.value())
        stats = sync.get("timestamp_delta_ms", {})
        recs = capture.get("recommendations", [])
        rec_text = " / ".join(str(item) for item in recs[:3]) if recs else "현재 추천 없음"
        self.capture_coach_label.setText(
            "CAPTURE COACH\n"
            f"Usable pairs: {capture.get('usable_pairs', 0)} / {capture.get('target_pairs', 50)}, "
            f"dataset quality: {capture.get('dataset_quality_score', 0.0):.1f}%, "
            f"joint coverage: {capture.get('joint_coverage_score', 0.0):.1f}%, "
            f"ready: {capture.get('dataset_ready', False)}\n"
            f"Sync Guard: {sync.get('status', 'N/A')}, "
            f"median/p95/max Δt: {stats.get('median', 'N/A')}/"
            f"{stats.get('p95', 'N/A')}/{stats.get('max', 'N/A')} ms\n"
            f"Recommendation: {rec_text}"
        )

    def _quality_summary(self, pair: StereoPairObservation) -> str:
        c = pair.quality_components
        if not c:
            return f"{pair.quality_score:.0f} ({pair.quality_status})"
        return (
            f"{pair.quality_score:.0f} ({pair.quality_status}) "
            f"C{c.get('common_corners', 0):.0f}/S{c.get('board_size', 0):.0f}/"
            f"P{c.get('board_position', 0):.0f}/D{c.get('pose_diversity', 0):.0f}"
        )

    def _quality_detail(self, pair: StereoPairObservation) -> str:
        c = pair.quality_components
        lines = [
            f"Overall: {pair.quality_score:.1f} ({pair.quality_status})",
            f"Common corners: {c.get('common_corners', 0):.1f}",
            f"Board size: {c.get('board_size', 0):.1f}",
            f"Board position: {c.get('board_position', 0):.1f}",
            f"Pose diversity: {c.get('pose_diversity', 0):.1f}",
            f"Timestamp sync: {c.get('timestamp_sync', 0):.1f}",
            f"Detection confidence: {c.get('detection_confidence', 0):.1f}",
        ]
        if pair.quality_warnings:
            lines.append("Warnings: " + "; ".join(pair.quality_warnings))
        return "\n".join(lines)

    def _show_unmatched_preview(self) -> None:
        cam1 = self._last_unmatched_camera1_paths
        cam2 = self._last_unmatched_camera2_paths
        if not cam1 and not cam2:
            QMessageBox.information(self, "Unmatched files", "현재 pairing에서 제외된 파일이 없습니다.")
            return
        preview1 = "\n".join(cam1[:20]) + (f"\n... 외 {len(cam1) - 20}개" if len(cam1) > 20 else "")
        preview2 = "\n".join(cam2[:20]) + (f"\n... 외 {len(cam2) - 20}개" if len(cam2) > 20 else "")
        QMessageBox.information(
            self,
            "Unmatched files",
            f"Camera 1 unmatched ({len(cam1)}):\n{preview1 or '-'}\n\n"
            f"Camera 2 unmatched ({len(cam2)}):\n{preview2 or '-'}",
        )

    def _open_manual_pair_dialog(self) -> None:
        if not self._last_unmatched_camera1_paths or not self._last_unmatched_camera2_paths:
            QMessageBox.information(self, "Manual Pair", "수동으로 묶을 unmatched 파일이 양쪽 모두에 있어야 합니다.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Manual Stereo Pair")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Camera 1 unmatched file"))
        cam1_combo = QListWidget()
        for path in self._last_unmatched_camera1_paths:
            cam1_combo.addItem(Path(path).name)
            cam1_combo.item(cam1_combo.count() - 1).setData(Qt.UserRole, path)
        if cam1_combo.count():
            cam1_combo.setCurrentRow(0)
        layout.addWidget(cam1_combo)
        layout.addWidget(QLabel("Camera 2 unmatched file"))
        cam2_combo = QListWidget()
        for path in self._last_unmatched_camera2_paths:
            cam2_combo.addItem(Path(path).name)
            cam2_combo.item(cam2_combo.count() - 1).setData(Qt.UserRole, path)
        if cam2_combo.count():
            cam2_combo.setCurrentRow(0)
        layout.addWidget(cam2_combo)
        layout.addWidget(QLabel("선택한 Camera 1 파일과 Camera 2 파일을 하나의 stereo pair로 묶습니다."))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return
        item1 = cam1_combo.currentItem()
        item2 = cam2_combo.currentItem()
        path1 = item1.data(Qt.UserRole) if item1 is not None else None
        path2 = item2.data(Qt.UserRole) if item2 is not None else None
        if not path1 or not path2:
            return
        self._current_camera1_paths.append(path1)
        self._current_camera2_paths.append(path2)
        self._last_unmatched_camera1_paths = [p for p in self._last_unmatched_camera1_paths if p != path1]
        self._last_unmatched_camera2_paths = [p for p in self._last_unmatched_camera2_paths if p != path2]
        self.result_label.setText("Manual pair를 추가했습니다. Pair detection을 다시 실행합니다.")
        self._start_pair_detection(self._current_camera1_paths, self._current_camera2_paths)

    def _on_pair_selection_changed(self) -> None:
        self._render_pair_preview()
        self._render_rectification_preview()

    def _render_pair_preview(self) -> None:
        row = self.pair_table.currentRow()
        visible_pairs = self._visible_pairs()
        if row < 0 or row >= len(visible_pairs):
            self.cam1_preview_label.setText("Camera 1 preview")
            self.cam2_preview_label.setText("Camera 2 preview")
            return
        pair = visible_pairs[row]
        self._set_preview_pixmap(
            self.cam1_preview_label,
            pair.image_path_cam1,
            pair.detected_points_cam1 if pair.detected_points_cam1 is not None else pair.image_points_cam1,
            pair.detected_ids_cam1 if pair.detected_ids_cam1 is not None else pair.common_ids,
            pair.common_ids,
            rejected=not pair.used,
        )
        self._set_preview_pixmap(
            self.cam2_preview_label,
            pair.image_path_cam2,
            pair.detected_points_cam2 if pair.detected_points_cam2 is not None else pair.image_points_cam2,
            pair.detected_ids_cam2 if pair.detected_ids_cam2 is not None else pair.common_ids,
            pair.common_ids,
            rejected=not pair.used,
        )

    def _set_preview_pixmap(
        self,
        label: QLabel,
        path: str | None,
        points: np.ndarray,
        ids: np.ndarray,
        common_ids: np.ndarray,
        *,
        rejected: bool = False,
    ) -> None:
        if not path:
            label.setText("이미지 경로 없음")
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            label.setText(f"이미지 로드 실패:\n{path}")
            return
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        ids_flat = np.asarray(ids).reshape(-1)
        common_set = {int(v) for v in np.asarray(common_ids).reshape(-1).tolist()}
        for (x, y), corner_id in zip(pts, ids_flat):
            center = (int(round(x)), int(round(y)))
            if rejected:
                color = (40, 40, 230)
            elif int(corner_id) in common_set:
                color = (0, 220, 0)
            else:
                color = (0, 165, 255)
            cv2.circle(img, center, 4, color, -1)
            cv2.putText(
                img,
                str(int(corner_id)),
                (center[0] + 5, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        legend = [
            ((0, 220, 0), "common"),
            ((0, 165, 255), "non-common"),
            ((40, 40, 230), "rejected"),
        ]
        for index, (color, text) in enumerate(legend):
            y = 22 + index * 22
            cv2.circle(img, (12, y - 5), 5, color, -1)
            cv2.putText(img, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        label.setPixmap(self._bgr_to_pixmap(img, max_width=460))

    @staticmethod
    def _bgr_to_pixmap(img_bgr: np.ndarray, max_width: int = 460) -> QPixmap:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        if pixmap.width() > max_width:
            pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
        return pixmap

    def _render_result(self) -> None:
        if self.result is None:
            return
        r = self.result
        roll, pitch, yaw = r.roll_pitch_yaw_deg

        def fmt(value: float | None) -> str:
            return "N/A" if value is None else f"{value:.4f}"

        self.result_label.setText(
            f"STEREO CALIBRATION RESULT\n"
            f"Stereo RMS: {r.stereo_rms:.4f} px\n"
            f"Baseline: {r.baseline * 1000.0:.2f} mm\n"
            f"Translation Tx/Ty/Tz: {r.t_cam2_from_cam1.reshape(3).tolist()}\n"
            f"Rotation Roll/Pitch/Yaw: {roll:.3f}, {pitch:.3f}, {yaw:.3f} deg\n"
            f"Epipolar Mean/P95/Max: {fmt(r.epipolar_error.mean)}, {fmt(r.epipolar_error.p95)}, {fmt(r.epipolar_error.max)} px\n"
            f"Rectification Vertical RMSE/P95/Max: {fmt(r.rectification_vertical_error.rmse)}, "
            f"{fmt(r.rectification_vertical_error.p95)}, {fmt(r.rectification_vertical_error.max)} px\n"
            f"Hold-out Epipolar RMSE Train/Test/Gap: "
            f"{fmt(r.holdout_training_error.rmse if r.holdout_training_error else None)}, "
            f"{fmt(r.holdout_validation_error.rmse if r.holdout_validation_error else None)}, "
            f"{fmt(r.holdout_generalization_gap)} px\n"
            f"Used pairs: {r.used_pair_count}, Rejected pairs: {r.rejected_pair_count}, "
            f"Total common corners: {r.total_common_corners}"
        )
        self.evidence_label.setText(self._evidence_summary_text(r))
        if hasattr(self, "evidence_detail_label"):
            self.evidence_detail_label.setText(self._evidence_detail_text(r))
            self._render_evidence_cards(r)
        self.matrix_detail_label.setText(self._matrix_detail_text(r))
        self.validation_table.setRowCount(len(r.pair_validations))
        for row, item in enumerate(r.pair_validations):
            values = [
                item.pair_id,
                str(item.common_corners),
                "N/A" if item.epipolar_mean is None else f"{item.epipolar_mean:.4f}",
                "N/A" if item.vertical_mean is None else f"{item.vertical_mean:.4f}",
                item.status,
            ]
            for col, value in enumerate(values):
                self.validation_table.setItem(row, col, QTableWidgetItem(value))
        self._render_rectification_preview()

    def _evidence_summary_text(self, r: StereoCalibrationResult) -> str:
        def fmt(value: object, digits: int = 2) -> str:
            if value is None:
                return "N/A"
            if isinstance(value, (float, int)):
                return f"{float(value):.{digits}f}"
            return str(value)

        capture = r.capture_coach or {}
        sync = r.sync_guard or {}
        audit = r.calibration_audit or {}
        evidence = r.evidence_report or {}
        sync_stats = sync.get("timestamp_delta_ms", {}) if isinstance(sync.get("timestamp_delta_ms"), dict) else {}
        pose = audit.get("cross_camera_pose_consistency", {}) if isinstance(audit.get("cross_camera_pose_consistency"), dict) else {}
        pose_t = pose.get("translation_error_mm", {}) if isinstance(pose.get("translation_error_mm"), dict) else {}
        recon = audit.get("reconstruction", {}) if isinstance(audit.get("reconstruction"), dict) else {}
        recon_point = recon.get("point_to_pose_error_mm", {}) if isinstance(recon.get("point_to_pose_error_mm"), dict) else {}
        uncertainty = audit.get("stability_uncertainty", {}) if isinstance(audit.get("stability_uncertainty"), dict) else {}
        recs = capture.get("recommendations", [])
        warnings = evidence.get("warnings", [])
        rec_text = " / ".join(str(item) for item in recs[:3]) if recs else "No immediate capture recommendation"
        warn_text = " / ".join(str(item) for item in warnings[:3]) if warnings else "No evidence warning"
        return (
            "EVIDENCE REPORT\n"
            f"Confidence: {evidence.get('confidence', 'N/A')} "
            f"({evidence.get('passed_checks', 0)}/{evidence.get('total_checks', 0)} checks)\n"
            f"Dataset quality: {fmt(capture.get('dataset_quality_score'), 1)}%, "
            f"joint coverage: {fmt(capture.get('joint_coverage_score'), 1)}%, "
            f"ready: {capture.get('dataset_ready', False)}\n"
            f"Sync: {sync.get('status', 'N/A')}, "
            f"median/p95/max Δt: {fmt(sync_stats.get('median'))}/"
            f"{fmt(sync_stats.get('p95'))}/{fmt(sync_stats.get('max'))} ms\n"
            f"Pose consistency P95: {fmt(pose_t.get('p95'))} mm, "
            f"3D point RMSE: {fmt(recon_point.get('rmse'))} mm\n"
            f"Baseline 95% CI: {fmt((uncertainty.get('baseline_95ci_mm') or [None, None])[0])} ~ "
            f"{fmt((uncertainty.get('baseline_95ci_mm') or [None, None])[1])} mm\n"
            f"Recommendation: {rec_text}\n"
            f"Warning: {warn_text}\n"
            "Absolute accuracy: external ground truth is required."
        )

    def _evidence_detail_text(self, r: StereoCalibrationResult) -> str:
        return self._evidence_summary_text(r) + "\n\n" + (
            "Export에 포함되는 항목:\n"
            "- Capture Coach: coverage, common corners, recommendations\n"
            "- Sync Guard: timestamp delta statistics, jitter, suspect pairs\n"
            "- Calibration Auditor: epipolar, rectification, hold-out, pose consistency\n"
            "- Stability/Uncertainty: bootstrap baseline CI and R/T repeatability\n"
            "- 3D Geometry: triangulation point error, plane error, board scale error"
        )

    def _render_evidence_cards(self, r: StereoCalibrationResult) -> None:
        def fmt(value: object, suffix: str = "", digits: int = 1) -> str:
            if value is None:
                return "N/A"
            if isinstance(value, (float, int)):
                return f"{float(value):.{digits}f}{suffix}"
            return str(value)

        capture = r.capture_coach or {}
        sync = r.sync_guard or {}
        audit = r.calibration_audit or {}
        evidence = r.evidence_report or {}
        sync_stats = sync.get("timestamp_delta_ms", {}) if isinstance(sync.get("timestamp_delta_ms"), dict) else {}
        pose = audit.get("cross_camera_pose_consistency", {}) if isinstance(audit.get("cross_camera_pose_consistency"), dict) else {}
        pose_t = pose.get("translation_error_mm", {}) if isinstance(pose.get("translation_error_mm"), dict) else {}
        recon = audit.get("reconstruction", {}) if isinstance(audit.get("reconstruction"), dict) else {}
        scale = recon.get("local_board_scale_error_percent", {}) if isinstance(recon.get("local_board_scale_error_percent"), dict) else {}
        uncertainty = audit.get("stability_uncertainty", {}) if isinstance(audit.get("stability_uncertainty"), dict) else {}
        self.evidence_confidence_card.setText(
            f"Confidence\n{evidence.get('confidence', 'N/A')} "
            f"({evidence.get('passed_checks', 0)}/{evidence.get('total_checks', 0)})"
        )
        self.evidence_dataset_card.setText(
            f"Dataset\n{capture.get('usable_pairs', 0)}/{capture.get('target_pairs', 50)} pairs · "
            f"{fmt(capture.get('dataset_quality_score'), '%')}"
        )
        self.evidence_sync_card.setText(
            f"Sync\n{sync.get('status', 'N/A')} · p95 {fmt(sync_stats.get('p95'), ' ms')}"
        )
        self.evidence_geometry_card.setText(
            f"Geometry\npose p95 {fmt(pose_t.get('p95'), ' mm')} · scale p95 {fmt(scale.get('p95'), '%')}\n"
            f"{uncertainty.get('method', 'uncertainty N/A')}"
        )
        warnings = list(evidence.get("warnings", [])) + list(capture.get("recommendations", []))
        self.evidence_warning_label.setText(
            "Warnings / Recommendations\n" + ("\n".join(f"- {item}" for item in warnings[:8]) if warnings else "- 없음")
        )

    def _matrix_detail_text(self, r: StereoCalibrationResult) -> str:
        def mat(name: str, value: np.ndarray) -> str:
            return f"{name}:\n{np.array2string(np.asarray(value), precision=6, suppress_small=True)}"
        return "\n\n".join([
            mat("Camera1 K", r.camera1.camera_matrix),
            mat("Camera1 D", r.camera1.distortion.reshape(1, -1)),
            mat("Camera2 K", r.camera2.camera_matrix),
            mat("Camera2 D", r.camera2.distortion.reshape(1, -1)),
            mat("R_cam2_from_cam1", r.R_cam2_from_cam1),
            mat("t_cam2_from_cam1", r.t_cam2_from_cam1.reshape(3, 1)),
            mat("T_cam2_from_cam1", r.T_cam2_from_cam1),
            mat("T_cam1_from_cam2", r.T_cam1_from_cam2),
        ])

    def _render_rectification_preview(self) -> None:
        if self.result is None:
            self.rectified_cam1_label.setText("Rectified Cam1 preview")
            self.rectified_cam2_label.setText("Rectified Cam2 preview")
            return
        row = self.pair_table.currentRow()
        visible_pairs = self._visible_pairs()
        if row < 0 or row >= len(visible_pairs):
            row = 0 if visible_pairs else -1
        if row < 0:
            return
        pair = visible_pairs[row]
        self._set_rectified_preview(self.rectified_cam1_label, pair.image_path_cam1, camera_index=1)
        self._set_rectified_preview(self.rectified_cam2_label, pair.image_path_cam2, camera_index=2)

    def _set_rectified_preview(self, label: QLabel, path: str | None, *, camera_index: int) -> None:
        if self.result is None or not path:
            label.setText("Rectified preview unavailable")
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            label.setText(f"이미지 로드 실패:\n{path}")
            return
        result = self.result
        size = result.image_size
        if camera_index == 1:
            K, D, R, P = result.camera1.camera_matrix, result.camera1.distortion, result.R1, result.P1
            fisheye = result.camera1.model_name is not None and result.camera1.model_name.value == "fisheye"
        else:
            K, D, R, P = result.camera2.camera_matrix, result.camera2.distortion, result.R2, result.P2
            fisheye = result.camera2.model_name is not None and result.camera2.model_name.value == "fisheye"
        if fisheye:
            map_x, map_y = cv2.fisheye.initUndistortRectifyMap(K, D.reshape(-1, 1), R, P, size, cv2.CV_32FC1)
        else:
            map_x, map_y = cv2.initUndistortRectifyMap(K, D, R, P, size, cv2.CV_32FC1)
        rectified = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        h = rectified.shape[0]
        for y in np.linspace(h * 0.15, h * 0.85, 6):
            yy = int(round(y))
            cv2.line(rectified, (0, yy), (rectified.shape[1] - 1, yy), (0, 220, 220), 1)
        label.setPixmap(self._bgr_to_pixmap(rectified, max_width=460))

    def _export_yaml(self) -> None:
        self._export("yaml")

    def _export_json(self) -> None:
        self._export("json")

    def _export_html(self) -> None:
        self._export("html")

    def _export_kalibr(self) -> None:
        self._export("kalibr")

    def _export(self, kind: str) -> None:
        if self.result is None:
            QMessageBox.warning(self, "Export 불가", "먼저 Stereo Calibration을 실행하세요.")
            return
        suffix = ".yaml" if kind in {"yaml", "kalibr"} else ".html" if kind == "html" else ".json"
        path, _ = QFileDialog.getSaveFileName(self, "Stereo 결과 저장", f"stereo_result{suffix}")
        if not path:
            return
        options = StereoRoboticsExportOptions(
            parent_frame=self.parent_frame_edit.text().strip() or "camera1",
            child_frame=self.child_frame_edit.text().strip() or "camera2",
            rotation_format=self.rotation_format_combo.currentData() or "quaternion",
        )
        if kind == "yaml":
            export_stereo_yaml(self.result, path, robotics_options=options)
        elif kind == "html":
            export_stereo_html(self.result, path, robotics_options=options)
        elif kind == "kalibr":
            export_stereo_kalibr_camchain(self.result, path)
        else:
            export_stereo_json(self.result, path, robotics_options=options)

    def _update_state(self) -> None:
        intrinsics_ready = self.camera1 is not None and self.camera2 is not None
        pairs_ready = bool(self.pairs)
        result_ready = self.result is not None
        done_by_key = {
            "intrinsics": intrinsics_ready,
            "pairs": pairs_ready,
            "result": result_ready,
            "validation": result_ready,
            "rectification": result_ready,
            "evidence": result_ready,
        }
        current_key = self._step_order[self._current_step_index]
        self.step_label.setText(f"{self._step_labels[current_key]} 단계")
        for index, button in enumerate(self.step_buttons):
            key = self._step_order[index]
            done = done_by_key[key]
            is_current = index == self._current_step_index
            base_label = self._step_labels[key]
            button.setText(f"✓ {base_label}" if done else base_label)
            button.setEnabled(index == 0 or intrinsics_ready)
            if done:
                button.setStyleSheet(
                    f"color: {Theme.GOOD}; border-color: {Theme.GOOD}; font-weight: 700;"
                )
            elif is_current:
                button.setStyleSheet(
                    f"color: {Theme.TEXT_VALUE}; border-color: {Theme.ACCENT}; font-weight: 700;"
                )
            else:
                button.setStyleSheet("")
        self.cam1_status.setText(self._calibration_status("Camera 1", self.camera1))
        self.cam2_status.setText(self._calibration_status("Camera 2", self.camera2))
        ready = self.camera1 is not None and self.camera2 is not None
        busy = self._pair_thread is not None or self._calibration_thread is not None
        self.load_pairs_button.setEnabled(ready and not busy)
        self.live_capture_button.setEnabled(not busy)
        self.run_calibration_button.setEnabled(ready and bool(self.pairs) and not busy)
        can_export = self.result is not None
        self.export_yaml_button.setEnabled(can_export)
        self.export_json_button.setEnabled(can_export)
        self.export_html_button.setEnabled(can_export)
        self.export_kalibr_button.setEnabled(can_export)
        self.audit_mode_combo.setEnabled(not busy)
        self.unmatched_preview_button.setEnabled(
            bool(self._last_unmatched_camera1_paths or self._last_unmatched_camera2_paths)
        )
        self.manual_pair_button.setEnabled(
            bool(self._last_unmatched_camera1_paths and self._last_unmatched_camera2_paths)
        )
        self.step_back_button.setEnabled(self._current_step_index > 0)
        can_advance = self._current_step_index < len(self._step_order) - 1
        if self._current_step_index == 0:
            can_advance = can_advance and ready
        elif self._current_step_index == 1:
            can_advance = can_advance and pairs_ready
        elif self._current_step_index >= 2:
            can_advance = can_advance and result_ready
        self.step_next_button.setEnabled(can_advance)

    def _calibration_status(self, label: str, cal: StandardCalibration | None) -> str:
        if cal is None:
            return f"{label}: Intrinsic not ready"
        model = cal.model_name.value if cal.model_name else "?"
        coeffs = ", ".join(f"{v:.4g}" for v in np.asarray(cal.distortion).reshape(-1).tolist())
        return (
            f"✓ {label} Intrinsic Ready\n"
            f"Resolution: {cal.width or '?'} × {cal.height or '?'}\n"
            f"Model: {model}\n"
            f"fx={cal.fx:.3f}, fy={cal.fy:.3f}, cx={cal.cx:.3f}, cy={cal.cy:.3f}\n"
            f"Distortion: {coeffs or 'N/A'}"
        )
