"""
Scene Browser for the Camera-LiDAR FAST-Calib workspace: the MARKER
EXTRACTION button scans a whole bag's camera topic (camera_lidar.
scene_extraction.build_scene_candidates, via
calibration.camera_lidar_controller.CameraLidarController.
extract_scene_candidates) and shows every discovered SceneCandidate as a
filterable, checkbox-selectable table row.

Design constraint carried over from the feature spec: extraction and
selection are two separate concerns. The Filter combo (ALL / 4 MARKERS /
3 MARKERS) only controls which rows are VISIBLE (QTableWidget.setRowHidden)
-- it never touches SceneCandidate.is_selected, which lives on the
candidate object itself and therefore survives filter changes untouched.
Only candidates the user explicitly checks (or "Select All Visible") are
emitted via add_selected_requested when "ADD SELECTED TO SCENE MANAGER" is
clicked -- the tool only proposes candidates, the user makes the final call.
"""

from __future__ import annotations

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from camera_lidar.camera_detector import CameraDetectionResult, diagnose_dictionaries, render_marker_overlay
from camera_lidar.extraction_diagnostics import ExtractionDiagnosticSummary, format_extraction_diagnostics
from camera_lidar.target_config import CORNER_ORDER
from camera_lidar.types import SceneCandidate, SceneType

_FILTER_ALL = "all"
_FILTER_FULL = "full"
_FILTER_PARTIAL = "partial"

_TYPE_LABEL = {
    SceneType.VALID_FULL: "FULL (4/4)",
    SceneType.VALID_PARTIAL: "PARTIAL (3/4)",
}
_CORNER_ABBREV = {
    "top_left": "TL", "top_right": "TR", "bottom_right": "BR", "bottom_left": "BL",
}
_THUMB_SIZE = 64


def _to_pixmap(image, size: int | None = None) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimage = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage.copy())
    if size is not None:
        pixmap = pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pixmap


class CameraLidarSceneBrowser(QGroupBox):
    extraction_requested = Signal()
    # object, not list: PySide's meta-type marshalling for a typed `list`
    # signal tries to convert each element to a QVariant-compatible type,
    # which corrupts/double-frees these plain Python SceneCandidate dataclass
    # instances at interpreter teardown (segfault, reproduced empirically).
    # Every other signal in this codebase that carries a Python container of
    # custom objects (e.g. ui.worker's `result_ready = Signal(object)`) uses
    # `object` for the same reason -- this was the one inconsistent spot.
    add_selected_requested = Signal(object)  # list[SceneCandidate]
    test_current_frame_requested = Signal()
    apply_dictionary_requested = Signal(str)  # dictionary name, e.g. "DICT_6X6_50" -- emitted only on explicit user click
    cancel_extraction_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__("Scene Browser (MARKER EXTRACTION → filter → manual selection)", parent)
        self.candidates: list[SceneCandidate] = []
        self._refreshing = False  # guards itemChanged while _refresh_table() rebuilds rows programmatically

        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.extract_button = QPushButton("MARKER EXTRACTION")
        self.extract_button.setProperty("role", "primary")
        self.extract_button.clicked.connect(self.extraction_requested.emit)
        top_row.addWidget(self.extract_button)
        self.cancel_button = QPushButton("CANCEL")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        top_row.addWidget(self.cancel_button)
        self.test_frame_button = QPushButton("TEST CURRENT FRAME")
        self.test_frame_button.clicked.connect(self.test_current_frame_requested.emit)
        top_row.addWidget(self.test_frame_button)
        self.summary_label = QLabel("Total 0 · Full 0 · Partial 0")
        top_row.addWidget(self.summary_label, stretch=1)
        layout.addLayout(top_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("ALL", _FILTER_ALL)
        self.filter_combo.addItem("4 MARKERS (FULL)", _FILTER_FULL)
        self.filter_combo.addItem("3 MARKERS (PARTIAL)", _FILTER_PARTIAL)
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)
        filter_row.addWidget(self.filter_combo)
        select_all_button = QPushButton("Select All Visible")
        select_all_button.clicked.connect(self._on_select_all_visible)
        filter_row.addWidget(select_all_button)
        clear_button = QPushButton("Clear Selection")
        clear_button.clicked.connect(self._on_clear_selection)
        filter_row.addWidget(clear_button)
        filter_row.addStretch(1)
        layout.addLayout(filter_row)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Use", "Preview", "Type", "Detected", "Missing", "Time (s)", "Quality"]
        )
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemDoubleClicked.connect(lambda item: self._show_detail(item.row()))
        layout.addWidget(self.table)

        bottom_row = QHBoxLayout()
        view_button = QPushButton("View")
        view_button.clicked.connect(lambda: self._show_detail(self.table.currentRow()))
        bottom_row.addWidget(view_button)
        self.selected_summary_label = QLabel("Selected: 0 — Full: 0, Partial: 0")
        bottom_row.addWidget(self.selected_summary_label, stretch=1)
        self.add_selected_button = QPushButton("ADD SELECTED TO SCENE MANAGER")
        self.add_selected_button.setProperty("role", "primary")
        self.add_selected_button.setEnabled(False)
        self.add_selected_button.clicked.connect(self._on_add_selected)
        bottom_row.addWidget(self.add_selected_button)
        layout.addLayout(bottom_row)

        layout.addWidget(QLabel("EXTRACTION DIAGNOSTICS (where/why the funnel emptied out)"))
        self.diagnostics_text = QPlainTextEdit()
        self.diagnostics_text.setReadOnly(True)
        self.diagnostics_text.setMinimumHeight(220)
        self.diagnostics_text.setPlainText("Run MARKER EXTRACTION to see the stage-by-stage funnel here.")
        layout.addWidget(self.diagnostics_text)

    # ------------------------------------------------------------------
    # Extraction lifecycle
    # ------------------------------------------------------------------

    def set_extracting(self, extracting: bool) -> None:
        self.extract_button.setEnabled(not extracting)
        self.extract_button.setText("EXTRACTING..." if extracting else "MARKER EXTRACTION")
        self.cancel_button.setVisible(extracting)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("CANCEL")
        self.progress_bar.setVisible(extracting)
        self.status_label.setVisible(extracting)
        if extracting:
            self.progress_bar.setRange(0, 0)  # busy indicator until the first (done, total) arrives
            self.status_label.setText("Starting...")
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)

    def _on_cancel_clicked(self) -> None:
        # No further clicks needed/useful once cancellation has been
        # requested -- the worker checks cancel_check between frames and can
        # take a moment to actually stop, so give immediate feedback rather
        # than leaving the button looking unresponsive.
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("CANCELLING...")
        self.cancel_extraction_requested.emit()

    def set_progress_text(self, message: str) -> None:
        self.status_label.setText(message)

    def set_progress_value(self, done: int, total: int) -> None:
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(done)
        else:
            self.progress_bar.setRange(0, 0)  # total unknown -- stay a busy indicator

    def set_candidates(self, candidates: list) -> None:
        """Replaces the full candidate list and rebuilds the table.
        Re-used both after a fresh extraction and after committing an
        "ADD SELECTED" batch (to refresh the Selected/Add-button state)."""
        self.candidates = candidates
        self._refresh_table()

    def set_diagnostics_summary(self, summary: ExtractionDiagnosticSummary) -> None:
        """Populated after every MARKER EXTRACTION run (0 candidates or
        many) -- see camera_lidar.extraction_diagnostics.
        format_extraction_diagnostics for the funnel + auto-diagnosis text."""
        self.diagnostics_text.setPlainText(format_extraction_diagnostics(summary))

    # ------------------------------------------------------------------
    # Table rebuild / filter (view-only) / selection (data-only)
    # ------------------------------------------------------------------

    def _refresh_table(self) -> None:
        self._refreshing = True
        self.table.setRowCount(len(self.candidates))
        for row, candidate in enumerate(self.candidates):
            check_item = QTableWidgetItem()
            check_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            check_item.setCheckState(Qt.Checked if candidate.is_selected else Qt.Unchecked)
            self.table.setItem(row, 0, check_item)

            preview_label = QLabel()
            preview_label.setAlignment(Qt.AlignCenter)
            if candidate.image is not None:
                preview_label.setPixmap(_to_pixmap(candidate.image, _THUMB_SIZE))
            self.table.setCellWidget(row, 1, preview_label)

            detected_text = ", ".join(_CORNER_ABBREV[c] for c in CORNER_ORDER if c in candidate.detected_ids)
            missing_text = ", ".join(_CORNER_ABBREV[c] for c in CORNER_ORDER if c in candidate.missing_ids) or "-"
            cells = [
                _TYPE_LABEL.get(candidate.scene_type, "-"),
                detected_text,
                missing_text,
                f"{candidate.representative_timestamp_s:.2f}",
                f"{candidate.quality_score:.1f}",
            ]
            for col, text in enumerate(cells, start=2):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, col, item)
            self.table.setRowHeight(row, _THUMB_SIZE + 4)
        self._refreshing = False
        self._apply_filter()
        self._update_summaries()

    def _apply_filter(self) -> None:
        current_filter = self.filter_combo.currentData()
        for row, candidate in enumerate(self.candidates):
            visible = (
                current_filter == _FILTER_ALL
                or (current_filter == _FILTER_FULL and candidate.scene_type == SceneType.VALID_FULL)
                or (current_filter == _FILTER_PARTIAL and candidate.scene_type == SceneType.VALID_PARTIAL)
            )
            self.table.setRowHidden(row, not visible)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._refreshing or item.column() != 0:
            return
        row = item.row()
        if 0 <= row < len(self.candidates):
            self.candidates[row].is_selected = item.checkState() == Qt.Checked
            self._update_summaries()

    def _on_select_all_visible(self) -> None:
        for row, candidate in enumerate(self.candidates):
            if not self.table.isRowHidden(row):
                candidate.is_selected = True
        self._refresh_table()

    def _on_clear_selection(self) -> None:
        for candidate in self.candidates:
            candidate.is_selected = False
        self._refresh_table()

    def _update_summaries(self) -> None:
        total = len(self.candidates)
        full = sum(1 for c in self.candidates if c.scene_type == SceneType.VALID_FULL)
        partial = sum(1 for c in self.candidates if c.scene_type == SceneType.VALID_PARTIAL)
        self.summary_label.setText(f"Total {total} · Full {full} · Partial {partial}")

        selected = [c for c in self.candidates if c.is_selected]
        sel_full = sum(1 for c in selected if c.scene_type == SceneType.VALID_FULL)
        sel_partial = sum(1 for c in selected if c.scene_type == SceneType.VALID_PARTIAL)
        self.selected_summary_label.setText(
            f"Selected: {len(selected)} — Full: {sel_full}, Partial: {sel_partial}"
        )
        self.add_selected_button.setEnabled(len(selected) > 0)

    # ------------------------------------------------------------------
    # Detail dialog
    # ------------------------------------------------------------------

    def _show_detail(self, row: int) -> None:
        if row < 0 or row >= len(self.candidates):
            return
        candidate = self.candidates[row]
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Scene Candidate — {candidate.candidate_id}")
        layout = QVBoxLayout(dialog)

        if candidate.image is not None:
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setPixmap(_to_pixmap(candidate.image).scaledToWidth(480, Qt.SmoothTransformation))
            layout.addWidget(image_label)

        lines = [
            f"Segment: {candidate.segment_start_s:.2f}s → {candidate.segment_end_s:.2f}s "
            f"(representative t = {candidate.representative_timestamp_s:.2f}s)",
            f"Camera topic: {candidate.camera_topic}",
            f"LiDAR topic: {candidate.lidar_topic}",
            f"Type: {_TYPE_LABEL.get(candidate.scene_type, '-')}",
            f"Quality score: {candidate.quality_score:.1f}",
            "",
            "MARKER CHECKLIST (expected ids)",
        ]
        for cid in CORNER_ORDER:
            mark = "✓" if cid in candidate.detected_ids else "✕"
            lines.append(f"  {cid:14s} {mark}")
        lines.append("")
        if candidate.cloud_points is not None:
            sync_text = ""
            if candidate.cloud_timestamp_s is not None:
                sync_ms = abs(candidate.representative_timestamp_s - candidate.cloud_timestamp_s) * 1000.0
                sync_text = f"  (Δt = {sync_ms:.1f} ms)"
            lines.append(f"LiDAR pairing: {candidate.cloud_points.shape[0]} points{sync_text}")
        else:
            lines.append("LiDAR pairing: deferred until ADD SELECTED")

        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText("\n".join(lines))
        text.setMinimumHeight(200)
        layout.addWidget(text)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        dialog.exec()

    # ------------------------------------------------------------------
    # TEST CURRENT FRAME -- pure ArUco detector test, no Scene Extraction
    # ------------------------------------------------------------------

    def show_detector_test_result(self, image, result: CameraDetectionResult, expected_ids: list[int]) -> None:
        """Shows the raw ArUco detector's output on ONE frame, independent
        of Scene Extraction entirely -- the answer to "is the detector even
        working" (spec section 12-13). Raw detection is shown BEFORE/
        separate from the expected-id filter, per the feature's core
        requirement (never show only the post-filter result)."""
        dialog = QDialog(self)
        dialog.setWindowTitle("ArUco Detector Test — TEST CURRENT FRAME")
        layout = QVBoxLayout(dialog)

        overlay = render_marker_overlay(image, result)
        overlay_label = QLabel()
        overlay_label.setAlignment(Qt.AlignCenter)
        overlay_label.setPixmap(_to_pixmap(overlay).scaledToWidth(480, Qt.SmoothTransformation))
        layout.addWidget(overlay_label)

        h, w = image.shape[:2]
        raw_ids = ", ".join(str(i) for i in result.detected_marker_ids) or "(none)"
        expected_text = ", ".join(str(i) for i in sorted(expected_ids)) or "(none)"
        matched_text = ", ".join(str(i) for i in result.matched_marker_ids) or "(none)"
        missing_text = ", ".join(str(i) for i in result.missing_marker_ids) or "(none)"
        lines = [
            "ARUCO DETECTOR TEST",
            "-" * 24,
            "",
            "Image Decode          PASS",
            f"Resolution            {w} x {h}",
            f"Dictionary            {result.dictionary_name or '(unknown)'}",
            "",
            "RAW DETECTION (before expected-id filtering)",
            f"  Detected            {len(result.detected_marker_ids)}",
            f"  Raw IDs             {raw_ids}",
            f"  Rejected Candidates {result.rejected_candidate_count}",
            "",
            f"Expected IDs          {expected_text}",
            f"Matched IDs           {matched_text}",
            f"Missing IDs           {missing_text}",
            "",
            f"Detector Status       {'PASS' if result.success else 'FAIL'}",
        ]

        if len(result.detected_marker_ids) == 0:
            if result.rejected_candidate_count == 0:
                lines += [
                    "", "Marker-like quadrilateral candidates were not found.",
                    "Possible causes: marker too small, severe blur, exposure issue,",
                    "image decoding issue, strong distortion, target outside FOV.",
                ]
            else:
                lines += [
                    "", "Marker-like candidates were found, but marker decoding failed.",
                    "Possible causes: wrong dictionary, damaged marker border,",
                    "low resolution, motion blur, strong perspective, image rescaling.",
                ]
        elif len(result.matched_marker_ids) == 0:
            lines += [
                "", "TARGET ID MISMATCH",
                "ArUco detection is working, but detected IDs do not match",
                "the configured FAST-Calib target.",
                f"  Detected: {raw_ids}",
                f"  Expected: {expected_text}",
            ]

        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText("\n".join(lines))
        text.setMinimumHeight(260)
        layout.addWidget(text)

        # Dictionary diagnostic runs automatically as part of this same test
        # (it's just a handful of detectMarkers() calls on one already-in-
        # memory image, cheap enough not to need a separate button gating
        # it -- a click sitting between the detector-test text and the
        # dictionary results made the dialog read awkwardly).
        dict_results = diagnose_dictionaries(image)
        has_best = bool(dict_results) and dict_results[0][1] > 0
        best_dictionary = dict_results[0][0] if has_best else None
        dict_lines = ["DICTIONARY DIAGNOSTIC", "-" * 24, ""]
        for name, count in dict_results:
            dict_lines.append(f"{name:16s} {count} markers")
        dict_lines += [
            "", f"Best Candidate: {best_dictionary or '(none found any markers)'}",
        ]
        dict_results_text = QPlainTextEdit()
        dict_results_text.setReadOnly(True)
        dict_results_text.setPlainText("\n".join(dict_lines))
        dict_results_text.setMinimumHeight(160)
        layout.addWidget(dict_results_text)

        apply_button = QPushButton(f"APPLY BEST CANDIDATE ({best_dictionary})" if has_best else "APPLY BEST CANDIDATE")
        apply_button.setVisible(has_best)

        def _on_apply_best_candidate() -> None:
            self.apply_dictionary_requested.emit(best_dictionary)
            apply_button.setText(f"APPLIED: {best_dictionary}")
            apply_button.setEnabled(False)

        apply_button.clicked.connect(_on_apply_best_candidate)
        layout.addWidget(apply_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        dialog.exec()

    # ------------------------------------------------------------------
    # ADD SELECTED TO SCENE MANAGER
    # ------------------------------------------------------------------

    def _on_add_selected(self) -> None:
        selected = [c for c in self.candidates if c.is_selected]
        if selected:
            self.add_selected_requested.emit(selected)
