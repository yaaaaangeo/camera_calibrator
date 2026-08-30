"""
tests/test_camera_lidar_workspace_diagnostics.py
====================================================

Regression test for the GUIDED ROI diagnostics lifecycle hardening
(ui/camera_lidar_workspace.py): a capture worker's result diagnostics must
describe the ROI mode that worker actually ran with
(self._active_capture_roi_mode), not whatever the live ROI Mode combo box
has been changed to in the meantime.

Failure scenario this guards against:
    Worker starts with ROI mode = GUIDED
    User changes the ROI Mode combo box to AUTO while the worker still runs
    Worker finishes -- the computation was GUIDED throughout
    Diagnostics incorrectly display "ROI Mode: AUTO"
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="PySide6가 설치되어 있지 않음")

from PySide6.QtWidgets import QApplication

import numpy as np

from camera_lidar.target_config import TargetConfig
from camera_lidar.types import (
    CalibrationScene,
    CameraLidarCalibrationResult,
    FailureReason,
    ImageFrame,
    PointCloudFrame,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _dummy_scene() -> CalibrationScene:
    target = TargetConfig()
    image = ImageFrame(timestamp=0.0, image=np.zeros((4, 4, 3), dtype=np.uint8))
    cloud = PointCloudFrame(timestamp=0.0, points=np.zeros((10, 3)))
    return CalibrationScene(image=image, cloud=cloud, intrinsics=None, target=target)


def test_format_diagnostics_uses_explicit_roi_mode_over_live_combo(qapp):
    from ui.camera_lidar_workspace import CameraLidarWorkspace

    workspace = CameraLidarWorkspace()
    # Live combo box says AUTO...
    auto_index = workspace.roi_mode_combo.findData("auto")
    workspace.roi_mode_combo.setCurrentIndex(auto_index)
    assert workspace._current_roi_mode() == "auto"

    result = CameraLidarCalibrationResult(success=False, failure_reason=FailureReason.LIDAR_PLANE_NOT_FOUND)

    # ...but an explicit roi_mode="guided" must win.
    text = workspace._format_diagnostics(result, roi_mode="guided")
    assert "ROI Mode: GUIDED" in text
    assert "ROI Mode: AUTO" not in text

    # No override -> falls back to the live combo box (unchanged behavior
    # for other callers that don't pass roi_mode).
    text_default = workspace._format_diagnostics(result)
    assert "ROI Mode: AUTO" in text_default


def test_capture_result_diagnostics_reflect_worker_roi_mode_not_live_combo(qapp):
    from ui.camera_lidar_workspace import CameraLidarWorkspace

    workspace = CameraLidarWorkspace()

    # Worker started with GUIDED.
    workspace._active_capture_scene = _dummy_scene()
    workspace._active_capture_roi_mode = "guided"

    # User changes the live combo to AUTO while the worker is still "running".
    auto_index = workspace.roi_mode_combo.findData("auto")
    assert auto_index >= 0
    workspace.roi_mode_combo.setCurrentIndex(auto_index)
    assert workspace._current_roi_mode() == "auto"

    result = CameraLidarCalibrationResult(success=False, failure_reason=FailureReason.LIDAR_PLANE_NOT_FOUND)
    workspace._on_capture_result_ready(result)

    diagnostics_text = workspace.diagnostics_text.toPlainText()
    assert "ROI Mode: GUIDED" in diagnostics_text
    assert "ROI Mode: AUTO" not in diagnostics_text

    # The committed CapturedScene must also record "guided", matching what
    # the worker actually ran, not the combo box's current value.
    assert workspace.captured_scenes[-1].roi_mode == "guided"


def test_view_scene_diagnostics_use_captured_roi_mode(qapp):
    from ui.camera_lidar_workspace import CameraLidarWorkspace
    from camera_lidar.types import CapturedScene

    workspace = CameraLidarWorkspace()

    # A previously captured GUIDED scene...
    result = CameraLidarCalibrationResult(success=False, failure_reason=FailureReason.LIDAR_PLANE_NOT_FOUND)
    captured = CapturedScene(scene_id="scene_01", scene=_dummy_scene(), roi_mode="guided", detection=result)
    workspace.captured_scenes.append(captured)
    workspace._refresh_scene_table()
    workspace.scene_table.selectRow(0)

    # ...viewed while the live combo box currently shows AUTO.
    auto_index = workspace.roi_mode_combo.findData("auto")
    workspace.roi_mode_combo.setCurrentIndex(auto_index)

    workspace._on_view_scene()

    diagnostics_text = workspace.diagnostics_text.toPlainText()
    assert "ROI Mode: GUIDED" in diagnostics_text
    assert "ROI Mode: AUTO" not in diagnostics_text
