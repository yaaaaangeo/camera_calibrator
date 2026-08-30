"""Live Calibration Preview overlay and bounded worker regression tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from calibration.detector import build_detect_fn, maximum_pattern_corners
from calibration.types import DetectionResult, PatternConfig, PatternType


def _charuco_pattern() -> PatternConfig:
    return PatternConfig(
        type=PatternType.CHARUCO,
        squares_x=14,
        squares_y=9,
        square_size=0.04,
        marker_size=0.03,
        dictionary="DICT_5X5_100",
    )


def _detected_charuco() -> tuple[np.ndarray, DetectionResult]:
    pattern = _charuco_pattern()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (pattern.squares_x, pattern.squares_y),
        pattern.square_size,
        pattern.marker_size,
        dictionary,
    )
    gray = board.generateImage((1000, 650), marginSize=30, borderBits=1)
    raw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    detection = build_detect_fn(pattern)(raw, "live_test")
    assert detection.success
    return raw, detection


def test_charuco_detection_success_renders_overlay():
    pytest.importorskip("PySide6")
    from ui.live_capture_dialog import render_detection_overlay

    raw, detection = _detected_charuco()
    overlay = render_detection_overlay(raw, detection, maximum_pattern_corners(_charuco_pattern()))

    assert overlay.shape == raw.shape
    assert not np.array_equal(overlay, raw)
    # NVIDIA-green corner pixels are introduced by the renderer.
    assert np.any((overlay[:, :, 1] > overlay[:, :, 0]) & (overlay[:, :, 1] > overlay[:, :, 2]))


def test_detection_failure_returns_unchanged_raw_copy():
    pytest.importorskip("PySide6")
    from ui.live_capture_dialog import render_detection_overlay

    raw = np.full((80, 120, 3), 37, dtype=np.uint8)
    failed = DetectionResult(image_id="failed", success=False, num_corners=0)
    overlay = render_detection_overlay(raw, failed, 104)

    assert np.array_equal(overlay, raw)
    assert not np.shares_memory(overlay, raw)


def test_maximum_charuco_corner_count_is_board_internal_intersections():
    assert maximum_pattern_corners(_charuco_pattern()) == 13 * 8 == 104


def test_overlay_never_mutates_original_raw_frame():
    pytest.importorskip("PySide6")
    from ui.live_capture_dialog import render_detection_overlay

    raw, detection = _detected_charuco()
    before = raw.copy()
    overlay = render_detection_overlay(raw, detection, 104, show_ids=True)

    assert np.array_equal(raw, before)
    assert not np.shares_memory(overlay, raw)


def test_live_detection_worker_keeps_only_one_pending_frame():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from ui.worker import LiveDetectionWorker

    _app = QApplication.instance() or QApplication([])
    worker = LiveDetectionWorker(_charuco_pattern())
    for index in range(100):
        worker.submit_frame(np.full((8, 8, 3), index, dtype=np.uint8), float(index))

    stats = worker.buffer_stats()
    worker.request_stop()

    assert stats.received == 100
    assert stats.replaced == 99
    assert stats.delivered == 0


def test_live_dialog_accepts_legacy_string_pattern_type(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    import ui.live_capture_dialog as live_module

    _app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(live_module, "ROS_LIVE_BACKEND", None)
    pattern = _charuco_pattern()
    pattern.type = "charuco"  # 구버전 project/external caller 형태

    dialog = live_module.LiveCaptureDialog(str(tmp_path), pattern_config=pattern)
    try:
        assert dialog._maximum_corners == 104
        assert "104" in dialog.detection_status_label.text()
        label_texts = [label.text() for label in dialog.findChildren(live_module.QLabel)]
        assert "ChArUco" in label_texts
    finally:
        dialog.close()


def test_live_dialog_auto_captures_detected_novel_pose_only(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from calibration.types import CameraConfig
    import ui.live_capture_dialog as live_module

    _app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(live_module, "ROS_LIVE_BACKEND", None)
    raw, detection = _detected_charuco()
    dialog = live_module.LiveCaptureDialog(
        str(tmp_path),
        pattern_config=_charuco_pattern(),
        camera_config=CameraConfig(width=raw.shape[1], height=raw.shape[0]),
    )
    try:
        dialog._live_detection_active = True
        dialog._on_live_detection_frame_ready(raw, detection)
        assert len(dialog.captured_paths) == 1
        assert dialog.captured_image_size == (raw.shape[1], raw.shape[0])

        # 시간 간격이 지났어도 같은 자세는 중복 자동 저장하지 않는다.
        dialog._last_auto_capture_t = 0.0
        dialog._on_live_detection_frame_ready(raw, detection)
        assert len(dialog.captured_paths) == 1
        assert "같은 자세" in dialog.count_label.text()
    finally:
        dialog._live_detection_active = False
        dialog.close()


def test_live_dialog_is_resizable_and_maximizable(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    import ui.live_capture_dialog as live_module

    _app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(live_module, "ROS_LIVE_BACKEND", None)
    dialog = live_module.LiveCaptureDialog(str(tmp_path), pattern_config=_charuco_pattern())
    try:
        assert dialog.windowFlags() & Qt.WindowMaximizeButtonHint
        assert dialog.isSizeGripEnabled()
        before = dialog.size()
        dialog.resize(before.width() + 200, before.height() + 100)
        assert dialog.width() > before.width()
        assert dialog.height() > before.height()
    finally:
        dialog.close()
