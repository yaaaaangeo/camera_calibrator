"""
tests/test_direct_visual_bootstrap_ui.py
============================================

UI-level regression tests for the Targetless Bootstrap panel
(ui/camera_lidar_workspace.py): RUN/CANCEL button enable-state lifecycle,
and that success/failure/cancel route through _apply_targetless_prior()
(success only) without disturbing 1st-stage manual-load functionality.

Note: like tests/test_camera_lidar_workspace_diagnostics.py, this needs a
working PySide6 QtWidgets import -- environments where that fails (a
pre-existing, unrelated Windows DLL issue on some dev machines) will error
at collection despite pytest.importorskip("PySide6") succeeding, since
importorskip only checks the top-level `PySide6` package, not the
QtWidgets native extension specifically.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="PySide6가 설치되어 있지 않음")

from PySide6.QtWidgets import QApplication

from camera_lidar.types import TargetlessPrior
from integrations.direct_visual_runner import DirectVisualFailureReason


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_workspace():
    from ui.camera_lidar_workspace import CameraLidarWorkspace

    return CameraLidarWorkspace()


def test_run_button_disabled_on_non_linux(qapp, monkeypatch):
    import ui.camera_lidar_workspace as workspace_module
    monkeypatch.setattr(workspace_module.platform, "system", lambda: "Windows")

    workspace = _make_workspace()
    assert workspace._bootstrap_is_linux is False
    assert workspace.bootstrap_run_button.isEnabled() is False


def test_run_button_enabled_on_linux(qapp, monkeypatch):
    import ui.camera_lidar_workspace as workspace_module
    monkeypatch.setattr(workspace_module.platform, "system", lambda: "Linux")

    workspace = _make_workspace()
    assert workspace._bootstrap_is_linux is True
    assert workspace.bootstrap_run_button.isEnabled() is True


def test_busy_state_disables_run_and_shows_cancel(qapp):
    workspace = _make_workspace()
    workspace._bootstrap_is_linux = True
    workspace.bootstrap_run_button.setEnabled(True)

    workspace._set_bootstrap_busy(True)
    assert workspace.bootstrap_run_button.isEnabled() is False
    assert workspace.bootstrap_cancel_button.isVisible() is True

    workspace._set_bootstrap_busy(False)
    assert workspace.bootstrap_run_button.isEnabled() is True
    assert workspace.bootstrap_cancel_button.isVisible() is False


def test_succeeded_applies_prior_via_shared_path(qapp):
    workspace = _make_workspace()
    assert workspace.targetless_prior is None

    prior = TargetlessPrior(T_lidar_from_camera=__import__("numpy").eye(4), source_path="calib.json", source_key="init_T_lidar_camera_auto")
    workspace._on_bootstrap_succeeded(prior)

    assert workspace.targetless_prior is prior
    assert workspace.prior_status_label.text() == "READY"
    # Shared with manual load: GUIDED AUTO gets auto-selected.
    assert workspace._current_roi_mode() == "guided"


def test_failed_does_not_clear_existing_prior(qapp):
    workspace = _make_workspace()
    prior = TargetlessPrior(T_lidar_from_camera=__import__("numpy").eye(4), source_path="calib.json", source_key="T_lidar_camera")
    workspace._apply_targetless_prior(prior)
    assert workspace.targetless_prior is prior

    workspace._on_bootstrap_failed(DirectVisualFailureReason.PREPROCESS_FAILED.value, "synthetic failure output")

    assert workspace.targetless_prior is prior  # untouched


def test_cancelled_does_not_clear_existing_prior(qapp):
    workspace = _make_workspace()
    prior = TargetlessPrior(T_lidar_from_camera=__import__("numpy").eye(4), source_path="calib.json", source_key="T_lidar_camera")
    workspace._apply_targetless_prior(prior)

    workspace._on_bootstrap_cancelled()

    assert workspace.targetless_prior is prior  # untouched


def test_use_bag_topics_button_copies_from_bag_source(qapp):
    workspace = _make_workspace()
    workspace.bag_source.camera_topic_combo.addItem("/camera/image_raw", "/camera/image_raw")
    workspace.bag_source.lidar_topic_combo.addItem("/points_raw", "/points_raw")
    workspace.bag_source.camera_topic_combo.setCurrentIndex(workspace.bag_source.camera_topic_combo.count() - 1)
    workspace.bag_source.lidar_topic_combo.setCurrentIndex(workspace.bag_source.lidar_topic_combo.count() - 1)

    workspace._on_use_bag_topics_for_bootstrap()

    assert workspace.bootstrap_image_topic_edit.text() == "/camera/image_raw"
    assert workspace.bootstrap_points_topic_edit.text() == "/points_raw"
