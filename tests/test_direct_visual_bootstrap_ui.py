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

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="PySide6가 설치되어 있지 않음")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from camera_lidar.types import TargetlessPrior
from integrations.direct_visual_runner import DirectVisualBootstrapResult, DirectVisualFailureReason, RunnerStage


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_workspace():
    from ui.camera_lidar_workspace import CameraLidarWorkspace

    return CameraLidarWorkspace()


class _SyncThreadStub(QObject):
    """Stand-in for run_worker_in_thread's real QThread: runs the worker's
    run() synchronously on the calling (test) thread instead of moving it
    to a real QThread, so a RUN-button-driven test can assert on the
    resulting state immediately after calling _on_run_targetless_bootstrap()
    without needing to pump a real Qt event loop across threads. The
    worker's signal emissions still reach the workspace's connected slots
    (Qt delivers same-thread signals via a direct synchronous call)."""
    finished = Signal()

    def __init__(self, worker):
        super().__init__()
        self._worker = worker

    def start(self) -> None:
        self._worker.run()
        self.finished.emit()


def _prepare_workspace_for_run(workspace, monkeypatch, fake_pipeline) -> None:
    import ui.camera_lidar_workspace as workspace_module
    import ui.worker as worker_module

    workspace._bootstrap_is_linux = True
    workspace.bootstrap_run_button.setEnabled(True)
    workspace.bootstrap_bag_edit.setText(str(workspace_module.__file__))  # any isdir-passing override below
    workspace.bootstrap_output_edit.setText("/tmp/out")
    workspace._set_combo_free_text(workspace.bootstrap_image_topic_combo, "/camera/image_raw")
    workspace._set_combo_free_text(workspace.bootstrap_points_topic_combo, "/points_raw")

    class _FakeIntrinsics:
        camera_matrix = np.eye(3)
        distortion = np.zeros(5)

    workspace.intrinsics = _FakeIntrinsics()

    monkeypatch.setattr("os.path.isdir", lambda _p: True)
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(worker_module, "run_direct_visual_pipeline", fake_pipeline)
    monkeypatch.setattr(workspace_module, "run_worker_in_thread", lambda worker, parent: _SyncThreadStub(worker))


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

    # Matches the real invariant: _on_run_targetless_bootstrap() always sets
    # self._bootstrap_worker BEFORE calling _set_bootstrap_busy(True) (and
    # clears it BEFORE _set_bootstrap_busy(False)) -- _update_bootstrap_run_
    # button_enabled() derives "busy" from that, not from a separate flag.
    workspace._bootstrap_worker = object()
    workspace._set_bootstrap_busy(True)
    assert workspace.bootstrap_run_button.isEnabled() is False
    assert workspace.bootstrap_cancel_button.isVisible() is True

    workspace._bootstrap_worker = None
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


# ---------------------------------------------------------------------------
# §31-33: full RUN-button-driven integration -- a fake run_direct_visual_pipeline
# stands in for the real external-process pipeline (already covered
# end-to-end by tests/test_direct_visual_runner.py), so these tests
# exercise the actual wiring: button click -> DirectVisualConfig ->
# DirectVisualBootstrapWorker -> run_worker_in_thread -> signals ->
# _apply_targetless_prior() / prior preservation / button re-enable.
# ---------------------------------------------------------------------------

def test_run_button_coarse_success_applies_prior_and_switches_to_guided(qapp, monkeypatch):
    fake_prior = TargetlessPrior(
        T_lidar_from_camera=np.eye(4), source_path="/tmp/out/calib.json", source_key="init_T_lidar_camera_auto",
    )

    def fake_pipeline(config, on_progress=None, on_log=None, on_stage_started=None, on_stage_finished=None, cancel_check=None):
        for stage in (RunnerStage.PREPROCESS, RunnerStage.MATCHING, RunnerStage.INITIAL_GUESS):
            on_progress and on_progress(f"Stage: {stage.value}")
            on_stage_started and on_stage_started(stage)
            on_log and on_log(f"{stage.value} ok")
            on_stage_finished and on_stage_finished(stage)
        return DirectVisualBootstrapResult(success=True, prior=fake_prior)

    workspace = _make_workspace()
    _prepare_workspace_for_run(workspace, monkeypatch, fake_pipeline)
    assert workspace.targetless_prior is None

    workspace._on_run_targetless_bootstrap()

    assert workspace.targetless_prior is fake_prior
    assert workspace._current_roi_mode() == "guided"
    assert workspace.prior_status_label.text() == "READY"
    assert workspace.bootstrap_status_label.text() == "DONE"
    # RUN re-enabled, CANCEL hidden again, after the (synchronous) run finished.
    assert workspace.bootstrap_run_button.isEnabled() is True
    assert workspace.bootstrap_cancel_button.isVisible() is False


def test_run_button_failure_preserves_existing_prior(qapp, monkeypatch):
    existing_prior = TargetlessPrior(
        T_lidar_from_camera=np.eye(4), source_path="/prev/calib.json", source_key="T_lidar_camera",
    )

    def fake_pipeline(config, on_progress=None, on_log=None, on_stage_started=None, on_stage_finished=None, cancel_check=None):
        on_stage_started and on_stage_started(RunnerStage.PREPROCESS)
        return DirectVisualBootstrapResult(
            success=False, failure_reason=DirectVisualFailureReason.PREPROCESS_FAILED,
            failure_message="synthetic preprocess failure",
        )

    workspace = _make_workspace()
    workspace._apply_targetless_prior(existing_prior)
    _prepare_workspace_for_run(workspace, monkeypatch, fake_pipeline)
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: QMessageBox.Ok))

    workspace._on_run_targetless_bootstrap()

    # Existing prior untouched -- GUIDED AUTO remains usable exactly as before.
    assert workspace.targetless_prior is existing_prior
    assert workspace.prior_status_label.text() == "READY"
    assert "FAILED" in workspace.bootstrap_status_label.text()
    assert workspace.bootstrap_run_button.isEnabled() is True


def test_run_button_cancel_preserves_prior_and_runs_no_further_stage(qapp, monkeypatch):
    existing_prior = TargetlessPrior(
        T_lidar_from_camera=np.eye(4), source_path="/prev/calib.json", source_key="T_lidar_camera",
    )
    calls = []

    def fake_pipeline(config, on_progress=None, on_log=None, on_stage_started=None, on_stage_finished=None, cancel_check=None):
        calls.append("started")
        # Simulates cancellation being observed at the first opportunity --
        # the real stage-by-stage cancel_check honoring is covered by
        # tests/test_direct_visual_runner.py's pure-function tests.
        assert cancel_check is not None and cancel_check() is True
        return DirectVisualBootstrapResult(success=False, cancelled=True)

    workspace = _make_workspace()
    workspace._apply_targetless_prior(existing_prior)
    _prepare_workspace_for_run(workspace, monkeypatch, fake_pipeline)

    # Request cancel BEFORE the (synchronous, in this test) run starts --
    # DirectVisualBootstrapWorker.request_cancel() just flips a flag that
    # fake_pipeline's cancel_check() reads, exactly like the real worker.
    import ui.worker as worker_module
    original_init = worker_module.DirectVisualBootstrapWorker.__init__

    created_workers = []

    def spying_init(self, config):
        original_init(self, config)
        self.request_cancel()  # pre-cancel, simulating a CANCEL click that beat the run
        created_workers.append(self)

    monkeypatch.setattr(worker_module.DirectVisualBootstrapWorker, "__init__", spying_init)

    workspace._on_run_targetless_bootstrap()

    assert len(calls) == 1  # pipeline entered exactly once, no retry/second stage attempt
    assert workspace.targetless_prior is existing_prior  # untouched
    assert workspace.bootstrap_status_label.text() == "CANCELLED"
    assert workspace.bootstrap_run_button.isEnabled() is True
    assert workspace.bootstrap_cancel_button.isVisible() is False


def test_cancel_button_calls_worker_request_cancel(qapp):
    workspace = _make_workspace()
    from integrations.direct_visual_runner import DirectVisualConfig
    from ui.worker import DirectVisualBootstrapWorker

    worker = DirectVisualBootstrapWorker(DirectVisualConfig(input_bag_path="/x", output_path="/y"))
    workspace._bootstrap_worker = worker

    workspace._on_cancel_targetless_bootstrap()

    assert worker._cancelled is True
    assert workspace.bootstrap_status_label.text() == "CANCELLING..."


def test_use_bag_topics_button_copies_from_bag_source(qapp):
    workspace = _make_workspace()
    workspace.bag_source.camera_topic_combo.addItem("/camera/image_raw", "/camera/image_raw")
    workspace.bag_source.lidar_topic_combo.addItem("/points_raw", "/points_raw")
    workspace.bag_source.camera_topic_combo.setCurrentIndex(workspace.bag_source.camera_topic_combo.count() - 1)
    workspace.bag_source.lidar_topic_combo.setCurrentIndex(workspace.bag_source.lidar_topic_combo.count() - 1)

    workspace._on_use_bag_topics_for_bootstrap()

    assert workspace._combo_selected_value(workspace.bootstrap_image_topic_combo) == "/camera/image_raw"
    assert workspace._combo_selected_value(workspace.bootstrap_points_topic_combo) == "/points_raw"


# ---------------------------------------------------------------------------
# §6: bag-load auto-fills bootstrap topic defaults (fills empty fields only)
# ---------------------------------------------------------------------------

def test_bag_scene_loaded_prefills_empty_bootstrap_topics():
    workspace = _make_workspace()
    assert workspace._combo_selected_value(workspace.bootstrap_image_topic_combo) == ""

    workspace._prefill_bootstrap_defaults_from_bag("/camera/image_raw", "/ouster/points")

    assert workspace._combo_selected_value(workspace.bootstrap_image_topic_combo) == "/camera/image_raw"
    assert workspace._combo_selected_value(workspace.bootstrap_points_topic_combo) == "/ouster/points"


def test_bag_scene_loaded_never_overwrites_a_field_the_user_already_set():
    workspace = _make_workspace()
    workspace._set_combo_free_text(workspace.bootstrap_image_topic_combo, "/my/custom/topic")

    workspace._prefill_bootstrap_defaults_from_bag("/camera/image_raw", "/ouster/points")

    assert workspace._combo_selected_value(workspace.bootstrap_image_topic_combo) == "/my/custom/topic"  # untouched
    assert workspace._combo_selected_value(workspace.bootstrap_points_topic_combo) == "/ouster/points"   # was empty, filled


# ---------------------------------------------------------------------------
# §7/§24: RUN pre-flight validation blocks before any worker/process starts
# ---------------------------------------------------------------------------

def test_run_blocked_without_camera_intrinsic(qapp, monkeypatch):
    workspace = _make_workspace()
    workspace._bootstrap_is_linux = True
    workspace.intrinsics = None
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Ok))

    workspace._on_run_targetless_bootstrap()

    assert workspace._bootstrap_worker is None  # never started


def test_run_blocked_when_bag_path_is_not_a_directory(qapp, monkeypatch, tmp_path):
    workspace = _make_workspace()
    workspace._bootstrap_is_linux = True

    class _FakeIntrinsics:
        camera_matrix = np.eye(3)
        distortion = np.zeros(5)

    workspace.intrinsics = _FakeIntrinsics()
    bag_file = tmp_path / "scene01.bag"
    bag_file.write_bytes(b"fake")
    workspace.bootstrap_bag_edit.setText(str(bag_file))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Ok))

    workspace._on_run_targetless_bootstrap()

    assert workspace._bootstrap_worker is None  # never started
