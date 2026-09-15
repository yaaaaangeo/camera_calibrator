from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np
import pytest

from calibration.error_normalization import normalized_reprojection_error
from calibration.optimizer import (
    ScipyCalibrationOptimizer,
    apply_optimized_calibration,
    parameter_definition,
    restore_original_calibration,
)
from calibration.project_io import project_from_dict, project_to_dict
from calibration.types import (
    CalibrationProject, CalibrationResult, CameraConfig, CameraModelType,
    Dataset, DetectionResult, Frame, FrameStatus, ImageInfo, OptimizerResult,
    OptimizerSettings, PatternConfig, PatternType,
)


def _synthetic_dataset(seed=7, outlier=False):
    rng = np.random.default_rng(seed)
    camera = CameraConfig(width=1280, height=720)
    pattern = PatternConfig(PatternType.CHESSBOARD, 7, 6, 0.04)
    K = np.array([[820., 0., 641.], [0., 815., 359.], [0., 0., 1.]])
    D = np.array([-0.16, 0.045, 0.001, -0.0008, -0.006])
    grid = np.zeros((42, 1, 3), np.float32)
    grid[:, 0, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) * 0.04
    frames = []
    for index in range(12):
        rv = np.array([0.08 * np.sin(index), 0.18 * np.cos(index * .7), -0.12 + .02 * index])
        tv = np.array([-.13 + .025 * index, -.08 + .015 * (index % 5), 0.72 + .045 * (index % 4)])
        points, _ = cv2.projectPoints(grid, rv, tv, K, D)
        points = points.astype(np.float32)
        points += rng.normal(0, 0.18, points.shape).astype(np.float32)
        if outlier and index < 8:
            points[index % 10, 0] += np.array([25., -20.], np.float32)
        frames.append(Frame(
            ImageInfo(f"f{index}", f"/tmp/f{index}.png", 1280, 720),
            DetectionResult(f"f{index}", True, points, grid.copy(), num_corners=len(grid),
                            board_center_px=tuple(points.reshape(-1, 2).mean(axis=0)), board_area_ratio=.1),
            status=FrameStatus.DETECTED,
        ))
    return Dataset(frames), camera, pattern, K, D


def _run(dataset, camera, pattern, loss="huber", cancel_check=None):
    settings = OptimizerSettings(
        num_starts=4, robust_loss=loss, max_iterations_per_stage=100,
        final_joint_iterations=100,
    )
    return ScipyCalibrationOptimizer().optimize(
        dataset, camera, pattern, CameraModelType.BROWN_CONRADY,
        [f"f{i}" for i in range(9)], [f"f{i}" for i in range(9, 12)],
        settings, cancel_check=cancel_check,
    )


@pytest.fixture(scope="module")
def optimized_case():
    dataset, camera, pattern, K, D = _synthetic_dataset()
    return dataset, camera, pattern, K, D, _run(dataset, camera, pattern)


def test_no_holdout_leakage(optimized_case):
    dataset, camera, pattern, _K, _D, first = optimized_case
    changed = deepcopy(dataset)
    for frame in changed.frames[9:]:
        frame.detection.corners += 100.0
    second = _run(changed, camera, pattern)
    assert first.success and second.success
    # Iterative sparse LSMR can vary at the last few solver digits, but changing
    # hold-out measurements must not cause a material K/D change.
    np.testing.assert_allclose(first.optimized_calibration.camera_matrix,
                               second.optimized_calibration.camera_matrix, rtol=5e-4, atol=1e-3)
    np.testing.assert_allclose(first.optimized_calibration.distortion,
                               second.optimized_calibration.distortion, rtol=5e-3, atol=2e-3)


def test_synthetic_recovery_and_multistart(optimized_case):
    _dataset, _camera, _pattern, K, _D, result = optimized_case
    assert result.success, result.error_message
    assert result.after_metrics.train_rms <= result.before_metrics.train_rms * 1.01
    assert abs(result.optimized_calibration.camera_matrix[0, 0] - K[0, 0]) < 40
    assert sum(start.selected for start in result.starts) == 1
    assert next(start for start in result.starts if start.selected).converged
    assert all(np.isfinite(result.optimized_calibration.distortion))


def test_robust_huber_limits_extreme_corner_influence():
    dataset, camera, pattern, _K, _D = _synthetic_dataset(outlier=True)
    linear = _run(dataset, camera, pattern, "linear")
    huber = _run(dataset, camera, pattern, "huber")
    assert linear.success and huber.success
    # Frozen clean hold-out is the honest comparison; robust fitting should not
    # let a handful of train corners pull K/D farther away.
    assert huber.after_metrics.test_rms < linear.after_metrics.test_rms


def test_parameter_definition_and_final_joint_release(optimized_case):
    definition = parameter_definition(CameraModelType.EXTENDED_PINHOLE, 8)
    assert definition["distortion"] == ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
    assert definition["stages"][-1] == ["k4", "k5", "k6"]
    result = optimized_case[-1]
    assert result.stages[-1].name == "Final Joint"
    assert set(("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")) <= set(result.stages[-1].active_parameters)


def test_cancel_preserves_input_calibration():
    dataset, camera, pattern, _K, _D = _synthetic_dataset()
    original = CalibrationResult(CameraModelType.BROWN_CONRADY, np.eye(3), np.zeros((5, 1)), success=True)
    snapshot = deepcopy(original)
    result = _run(dataset, camera, pattern, cancel_check=lambda: True)
    assert result.cancelled and not result.success
    np.testing.assert_array_equal(original.camera_matrix, snapshot.camera_matrix)
    np.testing.assert_array_equal(original.distortion, snapshot.distortion)


def test_apply_restore_save_load_and_normalization(optimized_case, tmp_path):
    dataset, camera, pattern, _K, _D, optimizer = optimized_case
    current = CalibrationResult(CameraModelType.BROWN_CONRADY, np.eye(3) * 900,
                                np.ones((5, 1)), rms_error=2.0, success=True)
    applied = apply_optimized_calibration(current, optimizer)
    assert optimizer.applied
    np.testing.assert_array_equal(applied.camera_matrix, optimizer.optimized_calibration.camera_matrix)
    from export.opencv import export_opencv_yaml, load_opencv_yaml
    yaml_path = tmp_path / "optimized.yaml"
    export_opencv_yaml(applied, camera, pattern, str(yaml_path), calibration_source="optimized")
    assert load_opencv_yaml(str(yaml_path))["calibration_source"] == "optimized"
    restored = restore_original_calibration(optimizer)
    np.testing.assert_array_equal(restored.camera_matrix, current.camera_matrix)

    project = CalibrationProject("optimizer", camera, pattern, dataset=dataset,
                                 optimizer_results={CameraModelType.BROWN_CONRADY: optimizer})
    loaded = project_from_dict(project_to_dict(project))
    roundtrip = loaded.optimizer_results[CameraModelType.BROWN_CONRADY]
    np.testing.assert_allclose(roundtrip.optimized_calibration.distortion,
                               optimizer.optimized_calibration.distortion)
    assert roundtrip.train_frame_ids == optimizer.train_frame_ids

    legacy = project_to_dict(CalibrationProject("legacy", camera, pattern))
    assert project_from_dict(legacy).optimizer_results == {}
    raw = optimizer.after_metrics.test_rms
    expected = raw / np.mean([optimizer.optimized_calibration.camera_matrix[0, 0],
                              optimizer.optimized_calibration.camera_matrix[1, 1]])
    assert normalized_reprojection_error(raw, optimizer.optimized_calibration.camera_matrix) == pytest.approx(expected)


def test_optimizer_view_requires_opencv_and_holdout_context(optimized_case):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from ui.optimizer_view import OptimizerView

    app = QApplication.instance() or QApplication([])
    view = OptimizerView()
    view.select_model(CameraModelType.BROWN_CONRADY)
    assert not view.run_button.isEnabled()
    dataset, camera, _pattern, _K, _D, result = optimized_case
    # The existing frozen validation split is enough to enable Run; optimized
    # output remains unapplied until the separate button is clicked.
    from calibration.types import ValidationResult
    validation = ValidationResult(
        train_frame_ids=result.train_frame_ids,
        test_frame_ids=result.holdout_frame_ids,
    )
    view.set_context(
        {CameraModelType.BROWN_CONRADY: result.original_calibration},
        {CameraModelType.BROWN_CONRADY: validation},
        {CameraModelType.BROWN_CONRADY: result},
        camera,
    )
    view.select_model(CameraModelType.BROWN_CONRADY)
    app.processEvents()
    assert view.run_button.isEnabled()
    assert view.apply_button.isEnabled()
    assert not result.applied
    view.close()
