"""
Tests for camera_lidar.targetless_prior -- parsing a
direct_visual_lidar_calibration calib.json into a TargetlessPrior.

Load-bearing checks: quaternion ORDER (x, y, z, w -- never w-first) and
transform DIRECTION (Camera -> LiDAR, no extra inversion), since both are
easy, silent, hard-to-notice mistakes that would make GUIDED ROI search in
entirely the wrong place while still "looking like" it worked.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from camera_lidar.targetless_prior import load_direct_visual_calib
from geometry.transform import quaternion_to_rotation_matrix, transform_points


def _write_calib(tmp_path, results: dict) -> str:
    path = tmp_path / "calib.json"
    path.write_text(json.dumps({"results": results}))
    return str(path)


# ---------------------------------------------------------------------------
# TEST 1 -- final transform parse
# ---------------------------------------------------------------------------

def test_parses_final_transform(tmp_path):
    path = _write_calib(tmp_path, {"T_lidar_camera": [1, 2, 3, 0, 0, 0, 1]})
    prior = load_direct_visual_calib(path, source="auto")

    assert prior.source_key == "T_lidar_camera"
    assert np.allclose(prior.T_lidar_from_camera[:3, 3], [1, 2, 3])
    assert np.allclose(prior.T_lidar_from_camera[:3, :3], np.eye(3))
    assert prior.source_path == path


# ---------------------------------------------------------------------------
# TEST 2 -- quaternion order (x, y, z, w), never w-first
# ---------------------------------------------------------------------------

def test_quaternion_order_is_xyzw(tmp_path):
    # 90 degree rotation about Z: qx=0, qy=0, qz=sin(45deg), qw=cos(45deg).
    qz = float(np.sin(np.pi / 4))
    qw = float(np.cos(np.pi / 4))
    path = _write_calib(tmp_path, {"T_lidar_camera": [0, 0, 0, 0.0, 0.0, qz, qw]})
    prior = load_direct_visual_calib(path, source="final")

    expected_R = quaternion_to_rotation_matrix(0.0, 0.0, qz, qw)
    assert np.allclose(prior.T_lidar_from_camera[:3, :3], expected_R, atol=1e-9)

    # A 90 degree Z rotation must send +X to +Y, not +X to -Y (which is what
    # you'd get if qw/qx were swapped -- i.e. wxyz misread as xyzw).
    rotated = quaternion_to_rotation_matrix(0.0, 0.0, qz, qw) @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# TEST 3 -- transform direction: Camera -> LiDAR, no invert_transform()
# ---------------------------------------------------------------------------

def test_transform_direction_is_camera_to_lidar_uninverted(tmp_path):
    # A pure translation prior: LiDAR origin is 5m ahead (+X) of the camera.
    path = _write_calib(tmp_path, {"T_lidar_camera": [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]})
    prior = load_direct_visual_calib(path, source="final")

    camera_point = np.array([[1.0, 0.0, 0.0]])
    lidar_point = transform_points(prior.T_lidar_from_camera, camera_point)

    # p_lidar = T_lidar_from_camera @ p_camera -- NOT inverted.
    assert np.allclose(lidar_point, [[6.0, 0.0, 0.0]], atol=1e-9)


# ---------------------------------------------------------------------------
# TEST 4 -- AUTO SELECT priority
# ---------------------------------------------------------------------------

def test_auto_select_prefers_final_over_initial(tmp_path):
    path = _write_calib(tmp_path, {
        "T_lidar_camera": [1, 0, 0, 0, 0, 0, 1],
        "init_T_lidar_camera_auto": [2, 0, 0, 0, 0, 0, 1],
        "init_T_lidar_camera": [3, 0, 0, 0, 0, 0, 1],
    })
    prior = load_direct_visual_calib(path, source="auto")
    assert prior.source_key == "T_lidar_camera"
    assert np.allclose(prior.T_lidar_from_camera[:3, 3], [1, 0, 0])


def test_auto_select_falls_back_to_auto_initial(tmp_path):
    path = _write_calib(tmp_path, {
        "init_T_lidar_camera_auto": [2, 0, 0, 0, 0, 0, 1],
        "init_T_lidar_camera": [3, 0, 0, 0, 0, 0, 1],
    })
    prior = load_direct_visual_calib(path, source="auto")
    assert prior.source_key == "init_T_lidar_camera_auto"
    assert np.allclose(prior.T_lidar_from_camera[:3, 3], [2, 0, 0])


def test_auto_select_falls_back_to_manual_initial(tmp_path):
    path = _write_calib(tmp_path, {"init_T_lidar_camera": [3, 0, 0, 0, 0, 0, 1]})
    prior = load_direct_visual_calib(path, source="auto")
    assert prior.source_key == "init_T_lidar_camera"
    assert np.allclose(prior.T_lidar_from_camera[:3, 3], [3, 0, 0])


def test_auto_select_errors_when_no_key_present(tmp_path):
    path = _write_calib(tmp_path, {"something_else": [0, 0, 0, 0, 0, 0, 1]})
    with pytest.raises(ValueError):
        load_direct_visual_calib(path, source="auto")


# ---------------------------------------------------------------------------
# TEST 5 -- malformed JSON, every case must raise ValueError
# ---------------------------------------------------------------------------

def test_missing_results_key(tmp_path):
    path = tmp_path / "calib.json"
    path.write_text(json.dumps({"not_results": {}}))
    with pytest.raises(ValueError):
        load_direct_visual_calib(str(path), source="auto")


def test_missing_requested_key(tmp_path):
    path = _write_calib(tmp_path, {"init_T_lidar_camera": [0, 0, 0, 0, 0, 0, 1]})
    with pytest.raises(ValueError):
        load_direct_visual_calib(path, source="final")


def test_wrong_length_array(tmp_path):
    path = _write_calib(tmp_path, {"T_lidar_camera": [1, 2, 3, 0, 0, 1]})
    with pytest.raises(ValueError):
        load_direct_visual_calib(path, source="final")


def test_nan_value(tmp_path):
    path = _write_calib(tmp_path, {"T_lidar_camera": [float("nan"), 0, 0, 0, 0, 0, 1]})
    with pytest.raises(ValueError):
        load_direct_visual_calib(path, source="final")


def test_inf_value(tmp_path):
    path = _write_calib(tmp_path, {"T_lidar_camera": [float("inf"), 0, 0, 0, 0, 0, 1]})
    with pytest.raises(ValueError):
        load_direct_visual_calib(path, source="final")


def test_zero_quaternion(tmp_path):
    path = _write_calib(tmp_path, {"T_lidar_camera": [0, 0, 0, 0, 0, 0, 0]})
    with pytest.raises(ValueError):
        load_direct_visual_calib(path, source="final")


def test_invalid_value_type(tmp_path):
    path = _write_calib(tmp_path, {"T_lidar_camera": [1, 2, 3, 0, 0, 0, "not_a_number"]})
    with pytest.raises(ValueError):
        load_direct_visual_calib(path, source="final")


def test_root_not_a_dict(tmp_path):
    path = tmp_path / "calib.json"
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        load_direct_visual_calib(str(path), source="auto")


def test_results_not_a_dict(tmp_path):
    path = tmp_path / "calib.json"
    path.write_text(json.dumps({"results": [1, 2, 3]}))
    with pytest.raises(ValueError):
        load_direct_visual_calib(str(path), source="auto")


def test_file_not_found(tmp_path):
    with pytest.raises(ValueError):
        load_direct_visual_calib(str(tmp_path / "does_not_exist.json"), source="auto")


def test_invalid_source_argument(tmp_path):
    path = _write_calib(tmp_path, {"T_lidar_camera": [0, 0, 0, 0, 0, 0, 1]})
    with pytest.raises(ValueError):
        load_direct_visual_calib(path, source="not_a_real_source")
