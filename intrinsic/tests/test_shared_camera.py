"""
tests/test_shared_camera.py
===========================

Camera Intrinsic 결과 -> CAMERA–LiDAR 렌즈 프리셋 (shared.camera) 이 K·D 를 그대로 옮기는지.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from calibration.types import CalibrationResult, CameraConfig, CameraModelType, PatternConfig, PatternType
from shared import camera as sc

ROOM = Path(__file__).resolve().parents[2] / "extrinsic"
if str(ROOM) not in sys.path:
    sys.path.append(str(ROOM))

K = np.array([[1100.0, 0, 955.5], [0, 1098.0, 541.25], [0, 0, 1]])
CASES = [
    (CameraModelType.BROWN_CONRADY, np.array([-0.28, 0.1, 0.001, -0.002, -0.03]), "pinhole"),
    (CameraModelType.EXTENDED_PINHOLE, np.array([0.3, -0.1, 0.001, 0.0005, 0.02, 0.7, -0.05, 0.01]), "pinhole"),
    (CameraModelType.FISHEYE, np.array([0.05, -0.01, 0.002, -0.0005]), "fisheye"),
    (CameraModelType.PINHOLE, np.zeros(5), "pinhole"),
]


@pytest.mark.parametrize("model,D,kind", CASES, ids=[c[0].value for c in CASES])
def test_result_to_room_preset_round_trip(tmp_path, model, D, kind):
    from gui.core import camera_ops, presets

    result = CalibrationResult(model_name=model, camera_matrix=K, distortion=D, rms_error=0.3, success=True)
    path = tmp_path / "camera_intrinsics.yaml"
    sc.add_lens_preset("cam_x", sc.lens_from_calibration_result(result, 1920, 1080), "test", path)
    cam = {p.name: p.camera for p in presets.load(path)}["cam_x"]
    assert cam.model == kind
    np.testing.assert_allclose(cam.matrix(), K)
    np.testing.assert_allclose(np.ravel(cam.dist())[: D.size], D)

    # 같은 점을 같은 픽셀로 투영해야 한다 (인트린직 쪽 OpenCV 호출과 비교)
    import cv2

    pts = np.array([[0.3, -0.2, 2.0], [-0.5, 0.4, 3.0], [0.0, 0.0, 1.5]])
    got = camera_ops.project_points(cam, pts)
    if kind == "fisheye":
        want = cv2.fisheye.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    else:
        want = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_exported_yaml_file_can_be_read_back(tmp_path):
    from export.opencv import export_opencv_yaml

    model, D, _ = CASES[0]
    result = CalibrationResult(model_name=model, camera_matrix=K, distortion=D, rms_error=0.3, success=True)
    pattern = PatternConfig(type=PatternType.CHARUCO, squares_x=7, squares_y=5, square_size=0.04,
                            marker_size=0.03, dictionary="DICT_5X5_100")
    out = tmp_path / "camera_brown_conrady.yaml"
    export_opencv_yaml(result, CameraConfig(width=1920, height=1080), pattern, str(out))
    lens = sc.read_lens_file(out)
    assert (lens.model, lens.width, lens.height) == ("pinhole", 1920, 1080)
    assert lens.fx == pytest.approx(1100.0) and lens.cy == pytest.approx(541.25)
    assert [lens.coeffs[k] for k in ("k1", "k2", "p1", "p2", "k3")] == pytest.approx(list(D))


def test_same_name_replaces_and_other_presets_are_kept(tmp_path):
    path = tmp_path / "p.yaml"
    lens = sc.lens_from_kd(K, np.zeros(5), "pinhole")
    sc.add_lens_preset("a", lens, "", path)
    sc.add_lens_preset("b", lens, "", path)
    sc.add_lens_preset("a", sc.lens_from_kd(K * [[2, 1, 1], [1, 1, 1], [1, 1, 1]], np.zeros(5), "pinhole"), "", path)
    entries = sc.load_preset_entries(path)
    assert [e["name"] for e in entries] == ["a", "b"] and entries[0]["fx"] == 2200.0
    with pytest.raises(ValueError):
        sc.add_lens_preset("b", lens, "", path, replace=False)
