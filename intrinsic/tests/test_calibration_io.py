from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import yaml

from calibration.calibration_io import (
    StandardCalibration,
    export_standard_json,
    load_kalibr_camchain,
    load_opencv_calibration,
    load_ros_camera_info,
    load_standard_calibration,
    load_standard_json,
)
from calibration.types import CameraModelType


K = np.array([[800.0, 0.0, 640.0], [0.0, 802.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
D = np.array([-0.2, 0.05, 0.001, -0.002, 0.01], dtype=np.float64)


def test_standard_json_roundtrip(tmp_path):
    original = StandardCalibration(
        label="candidate",
        camera_matrix=K,
        distortion=D.reshape(-1, 1),
        model_name=CameraModelType.EXTENDED_PINHOLE,
        distortion_model="plumb_bob",
        width=1280,
        height=720,
        camera_name="front",
        source_format="manual",
    )
    path = tmp_path / "calibration.json"

    export_standard_json(original, str(path))
    loaded = load_standard_json(str(path))

    assert loaded.label == "candidate"
    assert loaded.width == 1280
    assert loaded.height == 720
    assert loaded.camera_name == "front"
    assert loaded.model_name == CameraModelType.EXTENDED_PINHOLE
    np.testing.assert_allclose(loaded.camera_matrix, K)
    np.testing.assert_allclose(loaded.distortion.reshape(-1), D)
    assert loaded.coefficient_order == ["k1", "k2", "p1", "p2", "k3"]


def test_auto_loader_reads_standard_json(tmp_path):
    path = tmp_path / "standard.json"
    path.write_text(
        json.dumps({
            "label": "reference",
            "camera": {"width": 1280, "height": 720, "name": "cam"},
            "model": {"type": "fisheye", "distortion_model": "equidistant"},
            "intrinsics": {"fx": 800, "fy": 801, "cx": 640, "cy": 360},
            "distortion": {"coefficients": [-0.01, 0.001, 0, 0]},
        }),
        encoding="utf-8",
    )

    loaded = load_standard_calibration(str(path))

    assert loaded.source_format == "standard_json"
    assert loaded.model_name == CameraModelType.FISHEYE
    assert loaded.distortion_model == "equidistant"


def test_load_opencv_filestorage_yaml(tmp_path):
    path = str(tmp_path / "opencv.yaml")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("calibration_model", CameraModelType.EXTENDED_PINHOLE.value)
    fs.write("image_width", 1280)
    fs.write("image_height", 720)
    fs.write("camera_matrix", K)
    fs.write("distortion_coefficients", D.reshape(-1, 1))
    fs.release()

    loaded = load_standard_calibration(path)

    assert loaded.source_format == "opencv_yaml"
    assert loaded.width == 1280
    assert loaded.height == 720
    assert loaded.model_name == CameraModelType.EXTENDED_PINHOLE
    np.testing.assert_allclose(loaded.camera_matrix, K)
    np.testing.assert_allclose(loaded.distortion.reshape(-1), D)


def test_load_ros_camera_info_yaml(tmp_path):
    path = tmp_path / "camera_info.yaml"
    data = {
        "image_width": 1280,
        "image_height": 720,
        "camera_name": "front_camera",
        "camera_matrix": {"rows": 3, "cols": 3, "data": K.reshape(-1).tolist()},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": D.tolist()},
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    loaded = load_ros_camera_info(str(path))
    auto = load_standard_calibration(str(path))

    for item in (loaded, auto):
        assert item.source_format == "ros_camera_info_yaml"
        assert item.camera_name == "front_camera"
        assert item.model_name == CameraModelType.EXTENDED_PINHOLE
        np.testing.assert_allclose(item.camera_matrix, K)
        np.testing.assert_allclose(item.distortion.reshape(-1), D)


def test_load_kalibr_camchain_yaml_defaults_to_cam0(tmp_path):
    path = tmp_path / "camchain.yaml"
    data = {
        "cam0": {
            "camera_model": "pinhole",
            "intrinsics": [800.0, 802.0, 640.0, 360.0],
            "distortion_model": "radtan",
            "distortion_coeffs": D.tolist(),
            "resolution": [1280, 720],
            "rostopic": "/camera/image_raw",
        }
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    loaded = load_kalibr_camchain(str(path))
    auto = load_standard_calibration(str(path))

    for item in (loaded, auto):
        assert item.source_format == "kalibr_camchain_yaml"
        assert item.label == "/camera/image_raw"
        assert item.width == 1280
        assert item.height == 720
        assert item.model_name == CameraModelType.EXTENDED_PINHOLE
        assert item.distortion_model == "radtan"
        np.testing.assert_allclose(item.camera_matrix, K)
        np.testing.assert_allclose(item.distortion.reshape(-1), D)


def test_loader_rejects_nan_intrinsics(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({
            "intrinsics": {"fx": float("nan"), "fy": 800, "cx": 640, "cy": 360},
            "distortion": {"coefficients": [0, 0, 0, 0, 0]},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="NaN|Inf"):
        load_standard_json(str(path))


def test_direct_opencv_loader_rejects_missing_matrix(tmp_path):
    path = str(tmp_path / "invalid.yaml")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("something_else", 1.0)
    fs.release()

    with pytest.raises(ValueError, match="camera_matrix"):
        load_opencv_calibration(path)


@pytest.mark.parametrize(
    ("k_key", "d_key"),
    [("K", "D"), ("cameraMatrix", "distCoeffs"), ("CameraMat", "DistCoeff")],
)
def test_auto_loader_accepts_common_plain_yaml_aliases(tmp_path, k_key, d_key):
    path = tmp_path / f"aliases_{k_key}.yaml"
    path.write_text(
        yaml.safe_dump({
            k_key: K.reshape(-1).tolist(),
            d_key: D.tolist(),
            "resolution": [1280, 720],
            "distortion_model": "plumb_bob",
        }),
        encoding="utf-8",
    )

    loaded = load_standard_calibration(str(path))

    assert loaded.source_format == "generic_yaml"
    assert loaded.width == 1280 and loaded.height == 720
    np.testing.assert_allclose(loaded.camera_matrix, K)
    np.testing.assert_allclose(loaded.distortion.reshape(-1), D)


def test_ros_yaml_accepts_plain_matrix_lists(tmp_path):
    path = tmp_path / "plain_ros.yaml"
    path.write_text(yaml.safe_dump({
        "image_width": 1280,
        "image_height": 720,
        "camera_matrix": K.reshape(-1).tolist(),
        "distortion_coefficients": D.tolist(),
        "distortion_model": "plumb_bob",
    }), encoding="utf-8")

    loaded = load_standard_calibration(str(path))

    np.testing.assert_allclose(loaded.camera_matrix, K)
    np.testing.assert_allclose(loaded.distortion.reshape(-1), D)


def test_loader_accepts_wrapped_comma_string_calibration(tmp_path):
    path = tmp_path / "bottom_center_calibration_data.yaml"
    path.write_text(yaml.safe_dump({
        "bottom_center_calibration_data": {
            "image_w": 1920,
            "image_h": 1536,
            "cameraMatrix": ", ".join(str(v) for v in K.reshape(-1)),
            "distCoeffs": ", ".join(str(v) for v in D),
        }
    }), encoding="utf-8")

    loaded = load_standard_calibration(str(path))

    assert loaded.label == "bottom_center_calibration_data"
    assert loaded.width == 1920 and loaded.height == 1536
    assert loaded.model_name == CameraModelType.EXTENDED_PINHOLE
    np.testing.assert_allclose(loaded.camera_matrix, K)
    np.testing.assert_allclose(loaded.distortion.reshape(-1), D)
