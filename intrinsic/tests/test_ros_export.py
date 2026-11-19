from __future__ import annotations

import numpy as np

from calibration.types import CalibrationResult, CameraConfig, CameraModelType
from export.ros import build_camera_info_dict


def _result(model: CameraModelType, distortion) -> CalibrationResult:
    return CalibrationResult(
        model_name=model,
        camera_matrix=np.eye(3, dtype=np.float64),
        distortion=np.asarray(distortion, dtype=np.float64).reshape(1, -1),
        success=True,
    )


def test_ros_distortion_model_comes_from_camera_model_type():
    camera = CameraConfig(width=640, height=480)

    brown = build_camera_info_dict(
        _result(CameraModelType.BROWN_CONRADY, [0.1, 0.2, 0.0, 0.0, 0.01]),
        camera,
    )
    rational = build_camera_info_dict(
        _result(CameraModelType.EXTENDED_PINHOLE, [0.1, 0.2, 0.0, 0.0, 0.01, 0.001, 0.002, 0.003]),
        camera,
    )
    pinhole = build_camera_info_dict(_result(CameraModelType.PINHOLE, [0, 0, 0, 0, 0]), camera)

    assert brown["distortion_model"] == "plumb_bob"
    assert rational["distortion_model"] == "rational_polynomial"
    assert pinhole["distortion_model"] == "plumb_bob"
