"""
camera_calibrator.export.ros
================================

설계 문서 11번 - ROS CameraInfo YAML export.

    "OpenCV 결과를 ROS CameraInfo 형식으로 바로 export하는 것을 강력 추천
     (Calibration Tool -> ROS -> CameraInfo 직결)."

distortion_model 필드는 ROS(sensor_msgs/CameraInfo, image_pipeline) 관례를 따른다:
    - Pinhole / Extended Pinhole(5계수)  -> "plumb_bob"
    - Extended Pinhole(rational, 8계수)  -> "rational_polynomial"
    - Fisheye(Kannala-Brandt, 4계수)     -> "equidistant"
      (image_pipeline의 fisheye 지원 관례. ROS1 기본 camera_info는 fisheye를
       직접 표준화하진 않지만, 이 문자열이 실무에서 가장 널리 쓰인다.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from calibration.types import CalibrationResult, CameraConfig, CameraModelType


def _distortion_model_name(model: CameraModelType, num_coeffs: int) -> str:
    if model == CameraModelType.FISHEYE:
        return "equidistant"
    if num_coeffs >= 8:
        return "rational_polynomial"
    return "plumb_bob"


def build_camera_info_dict(result: CalibrationResult, camera_config: CameraConfig) -> dict:
    """설계 문서 11번 예시와 동일한 구조의 dict를 만든다.

        image_width, image_height
        camera_matrix: {rows, cols, data}
        distortion_model
        distortion_coefficients: {rows, cols, data}
        rectification_matrix: {rows, cols, data}   (모노 카메라는 항등행렬)
        projection_matrix: {rows, cols, data}       (Tx=Ty=0, 스테레오 미사용 가정)
    """
    if not result.success or result.camera_matrix is None or result.distortion is None:
        raise ValueError(f"실패한 CalibrationResult는 export할 수 없습니다: {result.error_message}")

    K = result.camera_matrix
    D = result.distortion.ravel()
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    R = np.eye(3, dtype=np.float64)
    P = np.array(
        [[fx, 0, cx, 0],
         [0, fy, cy, 0],
         [0, 0, 1, 0]],
        dtype=np.float64,
    )

    return {
        "image_width": camera_config.width,
        "image_height": camera_config.height,
        "camera_name": camera_config.sensor_name or "camera",
        "camera_matrix": {
            "rows": 3, "cols": 3,
            "data": [round(float(v), 8) for v in K.ravel().tolist()],
        },
        "distortion_model": _distortion_model_name(result.model_name, D.size),
        "distortion_coefficients": {
            "rows": 1, "cols": int(D.size),
            "data": [round(float(v), 8) for v in D.tolist()],
        },
        "rectification_matrix": {
            "rows": 3, "cols": 3,
            "data": [round(float(v), 8) for v in R.ravel().tolist()],
        },
        "projection_matrix": {
            "rows": 3, "cols": 4,
            "data": [round(float(v), 8) for v in P.ravel().tolist()],
        },
    }


def export_ros_camera_info(
    result: CalibrationResult,
    camera_config: CameraConfig,
    path: str,
) -> str:
    data = build_camera_info_dict(result, camera_config)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None)
    return path
