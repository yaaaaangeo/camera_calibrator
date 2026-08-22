"""
camera_calibrator.export.opencv
===================================

설계 문서 11번 - OpenCV YAML export.

cv2.FileStorage로 쓰는 "진짜" OpenCV 매트릭스 YAML 포맷이다.
ROS의 camera_info YAML(export/ros.py)과는 스키마가 다르므로 분리했다
(설계 문서 12번 UI 목업에도 [Export ROS] [Export OpenCV]가 별도 버튼).
"""

from __future__ import annotations

from pathlib import Path

import cv2

from calibration.types import CalibrationResult, CameraConfig, PatternConfig
from calibration.models.common import distortion_coeff_labels


def export_opencv_yaml(
    result: CalibrationResult,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    path: str,
) -> str:
    """cv2.FileStorage 포맷으로 camera_matrix / distortion_coefficients를 저장.

    설계 문서 10번 원칙 - 왜곡 계수 "개수"를 명시적으로 저장 (k1,k2,p1,p2,k3 vs
    k1~k6,p1,p2는 다른 모델이므로) - distortion_coefficient_count를 별도 필드로 남긴다.
    패턴 메타정보도 함께 저장해 결과 파일만으로 재현 가능하게 한다.
    """
    if not result.success or result.camera_matrix is None or result.distortion is None:
        raise ValueError(f"실패한 CalibrationResult는 export할 수 없습니다: {result.error_message}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("calibration_model", result.model_name.value)
    fs.write("image_width", camera_config.width)
    fs.write("image_height", camera_config.height)
    fs.write("camera_matrix", result.camera_matrix)
    fs.write("distortion_coefficients", result.distortion)
    fs.write("distortion_coefficient_count", int(result.distortion.size))
    fs.write(
        "distortion_coefficient_order",
        ",".join(distortion_coeff_labels(result.model_name, int(result.distortion.size))),
    )
    fs.write("rms_reprojection_error", float(result.rms_error))

    # 패턴 메타정보 (설계 문서 10번)
    fs.write("pattern_type", pattern_config.type.value)
    fs.write("pattern_squares_x", pattern_config.squares_x)
    fs.write("pattern_squares_y", pattern_config.squares_y)
    fs.write("pattern_square_size", pattern_config.square_size)
    if pattern_config.marker_size is not None:
        fs.write("pattern_marker_size", pattern_config.marker_size)
    if pattern_config.dictionary is not None:
        fs.write("pattern_dictionary", pattern_config.dictionary)

    fs.release()
    return path


def load_opencv_yaml(path: str) -> dict:
    """저장한 파일을 다시 읽어 dict로 반환 (재현성 검증, 다른 도구 연동용)."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    data = {
        "calibration_model": fs.getNode("calibration_model").string(),
        "image_width": int(fs.getNode("image_width").real()),
        "image_height": int(fs.getNode("image_height").real()),
        "camera_matrix": fs.getNode("camera_matrix").mat(),
        "distortion_coefficients": fs.getNode("distortion_coefficients").mat(),
        "rms_reprojection_error": fs.getNode("rms_reprojection_error").real(),
    }
    fs.release()
    return data
