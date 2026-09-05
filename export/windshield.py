"""
camera_calibrator.export.windshield
========================================

Windshield Calibration 전용 YAML export. export/opencv.py::export_opencv_yaml
과 별개의 파일/스키마다 - Base Intrinsic(camera_matrix/distortion_coefficients)
YAML은 항상 그 기존 함수로만 만들고, 여기서는 절대 같은 파일에 덮어쓰거나
그 함수를 수정하지 않는다(사용자 스펙 19번 "Base Intrinsic과 Windshield
Calibration을 분리한다").

스키마:

    base_camera:
      camera_model: ...
      camera_matrix: ...
      distortion_coefficients: ...
    windshield:
      model: baseline | spherical | residual_ray | spline
      train_rms: ...
      test_rms: ...
      fitted_params: {}   # Baseline은 항상 비어 있음
"""

from __future__ import annotations

from pathlib import Path

import cv2

from calibration.models.common import distortion_coeff_labels
from calibration.types import CameraConfig
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldModelType


def export_windshield_yaml(
    result: WindshieldCalibrationResult,
    camera_config: CameraConfig,
    path: str,
) -> str:
    """WindshieldCalibrationResult를 base_camera/windshield 두 섹션으로 저장한다.

    Baseline(WindshieldModelType.BASELINE) 외의 모델은 fitted_params 스키마가
    아직 확정되지 않았으므로(Phase 2/3/4 미구현), 지금은 success=False인
    결과와 마찬가지로 export를 거부한다.
    """
    if not result.success or result.base_camera_matrix is None or result.base_distortion is None:
        raise ValueError(f"실패한 WindshieldCalibrationResult는 export할 수 없습니다: {result.error_message}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("format_version", 1)

    fs.startWriteStruct("base_camera", cv2.FileNode_MAP)
    fs.write("camera_model", result.base_model_name.value)
    fs.write("camera_matrix", result.base_camera_matrix)
    fs.write("distortion_coefficients", result.base_distortion)
    fs.write(
        "distortion_coefficient_order",
        ",".join(distortion_coeff_labels(result.base_model_name, int(result.base_distortion.size))),
    )
    fs.write("image_width", camera_config.width)
    fs.write("image_height", camera_config.height)
    fs.endWriteStruct()

    fs.startWriteStruct("windshield", cv2.FileNode_MAP)
    fs.write("model", result.windshield_model.value)
    fs.write("train_rms", float(result.residual_stats.rmse) if result.residual_stats and result.residual_stats.rmse is not None else -1.0)
    fs.write("test_rms", float(result.test_residual_stats.rmse) if result.test_residual_stats and result.test_residual_stats.rmse is not None else -1.0)
    fitted_param_keys = sorted(result.fitted_params.keys())
    fs.write("fitted_param_names", ",".join(fitted_param_keys))
    for key in fitted_param_keys:
        fs.write(f"fitted_param_{key}", float(result.fitted_params[key]))
    fs.endWriteStruct()

    fs.release()
    return path


def load_windshield_yaml(path: str) -> dict:
    """저장한 파일을 다시 읽어 dict로 반환 (재현성 검증/round-trip 테스트용)."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    base_node = fs.getNode("base_camera")
    windshield_node = fs.getNode("windshield")
    data = {
        "base_camera": {
            "camera_model": base_node.getNode("camera_model").string(),
            "camera_matrix": base_node.getNode("camera_matrix").mat(),
            "distortion_coefficients": base_node.getNode("distortion_coefficients").mat(),
            "image_width": int(base_node.getNode("image_width").real()),
            "image_height": int(base_node.getNode("image_height").real()),
        },
        "windshield": {
            "model": windshield_node.getNode("model").string(),
            "train_rms": windshield_node.getNode("train_rms").real(),
            "test_rms": windshield_node.getNode("test_rms").real(),
        },
    }
    fs.release()
    return data
