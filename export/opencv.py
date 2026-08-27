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
import numpy as np

from calibration.types import CalibrationResult, CameraConfig, CameraModelType, PatternConfig
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


# ---------------------------------------------------------------------------
# 외부 캘리브레이션 결과 불러오기 (calibration/external_compare.py에서 사용)
# ---------------------------------------------------------------------------
#
# "예전에 다른 사람/다른 툴로 구한 파라미터"와 지금 결과를 비교하는 기능의
# 입력 경로. load_opencv_yaml()과 다른 점: 이 툴이 만든 파일이 아니어도
# (calibration_model 필드가 없어도) camera_matrix/distortion_coefficients만
# 표준 OpenCV FileStorage 필드명으로 있으면 읽을 수 있게 관대하게 만든다.


def load_camera_matrix_and_distortion_from_opencv_yaml(
    path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """camera_matrix(3x3)와 distortion 벡터만 읽는다. "calibration_model"
    같은 이 툴 고유 필드는 없어도 된다 - ROS camera_calibration 패키지나
    다른 OpenCV 기반 툴이 만든 파일도 이 두 필드명은 표준으로 쓰는 경우가
    많아서, 그런 파일도 그대로 불러올 수 있게 하기 위함.
    """
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    try:
        cm_node = fs.getNode("camera_matrix")
        d_node = fs.getNode("distortion_coefficients")
        if cm_node.empty() or d_node.empty():
            raise ValueError(
                "이 YAML에서 'camera_matrix' 또는 'distortion_coefficients' 항목을 "
                "찾을 수 없습니다 (OpenCV FileStorage 표준 포맷이 아닌 것 같습니다)."
            )
        camera_matrix = cm_node.mat()
        distortion = d_node.mat()
    finally:
        fs.release()

    if camera_matrix is None or camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix가 3x3 형태가 아닙니다.")
    if distortion is None or distortion.size == 0:
        raise ValueError("distortion_coefficients가 비어 있습니다.")

    return (
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64).reshape(-1, 1),
    )


def detect_model_hint_from_opencv_yaml(path: str) -> CameraModelType | None:
    """이 툴이 남긴 "calibration_model" 필드가 파일 안에 있으면 그 값을
    그대로 돌려준다 (UI에서 모델 선택 콤보박스의 기본값으로만 쓴다).

    없다고 임의로 추측하지 않는 이유: distortion 배열 길이만으로는 예를 들어
    Pinhole(k1,k2,p1,p2 4개)과 Fisheye(k1~k4 4개)를 구분할 수 없다 - 잘못
    추측하면 재투영 계산 자체가 조용히 틀어진다. 그래서 힌트가 없으면
    사용자가 직접 고르게 하고, 이 함수는 None만 돌려준다.
    """
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    try:
        node = fs.getNode("calibration_model")
        if node.empty():
            return None
        value = node.string()
    finally:
        fs.release()
    if not value:
        return None
    try:
        return CameraModelType(value)
    except ValueError:
        return None
