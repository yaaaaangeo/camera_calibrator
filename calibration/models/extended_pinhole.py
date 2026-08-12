"""
camera_calibrator.calibration.models.extended_pinhole
========================================================

설계 문서 1번, 17번 Step4 - Pinhole + Brown-Conrady (방사 왜곡 k1~k6 + 접선 왜곡 p1,p2)

pinhole.py와 구조가 완전히 동일하다. 차이는 딱 하나:
Pinhole은 왜곡을 0으로 고정하는 플래그를 걸었지만, Extended Pinhole은
그 플래그를 빼서 cv2.calibrateCamera()가 왜곡 계수(k1,k2,p1,p2,k3)를
직접 추정하게 둔다.

k4~k6(rational model)은 UI에서 사용자가 켜고 끌 수 있는 고급 옵션으로
설계 문서에 명시되어 있어, use_rational_model 파라미터로 노출한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    ParameterUncertainty,
)
from calibration.models.common import (
    MIN_FRAMES_REQUIRED,
    collect_calibration_inputs,
    infer_image_size,
    compute_regional_error,
)


def calibrate_extended_pinhole(
    dataset: Dataset,
    camera_config: CameraConfig,
    use_rational_model: bool = False,
    fix_tangent_dist: bool = False,
) -> CalibrationResult:
    """Extended Pinhole (Brown-Conrady) 캘리브레이션 실행.

    Args:
        use_rational_model: True면 k4~k6까지 추정 (CALIB_RATIONAL_MODEL).
            왜곡이 매우 심한 렌즈(광각이지만 fisheye는 아닌 경우)에 유용.
        fix_tangent_dist: True면 접선 왜곡(p1,p2)을 0으로 고정.
            제조 공차가 좋은 렌즈는 접선 왜곡이 거의 없어 자유도를 줄이는 게
            오히려 안정적일 수 있다 (UI 고급 옵션으로 노출 예정).
    """
    frames, object_points, image_points = collect_calibration_inputs(dataset)

    if len(frames) < MIN_FRAMES_REQUIRED:
        return CalibrationResult(
            model_name=CameraModelType.EXTENDED_PINHOLE,
            success=False,
            error_message=(
                f"사용 가능한 프레임이 {len(frames)}장뿐입니다. "
                f"최소 {MIN_FRAMES_REQUIRED}장 이상 필요합니다."
            ),
        )

    image_size = infer_image_size(dataset, camera_config)

    flags = 0
    if use_rational_model:
        flags |= cv2.CALIB_RATIONAL_MODEL
    if fix_tangent_dist:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST

    try:
        (
            rms,
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
            std_intrinsics,
            std_extrinsics,
            per_view_errors,
        ) = cv2.calibrateCameraExtended(
            object_points,
            image_points,
            image_size,
            None,
            None,
            flags=flags,
        )
    except cv2.error as e:
        return CalibrationResult(
            model_name=CameraModelType.EXTENDED_PINHOLE,
            success=False,
            error_message=f"cv2.calibrateCameraExtended 실패: {e}",
        )

    per_frame_error = {
        frame.image_info.image_id: float(per_view_errors[i][0])
        for i, frame in enumerate(frames)
    }
    for frame in frames:
        frame.reprojection_error = per_frame_error[frame.image_info.image_id]

    regional_error = compute_regional_error(frames, per_frame_error, image_size)

    param_uncertainty = ParameterUncertainty(
        fx_std=float(std_intrinsics[0][0]),
        fy_std=float(std_intrinsics[1][0]),
        cx_std=float(std_intrinsics[2][0]),
        cy_std=float(std_intrinsics[3][0]),
    )

    return CalibrationResult(
        model_name=CameraModelType.EXTENDED_PINHOLE,
        camera_matrix=camera_matrix,
        distortion=dist_coeffs,
        rvecs=list(rvecs),
        tvecs=list(tvecs),
        rms_error=float(rms),
        per_frame_error=per_frame_error,
        regional_error=regional_error,
        param_uncertainty=param_uncertainty,
        success=True,
    )
