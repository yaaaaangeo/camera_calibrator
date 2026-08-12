"""
camera_calibrator.calibration.models.pinhole
==============================================

설계 문서 17번 Step4 - Pinhole 모델 캘리브레이션.

목표: 이미지 로드 -> 검출(Step2, 이미 완료) -> 이 모듈 -> K, D, RMS 출력까지
      엔드투엔드로 한 번 끝까지 도는 것.

Pinhole = "왜곡 없는 이상적인 핀홀". 설계 문서 1번에 따라
cv2.calibrateCamera()를 쓰되 왜곡 계수를 전부 0으로 고정하는 플래그를 사용한다.
(CALIB_ZERO_TANGENT_DIST | CALIB_FIX_K1 | CALIB_FIX_K2 | CALIB_FIX_K3)

Extended Pinhole(4단계)은 이 플래그들을 빼기만 하면 되므로,
optimizer 파라미터를 함수 인자로 분리해두면 확장이 자연스러워진다.
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
    fmt_optional,
)

# Pinhole 전용 플래그: 방사/접선 왜곡을 전부 0으로 고정
_PINHOLE_FLAGS = (
    cv2.CALIB_ZERO_TANGENT_DIST
    | cv2.CALIB_FIX_K1
    | cv2.CALIB_FIX_K2
    | cv2.CALIB_FIX_K3
)


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------

def calibrate_pinhole(
    dataset: Dataset,
    camera_config: CameraConfig,
) -> CalibrationResult:
    """Pinhole(왜곡 0 고정) 캘리브레이션 실행.

    cv2.calibrateCameraExtended()를 사용해 한 번의 호출로
    - camera_matrix, distortion
    - per-view(프레임별) RMS 오차 (perViewErrors)
    - fx/fy/cx/cy 표준편차 (stdDeviationsIntrinsics)
    를 모두 받아온다. 별도로 cv2.projectPoints를 프레임마다 돌릴 필요가 없다.
    """
    frames, object_points, image_points = collect_calibration_inputs(dataset)

    if len(frames) < MIN_FRAMES_REQUIRED:
        return CalibrationResult(
            model_name=CameraModelType.PINHOLE,
            success=False,
            error_message=(
                f"사용 가능한 프레임이 {len(frames)}장뿐입니다. "
                f"최소 {MIN_FRAMES_REQUIRED}장 이상 필요합니다."
            ),
        )

    image_size = infer_image_size(dataset, camera_config)

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
            flags=_PINHOLE_FLAGS,
        )
    except cv2.error as e:
        return CalibrationResult(
            model_name=CameraModelType.PINHOLE,
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

    # stdDeviationsIntrinsics 순서: fx, fy, cx, cy, k1, k2, p1, p2, k3, ...
    param_uncertainty = ParameterUncertainty(
        fx_std=float(std_intrinsics[0][0]),
        fy_std=float(std_intrinsics[1][0]),
        cx_std=float(std_intrinsics[2][0]),
        cy_std=float(std_intrinsics[3][0]),
    )

    return CalibrationResult(
        model_name=CameraModelType.PINHOLE,
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


# ---------------------------------------------------------------------------
# 터미널 확인용 요약 출력
# ---------------------------------------------------------------------------

def summarize_calibration(result: CalibrationResult) -> str:
    if not result.success:
        return f"[{result.model_name.value}] 캘리브레이션 실패: {result.error_message}"

    K = result.camera_matrix
    D = result.distortion.ravel()
    lines = [
        f"[{result.model_name.value}] RMS = {result.rms_error:.4f} px",
        f"  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}",
        f"  distortion = {np.round(D, 6).tolist()}",
    ]
    if result.regional_error:
        re = result.regional_error
        lines.append(
            "  regional RMS  "
            f"center={fmt_optional(re.center)} left={fmt_optional(re.left)} right={fmt_optional(re.right)} "
            f"top={fmt_optional(re.top)} bottom={fmt_optional(re.bottom)} corner={fmt_optional(re.corner)}"
        )
    if result.param_uncertainty:
        pu = result.param_uncertainty
        lines.append(f"  fx_std={pu.fx_std:.3f}  fy_std={pu.fy_std:.3f}")
    worst = sorted(result.per_frame_error.items(), key=lambda kv: kv[1], reverse=True)[:3]
    lines.append("  worst frames: " + ", ".join(f"{k}={v:.3f}px" for k, v in worst))
    return "\n".join(lines)
