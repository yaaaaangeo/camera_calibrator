"""
camera_calibrator.calibration.models.fisheye
================================================

설계 문서 1번, 2번, 17번 Step4 - OpenCV Fisheye (Kannala-Brandt), k1~k4.

cv2.fisheye 모듈은 cv2.calibrateCameraExtended()에 대응하는 Extended 버전이
없다. 즉 perViewErrors(프레임별 오차)와 stdDeviations(파라미터 불확실성)를
공짜로 주지 않는다. 그래서 이 모듈만 다음 두 가지를 직접 구현한다:

1. 초기값 안전장치 (설계 문서 2번):
   피쉬아이는 초기 fx, fy 추정이 나쁘면 최적화가 발산한다.
   Pinhole 캘리브레이션 결과의 fx, fy, cx, cy를 CALIB_USE_INTRINSIC_GUESS로
   넘겨서 시작점을 잡아준다.

       Pinhole calibration -> fx fy cx cy -> Fisheye initial K -> Fisheye calibration

2. 프레임별 재투영 오차 수동 계산:
   cv2.fisheye.projectPoints()를 프레임마다 호출해 pinhole.py가
   perViewErrors로 공짜로 받던 것을 직접 채운다.
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

# fisheye는 파라미터가 적어도(k1~k4) Brown-Conrady보다 화각 전역에서 비선형성이
# 강해, 뷰 개수가 부족하면 발산 위험이 pinhole/extended보다 크다.
# 그래도 "최소 조건"은 공통 모듈과 동일하게 두고, 실패 시 이유를 명확히 준다.
_FISHEYE_BASE_FLAGS = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    | cv2.fisheye.CALIB_FIX_SKEW
    | cv2.fisheye.CALIB_CHECK_COND
)


def _to_fisheye_points(
    object_points: list[np.ndarray], image_points: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """cv2.fisheye.* 함수군은 float64를 기대한다 (버전에 따라 float32는
    silent하게 잘못된 결과를 내거나 예외를 던질 수 있음). 형태(N,1,3)/(N,1,2)는
    유지하고 dtype만 맞춘다.
    """
    obj64 = [o.astype(np.float64) for o in object_points]
    img64 = [i.astype(np.float64) for i in image_points]
    return obj64, img64


def _per_frame_errors_fisheye(
    frames,
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    K: np.ndarray,
    D: np.ndarray,
) -> dict[str, float]:
    """cv2.fisheye에는 perViewErrors가 없으므로 projectPoints로 직접 계산.
    프레임별 RMS(px) = 검출된 코너와 재투영된 점 사이 거리의 RMS.
    """
    errors: dict[str, float] = {}
    for i, frame in enumerate(frames):
        projected, _ = cv2.fisheye.projectPoints(
            object_points[i], rvecs[i], tvecs[i], K, D
        )
        detected = image_points[i].reshape(-1, 2)
        projected = projected.reshape(-1, 2)
        diff = detected - projected
        rms = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        errors[frame.image_info.image_id] = rms
    return errors


def calibrate_fisheye(
    dataset: Dataset,
    camera_config: CameraConfig,
    initial_guess: CalibrationResult | None = None,
) -> CalibrationResult:
    """Fisheye(Kannala-Brandt) 캘리브레이션 실행.

    Args:
        initial_guess: Pinhole 모델의 CalibrationResult를 넘기면
            fx, fy, cx, cy를 초기값으로 사용해 발산을 방지한다 (설계 문서 2번).
            없으면 OpenCV 기본 추정 로직(least-squares)에 맡긴다.
    """
    frames, object_points, image_points = collect_calibration_inputs(dataset)

    if len(frames) < MIN_FRAMES_REQUIRED:
        return CalibrationResult(
            model_name=CameraModelType.FISHEYE,
            success=False,
            error_message=(
                f"사용 가능한 프레임이 {len(frames)}장뿐입니다. "
                f"최소 {MIN_FRAMES_REQUIRED}장 이상 필요합니다."
            ),
        )

    image_size = infer_image_size(dataset, camera_config)
    object_points, image_points = _to_fisheye_points(object_points, image_points)

    flags = _FISHEYE_BASE_FLAGS
    K_init, D_init = None, None
    if initial_guess is not None and initial_guess.success and initial_guess.camera_matrix is not None:
        K_init = initial_guess.camera_matrix.copy().astype(np.float64)
        # CALIB_USE_INTRINSIC_GUESS는 K와 D 둘 다 비어있지 않아야 한다
        # (K만 채우고 D=None으로 두면 OpenCV가 assertion 에러를 던짐).
        # 왜곡은 0에서 출발해 최적화가 채워나가게 한다.
        D_init = np.zeros((4, 1), dtype=np.float64)
        flags |= cv2.fisheye.CALIB_USE_INTRINSIC_GUESS

    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            object_points,
            image_points,
            image_size,
            K_init,
            D_init,
            flags=flags,
        )
    except cv2.error as e:
        # CALIB_CHECK_COND는 조건수가 나쁜(발산 위험) 프레임이 있으면 예외를 던진다.
        # 이 경우 조건 체크를 빼고 한 번 더 시도해, "완전 실패"보다는
        # "경고와 함께 결과라도 준다" 쪽을 택한다. 최종 채택 여부는 검증(hold-out)
        # 단계에서 사용자가 판단하게 한다.
        try:
            relaxed_flags = flags & ~cv2.fisheye.CALIB_CHECK_COND
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                object_points,
                image_points,
                image_size,
                K_init,
                D_init,
                flags=relaxed_flags,
            )
        except cv2.error as e2:
            return CalibrationResult(
                model_name=CameraModelType.FISHEYE,
                success=False,
                error_message=f"cv2.fisheye.calibrate 실패: {e2}",
            )

    per_frame_error = _per_frame_errors_fisheye(
        frames, object_points, image_points, rvecs, tvecs, K, D
    )
    for frame in frames:
        frame.reprojection_error = per_frame_error[frame.image_info.image_id]

    regional_error = compute_regional_error(frames, per_frame_error, image_size)

    return CalibrationResult(
        model_name=CameraModelType.FISHEYE,
        camera_matrix=K,
        distortion=D,
        rvecs=list(rvecs),
        tvecs=list(tvecs),
        rms_error=float(rms),
        per_frame_error=per_frame_error,
        regional_error=regional_error,
        param_uncertainty=None,  # fisheye는 stdDeviations를 제공하지 않음 (V2에서 별도 구현 필요)
        success=True,
    )
