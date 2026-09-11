"""
camera_calibrator.calibration.windshield.base_projection
=============================================================

Windshield 데이터셋 프레임들의 포즈(rvec, tvec)를 "고정된" Base Camera Model
K,D로 구한다. Phase 1(Baseline)뿐 아니라 Phase 2/3(Spherical/Residual Ray)도
자기 파라미터를 최적화하기 전의 초기 포즈 추정값으로 이 함수를 그대로
재사용할 것을 전제로 한다.

calibration.validation.py::_test_reprojection_errors와 동일한 계약을 따른다:
개별 프레임에서 solvePnP가 실패해도 예외를 던져 전체 배치를 죽이지 않고,
실패한 frame_id만 목록에 남긴다.
"""

from __future__ import annotations

import numpy as np

from calibration.models.common import solve_pnp_for_model_robust
from calibration.types import CameraModelType, Frame


def solve_poses_fixed_intrinsics(
    frames: list[Frame],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
) -> tuple[list[Frame], list[np.ndarray], list[np.ndarray], list[str]]:
    """프레임마다 solvePnP로 포즈만 구한다. camera_matrix/distortion은 절대
    수정하지 않는다(Windshield Calibration의 "Base K,D 고정" 원칙, 사용자
    스펙 3번) - 이 함수는 두 배열을 읽기만 하고 반환값도 rvec/tvec뿐이다.

    Returns:
        (ok_frames, rvecs, tvecs, failed_frame_ids) - 앞 세 개는 서로 순서/
        길이가 맞는 병렬 리스트다. calibration.radial_profile /
        calibration.spatial_error_map / calibration.models.common의 기존
        함수들이 기대하는 형태 그대로라, 그 함수들을 수정 없이 재사용할 수 있다.
    """
    ok_frames: list[Frame] = []
    rvecs: list[np.ndarray] = []
    tvecs: list[np.ndarray] = []
    failed_frame_ids: list[str] = []

    for frame in frames:
        det = frame.detection
        frame_id = frame.image_info.image_id
        if not det or not det.success or det.object_points is None or det.corners is None:
            failed_frame_ids.append(frame_id)
            continue

        # Fisheye input points must be reshaped to OpenCV's Nx1x{3,2}
        # convention.  The shared robust path performs that normalization and
        # retains the fixed-K,D fallback policy already used by intrinsic
        # Hold-out validation.
        ok, rvec, tvec, _reason = solve_pnp_for_model_robust(
            det.object_points, det.corners, camera_matrix, distortion, model
        )
        if not ok:
            failed_frame_ids.append(frame_id)
            continue

        ok_frames.append(frame)
        rvecs.append(rvec)
        tvecs.append(tvec)

    return ok_frames, rvecs, tvecs, failed_frame_ids
