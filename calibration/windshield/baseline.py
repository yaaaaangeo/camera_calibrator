"""
camera_calibrator.calibration.windshield.baseline
======================================================

Phase 1 - Baseline("None") Windshield 모델. 사용자 스펙 7/8/21번 STEP 1.

보정을 전혀 하지 않고(항등), 고정된 Base Camera Model K,D로만 예측한 픽셀과
실제 관측 픽셀의 차이를 측정한다:

    Known 3D target point
        -> Base Camera Model (K,D 고정, solvePnP로 포즈만 구함)
        -> Predicted Pixel
    Observed Pixel
        -> Residual: dx = observed_x - predicted_x, dy = observed_y - predicted_y

이 잔차가 "Windshield 때문에 생긴 기하학적 오차"의 기본 측정값이다. 이 값을
집계하는 계산(RMS/Median/P95/P99/Max, Regional/Radial/Edge, Spatial dx/dy Map)은
전부 기존 backend 함수(residual_stats.py, models/common.py, radial_profile.py,
spatial_error_map.py)를 그대로 재사용한다 - Baseline은 투영 방식이 기존
Standard 4모델과 동일(항등 보정)하므로 새 잔차 계산 로직이 필요 없다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from calibration.models.common import (
    compute_regional_error,
    infer_image_size,
    project_points_for_model,
)
from calibration.radial_profile import (
    collect_per_point_vectors,
    compute_radial_error_bands,
    compute_radial_error_profile,
)
from calibration.residual_stats import compute_residual_stats_for_calibration
from calibration.spatial_error_map import compute_spatial_error_map
from calibration.types import (
    CameraConfig,
    CameraModelType,
    Dataset,
    Frame,
    RadialErrorProfile,
    RegionalError,
    ResidualStats,
    SpatialErrorMap,
)
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldConfig, WindshieldModel, WindshieldModelType
from calibration.windshield.base_projection import solve_poses_fixed_intrinsics


def _subset_frames(dataset: Dataset, frame_ids: list[str]) -> list[Frame]:
    id_set = set(frame_ids)
    return [f for f in dataset.frames if f.image_info.image_id in id_set]


@dataclass
class _EvalOutcome:
    """_evaluate_frames()의 중간 결과 - Train/Test 양쪽에서 동일하게 필요한
    묶음이라 반환 타입을 하나로 정리했다. 외부에 노출하지 않는다."""
    ok_frames: list[Frame]
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    failed_frame_ids: list[str]
    per_frame_error: dict[str, float]
    residual_stats: ResidualStats
    regional_error: RegionalError
    radial_profile: RadialErrorProfile
    radial_bands: RadialErrorProfile
    spatial_error_map: SpatialErrorMap
    mean_dx: Optional[float]
    mean_dy: Optional[float]


def _evaluate_frames(
    frames: list[Frame],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
    image_size: tuple[int, int],
) -> _EvalOutcome:
    ok_frames, rvecs, tvecs, failed_ids = solve_poses_fixed_intrinsics(
        frames, camera_matrix, distortion, model
    )

    per_frame_error: dict[str, float] = {}
    for frame, rvec, tvec in zip(ok_frames, rvecs, tvecs):
        det = frame.detection
        projected = project_points_for_model(det.object_points, rvec, tvec, camera_matrix, distortion, model)
        detected = det.corners.reshape(-1, 2)
        diff = detected - projected
        per_point = np.hypot(diff[:, 0], diff[:, 1])
        per_frame_error[frame.image_info.image_id] = float(np.sqrt(np.mean(per_point ** 2)))

    residual_stats = compute_residual_stats_for_calibration(
        ok_frames, rvecs, tvecs, camera_matrix, distortion, image_size, model
    )
    regional_error = compute_regional_error(ok_frames, per_frame_error, image_size)
    radial_profile = compute_radial_error_profile(
        ok_frames, rvecs, tvecs, camera_matrix, distortion, image_size, model
    )
    radial_bands = compute_radial_error_bands(
        ok_frames, rvecs, tvecs, camera_matrix, distortion, image_size, model
    )
    spatial_map = compute_spatial_error_map(
        ok_frames, rvecs, tvecs, camera_matrix, distortion, image_size, model
    )

    _, _, dxs, dys = collect_per_point_vectors(ok_frames, rvecs, tvecs, camera_matrix, distortion, model)
    mean_dx = float(dxs.mean()) if dxs.size else None
    mean_dy = float(dys.mean()) if dys.size else None

    return _EvalOutcome(
        ok_frames=ok_frames,
        rvecs=rvecs,
        tvecs=tvecs,
        failed_frame_ids=failed_ids,
        per_frame_error=per_frame_error,
        residual_stats=residual_stats,
        regional_error=regional_error,
        radial_profile=radial_profile,
        radial_bands=radial_bands,
        spatial_error_map=spatial_map,
        mean_dx=mean_dx,
        mean_dy=mean_dy,
    )


def calibrate_baseline(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    train_ids: list[str],
    test_ids: list[str],
) -> WindshieldCalibrationResult:
    """Baseline("None") Windshield 모델을 계산한다. config.base_camera_matrix/
    base_distortion/base_model_name은 절대 재추정하지 않고 그대로 쓴다.
    """
    K, D, model = config.base_camera_matrix, config.base_distortion, config.base_model_name
    image_size = infer_image_size(windshield_dataset, camera_config)

    train_frames = _subset_frames(windshield_dataset, train_ids)
    if not train_frames:
        return WindshieldCalibrationResult(
            windshield_model=WindshieldModelType.BASELINE,
            base_model_name=model,
            base_camera_matrix=K,
            base_distortion=D,
            train_frame_ids=train_ids,
            test_frame_ids=test_ids,
            success=False,
            error_message="Train 프레임이 없습니다.",
        )

    train_outcome = _evaluate_frames(train_frames, K, D, model, image_size)

    result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType.BASELINE,
        base_model_name=model,
        base_camera_matrix=K,
        base_distortion=D,
        train_frame_ids=train_ids,
        test_frame_ids=test_ids,
        failed_frame_ids=list(train_outcome.failed_frame_ids),
        per_frame_error=train_outcome.per_frame_error,
        residual_stats=train_outcome.residual_stats,
        regional_error=train_outcome.regional_error,
        radial_profile=train_outcome.radial_profile,
        radial_bands=train_outcome.radial_bands,
        spatial_error_map=train_outcome.spatial_error_map,
        mean_dx=train_outcome.mean_dx,
        mean_dy=train_outcome.mean_dy,
        fitted_params={},
        success=True,
    )

    if test_ids:
        test_frames = _subset_frames(windshield_dataset, test_ids)
        if test_frames:
            test_outcome = _evaluate_frames(test_frames, K, D, model, image_size)
            result.test_residual_stats = test_outcome.residual_stats
            result.test_regional_error = test_outcome.regional_error
            result.test_radial_profile = test_outcome.radial_profile
            result.test_radial_bands = test_outcome.radial_bands
            result.test_spatial_error_map = test_outcome.spatial_error_map
            result.test_mean_dx = test_outcome.mean_dx
            result.test_mean_dy = test_outcome.mean_dy
            for fid in test_outcome.failed_frame_ids:
                if fid not in result.failed_frame_ids:
                    result.failed_frame_ids.append(fid)
        else:
            result.warning_message = "Test 프레임에서 유효한 검출 결과를 찾지 못했습니다."

    return result


class BaselineWindshieldModel(WindshieldModel):
    """Windshield 보정 없음 - Base Camera Model 그대로 투영/역투영한다.
    project_point(x,y,z)는 카메라 좌표계에 이미 있는 점을 가정하므로 rvec=tvec=0
    으로 project_points_for_model을 호출하는 것과 완전히 동일하다.
    """

    def __init__(self, camera_matrix: np.ndarray, distortion: np.ndarray, model: CameraModelType):
        self._K = camera_matrix
        self._D = distortion
        self._model = model
        self._zero = np.zeros(3, dtype=np.float64)

    def project_point(self, x: float, y: float, z: float) -> tuple[float, float]:
        obj = np.array([[[x, y, z]]], dtype=np.float64)
        projected = project_points_for_model(obj, self._zero, self._zero, self._K, self._D, self._model)
        u, v = projected[0]
        return float(u), float(v)

    def unproject_pixel(self, u: float, v: float) -> tuple[float, float, float]:
        pt = np.array([[[u, v]]], dtype=np.float64)
        if self._model == CameraModelType.FISHEYE:
            undistorted = cv2.fisheye.undistortPoints(pt, self._K, self._D)
        else:
            undistorted = cv2.undistortPoints(pt, self._K, self._D)
        x, y = undistorted[0, 0]
        norm = math.sqrt(x * x + y * y + 1.0)
        return float(x / norm), float(y / norm), float(1.0 / norm)
