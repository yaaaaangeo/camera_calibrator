"""
camera_calibrator.calibration.windshield.projection
========================================================

WindshieldCalibrationResult -> 실행 가능한 WindshieldModel 인스턴스로 바꾸는
단일 진입점(사용자 스펙 17번 "Projection API 통일"). Windshield Calibration
UI뿐 아니라, 향후 Camera-LiDAR 프로젝션처럼 Calibration 과정과 완전히 무관하게
"이미 계산된 결과로 좌표만 변환하고 싶은" 런타임 사용처가 전부 이 함수 하나로
windshield model을 얻어야 한다 - Calibration 내부 전용 임시 함수를 따로
만들지 않는다.
"""

from __future__ import annotations

import numpy as np

from calibration.windshield.base import WindshieldCalibrationResult, WindshieldModel, WindshieldModelType
from calibration.windshield.baseline import BaselineWindshieldModel
from calibration.windshield.residual_ray import ResidualRayWindshieldModel
from calibration.windshield.residual_rbf import build_residual_rbf_model_from_fitted_params
from calibration.windshield.spherical import (
    DEFAULT_AIR_REFRACTIVE_INDEX,
    DEFAULT_GLASS_REFRACTIVE_INDEX,
    DEFAULT_GLASS_THICKNESS_M,
    SphericalWindshieldModel,
)


def build_projector(result: WindshieldCalibrationResult) -> WindshieldModel:
    if result.windshield_model == WindshieldModelType.BASELINE:
        return BaselineWindshieldModel(
            result.base_camera_matrix, result.base_distortion, result.base_model_name
        )
    if result.windshield_model == WindshieldModelType.SPHERICAL:
        fp = result.fitted_params
        return SphericalWindshieldModel(
            result.base_camera_matrix,
            result.base_distortion,
            result.base_model_name,
            sphere_center=np.array(
                [fp["sphere_center_x"], fp["sphere_center_y"], fp["sphere_center_z"]]
            ),
            sphere_radius=fp["sphere_radius"],
            n_air=fp.get("air_refractive_index", DEFAULT_AIR_REFRACTIVE_INDEX),
            n_glass=fp.get("glass_refractive_index", DEFAULT_GLASS_REFRACTIVE_INDEX),
            glass_thickness_m=fp.get("glass_thickness_m", DEFAULT_GLASS_THICKNESS_M),
        )
    if result.windshield_model == WindshieldModelType.RESIDUAL_RAY:
        fp = result.fitted_params
        if fp.get("residual_ray_method", 0.0) == 1.0:
            return build_residual_rbf_model_from_fitted_params(
                result.base_camera_matrix,
                result.base_distortion,
                result.base_model_name,
                fp,
            )
        rows, cols = int(fp["grid_rows"]), int(fp["grid_cols"])
        grid = np.zeros((rows, cols, 3), dtype=np.float64)
        for r in range(rows):
            for c in range(cols):
                grid[r, c, 0] = fp[f"grid_dx_{r}_{c}"]
                grid[r, c, 1] = fp[f"grid_dy_{r}_{c}"]
                grid[r, c, 2] = fp[f"grid_dz_{r}_{c}"]
        return ResidualRayWindshieldModel(
            result.base_camera_matrix,
            result.base_distortion,
            result.base_model_name,
            grid=grid,
            image_width=fp["image_width"],
            image_height=fp["image_height"],
        )
    if result.windshield_model == WindshieldModelType.SPLINE:
        raise NotImplementedError("Spline windshield model is not implemented yet (Phase 4).")
    raise ValueError(f"알 수 없는 windshield 모델: {result.windshield_model}")
