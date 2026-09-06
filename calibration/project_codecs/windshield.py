"""
camera_calibrator.calibration.project_codecs.windshield
==============================================================

Phase D-2 안정화 - Windshield Calibration 관련 dict -> dataclass 디코더.
`calibration/project_io.py`에서 로직 변경 없이 그대로 옮겨왔다.

`_windshield_calibration_result_from_dict()`는 intrinsic 모듈이 이미
정의한 `_residual_stats_from_dict`/`_regional_error_from_dict`/
`_radial_profile_from_dict`/`_spatial_error_map_from_dict`를 재사용한다
(Windshield 전용 Residual/Regional/Radial/Spatial 재구성 로직을 새로
만들지 않는다 - Camera Intrinsic과 완전히 같은 통계 타입을 그대로 쓴다).
"""

from __future__ import annotations

import numpy as np

from calibration.project_codecs.common import _arr
from calibration.project_codecs.intrinsic import (
    _radial_profile_from_dict,
    _regional_error_from_dict,
    _residual_stats_from_dict,
    _spatial_error_map_from_dict,
)
from calibration.types import CameraModelType
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldConfig, WindshieldModelType

def _windshield_config_from_dict(d) -> WindshieldConfig | None:
    if d is None:
        return None
    return WindshieldConfig(
        base_model_name=CameraModelType(d["base_model_name"]),
        base_camera_matrix=_arr(d.get("base_camera_matrix"), np.float64),
        base_distortion=_arr(d.get("base_distortion"), np.float64),
        windshield_model=WindshieldModelType(d.get("windshield_model", "baseline")),
        test_ratio=d.get("test_ratio", 0.25),
        split_seed=d.get("split_seed", 42),
        glass_refractive_index=d.get("glass_refractive_index"),
        glass_thickness_m=d.get("glass_thickness_m"),
        windshield_position_hint=d.get("windshield_position_hint"),
        # STEP 3-A 때 필드는 추가됐지만 여기 복원 코드가 빠져 있던 기존 버그 -
        # residual_ray_hint(method/AUTO/manual 설정)가 프로젝트 저장 후
        # 재로드 시 사라지는 문제였다. spline_hint도 같은 패턴이라 함께 추가.
        residual_ray_hint=d.get("residual_ray_hint"),
        spline_hint=d.get("spline_hint"),
    )


def _windshield_calibration_result_from_dict(d) -> WindshieldCalibrationResult | None:
    if d is None:
        return None
    return WindshieldCalibrationResult(
        windshield_model=WindshieldModelType(d["windshield_model"]),
        base_model_name=CameraModelType(d["base_model_name"]),
        base_camera_matrix=_arr(d.get("base_camera_matrix"), np.float64),
        base_distortion=_arr(d.get("base_distortion"), np.float64),
        train_frame_ids=d.get("train_frame_ids", []),
        test_frame_ids=d.get("test_frame_ids", []),
        failed_frame_ids=d.get("failed_frame_ids", []),
        per_frame_error=d.get("per_frame_error", {}),
        residual_stats=_residual_stats_from_dict(d.get("residual_stats")),
        test_residual_stats=_residual_stats_from_dict(d.get("test_residual_stats")),
        regional_error=_regional_error_from_dict(d.get("regional_error")),
        radial_profile=_radial_profile_from_dict(d.get("radial_profile")),
        radial_bands=_radial_profile_from_dict(d.get("radial_bands")),
        spatial_error_map=_spatial_error_map_from_dict(d.get("spatial_error_map")),
        mean_dx=d.get("mean_dx"),
        mean_dy=d.get("mean_dy"),
        test_regional_error=_regional_error_from_dict(d.get("test_regional_error")),
        test_radial_profile=_radial_profile_from_dict(d.get("test_radial_profile")),
        test_radial_bands=_radial_profile_from_dict(d.get("test_radial_bands")),
        test_spatial_error_map=_spatial_error_map_from_dict(d.get("test_spatial_error_map")),
        test_mean_dx=d.get("test_mean_dx"),
        test_mean_dy=d.get("test_mean_dy"),
        ray_angular_error_deg=d.get("ray_angular_error_deg"),
        test_ray_angular_error_deg=d.get("test_ray_angular_error_deg"),
        fitted_params=d.get("fitted_params", {}),
        success=d.get("success", False),
        error_message=d.get("error_message"),
        warning_message=d.get("warning_message"),
        # STEP 5(Neural Residual) 전용 - 학습된 state_dict(base64 문자열).
        # 다른 모델 결과에서는 항상 None(dataclasses.asdict 기반 write
        # 경로는 이미 이 필드를 자동으로 직렬화한다 - project_to_dict 참고).
        neural_state_dict_b64=d.get("neural_state_dict_b64"),
    )


