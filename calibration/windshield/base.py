"""
camera_calibrator.calibration.windshield.base
==================================================

Windshield Refraction Calibration의 공용 타입.

핵심 설계 원칙(사용자 스펙 1/3번, 반드시 지켜야 함):

    Camera -> Base Camera Model(K,D) -> Windshield Correction -> Corrected Projection

Windshield는 calibration.types.CameraModelType에 들어가는 5번째 렌즈 모델이
아니다 - 카메라 자체(Pinhole/Brown-Conrady/Rational/Fisheye)와 완전히 분리된,
그 "뒤"에 있는 별도 광학 계층이다. 그래서 WindshieldModelType은 별도 Enum으로
두고, Base Intrinsic(K,D,base model)은 이 패키지의 어떤 함수도 절대
재최적화하지 않는다 - WindshieldConfig.base_camera_matrix/base_distortion은
항상 이미 확정된 CalibrationResult에서 그대로 스냅샷(.copy())해온 고정값이다.

이 파일은 calibration.types의 기존 타입(Dataset/ResidualStats/RegionalError/
RadialErrorProfile/SpatialErrorMap)을 참조만 하고 새로 정의하지 않는다 -
Baseline의 residual 계산이 그 타입들을 계산하는 기존 함수(residual_stats.py,
radial_profile.py, spatial_error_map.py, models/common.py)를 그대로 재사용하기
때문이다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TypeAlias

import numpy as np

from calibration.types import (
    CameraModelType,
    RadialErrorProfile,
    RegionalError,
    ResidualStats,
    SpatialErrorMap,
)


class WindshieldModelType(str, Enum):
    """Windshield 보정 모델. calibration.types.CameraModelType과 절대 섞지
    않는다 - Windshield는 카메라 모델이 아니라 카메라 뒤에 있는 별도 계층이다.
    """
    BASELINE = "baseline"          # Phase 1 - 보정 없음(항등), 순수 측정 용도
    SPHERICAL = "spherical"        # Phase 2 - Snell 굴절 + 구면 근사
    RESIDUAL_RAY = "residual_ray"  # Phase 3 - Residual Grid(3-A)/RBF(3-B), residual_ray_hint["method"]로 구분
    SPLINE = "spline"              # Phase 4 - Base Sphere + Spline local surface deformation


WindshieldResultKey: TypeAlias = str | WindshieldModelType | tuple[WindshieldModelType, str]


def residual_ray_variant_from_params(fitted_params: dict[str, float] | None) -> str:
    if not fitted_params:
        return "grid"
    method_code = fitted_params.get("residual_ray_method", 0.0)
    if method_code == 1.0:
        return "rbf"
    if method_code == 2.0:
        return "neural"
    return "grid"


def windshield_result_key(model: WindshieldModelType, variant: str = "") -> WindshieldResultKey:
    if model == WindshieldModelType.RESIDUAL_RAY:
        return (model, (variant or "grid").lower())
    return model


def windshield_result_key_for_result(result: "WindshieldCalibrationResult") -> WindshieldResultKey:
    return windshield_result_key(
        result.windshield_model,
        residual_ray_variant_from_params(result.fitted_params),
    )


def windshield_result_key_to_storage(key: WindshieldResultKey) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, tuple):
        model, variant = key
        return f"{model.value}:{variant or 'grid'}"
    return key.value


def windshield_result_key_from_storage(value: str) -> WindshieldResultKey:
    if ":" in value:
        model_value, variant = value.split(":", 1)
        return windshield_result_key(WindshieldModelType(model_value), variant)
    model = WindshieldModelType(value)
    if model == WindshieldModelType.RESIDUAL_RAY:
        return windshield_result_key(model, "grid")
    return model


def windshield_result_key_label(key: WindshieldResultKey) -> str:
    if isinstance(key, tuple):
        _model, variant = key
        if variant == "rbf":
            return "Residual RBF"
        if variant == "neural":
            return "Residual Neural"
        return "Residual Grid"
    labels = {
        WindshieldModelType.BASELINE: "Baseline",
        WindshieldModelType.SPHERICAL: "Spherical",
        WindshieldModelType.SPLINE: "Spline [Advanced]",
    }
    return labels.get(key, key.value)


class WindshieldModel(ABC):
    """모든 Windshield 모델(Baseline/Spherical/Residual Ray/Spline)이 구현해야
    하는 런타임 API. Calibration UI 안에서만 쓰는 임시 함수가 아니라, 향후
    Camera-LiDAR 프로젝션처럼 Calibration 과정과 무관하게 "이미 계산된 결과로
    좌표만 변환하고 싶은" 런타임 사용처가 재사용할 것을 전제로 설계한다
    (사용자 스펙 17번).
    """

    @abstractmethod
    def project_point(self, x: float, y: float, z: float) -> tuple[float, float]:
        """카메라 좌표계의 3D 점(x, y, z) -> Windshield 보정을 반영한 픽셀(u, v)."""
        raise NotImplementedError

    @abstractmethod
    def unproject_pixel(self, u: float, v: float) -> tuple[float, float, float]:
        """픽셀(u, v) -> 카메라 좌표계에서 이 픽셀이 가리키는 정규화된 광선
        방향(dx, dy, dz) (단위 벡터, Windshield 굴절을 반영)."""
        raise NotImplementedError


@dataclass
class WindshieldConfig:
    """Windshield Calibration 실행에 필요한 고정 입력.

    base_camera_matrix/base_distortion/base_model_name은 이미 확정된
    Camera Intrinsic Calibration 결과의 스냅샷이다 - 이 패키지의 어떤 함수도
    이 세 값을 다시 추정하지 않는다(사용자 스펙 3번 핵심 원칙). UI는 이 값을
    읽기 전용으로 표시하고 "Base K,D fixed" 표시를 해야 한다.
    """
    base_model_name: CameraModelType
    base_camera_matrix: np.ndarray
    base_distortion: np.ndarray
    windshield_model: WindshieldModelType = WindshieldModelType.BASELINE
    test_ratio: float = 0.25
    split_seed: int = 42
    # Phase 2+ (Spherical) 전용 고정 파라미터 자리. 지금은 아무 함수도 읽지
    # 않지만, 나중에 스키마를 또 바꾸지 않도록 미리 선언해둔다(사용자 스펙
    # 9번 - n_air/n_glass/glass_thickness는 "처음부터 자동 최적화하지 않는다").
    glass_refractive_index: Optional[float] = None
    glass_thickness_m: Optional[float] = None
    windshield_position_hint: Optional[dict[str, float]] = None
    # Phase 3 (Residual Ray) 전용 설정 자리 - windshield_position_hint와 같은
    # 패턴(범용 dict, additive) - grid_rows/grid_cols/lambda_mag/lambda_smooth
    # 키를 선택적으로 덮어쓸 수 있다(calibration/windshield/residual_ray.py 참고).
    residual_ray_hint: Optional[dict[str, object]] = None
    # Phase 4 (Spline) 전용 설정 자리 - 동일한 패턴(범용 dict, additive).
    # spline_rows/spline_cols/lambda_mag/lambda_smooth/lambda_curve/
    # max_displacement_m/auto_spline 키를 선택적으로 덮어쓸 수 있다
    # (calibration/windshield/spline.py 참고).
    spline_hint: Optional[dict[str, object]] = None


@dataclass
class WindshieldCalibrationResult:
    """Windshield Calibration 한 번 실행(한 모델)의 결과.

    calibration.types.CalibrationResult와 구조를 의도적으로 비슷하게 맞췄다
    (residual_stats/regional_error/radial_profile/spatial_error_map 등 같은
    이름) - 기존 UI 컴포넌트(_PageScrollTableWidget, RadialProfileChartWidget
    등)가 이 결과도 거의 그대로 표시할 수 있게 하기 위함. 다만 CalibrationResult
    자체를 재사용하지 않고 별도 타입으로 둔 이유는, camera_matrix/distortion이
    "이번에 추정한 값"이 아니라 "고정하고 빌려온 값"이라는 의미 차이를 필드
    이름(base_camera_matrix/base_distortion)으로 명확히 구분하기 위함이다.
    """
    windshield_model: WindshieldModelType
    base_model_name: CameraModelType
    base_camera_matrix: np.ndarray
    base_distortion: np.ndarray
    train_frame_ids: list[str] = field(default_factory=list)
    test_frame_ids: list[str] = field(default_factory=list)
    failed_frame_ids: list[str] = field(default_factory=list)
    per_frame_error: dict[str, float] = field(default_factory=dict)
    residual_stats: Optional[ResidualStats] = None          # Train
    test_residual_stats: Optional[ResidualStats] = None     # Hold-out
    regional_error: Optional[RegionalError] = None
    radial_profile: Optional[RadialErrorProfile] = None
    radial_bands: Optional[RadialErrorProfile] = None
    spatial_error_map: Optional[SpatialErrorMap] = None      # dx/dy vector field 데이터 소스도 겸함
    mean_dx: Optional[float] = None
    mean_dy: Optional[float] = None
    # Hold-out(Test) 쪽 Regional/Radial/Spatial/mean dx,dy - STEP 1(Baseline)
    # 구현 당시 test_residual_stats만 저장하고 나머지는 빠뜨렸던 것을 STEP 2에서
    # 추가(additive) - 계산 자체는 이미 하고 있었으므로 Baseline도 이 필드들을
    # 채우도록 함께 고쳤다(baseline.py 참고, 로직 자체를 바꾸는 게 아니라
    # 저장 누락을 보완하는 수정).
    test_regional_error: Optional[RegionalError] = None
    test_radial_profile: Optional[RadialErrorProfile] = None
    test_radial_bands: Optional[RadialErrorProfile] = None
    test_spatial_error_map: Optional[SpatialErrorMap] = None
    test_mean_dx: Optional[float] = None
    test_mean_dy: Optional[float] = None
    # Ray Angular Error(도) - Windshield 모델이 "관측 픽셀의 굴절 광선"과
    # "카메라 좌표계의 실제 목표점 방향" 사이의 각도로 표현한 잔차. Baseline은
    # 굴절 지점(exit point)이라는 개념 자체가 없어 의미 있게 정의할 수 없으므로
    # 항상 None으로 둔다(값을 억지로 만들지 않는다) - Spherical만 채운다.
    ray_angular_error_deg: Optional[float] = None
    test_ray_angular_error_deg: Optional[float] = None
    # Baseline은 항상 빈 dict - Phase 2+(Spherical의 sphere_center/radius,
    # Residual Ray의 grid 참조, Spline의 control point 등)가 여기 채워진다.
    # ModelScore.components와 같은 패턴(범용 dict)을 써서, Phase 2 설계가
    # 확정되기 전에 스키마를 미리 고정하지 않는다.
    fitted_params: dict[str, float] = field(default_factory=dict)
    success: bool = False
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    # Residual Ray Neural(STEP 5) 전용 - 학습된 PyTorch state_dict를 base64
    # 문자열로 인코딩해 담는다. fitted_params는 flat float dict 계약을
    # 유지해야 하므로(YAML export가 모든 값을 float로 쓴다), 실제 weight
    # blob은 여기 별도 필드에 둔다 - "float key 수천 개로 펼치지 않는다"는
    # 요구사항(calibration/windshield/neural_residual.py 모듈 docstring 참고).
    # Grid/RBF/Spline/Baseline/Spherical 결과에서는 항상 None이다.
    neural_state_dict_b64: Optional[str] = None
