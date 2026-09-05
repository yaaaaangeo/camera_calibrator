"""
camera_calibrator.calibration.windshield.validation
========================================================

Windshield 전용 Hold-out 오케스트레이션.

Train/Test 분할 로직 자체는 calibration.validation.split_train_test를 그대로
재사용한다 - stratified split은 투영 방식과 무관한 로직이라 Windshield용으로
다시 만들 이유가 없다(사용자 스펙 13번 "Split Logic -> 재사용 가능" 원칙).
이 모듈은 그 위에 windshield_model(Baseline/Spherical/Residual Ray/Spline)
dispatch만 얹는다 - 새 모델이 추가될 때 UI가 아니라 이 한 곳만 고치면 된다.
"""

from __future__ import annotations

from calibration.types import CameraConfig, Dataset
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldConfig, WindshieldModelType
from calibration.windshield.baseline import calibrate_baseline


def split_windshield_train_test(
    windshield_dataset: Dataset,
    camera_config: CameraConfig,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """calibration.validation.split_train_test의 windshield 네임스페이스
    래퍼. 지금은 순수 pass-through지만, Phase 2/3에서 windshield 전용 기준
    (예: 유리를 통과하는 입사각)으로 stratify해야 할 때 calibration/validation.py를
    건드리지 않고 여기만 바꿀 수 있게 별도 진입점으로 둔다.
    """
    return split_train_test(windshield_dataset, camera_config, test_ratio, seed)


def run_windshield_calibration(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
) -> WindshieldCalibrationResult:
    """UI가 호출하는 단일 진입점. Split -> windshield_model dispatch -> 결과."""
    train_ids, test_ids = split_windshield_train_test(
        windshield_dataset, camera_config, config.test_ratio, config.split_seed
    )

    if config.windshield_model == WindshieldModelType.BASELINE:
        return calibrate_baseline(windshield_dataset, config, camera_config, train_ids, test_ids)

    if config.windshield_model == WindshieldModelType.SPHERICAL:
        from calibration.windshield.spherical import calibrate_spherical
        return calibrate_spherical(windshield_dataset, config, camera_config, train_ids, test_ids)

    if config.windshield_model == WindshieldModelType.RESIDUAL_RAY:
        hint = config.residual_ray_hint or {}
        method = str(hint.get("method", "grid")).lower()
        if method == "rbf":
            from calibration.windshield.residual_rbf import calibrate_residual_rbf
            return calibrate_residual_rbf(windshield_dataset, config, camera_config, train_ids, test_ids)
        if method == "neural":
            from calibration.windshield.neural_residual import calibrate_neural_residual
            return calibrate_neural_residual(windshield_dataset, config, camera_config, train_ids, test_ids)
        from calibration.windshield.residual_ray import calibrate_residual_ray
        return calibrate_residual_ray(windshield_dataset, config, camera_config, train_ids, test_ids)

    if config.windshield_model == WindshieldModelType.SPLINE:
        from calibration.windshield.spline import calibrate_spline
        return calibrate_spline(windshield_dataset, config, camera_config, train_ids, test_ids)

    raise ValueError(f"알 수 없는 windshield 모델: {config.windshield_model}")
