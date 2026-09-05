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

from calibration.windshield.base import WindshieldCalibrationResult, WindshieldModel, WindshieldModelType
from calibration.windshield.baseline import BaselineWindshieldModel


def build_projector(result: WindshieldCalibrationResult) -> WindshieldModel:
    if result.windshield_model == WindshieldModelType.BASELINE:
        return BaselineWindshieldModel(
            result.base_camera_matrix, result.base_distortion, result.base_model_name
        )
    if result.windshield_model == WindshieldModelType.SPHERICAL:
        raise NotImplementedError("Spherical windshield model is not implemented yet (Phase 2).")
    if result.windshield_model == WindshieldModelType.RESIDUAL_RAY:
        raise NotImplementedError("Residual-Ray windshield model is not implemented yet (Phase 3).")
    if result.windshield_model == WindshieldModelType.SPLINE:
        raise NotImplementedError("Spline windshield model is not implemented yet (Phase 4).")
    raise ValueError(f"알 수 없는 windshield 모델: {result.windshield_model}")
