"""
camera_calibrator.calibration.windshield
=============================================

Windshield Refraction Calibration - Base Camera Model(calibration.types.
CameraModelType)과 완전히 분리된 별도 광학 계층. 이미 확정된 Camera Intrinsic
Calibration 결과(K, D, base model)를 고정한 채, 앞유리 굴절로 생기는
기하학적(geometric) 픽셀 변위만 독립적으로 측정/보정한다.

    Camera -> Base Camera Model(K,D 고정) -> Windshield Correction -> Corrected Projection

Windshield Reflection(글레어/고스트/이중상)은 이 패키지가 다루지 않는다 -
그건 광도(photometric) 문제이고, 이건 순수 기하 문제다(별도 모듈 예정).
"""

from __future__ import annotations

from calibration.windshield.base import (
    WindshieldCalibrationResult,
    WindshieldConfig,
    WindshieldModel,
    WindshieldModelType,
)
from calibration.windshield.baseline import BaselineWindshieldModel, calibrate_baseline
from calibration.windshield.projection import build_projector
from calibration.windshield.validation import run_windshield_calibration, split_windshield_train_test

__all__ = [
    "WindshieldCalibrationResult",
    "WindshieldConfig",
    "WindshieldModel",
    "WindshieldModelType",
    "BaselineWindshieldModel",
    "calibrate_baseline",
    "build_projector",
    "run_windshield_calibration",
    "split_windshield_train_test",
]
