"""
camera_calibrator.calibration.windshield.spline
====================================================

Phase 4 (Advanced, 미구현) - Spherical 보정 후에도 특정 image region에
systematic residual이 남을 때 쓰는 spline control point 기반 surface refinement.

TODO (Phase 4, 구현 시 지킬 것):
    - 우선순위가 가장 낮다 - Baseline/Spherical/Residual Ray가 안정적으로
      동작한 뒤에만 착수한다(사용자 스펙 21번 순서).
    - Spherical(Phase 2) 결과의 Regional/Radial/Spatial Error Map에서 특정
      영역(예: Left/Right가 Center보다 뚜렷하게 큰 패턴)이 계속 남는지부터
      진단하고, 그 근거가 있을 때만 control point 개수/배치를 정한다.
    - Residual Ray(Phase 3)와 마찬가지로 Base K,D는 계속 고정한다.
"""

from __future__ import annotations

from calibration.types import CameraConfig, Dataset
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldConfig, WindshieldModel


class SplineWindshieldModel(WindshieldModel):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Spline windshield surface model is planned for Phase 4 (Advanced) "
            "and is not implemented yet. See calibration/windshield/spline.py."
        )

    def project_point(self, x: float, y: float, z: float) -> tuple[float, float]:
        raise NotImplementedError("Phase 4 (Spline) is not implemented yet.")

    def unproject_pixel(self, u: float, v: float) -> tuple[float, float, float]:
        raise NotImplementedError("Phase 4 (Spline) is not implemented yet.")


def calibrate_spline(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    train_ids: list[str],
    test_ids: list[str],
) -> WindshieldCalibrationResult:
    raise NotImplementedError(
        "Spline windshield surface calibration is planned for Phase 4 (Advanced) "
        "and is not implemented yet. See calibration/windshield/spline.py."
    )
