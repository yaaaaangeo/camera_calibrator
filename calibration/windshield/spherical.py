"""
camera_calibrator.calibration.windshield.spherical
=======================================================

Phase 2 (미구현) - Windshield를 구면(sphere)으로 근사하고 Snell의 법칙으로
굴절을 계산하는 모델.

    3D Point -> Windshield Surface(구면) -> Snell Refraction -> Camera Ray
             -> 고정된 Base K,D -> Pixel

TODO (Phase 2, 구현 시 지킬 것):
    - 미지수는 sphere_center, sphere_radius, windshield pose(orientation)
      정도로 시작한다. n_air/n_glass/glass_thickness는
      calibration.windshield.base.WindshieldConfig에 이미 자리를 마련해뒀고,
      처음에는 Config에서 고정값으로 받아 최적화 대상에 넣지 않는다
      (사용자 스펙 9번 원칙).
    - 초기 포즈(rvec/tvec)는 calibration.windshield.base_projection.
      solve_poses_fixed_intrinsics()로 구한 값을 그대로 시작점으로 쓴다 -
      Base Camera Model K,D는 이 단계에서도 절대 재최적화하지 않는다.
    - Optimization은 scipy.optimize.least_squares() 사용을 권장(사용자 스펙 9번).
    - Validation은 calibration/validation.py의 cv2.solvePnP/projectPoints
      기반 함수를 그대로 쓰면 이론적으로 틀린다(중심 투영을 전제로 함) -
      calibration/windshield/validation.py에 이 모델 전용 평가 경로를 추가해야 한다.
"""

from __future__ import annotations

from calibration.types import CameraConfig, Dataset
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldConfig, WindshieldModel


class SphericalWindshieldModel(WindshieldModel):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Spherical windshield refraction model (sphere fit + Snell's law) is "
            "planned for Phase 2 and is not implemented yet. "
            "See calibration/windshield/spherical.py."
        )

    def project_point(self, x: float, y: float, z: float) -> tuple[float, float]:
        raise NotImplementedError("Phase 2 (Spherical) is not implemented yet.")

    def unproject_pixel(self, u: float, v: float) -> tuple[float, float, float]:
        raise NotImplementedError("Phase 2 (Spherical) is not implemented yet.")


def calibrate_spherical(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    train_ids: list[str],
    test_ids: list[str],
) -> WindshieldCalibrationResult:
    raise NotImplementedError(
        "Spherical windshield calibration (sphere center/radius + Snell's law fit "
        "via scipy.optimize.least_squares, base K/D fixed) is planned for Phase 2 "
        "and is not implemented yet. See calibration/windshield/spherical.py."
    )
