"""
camera_calibrator.calibration.windshield.residual_ray
==========================================================

Phase 3 (미구현) - 정확한 Windshield CAD/곡률을 모르는 실차에서도 쓸 수 있는
"Base Ray + Residual Correction" 모델.

    Pixel -> 고정된 Base K,D -> Base Ray -> Residual Ray Correction -> Corrected Ray

수학적으로 최소한 다음 형태를 지원해야 한다:

    d_corrected = normalize(d_base + Δd(u, v))

TODO (Phase 3, 구현 시 지킬 것):
    - Neural Network는 쓰지 않는다(사용자 스펙 11번) - 우선 Residual Grid +
      Interpolation, 또는 RBF interpolation 같은 deterministic 방식으로
      (u, v) -> local ray correction Δd(u, v)를 추정한다.
    - Base Ray는 calibration.windshield.baseline.BaselineWindshieldModel.
      unproject_pixel()이 이미 계산하는 것과 동일한 정의(고정 K,D 기반 단위
      벡터)를 재사용한다 - Residual Ray 전용으로 다시 정의하지 않는다.
    - 초기 포즈는 calibration.windshield.base_projection.
      solve_poses_fixed_intrinsics()로 구한다.
    - Grid 해상도, 보간 방식(bilinear/RBF)은 fitted_params에 기록해서
      Validation/Comparison 화면과 export/windshield.py가 값을 읽을 수 있게 한다.
"""

from __future__ import annotations

from calibration.types import CameraConfig, Dataset
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldConfig, WindshieldModel


class ResidualRayWindshieldModel(WindshieldModel):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Residual-Ray windshield model (grid/RBF ray correction) is planned "
            "for Phase 3 and is not implemented yet. "
            "See calibration/windshield/residual_ray.py."
        )

    def project_point(self, x: float, y: float, z: float) -> tuple[float, float]:
        raise NotImplementedError("Phase 3 (Residual Ray) is not implemented yet.")

    def unproject_pixel(self, u: float, v: float) -> tuple[float, float, float]:
        raise NotImplementedError("Phase 3 (Residual Ray) is not implemented yet.")


def calibrate_residual_ray(
    windshield_dataset: Dataset,
    config: WindshieldConfig,
    camera_config: CameraConfig,
    train_ids: list[str],
    test_ids: list[str],
) -> WindshieldCalibrationResult:
    raise NotImplementedError(
        "Residual-Ray windshield calibration (deterministic grid + interpolation "
        "or RBF interpolation of per-pixel ray correction, base K/D fixed) is "
        "planned for Phase 3 and is not implemented yet. "
        "See calibration/windshield/residual_ray.py."
    )
