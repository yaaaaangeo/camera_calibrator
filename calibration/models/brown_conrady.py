from __future__ import annotations

from calibration.models.extended_pinhole import _calibrate_extended_pinhole_core
from calibration.types import CalibrationMethod, CalibrationResult, CameraConfig, CameraModelType, Dataset


def calibrate_brown_conrady(
    dataset: Dataset,
    camera_config: CameraConfig,
    *,
    fix_tangent_dist: bool = False,
    estimate_uncertainty_bootstrap: bool = False,
    n_bootstrap: int = 20,
    bootstrap_seed: int = 42,
    bootstrap_jobs: int = 1,
) -> CalibrationResult:
    """Standard Brown-Conrady 5-coefficient pinhole calibration.

    Estimates fx, fy, cx, cy plus k1, k2, p1, p2, k3. Rational k4-k6
    coefficients are always disabled - this is a fixed property of the
    Brown-Conrady model, not a runtime option (there is no way to make this
    function produce an 8-coefficient result; use calibrate_extended_pinhole()
    for that).
    """
    result = _calibrate_extended_pinhole_core(
        dataset,
        camera_config,
        calibration_flags=0,
        fix_tangent_dist=fix_tangent_dist,
        estimate_uncertainty_bootstrap=estimate_uncertainty_bootstrap,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        bootstrap_jobs=bootstrap_jobs,
    )
    result.model_name = CameraModelType.BROWN_CONRADY
    result.calibration_method = CalibrationMethod.STANDARD
    return result
