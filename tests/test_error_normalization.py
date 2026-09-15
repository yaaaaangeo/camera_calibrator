from __future__ import annotations

import numpy as np
import pytest

from calibration.diagnosis import diagnose_calibration
from calibration.error_normalization import (
    fhd_equivalent_error,
    mean_focal_length,
    normalized_reprojection_error,
    reference_equivalent_error,
)
from calibration.recommender import compute_final_result
from calibration.sanity_check import run_sanity_check
from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    ValidationResult,
)


def _matrix(focal: float, width: int, height: int) -> np.ndarray:
    return np.array(
        [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
        dtype=float,
    )


@pytest.mark.parametrize(
    ("size", "focal", "error"),
    [
        ((1920, 1080), 950.0, 0.4),
        ((3840, 2160), 1900.0, 0.8),
        ((1280, 720), 950.0 * 2 / 3, 0.4 * 2 / 3),
    ],
)
def test_same_scene_scale_has_same_normalized_and_fhd_equivalent_error(size, focal, error):
    matrix = _matrix(focal, *size)
    assert normalized_reprojection_error(error, matrix) == pytest.approx(0.4 / 950.0)
    assert reference_equivalent_error(error, matrix) == pytest.approx(0.4)
    assert fhd_equivalent_error(error, size) == pytest.approx(0.4)


def test_non_16_by_9_is_normalized_but_not_fhd_scaled():
    matrix = _matrix(800.0, 1280, 800)
    assert normalized_reprojection_error(0.4, matrix) == pytest.approx(0.0005)
    assert fhd_equivalent_error(0.4, (1280, 800)) is None


@pytest.mark.parametrize(
    "matrix",
    [None, np.eye(2), np.zeros((3, 3)), np.diag([np.nan, 900.0, 1.0]), np.diag([-1.0, 900.0, 1.0])],
)
def test_invalid_or_legacy_camera_matrix_is_safe(matrix):
    assert mean_focal_length(matrix) is None
    assert normalized_reprojection_error(0.5, matrix) is None
    assert reference_equivalent_error(0.5, matrix) is None


def _result(size: tuple[int, int], focal: float, scale: float):
    model = CameraModelType.BROWN_CONRADY
    cal = CalibrationResult(
        model_name=model,
        camera_matrix=_matrix(focal, *size),
        distortion=np.zeros(5),
        rms_error=0.4 * scale,
        success=True,
    )
    val = ValidationResult(
        train_rms=0.4 * scale,
        test_rms=0.9 * scale,
        edge_rms=1.1 * scale,
        straightness_residual=0.3 * scale,
        success=True,
    )
    return model, cal, val


def test_absolute_diagnosis_grade_and_confidence_are_resolution_invariant():
    fhd = _result((1920, 1080), 950.0, 1.0)
    uhd = _result((3840, 2160), 1900.0, 2.0)

    reports = [diagnose_calibration(cal, val) for _, cal, val in (fhd, uhd)]
    assert [{p.code for p in report.patterns} for report in reports][0] == {
        p.code for p in reports[1].patterns
    }

    finals = [
        compute_final_result(model, {model: cal}, {model: val})
        for model, cal, val in (fhd, uhd)
    ]
    assert finals[0].overall_grade == finals[1].overall_grade
    assert finals[0].confidence.components == finals[1].confidence.components


def test_sanity_rms_threshold_is_resolution_invariant():
    outputs = []
    for size, focal, scale in (((1920, 1080), 950.0, 1.0), ((3840, 2160), 1900.0, 2.0)):
        _, cal, _ = _result(size, focal, scale)
        config = CameraConfig(width=size[0], height=size[1])
        outputs.append({issue.code for issue in run_sanity_check(cal, config).issues})
    assert outputs[0] == outputs[1]


def test_missing_image_size_keeps_raw_working_and_fhd_unavailable():
    matrix = _matrix(950.0, 1920, 1080)
    assert normalized_reprojection_error(0.4, matrix) == pytest.approx(0.4 / 950.0)
    assert fhd_equivalent_error(0.4, None) is None


def test_resolution_invariance_holds_after_corner_outlier_exclusion():
    """Test 9 보강 (계획 문서 11번) - corner-level outlier(excluded_corner_indices)
    로 코너 하나를 제외한 뒤에도, 1920x1080과 정확히 2배 스케일된
    3840x2160에서 normalized RMS/P95가 여전히 동일해야 한다. Problem 4의
    corner-exclusion 일관성 수정(active_correspondences)과 Problem 8의
    해상도 정규화가 같은 경로에서 함께 올바르게 작동하는지 확인한다.
    """
    from calibration.models.common import active_correspondences
    from calibration.residual_stats import compute_residual_stats
    from calibration.types import DetectionResult

    rng = np.random.default_rng(0)
    n_points = 20
    object_points = rng.uniform(-0.1, 0.1, (n_points, 3)).astype(np.float64)
    object_points[:, 2] = 0.0
    base_errors_px = rng.uniform(0.1, 1.0, n_points)
    excluded_index = 5

    def _stats_for_scale(scale: float):
        corners = rng.uniform(0, 1920 * scale, (n_points, 2)).astype(np.float64)
        det = DetectionResult(
            image_id="f", success=True,
            object_points=object_points.reshape(-1, 1, 3),
            corners=corners.reshape(-1, 1, 2),
            num_corners=n_points,
            excluded_corner_indices=[excluded_index],
        )
        _, active_corners = active_correspondences(det)
        assert active_corners.shape[0] == n_points - 1
        errors = np.delete(base_errors_px, excluded_index) * scale
        return compute_residual_stats(errors.tolist())

    stats_1x = _stats_for_scale(1.0)
    stats_2x = _stats_for_scale(2.0)

    K_1x = _matrix(950.0, 1920, 1080)
    K_2x = _matrix(1900.0, 3840, 2160)

    assert normalized_reprojection_error(stats_1x.rmse, K_1x) == pytest.approx(
        normalized_reprojection_error(stats_2x.rmse, K_2x), rel=1e-9
    )
    assert normalized_reprojection_error(stats_1x.p95, K_1x) == pytest.approx(
        normalized_reprojection_error(stats_2x.p95, K_2x), rel=1e-6
    )

    # And the exclusion must have actually removed exactly one point's worth
    # of data relative to not excluding anything - otherwise this test
    # wouldn't be exercising Problem 4's fix at all.
    full_stats_1x = compute_residual_stats(base_errors_px.tolist())
    assert stats_1x.n == full_stats_1x.n - 1
