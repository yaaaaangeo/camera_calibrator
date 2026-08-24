from __future__ import annotations

import numpy as np
import pytest

from calibration.model_refitting import (
    pixels_to_normalized_rays,
    project_rational_pinhole,
    refit_extended_pinhole_to_pinhole,
    sample_image_points,
)


K = np.array([[850.0, 0.0, 640.0], [0.0, 840.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
IMAGE_SIZE = (1280, 720)


def test_projection_matches_cv2_project_points():
    cv2 = pytest.importorskip("cv2")
    pixels = sample_image_points(IMAGE_SIZE, grid_size=(12, 8))
    rays = pixels_to_normalized_rays(pixels, K)
    D8 = np.array([-0.28, 0.12, 0.001, -0.002, -0.03, 0.04, -0.02, 0.01])

    ours = project_rational_pinhole(rays, K, D8)
    obj = rays.reshape(-1, 1, 3)
    expected, _ = cv2.projectPoints(
        obj,
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        K,
        D8.reshape(-1, 1),
    )
    np.testing.assert_allclose(ours, expected.reshape(-1, 2), rtol=1e-10, atol=1e-10)


def test_zero_denominator_terms_recover_original_5_coefficients():
    pytest.importorskip("scipy")
    D8 = np.array([-0.22, 0.08, 0.001, -0.0015, -0.015, 0.0, 0.0, 0.0])

    result = refit_extended_pinhole_to_pinhole(
        K, D8, IMAGE_SIZE, mode="distortion_only", grid_size=(30, 18)
    )

    np.testing.assert_allclose(result.K_refitted, K, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.D_refitted.reshape(-1), D8[:5], rtol=1e-5, atol=1e-7)
    assert result.error.rmse_px < 1e-6


def test_optimized_refit_beats_naive_truncation():
    pytest.importorskip("scipy")
    D8 = np.array([-0.36, 0.18, 0.001, -0.001, -0.04, 0.12, -0.08, 0.025])

    result = refit_extended_pinhole_to_pinhole(
        K, D8, IMAGE_SIZE, mode="full", grid_size=(50, 30), regularization=1e-4
    )

    assert result.optimization.success
    assert result.error.rmse_px < result.naive_error.rmse_px
    assert result.region_error["edge"].rmse_px < result.naive_region_error["edge"].rmse_px


def test_strong_distortion_finishes_with_finite_result():
    pytest.importorskip("scipy")
    D8 = np.array([-0.65, 0.35, 0.002, -0.002, -0.12, 0.22, -0.10, 0.04])

    result = refit_extended_pinhole_to_pinhole(
        K, D8, IMAGE_SIZE, mode="full", grid_size=(40, 24), edge_weighting=True, loss="soft_l1"
    )

    assert np.all(np.isfinite(result.K_refitted))
    assert np.all(np.isfinite(result.D_refitted))
    assert np.isfinite(result.error.rmse_px)
    assert np.isfinite(result.region_error["edge"].p95_px)


def test_full_and_distortion_only_modes():
    pytest.importorskip("scipy")
    D8 = np.array([-0.32, 0.16, 0.001, -0.001, -0.03, 0.08, -0.035, 0.01])

    dist_only = refit_extended_pinhole_to_pinhole(
        K, D8, IMAGE_SIZE, mode="distortion_only", grid_size=(32, 20)
    )
    full = refit_extended_pinhole_to_pinhole(
        K, D8, IMAGE_SIZE, mode="full", grid_size=(32, 20)
    )

    np.testing.assert_allclose(dist_only.K_refitted, K, rtol=0, atol=1e-12)
    assert full.K_refitted.shape == (3, 3)
    assert full.D_refitted.reshape(-1).shape == (5,)
    assert full.error.rmse_px <= dist_only.error.rmse_px + 1e-6
