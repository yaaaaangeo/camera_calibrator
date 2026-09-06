"""
STEP 8B - Ghost Suppression 테스트.

`tests/test_reflection_suppression.py`(Reflection Suppression)와 완전히
독립된 테스트 파일이다 - Ghost Suppression은 Reflection Suppression 모델을
절대 재사용하지 않는다.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from calibration.windshield.ghost.evaluator import evaluate_ghost_point_source
from calibration.windshield.ghost.suppression import (
    build_dense_fields,
    fit_ghost_field_constant,
    fit_ghost_field_from_spatial_map,
    load_ghost_model,
    save_ghost_model,
    suppress_ghost,
)
from calibration.windshield.ghost.synthetic import make_constant_offset_ghost_sample
from calibration.windshield.ghost.types import GhostEvaluationConfig, GhostField, GhostSpatialCell


def _point_source_config(**overrides) -> GhostEvaluationConfig:
    base = dict(mode="point_source", bright_source_threshold=30.0, gaussian_sigma=0.4, min_peak_distance_px=1.5)
    base.update(overrides)
    return GhostEvaluationConfig(**base)


def _two_dot_image(h=240, w=320, centers=((100, 120), (220, 80)), radius=1) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.float32)
    for cx, cy in centers:
        cv2.circle(img, (cx, cy), radius, (255.0, 255.0, 255.0), -1)
    return img


def _ghosted_observed_uint8(dx=4.0, dy=-2.0, alpha=0.15):
    clean = _two_dot_image()
    sample = make_constant_offset_ghost_sample(clean, dx=dx, dy=dy, alpha=alpha)
    return np.clip(sample.observed, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Core suppress_ghost() behavior
# ---------------------------------------------------------------------------

def test_suppress_ghost_reduces_ghost_strength_while_keeping_main_edge():
    observed = _ghosted_observed_uint8()
    field = fit_ghost_field_constant(
        image_width=320, image_height=240, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.15,
    )
    supp = suppress_ghost(observed, field)
    assert supp.success
    assert supp.suppressed_image is not None

    cfg = _point_source_config()
    before = evaluate_ghost_point_source(observed, cfg)
    after = evaluate_ghost_point_source(supp.suppressed_image, cfg)

    # Ghost strength가 줄어들어야 한다(Before > After) - 사용자 스펙 required test.
    before_strength = before.mean_strength_ratio or 0.0
    after_strength = after.mean_strength_ratio or 0.0
    assert after_strength < before_strength

    # Main edge(=main peak 자체)는 유지되어야 한다 - suppressed 이미지에서
    # main light source 위치의 밝기가 크게 손실되면 안 된다.
    main_before = float(cv2.cvtColor(observed, cv2.COLOR_BGR2GRAY)[120, 100])
    main_after = float(cv2.cvtColor(supp.suppressed_image, cv2.COLOR_BGR2GRAY)[120, 100])
    assert main_after > main_before * 0.7


def test_suppress_ghost_iterations_zero_is_identity():
    observed = _ghosted_observed_uint8()
    field = fit_ghost_field_constant(
        image_width=320, image_height=240, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.15,
    )
    supp = suppress_ghost(observed, field, iterations=0)
    assert supp.success
    assert np.array_equal(supp.suppressed_image, observed)
    assert supp.mean_correction == pytest.approx(0.0)


def test_suppress_ghost_returns_predicted_ghost_and_correction_map():
    observed = _ghosted_observed_uint8()
    field = fit_ghost_field_constant(
        image_width=320, image_height=240, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.15,
    )
    supp = suppress_ghost(observed, field)
    assert supp.predicted_ghost_image is not None
    assert supp.correction_map is not None
    assert supp.correction_map.shape[:2] == observed.shape[:2]


# ---------------------------------------------------------------------------
# Safety Guard - NaN/Inf, size mismatch, exception -> fallback to original.
# ---------------------------------------------------------------------------

def test_suppress_ghost_falls_back_on_nan_input():
    bad = np.full((240, 320, 3), np.nan, dtype=np.float32)
    field = fit_ghost_field_constant(image_width=320, image_height=240, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.15)
    result = suppress_ghost(bad, field)
    assert result.success is False
    assert result.fell_back_to_original is True


def test_suppress_ghost_falls_back_on_size_mismatch():
    observed = _ghosted_observed_uint8()
    field = fit_ghost_field_constant(image_width=999.0, image_height=999.0, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.15)
    result = suppress_ghost(observed, field)
    assert result.success is False
    assert result.fell_back_to_original is True
    assert np.array_equal(result.suppressed_image, observed)


def test_suppress_ghost_correction_never_exceeds_configured_max():
    observed = _ghosted_observed_uint8(alpha=0.9)
    field = fit_ghost_field_constant(image_width=320, image_height=240, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.9)
    max_correction = 0.1
    supp = suppress_ghost(observed, field, max_correction=max_correction)
    assert supp.success
    # correction_map은 [0,1] 정규화가 아닌 uint8 scale(0-255)로 반환되므로
    # max_correction*255 기준(약간의 quantization 여유 포함)으로 비교한다.
    assert np.max(supp.correction_map) <= max_correction * 255.0 + 1.0


# ---------------------------------------------------------------------------
# GhostField 구성 - constant / spatial map 기반 fit.
# ---------------------------------------------------------------------------

def test_fit_ghost_field_constant_broadcasts_single_value():
    field = fit_ghost_field_constant(image_width=320, image_height=240, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.15)
    assert field.offset_x.shape == (1, 1)
    dx_dense, dy_dense, alpha_dense = build_dense_fields(field)
    assert dx_dense.shape == (240, 320)
    assert np.all(dx_dense == 4.0)
    assert np.all(dy_dense == -2.0)
    assert np.all(alpha_dense == pytest.approx(0.15))


def test_fit_ghost_field_from_spatial_map_preserves_per_cell_values():
    cells = [
        GhostSpatialCell(row=0, col=0, mean_offset_x_px=3.0, mean_offset_y_px=0.0, mean_distance_px=3.0, mean_strength_ratio=0.1, sample_count=1),
        GhostSpatialCell(row=0, col=1, mean_offset_x_px=0.0, mean_offset_y_px=0.0, mean_distance_px=0.0, mean_strength_ratio=0.0, sample_count=0),
        GhostSpatialCell(row=0, col=2, mean_offset_x_px=7.0, mean_offset_y_px=0.0, mean_distance_px=7.0, mean_strength_ratio=0.2, sample_count=1),
    ]
    field = fit_ghost_field_from_spatial_map(
        cells, image_width=300, image_height=100, rows=1, cols=3,
        default_offset_x=5.0, default_offset_y=0.0, default_strength=0.15,
    )
    assert field.offset_x[0, 0] == pytest.approx(3.0)
    assert field.offset_x[0, 1] == pytest.approx(5.0)  # empty cell -> default
    assert field.offset_x[0, 2] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# YAML round-trip persistence.
# ---------------------------------------------------------------------------

def test_save_and_load_ghost_model_round_trips():
    field = fit_ghost_field_constant(image_width=320, image_height=240, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.15)
    tmp = tempfile.mktemp(suffix=".yml")
    try:
        save_ghost_model(field, tmp, metadata={"dataset": "unit_test"})
        loaded = load_ghost_model(tmp)
        assert np.allclose(loaded.offset_x, field.offset_x)
        assert np.allclose(loaded.offset_y, field.offset_y)
        assert np.allclose(loaded.strength, field.strength)
        assert loaded.image_width == field.image_width
        assert loaded.image_height == field.image_height
        assert loaded.model_version == field.model_version
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def test_load_ghost_model_missing_file_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        load_ghost_model("this_file_does_not_exist_ghost_model.yml")


# ---------------------------------------------------------------------------
# Before/After using the SAME STEP 8A evaluator (required regression test).
# ---------------------------------------------------------------------------

def test_before_after_suppression_uses_same_evaluator_and_shows_improvement():
    observed = _ghosted_observed_uint8()
    field = fit_ghost_field_constant(image_width=320, image_height=240, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.15)
    cfg = _point_source_config()

    before = evaluate_ghost_point_source(observed, cfg)
    supp = suppress_ghost(observed, field)
    after = evaluate_ghost_point_source(supp.suppressed_image, cfg)

    assert before.detection_count >= after.detection_count
    if before.mean_strength_ratio is not None and after.mean_strength_ratio is not None:
        assert after.mean_strength_ratio < before.mean_strength_ratio
