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


# ===========================================================================
# STEP 8 stabilization 7 - Detail Retention / Over-Suppression Metrics
# ===========================================================================

from calibration.windshield.ghost.suppression import compute_reconstruction_metrics


def test_suppress_ghost_populates_detail_retention_metrics():
    observed = _ghosted_observed_uint8()
    field = fit_ghost_field_constant(image_width=320, image_height=240, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.15)
    result = suppress_ghost(observed, field)
    assert result.success
    assert result.edge_retention is not None
    assert result.clean_region_change is not None
    assert result.over_suppression_score is not None
    # Main edge가 어느 정도 유지되어야 한다(과도한 스무딩으로 완전히
    # 사라지면 안 됨) - 완벽한 1.0을 요구하지 않는다.
    assert result.edge_retention > 0.5


def test_suppress_ghost_over_suppression_score_is_bounded_and_finite():
    observed = _ghosted_observed_uint8(alpha=0.15)
    field = fit_ghost_field_constant(image_width=320, image_height=240, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.15)
    result = suppress_ghost(observed, field)
    assert np.isfinite(result.over_suppression_score)
    assert result.over_suppression_score >= 0.0


def test_suppress_ghost_zero_iterations_has_perfect_edge_retention_and_zero_change():
    """iterations=0이면 이미지가 전혀 바뀌지 않으므로 edge_retention은
    정확히 1.0, clean_region_change/over_suppression_score는 0에 가까워야
    한다."""
    observed = _ghosted_observed_uint8()
    field = fit_ghost_field_constant(image_width=320, image_height=240, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.15)
    result = suppress_ghost(observed, field, iterations=0)
    assert result.edge_retention == pytest.approx(1.0, abs=1e-4)
    assert result.clean_region_change == pytest.approx(0.0, abs=1e-6)
    assert result.over_suppression_score == pytest.approx(0.0, abs=1e-4)


def test_compute_reconstruction_metrics_zero_for_identical_images():
    clean = _ghosted_observed_uint8()
    metrics = compute_reconstruction_metrics(clean, clean)
    assert metrics.mae == pytest.approx(0.0)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.psnr_db is None  # mse==0이면 PSNR은 정의되지 않음(무한대) - None으로 남긴다


def test_compute_reconstruction_metrics_nonzero_for_different_images():
    clean = np.zeros((32, 32, 3), dtype=np.uint8)
    other = np.full((32, 32, 3), 10, dtype=np.uint8)
    metrics = compute_reconstruction_metrics(clean, other)
    assert metrics.mae == pytest.approx(10.0)
    assert metrics.rmse == pytest.approx(10.0)
    assert metrics.psnr_db is not None and metrics.psnr_db > 0


def test_suppression_success_requires_ghost_down_and_edge_retained_simultaneously():
    """사용자 스펙 7-E번 - Ghost Strength만 내려가는 것으로는 부족하다:
    같은 suppress_ghost() 호출 하나의 결과 안에서 Ghost 감소와 edge 유지가
    동시에 확인되어야 한다."""
    observed = _ghosted_observed_uint8()
    field = fit_ghost_field_constant(image_width=320, image_height=240, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.15)
    cfg = _point_source_config()

    before = evaluate_ghost_point_source(observed, cfg)
    result = suppress_ghost(observed, field)
    after = evaluate_ghost_point_source(result.suppressed_image, cfg)

    ghost_reduced = (after.mean_strength_ratio or 0.0) < (before.mean_strength_ratio or 0.0)
    edge_retained = result.edge_retention > 0.5
    assert ghost_reduced and edge_retained


# ===========================================================================
# STEP 8 stabilization 3 - Dataset-level GhostField fit + diagnostics round-trip
# ===========================================================================

from calibration.windshield.ghost.evaluator import evaluate_ghost_dataset
from calibration.windshield.ghost.suppression import fit_ghost_field_from_dataset
from calibration.windshield.ghost.types import GhostFieldDiagnostics


def test_dataset_ghost_field_yaml_round_trips_with_diagnostics():
    rng = np.random.default_rng(3)
    per_frame = []
    cfg = _point_source_config(bright_source_threshold=20.0, spatial_rows=1, spatial_cols=1)
    for i in range(10):
        observed = _ghosted_observed_uint8(
            dx=4.0 + rng.normal(0, 0.1), dy=-2.0 + rng.normal(0, 0.1), alpha=0.15 + rng.normal(0, 0.005),
        )
        per_frame.append(evaluate_ghost_point_source(observed, cfg, pair_id=f"f{i}"))
    dataset_result = evaluate_ghost_dataset(per_frame, mode="point_source")

    field = fit_ghost_field_from_dataset(dataset_result, image_width=320, image_height=240, rows=1, cols=1)
    assert field.diagnostics is not None
    assert field.diagnostics.num_frames == 10

    tmp = tempfile.mktemp(suffix=".yml")
    try:
        save_ghost_model(field, tmp)
        loaded = load_ghost_model(tmp)
        assert loaded.diagnostics is not None
        assert loaded.diagnostics.num_frames == field.diagnostics.num_frames
        assert loaded.diagnostics.fit_stability == pytest.approx(field.diagnostics.fit_stability)
        np.testing.assert_allclose(loaded.offset_x, field.offset_x)
        np.testing.assert_allclose(loaded.offset_y, field.offset_y)
        np.testing.assert_allclose(loaded.strength, field.strength)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def test_ghost_model_without_diagnostics_still_round_trips():
    """`GhostFieldDiagnostics` 없이 만든(예: 예전 constant-fit) `GhostField`도
    여전히 저장/복원 가능해야 한다 - diagnostics는 Optional이다."""
    field = fit_ghost_field_constant(image_width=100, image_height=100, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.15)
    assert field.diagnostics is None
    tmp = tempfile.mktemp(suffix=".yml")
    try:
        save_ghost_model(field, tmp)
        loaded = load_ghost_model(tmp)
        assert loaded.diagnostics is None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ===========================================================================
# STEP 8 stabilization 8 - Suppression evaluator mode dispatch
# ===========================================================================

from calibration.windshield.ghost.evaluator import evaluate_ghost_image


def test_suppression_before_after_can_use_shared_dispatcher():
    """Suppression Worker가 evaluate_ghost_point_source를 하드코딩하는
    대신 공용 `evaluate_ghost_image` dispatcher를 통해 Before/After를
    평가할 수 있어야 한다(사용자 스펙 8번, 중복 branching 제거)."""
    observed = _ghosted_observed_uint8()
    field = fit_ghost_field_constant(image_width=320, image_height=240, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.15)
    cfg = _point_source_config()

    before = evaluate_ghost_image(observed, cfg)
    supp = suppress_ghost(observed, field)
    after = evaluate_ghost_image(supp.suppressed_image, cfg)

    assert before.mode == "point_source"
    assert after.mode == "point_source"
    assert (after.mean_strength_ratio or 0.0) < (before.mean_strength_ratio or 0.0)


# ===========================================================================
# STEP 8 semantic fix 2 - General Suppression -> Likelihood Before/After
# ===========================================================================

from calibration.windshield.ghost.evaluator import evaluate_ghost_general_likelihood
from calibration.windshield.ghost.suppression import build_suppression_evaluation


def _fake_suppression_result(*, over_suppression_score=0.05) -> object:
    from calibration.windshield.ghost.types import GhostSuppressionResult

    return GhostSuppressionResult(success=True, over_suppression_score=over_suppression_score)


def test_point_source_suppression_populates_strength_reduction_not_likelihood():
    """build_suppression_evaluation()의 mode 분기 로직 자체를 검증한다 -
    suppress_ghost()의 실제 수치(완전히 제거되면 after.mean_strength_ratio
    가 None이 될 수 있음)와 무관하게, before/after 둘 다 값이 있을 때
    strength_reduction이 채워지고 likelihood_reduction은 절대 채워지지
    않아야 한다."""
    from calibration.windshield.ghost.types import GhostEvaluationResult

    before = GhostEvaluationResult(success=True, mode="point_source", detection_count=2, mean_strength_ratio=0.15)
    after = GhostEvaluationResult(success=True, mode="point_source", detection_count=1, mean_strength_ratio=0.05)
    supp = _fake_suppression_result()

    evaln = build_suppression_evaluation(before, after, supp)
    assert evaln.strength_reduction == pytest.approx(0.10)
    assert evaln.detection_reduction == 1
    assert evaln.likelihood_reduction is None


def test_edge_target_suppression_populates_strength_reduction_not_likelihood():
    from calibration.windshield.ghost.types import GhostEvaluationResult

    before = GhostEvaluationResult(success=True, mode="edge_target", detection_count=2, mean_strength_ratio=0.20, edge_offset_median_px=6.0)
    after = GhostEvaluationResult(success=True, mode="edge_target", detection_count=1, mean_strength_ratio=0.08, edge_offset_median_px=6.0)
    supp = _fake_suppression_result()

    evaln = build_suppression_evaluation(before, after, supp)
    assert evaln.strength_reduction == pytest.approx(0.12)
    assert evaln.likelihood_reduction is None


def test_general_likelihood_suppression_populates_likelihood_reduction_not_strength():
    img_before = np.zeros((64, 64, 3), dtype=np.uint8)
    img_before[:, 20:] = 200
    img_before[:, 26:] += 30  # 뚜렷한 double-edge 반복 -> likelihood 높음

    img_after = np.zeros((64, 64, 3), dtype=np.uint8)
    img_after[:, 20:] = 200  # double-edge 제거됨 -> likelihood 낮음

    before = evaluate_ghost_general_likelihood(img_before)
    after = evaluate_ghost_general_likelihood(img_after)
    supp = _fake_suppression_result()

    evaln = build_suppression_evaluation(before, after, supp)
    assert evaln.likelihood_reduction is not None
    assert evaln.likelihood_reduction == pytest.approx(before.ghost_likelihood - after.ghost_likelihood)
    assert evaln.likelihood_reduction > 0.0  # synthetic fixture가 실제로 likelihood를 낮추도록 구성됨
    # General mode에서는 strength/detection reduction 의미를 절대 혼용하지 않는다.
    assert evaln.strength_reduction is None
    assert evaln.detection_reduction is None


def test_suppression_evaluation_before_after_share_same_mode():
    """Before/After는 항상 같은 evaluation mode여야 한다 - dispatcher가
    양쪽에 동일한 config를 쓰므로 자연히 보장되지만, 결과 타입 레벨에서도
    확인한다."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    before = evaluate_ghost_general_likelihood(img)
    after = evaluate_ghost_general_likelihood(img)
    assert before.mode == after.mode == "general_likelihood"


# ===========================================================================
# STEP 8 semantic/safety fix 3 - GhostField Fit은 Point Source만 허용
# ===========================================================================

from calibration.windshield.ghost.evaluator import evaluate_ghost_dataset


def test_fit_ghost_field_from_dataset_allows_point_source():
    observed = _ghosted_observed_uint8()
    frame = evaluate_ghost_point_source(observed, _point_source_config(bright_source_threshold=20.0), pair_id="f1")
    dataset_result = evaluate_ghost_dataset([frame], mode="point_source")
    field = fit_ghost_field_from_dataset(dataset_result, image_width=320, image_height=240, rows=1, cols=1)
    assert field is not None


def test_fit_ghost_field_from_dataset_rejects_edge_target():
    n = 100
    profile = np.zeros(n)
    profile[40:] = 200.0
    profile[46:] += 30.0
    from calibration.windshield.ghost.evaluator import evaluate_ghost_edge_target

    frame = evaluate_ghost_edge_target([profile], pair_id="f1")
    dataset_result = evaluate_ghost_dataset([frame], mode="edge_target")
    with pytest.raises(ValueError):
        fit_ghost_field_from_dataset(dataset_result, image_width=320, image_height=240, rows=1, cols=1)


def test_fit_ghost_field_from_dataset_rejects_general_likelihood():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    frame = evaluate_ghost_general_likelihood(img, pair_id="f1")
    dataset_result = evaluate_ghost_dataset([frame], mode="general_likelihood")
    with pytest.raises(ValueError):
        fit_ghost_field_from_dataset(dataset_result, image_width=64, image_height=64, rows=1, cols=1)


def test_fit_ghost_field_from_dataset_resolution_cross_check():
    """Worker의 mixed-resolution gate를 통과한 dataset이라도, fit에
    넘겨지는 image_width/height가 dataset이 실제 평가된 해상도와 다르면
    거부한다(사용자 스펙 4-F번)."""
    observed = _ghosted_observed_uint8()
    frame = evaluate_ghost_point_source(observed, _point_source_config(bright_source_threshold=20.0), pair_id="f1")
    dataset_result = evaluate_ghost_dataset([frame], mode="point_source", image_width=320, image_height=240)
    with pytest.raises(ValueError):
        fit_ghost_field_from_dataset(dataset_result, image_width=999, image_height=999, rows=1, cols=1)
