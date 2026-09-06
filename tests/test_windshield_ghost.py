"""
STEP 8A - Ghost / Double Image Evaluation 테스트.

Reflection Evaluation(tests/test_windshield_reflection_evaluation.py)과
완전히 독립된 테스트 파일이다 - Ghost 관련 assertion을 Reflection 테스트에
절대 섞지 않는다.
"""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from calibration.types import CameraModelType
from calibration.windshield.ghost.edge_detector import detect_edge_ghost
from calibration.windshield.ghost.evaluator import (
    evaluate_ghost_dataset,
    evaluate_ghost_edge_target,
    evaluate_ghost_general_likelihood,
    evaluate_ghost_point_source,
)
from calibration.windshield.ghost.point_detector import detect_bright_blobs, pair_main_and_ghost_blobs
from calibration.windshield.ghost.spatial_model import build_spatial_map, compute_region_metrics
from calibration.windshield.ghost.synthetic import (
    make_constant_offset_ghost_sample,
    make_left_center_right_offset_fields,
    make_spatially_varying_ghost_sample,
)
from calibration.windshield.ghost.types import GhostEvaluationConfig, GhostPointDetection


def _point_source_config(**overrides) -> GhostEvaluationConfig:
    base = dict(mode="point_source", bright_source_threshold=30.0, gaussian_sigma=0.4, min_peak_distance_px=1.5)
    base.update(overrides)
    return GhostEvaluationConfig(**base)


def _two_dot_image(h=240, w=320, centers=((100, 120), (220, 80)), radius=1) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.float32)
    for cx, cy in centers:
        cv2.circle(img, (cx, cy), radius, (255.0, 255.0, 255.0), -1)
    return img


# ---------------------------------------------------------------------------
# Test A - No Ghost: 입력=clean -> detection ~= 0, strength ~= 0.
# ---------------------------------------------------------------------------

def test_a_no_ghost_produces_no_detections():
    clean = _two_dot_image().astype(np.uint8)
    result = evaluate_ghost_point_source(clean, _point_source_config())
    assert result.success
    assert result.detection_count == 0
    assert result.candidate_count == 2
    assert result.mean_strength_ratio is None


# ---------------------------------------------------------------------------
# Test B - Known Offset: dx=+4, dy=-2 -> 추정 dx~=4, dy~=-2.
# ---------------------------------------------------------------------------

def test_b_known_offset_is_recovered():
    clean = _two_dot_image()
    sample = make_constant_offset_ghost_sample(clean, dx=4.0, dy=-2.0, alpha=0.15)
    observed = np.clip(sample.observed, 0, 255).astype(np.uint8)

    result = evaluate_ghost_point_source(observed, _point_source_config())
    assert result.success
    assert result.detection_count == 2
    assert result.mean_offset_x_px == pytest.approx(4.0, abs=0.5)
    assert result.mean_offset_y_px == pytest.approx(-2.0, abs=0.5)


# ---------------------------------------------------------------------------
# Test C - Known Strength: alpha=0.10 -> strength가 기대 범위 안에 있다.
# ---------------------------------------------------------------------------

def test_c_known_strength_ratio_is_recovered():
    clean = _two_dot_image()
    sample = make_constant_offset_ghost_sample(clean, dx=5.0, dy=0.0, alpha=0.10)
    observed = np.clip(sample.observed, 0, 255).astype(np.uint8)

    # alpha=0.10인 ghost peak의 절대 밝기(~25)가 기본 threshold(30)보다
    # 낮아질 수 있어 이 테스트에서만 threshold를 낮춘다 - detection 자체가
    # 아니라 "일단 검출된 pair의 strength_ratio가 맞는가"를 검증하는 것이
    # 이 테스트의 목적이다.
    result = evaluate_ghost_point_source(observed, _point_source_config(bright_source_threshold=20.0))
    assert result.success
    assert result.detection_count == 2
    assert result.mean_strength_ratio == pytest.approx(0.10, abs=0.03)


# ---------------------------------------------------------------------------
# Test D - Spatially Varying Ghost: Left/Center/Right가 서로 다른 offset ->
# vector field가 그 경향을 recover해야 한다.
# ---------------------------------------------------------------------------

def test_d_spatially_varying_offset_trend_is_recovered():
    h, w = 240, 320
    clean = _two_dot_image(h, w, centers=((40, 120), (160, 120), (280, 120)))
    dx_fn, dy_fn = make_left_center_right_offset_fields(w, left_dx=3.0, center_dx=5.0, right_dx=7.0, dy=0.0)
    sample = make_spatially_varying_ghost_sample(clean, dx_fn=dx_fn, dy_fn=dy_fn, alpha=0.3)
    observed = np.clip(sample.observed, 0, 255).astype(np.uint8)

    cfg = _point_source_config(spatial_rows=1, spatial_cols=3)
    result = evaluate_ghost_point_source(observed, cfg)
    assert result.success
    assert result.detection_count == 3

    cells = {c.col: c for c in result.spatial_map}
    assert cells[0].sample_count == 1
    assert cells[1].sample_count == 1
    assert cells[2].sample_count == 1
    # 왼쪽 < 가운데 < 오른쪽 경향이 유지되어야 한다(정확한 값 일치는 요구하지 않음).
    assert cells[0].mean_offset_x_px < cells[1].mean_offset_x_px < cells[2].mean_offset_x_px


# ---------------------------------------------------------------------------
# Test E - Blur vs Ghost: blur만 있는 이미지는 ghost로 over-trigger되면 안 된다.
# ---------------------------------------------------------------------------

def test_e_blur_only_image_does_not_falsely_trigger_point_ghost_detection():
    clean = _two_dot_image()
    blurred = cv2.GaussianBlur(clean, (15, 15), 6.0).astype(np.uint8)
    result = evaluate_ghost_point_source(blurred, _point_source_config())
    assert result.success
    assert result.detection_count == 0


def test_e_blur_only_edge_profile_does_not_trigger_edge_ghost_detection():
    n = 100
    step = np.zeros(n)
    step[40:] = 200.0
    blurred_profile = cv2.GaussianBlur(step.reshape(1, -1).astype(np.float32), (0, 0), sigmaX=6.0).flatten()
    det = detect_edge_ghost(blurred_profile)
    assert det.detected is False


def test_e_genuine_double_edge_is_detected_and_distinct_from_blur():
    n = 100
    profile = np.zeros(n)
    profile[40:] = 200.0
    profile[46:] += 30.0
    det = detect_edge_ghost(profile)
    assert det.detected is True
    assert det.offset_px == pytest.approx(6.0, abs=0.5)
    assert det.strength_ratio == pytest.approx(0.15, abs=0.02)


# ---------------------------------------------------------------------------
# Point detector unit tests
# ---------------------------------------------------------------------------

def test_point_detector_pairs_main_and_ghost_within_search_radius():
    from calibration.windshield.ghost.point_detector import BrightBlob

    blobs = [
        BrightBlob(x=100.0, y=100.0, energy=1000.0, area_px=10),
        BrightBlob(x=104.0, y=98.0, energy=150.0, area_px=10),
        BrightBlob(x=300.0, y=300.0, energy=10.0, area_px=5),
    ]
    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=60.0)
    detected = [d for d in detections if d.detected]
    assert len(detected) == 1
    assert detected[0].offset_x_px == pytest.approx(4.0)
    assert detected[0].offset_y_px == pytest.approx(-2.0)
    not_detected = [d for d in detections if not d.detected]
    assert len(not_detected) == 1
    assert not_detected[0].main_x == pytest.approx(300.0)


def test_point_detector_respects_max_search_radius():
    from calibration.windshield.ghost.point_detector import BrightBlob

    blobs = [
        BrightBlob(x=0.0, y=0.0, energy=1000.0, area_px=10),
        BrightBlob(x=200.0, y=0.0, energy=100.0, area_px=10),
    ]
    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=10.0)
    assert all(not d.detected for d in detections)


def test_detect_bright_blobs_returns_empty_for_dark_image():
    dark = np.zeros((64, 64, 3), dtype=np.uint8)
    blobs = detect_bright_blobs(dark)
    assert blobs == []


# ---------------------------------------------------------------------------
# Spatial model / region metrics unit tests
# ---------------------------------------------------------------------------

def test_build_spatial_map_ignores_undetected_points():
    detections = [
        GhostPointDetection(main_x=10, main_y=10, detected=False),
        GhostPointDetection(
            main_x=10, main_y=10, ghost_x=14, ghost_y=8, offset_x_px=4.0, offset_y_px=-2.0,
            distance_px=4.47, strength_ratio=0.15, detected=True,
        ),
    ]
    cells = build_spatial_map(detections, image_width=100, image_height=100, rows=2, cols=2)
    populated = [c for c in cells if c.sample_count > 0]
    assert len(populated) == 1
    assert populated[0].mean_offset_x_px == pytest.approx(4.0)


def test_compute_region_metrics_center_vs_corner():
    center_det = GhostPointDetection(
        main_x=50, main_y=50, ghost_x=54, ghost_y=50, offset_x_px=4.0, offset_y_px=0.0,
        distance_px=4.0, strength_ratio=0.2, detected=True,
    )
    corner_det = GhostPointDetection(main_x=2, main_y=2, detected=False)
    regions = compute_region_metrics([center_det, corner_det], image_width=100, image_height=100)
    assert regions["center"].detection_rate == pytest.approx(1.0)
    assert regions["corners"].detection_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Angular Ghost Separation (K, D read-only)
# ---------------------------------------------------------------------------

def test_angular_separation_is_positive_for_known_offset_and_zero_when_no_ghost():
    K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
    D = np.zeros(5)
    clean = _two_dot_image()
    sample = make_constant_offset_ghost_sample(clean, dx=4.0, dy=-2.0, alpha=0.15)
    observed = np.clip(sample.observed, 0, 255).astype(np.uint8)

    result = evaluate_ghost_point_source(
        observed, _point_source_config(),
        camera_matrix=K, distortion=D, camera_model=CameraModelType.PINHOLE,
    )
    assert result.mean_angular_separation_deg is not None
    assert result.mean_angular_separation_deg > 0.0

    clean_result = evaluate_ghost_point_source(
        clean.astype(np.uint8), _point_source_config(),
        camera_matrix=K, distortion=D, camera_model=CameraModelType.PINHOLE,
    )
    assert clean_result.mean_angular_separation_deg is None


def test_angular_separation_is_none_without_camera_intrinsics():
    clean = _two_dot_image()
    sample = make_constant_offset_ghost_sample(clean, dx=4.0, dy=-2.0, alpha=0.15)
    observed = np.clip(sample.observed, 0, 255).astype(np.uint8)
    result = evaluate_ghost_point_source(observed, _point_source_config())
    assert result.mean_angular_separation_deg is None


# ---------------------------------------------------------------------------
# General(No-Reference) Likelihood mode - never "ground truth".
# ---------------------------------------------------------------------------

def test_general_likelihood_mode_is_labeled_as_likelihood_not_ground_truth():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, 20:] = 200
    img[:, 26:] += 30
    result = evaluate_ghost_general_likelihood(img)
    assert result.success
    assert result.is_likelihood is True
    assert result.ghost_likelihood is not None
    assert result.warning_message is not None
    assert "ground truth" in result.warning_message.lower() or "Ground Truth" in result.warning_message
    # 반드시 "이것은 Ground Truth가 아니다"라고 명시적으로 부정해야 한다 -
    # 단순히 무관하게 등장하는 것이 아니라, 부정문 형태로 나타나야 한다.
    assert "아닙니다" in result.warning_message or "not" in result.warning_message.lower()


def test_general_likelihood_mode_never_populates_point_source_only_fields():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    result = evaluate_ghost_general_likelihood(img)
    assert result.mean_offset_x_px is None
    assert result.mean_offset_y_px is None


# ---------------------------------------------------------------------------
# Edge-target evaluation wrapper
# ---------------------------------------------------------------------------

def test_evaluate_ghost_edge_target_aggregates_multiple_profiles():
    n = 100
    clean_profile = np.zeros(n)
    clean_profile[40:] = 200.0
    ghost_profile = clean_profile.copy()
    ghost_profile[46:] += 30.0

    result = evaluate_ghost_edge_target([clean_profile, ghost_profile])
    assert result.success
    assert result.mode == "edge_target"
    assert result.candidate_count == 2
    assert result.detection_count == 1
    assert result.mean_strength_ratio == pytest.approx(0.15, abs=0.02)


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------

def test_evaluate_ghost_dataset_aggregates_per_frame_results():
    clean = _two_dot_image()
    sample = make_constant_offset_ghost_sample(clean, dx=4.0, dy=-2.0, alpha=0.15)
    observed = np.clip(sample.observed, 0, 255).astype(np.uint8)

    frame1 = evaluate_ghost_point_source(observed, _point_source_config(), pair_id="frame_1")
    frame2 = evaluate_ghost_point_source(clean.astype(np.uint8), _point_source_config(), pair_id="frame_2")

    dataset_result = evaluate_ghost_dataset([frame1, frame2], mode="point_source")
    assert dataset_result.success
    assert len(dataset_result.per_frame) == 2
    assert dataset_result.mean_strength == pytest.approx(0.15, abs=0.03)


def test_evaluate_ghost_dataset_handles_empty_input():
    dataset_result = evaluate_ghost_dataset([], mode="point_source")
    assert dataset_result.success is False
    assert dataset_result.warning_message is not None


# ---------------------------------------------------------------------------
# Project IO round-trip - ghost_results/ghost_models는 windshield_results/
# reflection_results와 완전히 별도 필드로 저장/복원되어야 한다.
# ---------------------------------------------------------------------------

def test_ghost_results_and_models_round_trip_through_project_dict():
    from calibration.project_io import project_from_dict, project_to_dict
    from calibration.types import CalibrationProject, CameraConfig, PatternConfig, PatternType
    from calibration.windshield.ghost.suppression import fit_ghost_field_constant

    clean = _two_dot_image()
    sample = make_constant_offset_ghost_sample(clean, dx=4.0, dy=-2.0, alpha=0.15)
    observed = np.clip(sample.observed, 0, 255).astype(np.uint8)
    frame = evaluate_ghost_point_source(observed, _point_source_config(), pair_id="frame_1")
    dataset_result = evaluate_ghost_dataset([frame], mode="point_source")
    field = fit_ghost_field_constant(image_width=320, image_height=240, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.15)

    project = CalibrationProject(
        project_name="ghost-test",
        camera_config=CameraConfig(width=320, height=240),
        pattern_config=PatternConfig(type=PatternType.CHARUCO, squares_x=5, squares_y=5, square_size=0.02),
        ghost_results={"latest": dataset_result},
        ghost_models={"latest": field},
    )

    restored = project_from_dict(project_to_dict(project))

    assert "latest" in restored.ghost_results
    restored_dataset = restored.ghost_results["latest"]
    assert restored_dataset.mode == "point_source"
    assert restored_dataset.per_frame[0].detection_count == frame.detection_count
    assert restored_dataset.per_frame[0].mean_offset_x_px == pytest.approx(frame.mean_offset_x_px)

    assert "latest" in restored.ghost_models
    restored_field = restored.ghost_models["latest"]
    assert isinstance(restored_field.offset_x, np.ndarray)
    assert np.allclose(restored_field.offset_x, field.offset_x)

    # windshield_results/reflection_results와 절대 섞이지 않는다.
    assert not restored.windshield_results
    assert not restored.reflection_results
