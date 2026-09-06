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
from calibration.windshield.ghost.types import GhostEvaluationConfig, GhostPointDetection, GhostSpatialCell


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


# ===========================================================================
# STEP 8 stabilization 1 - Multi-LED Pairing Robust화
# ===========================================================================

from calibration.windshield.ghost.point_detector import BrightBlob, estimate_dominant_ghost_vector


def test_dense_multi_led_prevents_main_main_mispair():
    """3개의 Main LED가 서로 3px 간격으로 촘촘하게 배치되어 있고, 각자의
    진짜 Ghost는 4.47px 떨어져 있다(즉 이웃 Main보다 자기 Ghost가 더 멀다) -
    순수 nearest-neighbor는 이웃 Main을 Ghost로 잘못 짝짓지만
    (사용자 스펙 1번 시나리오), global displacement consensus는 dataset
    전체가 공유하는 dominant vector(4,-2)를 이용해 이를 방지해야 한다."""
    blobs = [
        BrightBlob(x=100, y=100, energy=1000, area_px=10),  # M1
        BrightBlob(x=104, y=98, energy=150, area_px=10),  # G1 = M1+(4,-2)
        BrightBlob(x=103, y=100, energy=900, area_px=10),  # M2 (M1과 3px만 떨어짐 - mispair 위험)
        BrightBlob(x=107, y=98, energy=140, area_px=10),  # G2 = M2+(4,-2)
        BrightBlob(x=106, y=100, energy=800, area_px=10),  # M3 (M2와 3px만 떨어짐 - mispair 위험)
        BrightBlob(x=110, y=98, energy=130, area_px=10),  # G3 = M3+(4,-2)
    ]
    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=60.0)
    detected = {(round(d.main_x), round(d.main_y)): d for d in detections if d.detected}

    assert len(detected) == 3
    for (mx, my), det in detected.items():
        assert det.offset_x_px == pytest.approx(4.0, abs=0.5)
        assert det.offset_y_px == pytest.approx(-2.0, abs=0.5)
        # main-main mispair였다면 offset이 (3,0) 근처였을 것이다.
        assert det.offset_x_px != pytest.approx(3.0, abs=0.3)
        assert det.pair_residual_px is not None
        assert det.pair_residual_px < 1.0


def test_dense_multi_led_unequal_brightness_still_recovers_dominant_vector():
    """Main LED들의 밝기가 서로 달라도(255/210/180 스타일) dominant
    displacement vector는 동일하게 복원되어야 한다."""
    blobs = [
        BrightBlob(x=50, y=50, energy=255.0, area_px=10),
        BrightBlob(x=54, y=48, energy=40.0, area_px=10),  # ghost of the above
        BrightBlob(x=52, y=50, energy=210.0, area_px=10),  # neighboring main, close by
        BrightBlob(x=56, y=48, energy=33.0, area_px=10),  # its ghost
        BrightBlob(x=200, y=200, energy=180.0, area_px=10),  # far-away independent main
        BrightBlob(x=204, y=198, energy=28.0, area_px=10),  # its ghost
    ]
    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=60.0)
    detected = [d for d in detections if d.detected]
    assert len(detected) == 3
    for det in detected:
        assert det.offset_x_px == pytest.approx(4.0, abs=0.5)
        assert det.offset_y_px == pytest.approx(-2.0, abs=0.5)


def test_outlier_bright_point_does_not_corrupt_dominant_vector():
    """Ghost와 무관한 밝은 점 하나가 섞여도 dominant ghost vector 추정이
    흔들리면 안 된다."""
    candidates_source = [
        BrightBlob(x=10, y=10, energy=1000, area_px=10),
        BrightBlob(x=14, y=8, energy=150, area_px=10),
        BrightBlob(x=60, y=60, energy=900, area_px=10),
        BrightBlob(x=64, y=58, energy=140, area_px=10),
        BrightBlob(x=110, y=110, energy=800, area_px=10),
        BrightBlob(x=114, y=108, energy=120, area_px=10),
        # 관계 없는 outlier bright point(가장 가까운 main으로부터 60px 밖 - search radius 안에 안 들어옴)
        BrightBlob(x=300, y=300, energy=50, area_px=5),
    ]
    detections = pair_main_and_ghost_blobs(candidates_source, max_search_radius_px=60.0)
    detected = [d for d in detections if d.detected]
    assert len(detected) == 3
    for det in detected:
        assert det.offset_x_px == pytest.approx(4.0, abs=0.5)
        assert det.offset_y_px == pytest.approx(-2.0, abs=0.5)


def test_single_led_fallback_matches_local_nearest_neighbor():
    """Candidate가 `min_consensus_candidates`보다 적으면(단일 LED 케이스)
    global consensus 없이도 기존과 동일하게 동작해야 한다(사용자 스펙
    1-E번, no-consensus fallback)."""
    blobs = [
        BrightBlob(x=100.0, y=100.0, energy=1000.0, area_px=10),
        BrightBlob(x=104.0, y=98.0, energy=150.0, area_px=10),
    ]
    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=60.0, min_consensus_candidates=3)
    detected = [d for d in detections if d.detected]
    assert len(detected) == 1
    assert detected[0].offset_x_px == pytest.approx(4.0)
    assert detected[0].offset_y_px == pytest.approx(-2.0)
    assert detected[0].pair_residual_px is None  # consensus 없이 fallback했으므로 residual 정의 안 됨


def test_estimate_dominant_ghost_vector_returns_none_below_min_candidates():
    from calibration.windshield.ghost.point_detector import _CandidatePair

    few = [_CandidatePair(main_idx=0, ghost_idx=1, dx=4.0, dy=-2.0, distance=4.47)]
    assert estimate_dominant_ghost_vector(few, min_consensus_candidates=3) is None


def test_estimate_dominant_ghost_vector_recovers_true_vector_from_noisy_candidates():
    from calibration.windshield.ghost.point_detector import _CandidatePair

    candidates = [
        _CandidatePair(main_idx=i, ghost_idx=i + 100, dx=dx, dy=dy, distance=float(np.hypot(dx, dy)))
        for i, (dx, dy) in enumerate([(4.0, -2.0), (4.1, -1.9), (3.9, -2.1), (4.0, -2.0), (30.0, 30.0)])
    ]
    dominant = estimate_dominant_ghost_vector(candidates, consensus_radius_px=3.0, min_consensus_candidates=3)
    assert dominant is not None
    dx, dy, support = dominant
    assert dx == pytest.approx(4.0, abs=0.2)
    assert dy == pytest.approx(-2.0, abs=0.2)
    assert support == 4  # outlier (30,30)는 inlier에서 제외되어야 한다


def test_full_pipeline_dense_scene_via_synthetic_image():
    """Blob 리스트를 직접 구성하는 단위 테스트뿐 아니라, 실제 이미지
    파이프라인(detect_bright_blobs -> pair_main_and_ghost_blobs)에서도
    여러 LED가 한 이미지 안에 있을 때 정상적으로 동작하는지 확인한다."""
    h, w = 240, 320
    clean = _two_dot_image(h, w, centers=((60, 60), (160, 160), (260, 60)))
    sample = make_constant_offset_ghost_sample(clean, dx=4.0, dy=-2.0, alpha=0.2)
    observed = np.clip(sample.observed, 0, 255).astype(np.uint8)
    result = evaluate_ghost_point_source(observed, _point_source_config())
    assert result.detection_count == 3
    assert result.mean_offset_x_px == pytest.approx(4.0, abs=0.5)
    assert result.mean_offset_y_px == pytest.approx(-2.0, abs=0.5)


# ===========================================================================
# STEP 8 stabilization 2 - Edge Target: Image -> 1D profile 추출
# ===========================================================================

from calibration.windshield.ghost.edge_detector import extract_edge_profiles
from calibration.windshield.ghost.evaluator import evaluate_ghost_image


def _vertical_double_edge_image(h=200, w=200, offset_px=6.0, strength=0.15, primary_x=100):
    img = np.zeros((h, w), dtype=np.float32)
    img[:, primary_x:] = 200.0
    img[:, primary_x + int(round(offset_px)):] += strength * 200.0
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


def test_extract_edge_profiles_from_image_returns_finite_profiles():
    img = _vertical_double_edge_image()
    cfg = GhostEvaluationConfig(mode="edge_target", edge_axis="vertical")
    profiles = extract_edge_profiles(img, cfg)
    assert len(profiles) >= 3
    for p in profiles:
        assert np.all(np.isfinite(p))


def test_extract_edge_profiles_respects_edge_axis_orientation():
    img = _vertical_double_edge_image()
    cfg_wrong_axis = GhostEvaluationConfig(mode="edge_target", edge_axis="horizontal")
    assert extract_edge_profiles(img, cfg_wrong_axis) == []

    cfg_auto = GhostEvaluationConfig(mode="edge_target", edge_axis="auto")
    assert len(extract_edge_profiles(img, cfg_auto)) >= 3


def test_extract_edge_profiles_empty_for_blank_image():
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    cfg = GhostEvaluationConfig(mode="edge_target")
    assert extract_edge_profiles(blank, cfg) == []


def test_known_plus6px_edge_ghost_recovered_from_image():
    img = _vertical_double_edge_image(offset_px=6.0, strength=0.15)
    cfg = GhostEvaluationConfig(mode="edge_target", edge_axis="vertical")
    result = evaluate_ghost_edge_target(extract_edge_profiles(img, cfg), cfg)
    assert result.success
    assert result.detection_count > 0
    assert result.edge_offset_median_px == pytest.approx(6.0, abs=0.5)
    assert result.mean_strength_ratio == pytest.approx(0.15, abs=0.02)
    # Point-source 전용 필드는 edge 모드에서 채워지지 않는다.
    assert result.mean_offset_x_px is None


def test_edge_target_blur_only_image_does_not_over_trigger():
    img = _vertical_double_edge_image(offset_px=0.0, strength=0.0)
    blurred = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=8.0).astype(np.uint8)
    cfg = GhostEvaluationConfig(mode="edge_target", edge_axis="vertical")
    result = evaluate_ghost_edge_target(extract_edge_profiles(blurred, cfg), cfg)
    assert result.detection_count == 0


def test_evaluate_ghost_image_dispatches_to_edge_target():
    """UI/Worker가 공용으로 쓸 단일 dispatcher(사용자 스펙 8/2-D번) -
    mode에 따라 올바른 evaluator로 라우팅되어야 한다."""
    img = _vertical_double_edge_image()
    cfg = GhostEvaluationConfig(mode="edge_target", edge_axis="vertical")
    result = evaluate_ghost_image(img, cfg)
    assert result.mode == "edge_target"
    assert result.edge_offset_median_px == pytest.approx(6.0, abs=0.5)


def test_evaluate_ghost_image_dispatches_to_point_source_and_general_likelihood():
    clean = _two_dot_image()
    cfg_point = _point_source_config()
    result_point = evaluate_ghost_image(clean.astype(np.uint8), cfg_point)
    assert result_point.mode == "point_source"

    cfg_general = GhostEvaluationConfig(mode="general_likelihood")
    result_general = evaluate_ghost_image(clean.astype(np.uint8), cfg_general)
    assert result_general.mode == "general_likelihood"
    assert result_general.is_likelihood is True


def test_evaluate_ghost_image_raises_on_unknown_mode():
    cfg = GhostEvaluationConfig(mode="not_a_real_mode")
    with pytest.raises(ValueError):
        evaluate_ghost_image(_two_dot_image().astype(np.uint8), cfg)


# ===========================================================================
# STEP 8 stabilization 3 - Multi-frame Dataset -> Robust GhostField Fit
# ===========================================================================

from calibration.windshield.ghost.spatial_model import build_robust_spatial_map_from_detections, fill_empty_spatial_cells
from calibration.windshield.ghost.suppression import fit_ghost_field_from_dataset


def _make_dataset_result_with_outliers(num_frames=20, outlier_indices=(5, 12)) -> "GhostDatasetResult":
    rng = np.random.default_rng(7)
    per_frame = []
    cfg = _point_source_config(bright_source_threshold=20.0, spatial_rows=1, spatial_cols=1)
    for i in range(num_frames):
        clean = _two_dot_image()
        dx = 4.0 + rng.normal(0, 0.2)
        dy = -2.0 + rng.normal(0, 0.2)
        alpha = 0.15 + rng.normal(0, 0.005)
        if i in outlier_indices:
            dx = 25.0  # 명백한 outlier displacement
        sample = make_constant_offset_ghost_sample(clean, dx=dx, dy=dy, alpha=alpha)
        observed = np.clip(sample.observed, 0, 255).astype(np.uint8)
        per_frame.append(evaluate_ghost_point_source(observed, cfg, pair_id=f"frame_{i}"))
    return evaluate_ghost_dataset(per_frame, mode="point_source")


def test_dataset_fit_uses_all_frames_not_only_first():
    """`fit_ghost_field_from_dataset`는 `per_frame[0]`만 쓰지 않는다 -
    첫 프레임을 의도적으로 다른 값으로 바꿔도 dataset 전체 median에는 거의
    영향이 없어야 한다."""
    dataset_result = _make_dataset_result_with_outliers(num_frames=20, outlier_indices=())
    # 첫 프레임의 detection들을 인위적으로 크게 왜곡한다.
    for det in dataset_result.per_frame[0].detections:
        if det.detected:
            det.offset_x_px = 99.0

    field = fit_ghost_field_from_dataset(dataset_result, image_width=320, image_height=240, rows=1, cols=1)
    # per_frame[0]만 썼다면 offset_x가 99 근처였을 것이다.
    assert field.offset_x[0, 0] == pytest.approx(4.0, abs=0.5)


def test_dataset_fit_rejects_outlier_frames_via_mad():
    dataset_result = _make_dataset_result_with_outliers(num_frames=20, outlier_indices=(5, 12))
    field = fit_ghost_field_from_dataset(dataset_result, image_width=320, image_height=240, rows=1, cols=1)
    assert field.offset_x[0, 0] == pytest.approx(4.0, abs=0.5)
    assert field.offset_y[0, 0] == pytest.approx(-2.0, abs=0.5)
    assert field.diagnostics is not None
    assert field.diagnostics.num_frames == 20


def test_build_robust_spatial_map_per_cell_median_and_mad():
    detections = [
        GhostPointDetection(main_x=10, main_y=10, ghost_x=14, ghost_y=8, offset_x_px=4.0, offset_y_px=-2.0, distance_px=4.47, strength_ratio=0.15, detected=True)
        for _ in range(5)
    ] + [
        GhostPointDetection(main_x=10, main_y=10, ghost_x=40, ghost_y=8, offset_x_px=30.0, offset_y_px=-2.0, distance_px=30.0, strength_ratio=0.9, detected=True),
    ]
    cells = build_robust_spatial_map_from_detections(detections, image_width=100, image_height=100, rows=1, cols=1)
    cell = cells[0]
    assert cell.sample_count == 6
    # median/MAD 기반 aggregation이 outlier(30.0)를 배제해야 한다.
    assert cell.mean_offset_x_px == pytest.approx(4.0, abs=0.1)
    assert cell.outlier_rejected_count >= 1


def test_fill_empty_spatial_cells_uses_nearest_valid_neighbor():
    from calibration.windshield.ghost.types import GhostSpatialCell

    cells = [
        GhostSpatialCell(row=0, col=0, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.15, sample_count=5),
        GhostSpatialCell(row=0, col=1, sample_count=0),
        GhostSpatialCell(row=1, col=0, sample_count=0),
        GhostSpatialCell(row=1, col=1, mean_offset_x_px=8.0, mean_offset_y_px=-4.0, mean_strength_ratio=0.30, sample_count=5),
    ]
    filled = fill_empty_spatial_cells(cells)
    by_pos = {(c.row, c.col): c for c in filled}
    assert by_pos[(0, 1)].is_filled is True
    assert by_pos[(0, 1)].mean_offset_x_px in (4.0, 8.0)  # 둘 중 하나의 인접 valid cell 값
    assert by_pos[(1, 0)].is_filled is True
    # 이미 값이 있던 cell은 is_filled=False로 남아야 한다.
    assert by_pos[(0, 0)].is_filled is False


def test_fill_empty_spatial_cells_no_valid_cells_leaves_unfilled():
    from calibration.windshield.ghost.types import GhostSpatialCell

    cells = [GhostSpatialCell(row=0, col=0, sample_count=0), GhostSpatialCell(row=0, col=1, sample_count=0)]
    filled = fill_empty_spatial_cells(cells)
    assert all(c.sample_count == 0 and not c.is_filled for c in filled)


# ===========================================================================
# STEP 8 stabilization 6 - 실제 Visualization (Overlay / Vector Field / Heatmap)
# ===========================================================================

from calibration.windshield.ghost.visualization import (
    render_ghost_point_overlay,
    render_strength_heatmap_image,
    render_vector_field_image,
)


def test_render_ghost_point_overlay_draws_on_a_copy_without_mutating_input():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    original = img.copy()
    dets = [
        GhostPointDetection(
            main_x=50, main_y=50, ghost_x=54, ghost_y=48, offset_x_px=4.0, offset_y_px=-2.0,
            distance_px=4.47, strength_ratio=0.15, detected=True,
        )
    ]
    overlay = render_ghost_point_overlay(img, dets)
    assert overlay.shape == img.shape
    assert np.array_equal(img, original)  # 입력을 변형하지 않는다
    assert np.count_nonzero(overlay) > 0  # 실제로 뭔가 그려졌다


def test_render_ghost_point_overlay_draws_main_only_for_undetected():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = [GhostPointDetection(main_x=50, main_y=50, detected=False)]
    overlay = render_ghost_point_overlay(img, dets)
    assert np.count_nonzero(overlay) > 0


def test_render_vector_field_image_and_heatmap_are_separate_images():
    """사용자 스펙 6-E번 - Vector Field와 Strength Heatmap을 하나의
    이미지로 합치지 않는다: 서로 다른 shape/content를 갖는 두 개의 독립된
    이미지여야 한다."""
    cells = [
        GhostSpatialCell(row=0, col=0, mean_offset_x_px=4.0, mean_offset_y_px=-2.0, mean_strength_ratio=0.9, sample_count=3),
        GhostSpatialCell(row=0, col=1, sample_count=0),
    ]
    vector_field_img = render_vector_field_image(cells, rows=1, cols=2, cell_size_px=40)
    heatmap_img = render_strength_heatmap_image(cells, rows=1, cols=2, cell_size_px=40)
    assert vector_field_img.shape == (40, 80, 3)
    assert heatmap_img.shape == (40, 80, 3)
    # 두 렌더러가 서로 다른 함수/내용을 만들어낸다 - 완전히 동일한 이미지가
    # 아니어야 한다(합쳐지지 않았다는 방증).
    assert not np.array_equal(vector_field_img, heatmap_img)


def test_render_vector_field_image_marks_empty_cells_differently():
    cells = [GhostSpatialCell(row=0, col=0, sample_count=0)]
    img = render_vector_field_image(cells, rows=1, cols=1, cell_size_px=30)
    assert img.shape == (30, 30, 3)
    assert np.count_nonzero(img != 255) > 0  # marker가 그려졌다(전부 흰 배경은 아님)


# ===========================================================================
# STEP 8 stabilization 9 - Metric/Model Version bump은 하위 호환을 깨면 안 된다
# ===========================================================================

def test_metric_version_bumped_to_2_for_new_results():
    from calibration.windshield.ghost.config import GHOST_METRIC_VERSION, GHOST_MODEL_VERSION

    assert GHOST_METRIC_VERSION == 2
    assert GHOST_MODEL_VERSION == 2


def test_old_v1_style_ghost_project_payload_still_loads_without_crashing():
    """pairing/aggregation 정의가 바뀌어 metric_version을 2로 올렸지만,
    v1 시절에 저장된(신규 필드가 아예 없는) raw dict도 project_from_dict()
    이후 예외 없이 복원되어야 한다(사용자 스펙 9번, "기존 project load가
    깨지면 안 된다")."""
    from calibration.project_io import (
        _ghost_dataset_result_from_dict,
        _ghost_evaluation_result_from_dict,
        _ghost_field_from_dict,
        _ghost_point_detection_from_dict,
        _ghost_spatial_cell_from_dict,
    )

    v1_point_detection = {
        "main_x": 10.0, "main_y": 10.0, "ghost_x": 14.0, "ghost_y": 8.0,
        "offset_x_px": 4.0, "offset_y_px": -2.0, "distance_px": 4.47,
        "strength_ratio": 0.15, "detected": True,
        # pair_residual_px 키 자체가 없음(v1) - v2에서 새로 추가된 필드.
    }
    restored_det = _ghost_point_detection_from_dict(v1_point_detection)
    assert restored_det.pair_residual_px is None

    v1_spatial_cell = {"row": 0, "col": 0, "mean_offset_x_px": 4.0, "sample_count": 3}
    restored_cell = _ghost_spatial_cell_from_dict(v1_spatial_cell)
    assert restored_cell.is_filled is False
    assert restored_cell.mad_offset_x_px is None

    v1_eval_result = {
        "success": True, "mode": "point_source", "metric_version": 1,
        "detection_count": 2, "mean_offset_x_px": 4.0,
        # edge_offset_median_px 등 v2 신규 필드 없음.
    }
    restored_eval = _ghost_evaluation_result_from_dict(v1_eval_result)
    assert restored_eval.metric_version == 1  # 저장된 값 그대로 보존(강제 마이그레이션 없음)
    assert restored_eval.edge_offset_median_px is None

    v1_dataset = {"mode": "point_source", "metric_version": 1, "per_frame": [v1_eval_result]}
    restored_dataset = _ghost_dataset_result_from_dict(v1_dataset)
    assert restored_dataset.metric_version == 1

    v1_field = {
        "offset_x": [[4.0]], "offset_y": [[-2.0]], "strength": [[0.15]],
        "image_width": 320.0, "image_height": 240.0, "model_version": 1,
        # diagnostics 키 자체가 없음(v1).
    }
    restored_field = _ghost_field_from_dict(v1_field)
    assert restored_field.diagnostics is None
    assert restored_field.model_version == 1
