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
# Priority 3 안정화 - Ghost 2-pass spatial pairing이 실제 Point Source
# production evaluator(evaluate_ghost_point_source)에 연결돼 있는지 확인.
# 함수와 단위 테스트가 존재하는 것만으로는 "연결됐다"고 볼 수 없으므로,
# (1) 소스 상에서 실제 호출부를 확인하고, (2) 그 함수가 실제로 호출되는지
# monkeypatch로 행동 검증한다.
# ---------------------------------------------------------------------------

def test_evaluate_ghost_point_source_calls_two_pass_pairing_not_single_pass():
    """evaluator.py 소스 자체를 확인한다 - `pair_main_and_ghost_blobs_two_
    pass`를 import/호출하고, 예전 single-pass 함수(`pair_main_and_ghost_
    blobs(`처럼 뒤에 여는 괄호가 바로 오는 호출 형태)는 더 이상 호출하지
    않아야 한다."""
    from pathlib import Path

    source = Path(
        __file__
    ).resolve().parents[1].joinpath(
        "calibration", "windshield", "ghost", "evaluator.py"
    ).read_text(encoding="utf-8")

    assert "pair_main_and_ghost_blobs_two_pass" in source
    assert "pair_main_and_ghost_blobs(" not in source


def test_evaluate_ghost_point_source_actually_invokes_two_pass_function(monkeypatch):
    """소스 검사만으로는 실제 실행 경로를 증명하지 못하므로, 진짜
    `pair_main_and_ghost_blobs_two_pass()`를 monkeypatch해서 실제로
    호출되는지, 그리고 이미지 크기(image_width/image_height)가 올바르게
    전달되는지까지 행동으로 검증한다."""
    import calibration.windshield.ghost.evaluator as evaluator_module

    calls = []
    real_two_pass = evaluator_module.pair_main_and_ghost_blobs_two_pass

    def _fake_two_pass(blobs, *, image_width, image_height, **kwargs):
        calls.append((image_width, image_height))
        return real_two_pass(blobs, image_width=image_width, image_height=image_height, **kwargs)

    monkeypatch.setattr(evaluator_module, "pair_main_and_ghost_blobs_two_pass", _fake_two_pass)

    clean = _two_dot_image(h=240, w=320)
    sample = make_constant_offset_ghost_sample(clean, dx=4.0, dy=-2.0, alpha=0.15)
    observed = np.clip(sample.observed, 0, 255).astype(np.uint8)

    result = evaluate_ghost_point_source(observed, _point_source_config())

    assert len(calls) == 1, "pair_main_and_ghost_blobs_two_pass()가 정확히 한 번 호출돼야 한다."
    assert calls[0] == (320, 240), "image_width/image_height가 실제 이미지 크기와 일치해야 한다."
    assert result.success
    assert result.detection_count == 2


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
# Phase B-5 - column-profile expansion (row-profile만으로는 세로 방향
# double-edge를 놓친다).
# ---------------------------------------------------------------------------

def test_general_likelihood_row_only_pattern_reports_row_but_not_column_detection():
    """가로 스캔(row-profile)으로만 잡히는 패턴(수직선 형태의 double-edge,
    모든 row에서 x축을 따라 밝기가 계단식으로 변함) - column 방향으로는
    각 열이 y축 방향으로 상수이므로 검출되면 안 된다."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, 20:] = 200
    img[:, 26:] += 30
    result = evaluate_ghost_general_likelihood(img)
    assert result.likelihood_row_detection_rate is not None
    assert result.likelihood_column_detection_rate is not None
    assert result.likelihood_row_detection_rate > 0.0
    assert result.likelihood_column_detection_rate == pytest.approx(0.0)


def test_general_likelihood_column_only_pattern_is_recovered_by_column_profile():
    """Phase B-5 핵심 회귀 - row-profile만 쓰던 이전 구현이라면 이 패턴(모든
    column에서 y축을 따라 밝기가 계단식으로 변하고, 각 row는 x축 방향으로
    상수라 row-profile로는 전혀 검출되지 않음)에서 ghost_likelihood가
    0이었을 것이다. Column-profile을 추가한 뒤에는 검출되어야 한다."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[20:, :] = 200
    img[26:, :] += 30
    result = evaluate_ghost_general_likelihood(img)
    assert result.success
    assert result.likelihood_row_detection_rate == pytest.approx(0.0)
    assert result.likelihood_column_detection_rate is not None
    assert result.likelihood_column_detection_rate > 0.0
    # 핵심 assertion: row-profile만 썼다면 0이었을 전체 ghost_likelihood가
    # column-profile 덕분에 0보다 커야 한다.
    assert result.ghost_likelihood > 0.0
    assert result.is_likelihood is True
    assert result.warning_message is not None


def test_general_likelihood_row_and_column_rates_never_appear_in_other_modes():
    """Point Source/Edge Target 모드에서는 이 두 필드가 절대 채워지면 안
    된다(사용자 스펙 - General Likelihood 전용 필드가 다른 모드로 새면
    안 된다).

    이전 버전은 실수로 `evaluate_ghost_point_source()`에 `BrightBlob`
    리스트를 직접 넘겼는데(이미지가 아님), 그 시점엔 `detect_bright_blobs`/
    `pair_main_and_ghost_blobs`가 던진 예외가 함수 내부 try/except에 잡혀
    success=False인 채로 모든 필드가 기본값(None)인 결과가 돌아온 덕분에
    "우연히" 통과하고 있었다 - 실제로는 point-source 파이프라인을 전혀
    실행하지 않은 것이다. 이제는 실제 이미지를 넘겨 진짜 point-source
    평가가 성공적으로 끝난 뒤에도 likelihood 필드가 비어있는지 검증한다."""
    img = _two_dot_image()
    result = evaluate_ghost_point_source(img)
    assert result.success
    assert result.mode == "point_source"
    assert result.likelihood_row_detection_rate is None
    assert result.likelihood_column_detection_rate is None


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

    few = [_CandidatePair(main_idx=0, ghost_idx=1, dx=4.0, dy=-2.0, distance=4.47, energy_ratio=0.15)]
    assert estimate_dominant_ghost_vector(few, min_consensus_candidates=3) is None


def test_estimate_dominant_ghost_vector_recovers_true_vector_from_noisy_candidates():
    from calibration.windshield.ghost.point_detector import _CandidatePair

    candidates = [
        _CandidatePair(main_idx=i, ghost_idx=i + 100, dx=dx, dy=dy, distance=float(np.hypot(dx, dy)), energy_ratio=0.15)
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


def test_pick_dominant_line_handles_both_opencv_houghlinesp_shapes():
    """cv2.HoughLinesP()의 반환 shape은 OpenCV 버전에 따라 (N,1,4) 또는
    (N,4)일 수 있다(OpenCV 5 CI에서 (N,4)로 확인됨 - 이 sandbox의 OpenCV
    4.11은 항상 (N,1,4)만 내므로 실제 호출로는 재현이 안 되고, 여기서
    두 shape을 직접 구성해 `_pick_dominant_line()`이 둘 다 같은 결과를
    내는지 검증한다). 이전에는 `line[0]`이 (N,4) 입력에서 scalar가 되어
    `TypeError: 'numpy.int32' object is not iterable`가 났다."""
    from calibration.windshield.ghost.edge_detector import _pick_dominant_line

    lines_n14 = np.array([[[1, 2, 3, 4]]], dtype=np.int32)
    lines_n4 = np.array([[1, 2, 3, 4]], dtype=np.int32)

    result_n14 = _pick_dominant_line(lines_n14, "auto", 20.0)
    result_n4 = _pick_dominant_line(lines_n4, "auto", 20.0)

    assert result_n14 is not None
    assert result_n14 == result_n4 == (1.0, 2.0, 3.0, 4.0)


def test_pick_dominant_line_picks_longest_matching_line_regardless_of_shape():
    """단일 라인이 아니라 여러 라인 중 axis 제약을 만족하는 가장 긴
    라인을 고르는 기존 로직이 (N,4) shape에서도 그대로 유지되는지 확인."""
    from calibration.windshield.ghost.edge_detector import _pick_dominant_line

    # 수직에 가까운 짧은 라인 하나 + 수직에 가까운 긴 라인 하나(N,4).
    lines_n4 = np.array([
        [10, 0, 10, 5],     # length ~5, vertical
        [50, 0, 50, 100],   # length ~100, vertical - dominant여야 함
    ], dtype=np.int32)
    result = _pick_dominant_line(lines_n4, "vertical", 20.0)
    assert result == (50.0, 0.0, 50.0, 100.0)


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


# ===========================================================================
# STEP 8 semantic fix 1 - Dataset Ghost Likelihood aggregation
# ===========================================================================

def _general_likelihood_frame(pair_id: str, *, likelihood: float, strength: float) -> GhostEvaluationResult:
    from calibration.windshield.ghost.types import GhostEvaluationResult

    return GhostEvaluationResult(
        success=True,
        mode="general_likelihood",
        pair_id=pair_id,
        detection_count=3,
        candidate_count=10,
        mean_strength_ratio=strength,
        ghost_likelihood=likelihood,
        is_likelihood=True,
    )


def test_dataset_ghost_likelihood_aggregation_mean_median_p95():
    frames = [
        _general_likelihood_frame("f1", likelihood=0.1, strength=0.5),
        _general_likelihood_frame("f2", likelihood=0.3, strength=0.5),
        _general_likelihood_frame("f3", likelihood=0.8, strength=0.5),
    ]
    dataset_result = evaluate_ghost_dataset(frames, mode="general_likelihood")

    assert dataset_result.mean_ghost_likelihood == pytest.approx(0.4, abs=1e-9)
    assert dataset_result.median_ghost_likelihood == pytest.approx(0.3, abs=1e-9)
    assert dataset_result.p95_ghost_likelihood == pytest.approx(np.percentile([0.1, 0.3, 0.8], 95), abs=1e-9)


def test_dataset_ghost_likelihood_is_never_confused_with_strength():
    """Ghost Likelihood(double-edge 패턴이 나타난 profile 비율)와 Ghost
    Strength(검출된 secondary edge의 상대 강도)는 서로 다른 값이다(사용자
    스펙 1-A번) - 우연히 같은 숫자가 아니라 애초에 다른 필드/다른 계산
    경로에서 나와야 한다."""
    frames = [
        _general_likelihood_frame("f1", likelihood=0.70, strength=0.12),
        _general_likelihood_frame("f2", likelihood=0.65, strength=0.11),
    ]
    dataset_result = evaluate_ghost_dataset(frames, mode="general_likelihood")

    assert dataset_result.mean_ghost_likelihood != pytest.approx(dataset_result.mean_strength)
    assert dataset_result.mean_ghost_likelihood == pytest.approx(0.675, abs=1e-9)
    assert dataset_result.mean_strength == pytest.approx(0.115, abs=1e-9)


def test_point_source_and_edge_target_datasets_never_populate_likelihood_aggregate():
    clean = _two_dot_image()
    sample = make_constant_offset_ghost_sample(clean, dx=4.0, dy=-2.0, alpha=0.15)
    observed = np.clip(sample.observed, 0, 255).astype(np.uint8)
    frame = evaluate_ghost_point_source(observed, _point_source_config(), pair_id="f1")
    dataset_result = evaluate_ghost_dataset([frame], mode="point_source")
    assert dataset_result.mean_ghost_likelihood is None
    assert dataset_result.median_ghost_likelihood is None
    assert dataset_result.p95_ghost_likelihood is None


def test_real_general_likelihood_evaluation_end_to_end_dataset_aggregate():
    """실제 이미지 파이프라인(evaluate_ghost_general_likelihood)에서 나온
    per-frame 결과로도 dataset aggregate가 올바르게 계산되는지 확인한다."""
    img_low = np.zeros((64, 64, 3), dtype=np.uint8)  # no edges at all -> likelihood ~= 0
    img_high = np.zeros((64, 64, 3), dtype=np.uint8)
    img_high[:, 20:] = 200
    img_high[:, 26:] += 30  # 반복된 double-edge -> likelihood 높음

    frame_low = evaluate_ghost_general_likelihood(img_low, pair_id="low")
    frame_high = evaluate_ghost_general_likelihood(img_high, pair_id="high")
    dataset_result = evaluate_ghost_dataset([frame_low, frame_high], mode="general_likelihood")

    assert dataset_result.mean_ghost_likelihood is not None
    assert frame_high.ghost_likelihood > frame_low.ghost_likelihood
    assert dataset_result.mean_ghost_likelihood == pytest.approx(
        (frame_low.ghost_likelihood + frame_high.ghost_likelihood) / 2.0, abs=1e-9,
    )


# ===========================================================================
# STEP 8 semantic/safety fix 4 - Multi-frame Resolution Consistency Gate
# ===========================================================================

from calibration.windshield.ghost.evaluator import evaluate_ghost_dataset_from_paths


def _write_dot_image(path: str, *, width: int, height: int, dx: float = 4.0, dy: float = -2.0, alpha: float = 0.2) -> None:
    clean = np.zeros((height, width, 3), dtype=np.float32)
    cv2.circle(clean, (width // 3, height // 2), 1, (255, 255, 255), -1)
    cv2.circle(clean, (2 * width // 3, height // 2), 1, (255, 255, 255), -1)
    sample = make_constant_offset_ghost_sample(clean, dx=dx, dy=dy, alpha=alpha)
    observed = np.clip(sample.observed, 0, 255).astype(np.uint8)
    cv2.imwrite(path, observed)


def test_same_resolution_multi_frame_dataset_passes(tmp_path):
    paths = []
    for i in range(5):
        p = tmp_path / f"frame_{i}.png"
        _write_dot_image(str(p), width=320, height=240)
        paths.append(str(p))

    result = evaluate_ghost_dataset_from_paths(paths, _point_source_config(bright_source_threshold=20.0))
    assert result.success
    assert result.num_input_frames == 5
    assert result.num_valid_frames == 5
    assert result.image_width == 320
    assert result.image_height == 240


def test_mixed_resolution_dataset_is_rejected_with_descriptive_error(tmp_path):
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    p3 = tmp_path / "c.png"
    _write_dot_image(str(p1), width=320, height=240)
    _write_dot_image(str(p2), width=320, height=240)
    _write_dot_image(str(p3), width=640, height=480)

    with pytest.raises(ValueError) as excinfo:
        evaluate_ghost_dataset_from_paths([str(p1), str(p2), str(p3)], _point_source_config(bright_source_threshold=20.0))

    message = str(excinfo.value).lower()
    assert "mixed" in message
    assert "resolution" in message
    assert "c.png" in str(excinfo.value)  # 어떤 파일이 문제인지 명시


def test_mixed_resolution_does_not_silently_skip_or_resize(tmp_path):
    """Mixed-resolution frame을 조용히 건너뛰거나 resize하지 않고 dataset
    전체를 실패시켜야 한다(사용자 스펙 4-D번) - 예외 자체가 이를 보장한다
    (부분 결과를 반환하지 않음)."""
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    _write_dot_image(str(p1), width=320, height=240)
    _write_dot_image(str(p2), width=160, height=120)

    with pytest.raises(ValueError):
        evaluate_ghost_dataset_from_paths([str(p1), str(p2)], _point_source_config(bright_source_threshold=20.0))


def test_resolution_gate_progress_callback_still_fires_before_failure(tmp_path):
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    _write_dot_image(str(p1), width=320, height=240)
    _write_dot_image(str(p2), width=640, height=480)

    messages = []
    with pytest.raises(ValueError):
        evaluate_ghost_dataset_from_paths(
            [str(p1), str(p2)], _point_source_config(bright_source_threshold=20.0),
            progress_callback=messages.append,
        )
    assert len(messages) >= 1


# ===========================================================================
# Phase A-5 - num_valid_frames must count only *successfully evaluated*
# frames, not merely "attempted" frames.
# ===========================================================================

def test_num_valid_frames_excludes_failed_evaluations():
    from calibration.windshield.ghost.types import GhostEvaluationResult

    good = GhostEvaluationResult(success=True, mode="point_source", detection_count=1)
    bad = GhostEvaluationResult(success=False, mode="point_source", error_message="boom")
    dataset_result = evaluate_ghost_dataset([good, good, bad], mode="point_source", num_input_frames=3)

    assert dataset_result.num_input_frames == 3
    assert dataset_result.num_valid_frames == 2
    assert dataset_result.num_failed_frames == 1


def test_dataset_from_paths_counts_unreadable_image_as_failed_not_silently_dropped(tmp_path):
    """읽을 수 없는 이미지는 조용히 건너뛰지 않고, 실패한 frame으로
    기록되어야 한다(Phase A-5) - 그래야 num_input_frames == len(per_frame)
    불변식이 유지되고, 실패 원인도 결과에 남는다."""
    good_path = tmp_path / "good.png"
    _write_dot_image(str(good_path), width=320, height=240)
    bad_path = tmp_path / "not_an_image.png"
    bad_path.write_bytes(b"not a real image file")

    result = evaluate_ghost_dataset_from_paths(
        [str(good_path), str(bad_path)], _point_source_config(bright_source_threshold=20.0),
    )
    assert result.num_input_frames == 2
    assert len(result.per_frame) == 2  # 실패한 이미지도 per_frame 항목으로 남는다
    assert result.num_valid_frames == 1
    assert result.num_failed_frames == 1
    failed = [f for f in result.per_frame if not f.success]
    assert len(failed) == 1
    assert failed[0].error_message is not None


def test_all_frames_failed_dataset_reports_zero_valid():
    from calibration.windshield.ghost.types import GhostEvaluationResult

    bad = GhostEvaluationResult(success=False, mode="point_source", error_message="boom")
    dataset_result = evaluate_ghost_dataset([bad, bad], mode="point_source")
    assert dataset_result.num_valid_frames == 0
    assert dataset_result.num_failed_frames == 2
    assert dataset_result.success is False


# ===========================================================================
# Phase A-6 - frame_ids/image_paths length mismatch must not silently
# truncate the dataset.
# ===========================================================================

def test_frame_ids_length_mismatch_raises_instead_of_silent_truncation(tmp_path):
    paths = []
    for i in range(4):
        p = tmp_path / f"frame_{i}.png"
        _write_dot_image(str(p), width=320, height=240)
        paths.append(str(p))

    with pytest.raises(ValueError, match="frame_ids"):
        evaluate_ghost_dataset_from_paths(
            paths, _point_source_config(bright_source_threshold=20.0),
            frame_ids=["a", "b"],  # 4개 중 2개만 - 길이 불일치
        )


def test_frame_ids_matching_length_is_accepted(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"frame_{i}.png"
        _write_dot_image(str(p), width=320, height=240)
        paths.append(str(p))

    result = evaluate_ghost_dataset_from_paths(
        paths, _point_source_config(bright_source_threshold=20.0),
        frame_ids=["a", "b", "c"],
    )
    assert result.success
    assert [f.pair_id for f in result.per_frame] == ["a", "b", "c"]


# ===========================================================================
# STEP 8 semantic/safety fix 5 - Pair Score energy-ratio consistency
# ===========================================================================

from calibration.windshield.ghost.point_detector import (
    _CandidatePair,
    _generate_candidate_pairs,
    estimate_dominant_energy_ratio,
)


def test_bright_main_neighbour_trap_energy_consistency_prefers_real_ghost():
    """사용자 스펙 5-G번, "Bright Main Neighbour Trap" - 이웃 Main이
    ghost<main energy gate만 놓고 보면 candidate가 될 수 있지만, 실제
    ghost와 energy ratio가 크게 다르면(dominant ratio~=0.15 vs
    후보~=0.78) 걸러져야 한다. 이 시나리오는 vector_error만으로도 이미
    구분되지만(진짜 ghost가 훨씬 dominant vector에 가까움), energy 항이
    추가되어도 여전히 진짜 ghost가 선택되어야 한다(회귀 방지)."""
    blobs = [
        BrightBlob(x=10, y=10, energy=255, area_px=10),   # Main A
        BrightBlob(x=14, y=8, energy=40, area_px=10),     # Ghost A (real, ratio ~0.157)
        BrightBlob(x=12, y=10, energy=200, area_px=10),   # Main B - darker neighbour, ghost<main gate 통과
        BrightBlob(x=16, y=8, energy=32, area_px=10),     # Ghost B (real, ratio ~0.16)
        BrightBlob(x=60, y=60, energy=180, area_px=10),   # Main C - independent, far away
        BrightBlob(x=64, y=58, energy=28, area_px=10),    # Ghost C (real, ratio ~0.156)
    ]
    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=60.0)
    detected = {(round(d.main_x), round(d.main_y)): d for d in detections if d.detected}
    assert (10, 10) in detected
    main_a = detected[(10, 10)]
    # Main A는 이웃 Main B(200,dist~2.83)가 아니라 자신의 진짜 Ghost A(dist~4.47)를 선택해야 한다.
    assert main_a.ghost_x == pytest.approx(14.0)
    assert main_a.ghost_y == pytest.approx(8.0)
    assert main_a.strength_ratio < 0.3  # Main B(ratio 0.78)를 골랐다면 훨씬 컸을 것


def test_unequal_main_brightness_yields_stable_dominant_energy_ratio():
    """사용자 스펙 5-G번, Unequal Main Brightness - Main 밝기가 달라도
    (255/210/180) 전부 ratio~=0.15라면 dominant energy ratio가
    안정적으로 ~=0.15로 추정되어야 한다.

    Main 위치를 불규칙한 간격(0, 47, 139)으로 둔다 - 균등 간격(0,50,100)을
    쓰면 "Main0->Main1->Main2"/"Ghost0->Ghost1->Ghost2" 사슬이 우연히 진짜
    ghost displacement와 무관한 자기들끼리의 공통 벡터 cluster를 만들어
    (예: 둘 다 (50,0)) displacement consensus 자체를 왜곡시킨다 - 이는
    테스트 fixture의 인위적 결함이지 알고리즘 버그가 아니므로, 불규칙
    간격으로 그런 우연한 정렬을 피한다."""
    blobs = [
        BrightBlob(x=0, y=0, energy=255, area_px=10), BrightBlob(x=4, y=-2, energy=38, area_px=10),
        BrightBlob(x=47, y=3, energy=210, area_px=10), BrightBlob(x=51, y=1, energy=31, area_px=10),
        BrightBlob(x=139, y=-5, energy=180, area_px=10), BrightBlob(x=143, y=-7, energy=27, area_px=10),
    ]
    candidates = _generate_candidate_pairs(blobs, max_search_radius_px=60.0)
    dominant = estimate_dominant_ghost_vector(candidates, min_consensus_candidates=3)
    assert dominant is not None
    dominant_ratio = estimate_dominant_energy_ratio(candidates, dominant)
    assert dominant_ratio is not None
    assert dominant_ratio == pytest.approx(0.15, abs=0.02)

    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=60.0)
    detected = [d for d in detections if d.detected]
    assert len(detected) == 3
    for det in detected:
        assert det.pair_energy_residual is not None
        assert det.pair_energy_residual < 0.05


def test_energy_ratio_outlier_does_not_dominate_robust_median():
    """사용자 스펙 5-G번, Energy outlier - 하나의 pair만 ratio가 크게
    달라도(0.60) robust median(dominant vector inlier들의 median)이 크게
    움직이면 안 된다."""
    candidates = [
        _CandidatePair(main_idx=i, ghost_idx=i + 100, dx=4.0, dy=-2.0, distance=4.47, energy_ratio=ratio)
        for i, ratio in enumerate([0.15, 0.16, 0.14, 0.15, 0.60])
    ]
    dominant = estimate_dominant_ghost_vector(candidates, consensus_radius_px=3.0, min_consensus_candidates=3)
    assert dominant is not None
    ratio = estimate_dominant_energy_ratio(candidates, dominant, consensus_radius_px=3.0)
    assert ratio is not None
    assert ratio == pytest.approx(0.15, abs=0.02)  # median of [.15,.16,.14,.15,.60] with radius-based inliers


def test_single_led_fallback_ignores_energy_ratio_term():
    """Consensus가 없으면(candidate 부족) energy-ratio 항도 사용하지
    않는다(사용자 스펙 5-E번) - 순수 거리 기반 fallback이 그대로 동작해야
    한다(기존 동작 유지, 회귀 없음)."""
    blobs = [
        BrightBlob(x=100.0, y=100.0, energy=1000.0, area_px=10),
        BrightBlob(x=104.0, y=98.0, energy=150.0, area_px=10),
    ]
    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=60.0, min_consensus_candidates=3)
    detected = [d for d in detections if d.detected]
    assert len(detected) == 1
    assert detected[0].offset_x_px == pytest.approx(4.0)
    assert detected[0].offset_y_px == pytest.approx(-2.0)
    assert detected[0].pair_energy_residual is None  # fallback이므로 energy 항 자체가 정의되지 않음


def test_estimate_dominant_energy_ratio_returns_none_without_dominant_vector():
    candidates = [
        _CandidatePair(main_idx=0, ghost_idx=1, dx=4.0, dy=-2.0, distance=4.47, energy_ratio=0.15),
    ]
    assert estimate_dominant_energy_ratio(candidates, None) is None


# ===========================================================================
# Phase B-3 - Regular LED-grid Ghost regression (실제 calibration target이
# 규칙적인 패턴일 가능성이 높으므로 반드시 검증해야 한다).
# ===========================================================================

def _regular_led_grid_blobs(
    *, rows=4, cols=5, spacing=30.0, dx=5.0, dy=-3.0, ratio=0.15, main_energy=1000.0,
    skip_positions=(), energy_variation=None, extra_outlier_ratio_at=None,
):
    """rows x cols 격자로 배치된 Main LED + 각자의 Ghost(dx,dy,ratio 공통)를
    만든다. `skip_positions`에 있는 (r,c)는 Main 자체를 빼서 "missing LED"를
    흉내낸다. `energy_variation`은 (r,c)->energy 배율 dict로 밝기를
    다르게 만든다. `extra_outlier_ratio_at`는 (r,c)의 ghost energy ratio만
    다르게(outlier) 만든다."""
    blobs = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in skip_positions:
                continue
            mx, my = c * spacing, r * spacing
            energy = main_energy * (energy_variation.get((r, c), 1.0) if energy_variation else 1.0)
            blobs.append(BrightBlob(x=mx, y=my, energy=energy, area_px=10))
            local_ratio = ratio
            if extra_outlier_ratio_at is not None and (r, c) == extra_outlier_ratio_at:
                local_ratio = extra_outlier_ratio_at[2] if len(extra_outlier_ratio_at) > 2 else 0.7
            blobs.append(BrightBlob(x=mx + dx, y=my + dy, energy=energy * local_ratio, area_px=10))
    return blobs


def test_regular_5x4_led_grid_recovers_dominant_offset_and_ratio():
    blobs = _regular_led_grid_blobs(rows=4, cols=5, spacing=30.0, dx=5.0, dy=-3.0, ratio=0.15)
    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=15.0)
    detected = [d for d in detections if d.detected]

    assert len(detected) == 20  # 4x5 격자 전부 자기 자신의 ghost와 짝지어져야 한다
    for det in detected:
        assert det.offset_x_px == pytest.approx(5.0, abs=0.2)
        assert det.offset_y_px == pytest.approx(-3.0, abs=0.2)
        assert det.strength_ratio == pytest.approx(0.15, abs=0.02)
        # 규칙적인 격자에서도 main-main mispair가 없어야 한다: 짝지어진
        # ghost는 항상 main으로부터 정확히 (5,-3) 근처여야지, 이웃 Main
        # (거리 30px)일 수 없다.
        assert det.distance_px < 10.0


def test_regular_led_grid_with_unequal_brightness():
    """격자 안에서 Main 밝기가 위치마다 달라도(예: 좌상단이 더 밝고
    우하단이 더 어두운 vignette 패턴) dominant offset/ratio는 흔들리면
    안 된다."""
    variation = {(r, c): 1.0 - 0.15 * (r + c) / 8.0 for r in range(4) for c in range(5)}
    blobs = _regular_led_grid_blobs(
        rows=4, cols=5, spacing=30.0, dx=5.0, dy=-3.0, ratio=0.15, energy_variation=variation,
    )
    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=15.0)
    detected = [d for d in detections if d.detected]
    assert len(detected) == 20
    for det in detected:
        assert det.offset_x_px == pytest.approx(5.0, abs=0.2)
        assert det.offset_y_px == pytest.approx(-3.0, abs=0.2)


def test_regular_led_grid_with_missing_leds():
    """일부 LED가 검출 실패/빠짐(missing)이어도 나머지는 정상적으로
    pairing되어야 한다."""
    missing = {(0, 0), (1, 3), (3, 4), (2, 2)}
    blobs = _regular_led_grid_blobs(rows=4, cols=5, spacing=30.0, dx=5.0, dy=-3.0, ratio=0.15, skip_positions=missing)
    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=15.0)
    detected = [d for d in detections if d.detected]
    assert len(detected) == 20 - len(missing)
    for det in detected:
        assert det.offset_x_px == pytest.approx(5.0, abs=0.2)
        assert det.offset_y_px == pytest.approx(-3.0, abs=0.2)


def test_regular_led_grid_with_one_energy_outlier():
    """한 위치의 ghost energy ratio만 크게 벗어나도(outlier) 그 위치의
    pairing 자체는 displacement가 여전히 정확하므로 정상적으로
    이루어져야 하고, dominant ratio(다른 19개 기준)는 흔들리면 안 된다."""
    outlier_pos = (2, 3)
    blobs = _regular_led_grid_blobs(
        rows=4, cols=5, spacing=30.0, dx=5.0, dy=-3.0, ratio=0.15,
        extra_outlier_ratio_at=(outlier_pos[0], outlier_pos[1], 0.7),
    )
    candidates = _generate_candidate_pairs(blobs, max_search_radius_px=15.0)
    dominant = estimate_dominant_ghost_vector(candidates, min_consensus_candidates=3)
    assert dominant is not None
    dominant_ratio = estimate_dominant_energy_ratio(candidates, dominant)
    assert dominant_ratio == pytest.approx(0.15, abs=0.02)  # outlier 하나가 median을 흔들면 안 된다

    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=15.0)
    detected = [d for d in detections if d.detected]
    assert len(detected) == 20
    for det in detected:
        # Outlier 위치라도 displacement 자체는 정확하다 - energy만 다르다.
        assert det.offset_x_px == pytest.approx(5.0, abs=0.2)
        assert det.offset_y_px == pytest.approx(-3.0, abs=0.2)


def test_regular_led_grid_with_one_unrelated_bright_neighbor():
    """격자와 무관한 밝은 점 하나가 격자 사이에 끼어 있어도(예: 다른
    반사광원) 격자 pairing 자체가 깨지면 안 된다."""
    blobs = _regular_led_grid_blobs(rows=4, cols=5, spacing=30.0, dx=5.0, dy=-3.0, ratio=0.15)
    # 격자 칸 사이(15,15 오프셋)에 무관한 밝은 점 하나 추가 - 어떤 Main과도
    # (5,-3) 벡터로 맞지 않는다.
    blobs.append(BrightBlob(x=15.0, y=15.0, energy=900.0, area_px=10))

    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=15.0)
    detected = [d for d in detections if d.detected]
    assert len(detected) == 20  # 무관한 점은 어느 Main과도 짝지어지지 않아야 한다
    for det in detected:
        assert det.offset_x_px == pytest.approx(5.0, abs=0.2)
        assert det.offset_y_px == pytest.approx(-3.0, abs=0.2)


# ===========================================================================
# Priority 4 안정화 - 위 테스트들은 전부 max_search_radius_px=15.0(< LED
# spacing 30px)라서 이웃 Main/이웃 Main의 Ghost가 애초에 candidate조차 될 수
# 없었다. 실제로 어려운 상황은 search radius가 LED 간격과 비슷하거나 더
# 커서(예: 60px) 이웃 Main과 이웃 Main의 Ghost까지 candidate set에 들어오는
# 경우다 - 이 섹션은 그 조건에서도 pairing이 정확한지 확인한다.
# ===========================================================================

def test_regular_led_grid_with_large_search_radius_still_pairs_own_ghost():
    """4x5 grid, spacing=30px, search radius=60px(> spacing) - 이웃 Main과
    이웃 Main의 Ghost가 전부 geometric하게 candidate가 되는 조건에서도,
    각 Main이 자기 자신의 Ghost와 정확히 pairing돼야 한다."""
    blobs = _regular_led_grid_blobs(rows=4, cols=5, spacing=30.0, dx=5.0, dy=-3.0, ratio=0.15)
    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=60.0)
    detected = [d for d in detections if d.detected]

    assert len(detected) == 20
    for det in detected:
        assert det.offset_x_px == pytest.approx(5.0, abs=0.5)
        assert det.offset_y_px == pytest.approx(-3.0, abs=0.5)
        # 이웃 Main 자체(거리 30px, offset이 (30,0) 근처)를 ghost로 선택하지
        # 않아야 한다 - 진짜 ghost distance(~5.8px)와 뚜렷이 구분된다.
        assert det.distance_px < 10.0


def test_regular_led_grid_large_radius_with_missing_leds():
    """Large search radius에서도 일부 LED가 빠졌을 때 나머지가 정상적으로
    pairing되어야 한다."""
    missing = {(0, 0), (1, 3), (3, 4), (2, 2)}
    blobs = _regular_led_grid_blobs(rows=4, cols=5, spacing=30.0, dx=5.0, dy=-3.0, ratio=0.15, skip_positions=missing)
    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=60.0)
    detected = [d for d in detections if d.detected]
    assert len(detected) == 20 - len(missing)
    for det in detected:
        assert det.offset_x_px == pytest.approx(5.0, abs=0.5)
        assert det.offset_y_px == pytest.approx(-3.0, abs=0.5)


def test_regular_led_grid_large_radius_with_energy_outlier():
    """Large search radius에서도 ghost energy ratio outlier 하나가 전체
    pairing을 깨지 않아야 한다."""
    blobs = _regular_led_grid_blobs(
        rows=4, cols=5, spacing=30.0, dx=5.0, dy=-3.0, ratio=0.15,
        extra_outlier_ratio_at=(2, 3, 0.7),
    )
    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=60.0)
    detected = [d for d in detections if d.detected]
    assert len(detected) == 20
    for det in detected:
        assert det.offset_x_px == pytest.approx(5.0, abs=0.5)
        assert det.offset_y_px == pytest.approx(-3.0, abs=0.5)


def test_regular_led_grid_large_radius_via_production_two_pass_pairing():
    """Priority 3(production wiring)과 Priority 4(large radius)를 함께
    검증한다 - 실제 production evaluator가 쓰는
    `pair_main_and_ghost_blobs_two_pass()` 경로도 large radius에서 동일하게
    정확해야 한다."""
    blobs = _regular_led_grid_blobs(rows=4, cols=5, spacing=30.0, dx=5.0, dy=-3.0, ratio=0.15)
    detections = pair_main_and_ghost_blobs_two_pass(
        blobs, image_width=200.0, image_height=200.0, max_search_radius_px=60.0,
    )
    detected = [d for d in detections if d.detected]
    assert len(detected) == 20
    for det in detected:
        assert det.offset_x_px == pytest.approx(5.0, abs=0.5)
        assert det.offset_y_px == pytest.approx(-3.0, abs=0.5)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "알려진 한계(Priority 4 조사 중 발견, 이번 안정화 라운드의 범위를 "
        "벗어남): LED grid 전체에 걸친 매끄러운 밝기 gradient(vignette)와 "
        "search radius가 LED 간격보다 클 때(예: 60px > 30px spacing), 밝은 "
        "Main이 그보다 살짝 어두운 이웃 Main 여러 개를 'ghost<main energy' "
        "게이트만으로 후보로 착각할 수 있다. 이 grid(20개 Main) 규모에서는 "
        "그런 Main-Main 후보 수가 실제 Main-Ghost 후보 수(20개)를 넘어서 "
        "estimate_dominant_ghost_vector()가 진짜 displacement(+5,-3) 대신 "
        "이웃 Main 방향(+30,0)을 dominant vector로 잘못 고른다. 새로운 "
        "consensus 알고리즘이나 임의 threshold를 발명하지 않고는 고칠 수 "
        "없어(이번 라운드의 '새 알고리즘 추가 금지' 범위 밖) 정직하게 "
        "xfail로 남긴다."
    ),
)
def test_regular_led_grid_large_radius_with_smooth_brightness_gradient_is_a_known_limitation():
    variation = {(r, c): 1.0 - 0.15 * (r + c) / 8.0 for r in range(4) for c in range(5)}
    blobs = _regular_led_grid_blobs(
        rows=4, cols=5, spacing=30.0, dx=5.0, dy=-3.0, ratio=0.15, energy_variation=variation,
    )
    detections = pair_main_and_ghost_blobs(blobs, max_search_radius_px=60.0)
    detected = [d for d in detections if d.detected]
    assert len(detected) == 20
    for det in detected:
        assert det.offset_x_px == pytest.approx(5.0, abs=0.5)
        assert det.offset_y_px == pytest.approx(-3.0, abs=0.5)


# ===========================================================================
# Phase B-4 - Ghost 2-pass Spatial Pairing.
# ===========================================================================

from calibration.windshield.ghost.point_detector import pair_main_and_ghost_blobs_two_pass


def test_two_pass_pairing_recovers_local_offset_that_global_vector_gets_wrong():
    """PASS 1(단일 global dominant vector)은 이미지 전체에서 가장 support가
    큰 하나의 displacement만 대표로 쓰므로, 소수 지역에서 실제 ghost
    displacement가 다르면(windshield 곡률 등) 그 지역의 애매한 pairing을
    틀리게 고를 수 있다. PASS 2(coarse spatial field 기반 재-pairing)는
    이 경우를 복구해야 한다."""
    image_width, image_height = 500.0, 200.0

    # 왼쪽: 4x5 규칙적 LED grid, true ghost vector A=(+5,-3) - global
    # consensus를 지배하는 다수 지역(20개 main, x<200이라 오른쪽과 같은
    # coarse cell에 절대 섞이지 않는다).
    blobs = _regular_led_grid_blobs(rows=4, cols=5, spacing=30.0, dx=5.0, dy=-3.0, ratio=0.15)

    # 오른쪽: 같은 coarse cell(3x3 grid의 (row=1,col=1)) 안에 놓인 4개의
    # Main - true ghost vector B=(+9,-7)로, A와는 displacement가 뚜렷이
    # 다른(consensus_radius=3.0보다 먼) 별도의 local cluster.
    m1, m2, m3, m4 = (300.0, 100.0), (330.0, 100.0), (300.0, 130.0), (330.0, 130.0)
    dx_b, dy_b, ratio = 9.0, -7.0, 0.15
    for mx, my in (m1, m2, m3):
        blobs.append(BrightBlob(x=mx, y=my, energy=1000.0, area_px=10))
        blobs.append(BrightBlob(x=mx + dx_b, y=my + dy_b, energy=1000.0 * ratio, area_px=10))

    # M4는 "애매한" main이다 - 진짜 local ghost(B)뿐 아니라, 우연히 global
    # dominant vector A와 정확히 일치하는 위치에 놓인 밝은 점(distractor,
    # 예: 다른 반사/광원)도 함께 있다. PASS 1은 global vector(A)로 점수를
    # 매기므로 이 distractor를 진짜 ghost로 착각해야 한다(이 assertion
    # 자체가 "2-pass 없이는 실제로 문제가 있다"를 먼저 증명한다).
    mx4, my4 = m4
    blobs.append(BrightBlob(x=mx4, y=my4, energy=1000.0, area_px=10))
    blobs.append(BrightBlob(x=mx4 + dx_b, y=my4 + dy_b, energy=1000.0 * ratio, area_px=10))  # true local ghost(B)
    blobs.append(BrightBlob(x=mx4 + 5.0, y=my4 - 3.0, energy=1000.0 * ratio, area_px=10))  # distractor at global A

    pass1 = pair_main_and_ghost_blobs(blobs, max_search_radius_px=15.0)
    pass1_detected = [d for d in pass1 if d.detected]
    assert len(pass1_detected) == 24  # 20(왼쪽) + 4(오른쪽)

    pass1_m4 = next(d for d in pass1_detected if d.main_x == mx4 and d.main_y == my4)
    assert pass1_m4.offset_x_px == pytest.approx(5.0, abs=0.5)
    assert pass1_m4.offset_y_px == pytest.approx(-3.0, abs=0.5)

    two_pass = pair_main_and_ghost_blobs_two_pass(
        blobs, image_width=image_width, image_height=image_height, max_search_radius_px=15.0,
    )
    two_pass_detected = [d for d in two_pass if d.detected]
    assert len(two_pass_detected) == 24  # PASS 2는 detected 여부 자체를 바꾸지 않는다

    two_pass_m4 = next(d for d in two_pass_detected if d.main_x == mx4 and d.main_y == my4)
    # PASS 2는 M4가 속한 coarse cell의 나머지 3개 clean main이 전부 진짜
    # local vector B로 pairing된 것을 보고, M4도 B로 재-pairing해야 한다.
    assert two_pass_m4.offset_x_px == pytest.approx(dx_b, abs=0.5)
    assert two_pass_m4.offset_y_px == pytest.approx(dy_b, abs=0.5)

    # 2-pass가 나머지(애매하지 않은) main들의 정확한 pairing을 망가뜨리지
    # 않았는지도 함께 확인한다(사용자 스펙: "PASS 2는 안전해야 한다").
    for det in two_pass_detected:
        if det.main_x == mx4 and det.main_y == my4:
            continue
        if det.main_x < 200.0:  # 왼쪽 LED grid
            assert det.offset_x_px == pytest.approx(5.0, abs=0.3)
            assert det.offset_y_px == pytest.approx(-3.0, abs=0.3)
        else:  # 오른쪽 clean main들
            assert det.offset_x_px == pytest.approx(dx_b, abs=0.3)
            assert det.offset_y_px == pytest.approx(dy_b, abs=0.3)


def test_two_pass_pairing_falls_back_to_pass_one_with_too_few_detections():
    """Candidate가 너무 적어 PASS 1 자체가 이미 global consensus 없이
    (순수 거리 기반) pairing된 경우, coarse spatial field를 만들 근거가
    없으므로 PASS 2는 PASS 1 결과를 그대로 반환해야 한다(fallback,
    사용자 스펙 B-4번)."""
    blobs = [
        BrightBlob(x=100.0, y=100.0, energy=1000.0, area_px=10),
        BrightBlob(x=104.0, y=98.0, energy=150.0, area_px=10),
    ]
    pass1 = pair_main_and_ghost_blobs(blobs, max_search_radius_px=60.0, min_consensus_candidates=3)
    two_pass = pair_main_and_ghost_blobs_two_pass(
        blobs, image_width=320.0, image_height=240.0, max_search_radius_px=60.0, min_consensus_candidates=3,
    )
    assert len(two_pass) == len(pass1)
    for a, b in zip(pass1, two_pass):
        assert a.detected == b.detected
        assert a.offset_x_px == b.offset_x_px
        assert a.offset_y_px == b.offset_y_px


def test_two_pass_pairing_handles_empty_blobs_without_crashing():
    two_pass = pair_main_and_ghost_blobs_two_pass(
        [], image_width=320.0, image_height=240.0,
    )
    assert two_pass == []


def test_two_pass_pairing_on_regular_grid_matches_single_pass_result():
    """단일하고 고른(spatial variation이 없는) 격자에서는 PASS 2가 PASS 1과
    동일한 결과를 내야 한다(회귀 없음 - local vector가 global vector와
    사실상 같아지므로)."""
    blobs = _regular_led_grid_blobs(rows=4, cols=5, spacing=30.0, dx=5.0, dy=-3.0, ratio=0.15)
    pass1 = pair_main_and_ghost_blobs(blobs, max_search_radius_px=15.0)
    two_pass = pair_main_and_ghost_blobs_two_pass(
        blobs, image_width=200.0, image_height=200.0, max_search_radius_px=15.0,
    )
    assert len(two_pass) == len(pass1) == 20
    for det in two_pass:
        assert det.detected
        assert det.offset_x_px == pytest.approx(5.0, abs=0.2)
        assert det.offset_y_px == pytest.approx(-3.0, abs=0.2)
