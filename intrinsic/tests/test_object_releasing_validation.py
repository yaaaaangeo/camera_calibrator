"""
tests/test_object_releasing_validation.py
=============================================

calibration/object_releasing_validation.py 검증.

test_smoke_pipeline.py와 같은 방식(3D->2D 직접 사영, 이미지 파일/실제 검출 없음)으로
빠르게 합성 데이터셋을 만든다 - conftest.py의 공용 synthetic_dataset fixture는
ChArUco 전용이라 Object-Releasing(Checkerboard/Circle Grid만 지원)에는 쓸 수 없다.

가장 중요하게 검증하는 것(사용자 스펙의 핵심 원칙):
    Train에서만 calibrate_object_releasing_brown_conrady()가 호출되고,
    Test는 그 결과(K/D/refined geometry)를 고정한 채 pose만 다시 구한다.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import calibration.object_releasing_validation as orv
from calibration.models.object_releasing import (
    calibrate_object_releasing_brown_conrady,
    expected_object_releasing_ids,
    expected_object_releasing_object_points,
)
from calibration.types import (
    CameraConfig,
    Dataset,
    DetectionResult,
    Frame,
    FrameStatus,
    ImageInfo,
    PatternConfig,
    PatternType,
)
from calibration.validation import _subset_dataset, validate_holdout
from calibration.project_io import project_from_dict, project_to_dict
from calibration.types import CalibrationProject

W, H = 640, 480
TRUE_K = np.array([[600.0, 0, W / 2], [0, 600.0, H / 2], [0, 0, 1]])
TRUE_D = np.array([-0.15, 0.04, 0.0, 0.0, 0.0])


def _chessboard_pattern() -> PatternConfig:
    return PatternConfig(type=PatternType.CHESSBOARD, squares_x=6, squares_y=5, square_size=0.03)


def _full_board_dataset(pattern: PatternConfig, n_frames: int, *, seed: int = 0) -> Dataset:
    """이미지 파일/검출 없이 3D->2D 직접 사영으로 full-board 프레임을 만든다."""
    obj_pts = expected_object_releasing_object_points(pattern).astype(np.float32)
    ids = expected_object_releasing_ids(pattern)
    n_pts = len(ids)

    rng = np.random.default_rng(seed)
    frames: list[Frame] = []
    attempts = 0
    while len(frames) < n_frames and attempts < n_frames * 40:
        attempts += 1
        rvec = (rng.random(3) - 0.5) * 0.6
        tvec = np.array([
            (rng.random() - 0.5) * 0.15,
            (rng.random() - 0.5) * 0.15,
            0.5 + rng.random() * 0.25,
        ])
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, TRUE_K, TRUE_D)
        proj = proj.reshape(-1, 2)
        if np.any(proj < 5) or np.any(proj[:, 0] > W - 5) or np.any(proj[:, 1] > H - 5):
            continue

        image_id = f"f{len(frames):03d}"
        det = DetectionResult(
            image_id=image_id,
            success=True,
            corners=proj.reshape(-1, 1, 2).astype(np.float32),
            object_points=obj_pts.reshape(-1, 1, 3),
            ids=ids.reshape(-1, 1),
            num_corners=n_pts,
        )
        frames.append(
            Frame(
                image_info=ImageInfo(image_id=image_id, path="-", width=W, height=H),
                detection=det,
                status=FrameStatus.DETECTED,
            )
        )

    assert len(frames) == n_frames, "합성 프레임 생성 실패 - 파라미터 조정 필요"
    return Dataset(frames=frames)


def _camera_config() -> CameraConfig:
    return CameraConfig(width=W, height=H)


# ---------------------------------------------------------------------------
# P1-A: validate_object_releasing_holdout
# ---------------------------------------------------------------------------


def test_holdout_succeeds_and_reports_expected_fields():
    pattern = _chessboard_pattern()
    dataset = _full_board_dataset(pattern, n_frames=20)
    camera_config = _camera_config()

    result = orv.validate_object_releasing_holdout(dataset, camera_config, pattern, seed=0)

    assert result.success, result.error_message
    assert len(result.train_frame_ids) >= 3
    assert len(result.test_frame_ids) >= 1
    assert result.train_rms is not None and result.train_rms >= 0
    assert result.test_rms is not None and result.test_rms >= 0
    assert result.test_residual_stats is not None
    assert result.test_residual_stats.n > 0
    assert result.test_residual_stats.rmse is not None
    assert result.test_residual_stats.median is not None
    assert result.test_residual_stats.p95 is not None
    assert result.test_residual_stats.p99 is not None
    assert result.test_residual_stats.max is not None
    assert result.target_geometry_refinement is not None
    assert not result.failed_test_frame_ids


def test_holdout_train_test_split_is_disjoint_and_deterministic():
    pattern = _chessboard_pattern()
    dataset = _full_board_dataset(pattern, n_frames=20)
    camera_config = _camera_config()

    r1 = orv.validate_object_releasing_holdout(dataset, camera_config, pattern, seed=7)
    r2 = orv.validate_object_releasing_holdout(dataset, camera_config, pattern, seed=7)

    assert r1.success and r2.success
    assert set(r1.train_frame_ids).isdisjoint(set(r1.test_frame_ids))
    assert r1.train_frame_ids == r2.train_frame_ids
    assert r1.test_frame_ids == r2.test_frame_ids


def test_holdout_train_rms_matches_independent_refit_on_train_subset():
    """가장 중요한 계약: train_rms는 test 평가와 완전히 독립적으로 재현 가능해야
    한다 - tests/test_validation.py의 표준 모델용 동등 테스트와 같은 취지.
    """
    pattern = _chessboard_pattern()
    dataset = _full_board_dataset(pattern, n_frames=20)
    camera_config = _camera_config()

    result = orv.validate_object_releasing_holdout(dataset, camera_config, pattern, seed=1)
    assert result.success, result.error_message

    train_subset = _subset_dataset(dataset, result.train_frame_ids)
    independent = calibrate_object_releasing_brown_conrady(train_subset, camera_config, pattern)

    assert independent.success, independent.error_message
    assert result.train_rms == pytest.approx(independent.rms_error, rel=1e-9, abs=1e-9)


def test_holdout_never_recalibrates_on_test_data(monkeypatch):
    """calibrate_object_releasing_brown_conrady (-> cv2.calibrateCameraRO/-Extended)가
    정확히 한 번만(Train에서만) 호출되는지 spy로 확인한다.
    """
    pattern = _chessboard_pattern()
    dataset = _full_board_dataset(pattern, n_frames=20)
    camera_config = _camera_config()

    call_count = {"n": 0}
    real = orv.calibrate_object_releasing_brown_conrady

    def counting_wrapper(*args, **kwargs):
        call_count["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(orv, "calibrate_object_releasing_brown_conrady", counting_wrapper)

    result = orv.validate_object_releasing_holdout(dataset, camera_config, pattern, seed=0)

    assert result.success, result.error_message
    assert call_count["n"] == 1


def test_holdout_insufficient_full_board_frames_fails_clearly():
    pattern = _chessboard_pattern()
    dataset = _full_board_dataset(pattern, n_frames=3)  # < MIN_FULL_BOARD_FRAMES_FOR_HOLDOUT
    camera_config = _camera_config()

    result = orv.validate_object_releasing_holdout(dataset, camera_config, pattern, seed=0)

    assert not result.success
    assert result.error_message
    assert "Insufficient" in result.error_message


def test_holdout_rejects_bad_test_frame_with_reason_not_silently():
    """caller가 실수로(또는 예상치 못한 경로로) full-board가 아닌 프레임의 id를
    test_ids에 넣더라도, 조용히 빠뜨리지 않고 failed_test_frame_ids에 이유와
    함께 기록해야 한다 - eligible pool 계산 없이 helper를 직접 호출해 이
    방어적 재검증 경로를 강제로 통과시킨다.
    """
    pattern = _chessboard_pattern()
    dataset = _full_board_dataset(pattern, n_frames=10)
    camera_config = _camera_config()

    # full-board가 아닌(포인트 하나 누락된) 프레임을 하나 추가한다.
    full_ids = expected_object_releasing_ids(pattern)
    obj_pts = expected_object_releasing_object_points(pattern).astype(np.float32)
    partial_ids = full_ids[:-1]
    partial_obj = obj_pts[:-1]
    proj, _ = cv2.projectPoints(partial_obj, np.zeros(3), np.array([0.0, 0.0, 0.6]), TRUE_K, TRUE_D)
    bad_id = "partial-bad"
    det = DetectionResult(
        image_id=bad_id,
        success=True,
        corners=proj.reshape(-1, 1, 2).astype(np.float32),
        object_points=partial_obj.reshape(-1, 1, 3),
        ids=partial_ids.reshape(-1, 1),
        num_corners=len(partial_ids),
    )
    dataset.frames.append(
        Frame(
            image_info=ImageInfo(image_id=bad_id, path="-", width=W, height=H),
            detection=det,
            status=FrameStatus.DETECTED,
        )
    )

    all_ids = [f.image_info.image_id for f in dataset.frames if f.image_info.image_id != bad_id]
    train_ids = all_ids[:6]
    good_test_ids = all_ids[6:9]

    train_result, validation = orv._run_object_releasing_train_test(
        dataset, camera_config, pattern, train_ids, good_test_ids + [bad_id]
    )

    assert train_result is not None and train_result.success
    assert validation.success  # 나머지 정상 test 프레임으로는 여전히 성공
    assert bad_id in validation.failed_test_frame_ids
    assert bad_id in validation.failed_test_reasons
    assert validation.failed_test_reasons[bad_id]


def test_holdout_object_releasing_validation_result_round_trips_through_project_io():
    pattern = _chessboard_pattern()
    dataset = _full_board_dataset(pattern, n_frames=20)
    camera_config = _camera_config()

    validation = orv.validate_object_releasing_holdout(dataset, camera_config, pattern, seed=0)
    assert validation.success

    project = CalibrationProject(
        project_name="ro-holdout-test",
        camera_config=camera_config,
        pattern_config=pattern,
        dataset=dataset,
        object_releasing_validation_result=validation,
    )
    payload = project_to_dict(project)
    restored = project_from_dict(payload)

    assert restored.object_releasing_validation_result is not None
    assert restored.object_releasing_validation_result.train_frame_ids == validation.train_frame_ids
    assert restored.object_releasing_validation_result.test_frame_ids == validation.test_frame_ids
    assert restored.object_releasing_validation_result.test_rms == pytest.approx(validation.test_rms)


def test_object_releasing_validation_result_missing_from_legacy_payload_loads_as_none():
    pattern = _chessboard_pattern()
    dataset = _full_board_dataset(pattern, n_frames=5)
    camera_config = _camera_config()

    project = CalibrationProject(
        project_name="legacy-like",
        camera_config=camera_config,
        pattern_config=pattern,
        dataset=dataset,
    )
    payload = project_to_dict(project)
    del payload["project"]["object_releasing_validation_result"]
    del payload["project"]["standard_vs_object_releasing_comparison"]

    restored = project_from_dict(payload)

    assert restored.object_releasing_validation_result is None
    assert restored.standard_vs_object_releasing_comparison is None


# ---------------------------------------------------------------------------
# P1-B: compare_standard_vs_object_releasing_brown
# ---------------------------------------------------------------------------


def test_comparison_uses_shared_split_for_both_arms():
    pattern = _chessboard_pattern()
    dataset = _full_board_dataset(pattern, n_frames=20)
    camera_config = _camera_config()

    comparison = orv.compare_standard_vs_object_releasing_brown(dataset, camera_config, pattern, seed=3)

    assert comparison.success, comparison.error_message
    assert comparison.standard_validation is not None
    assert comparison.object_releasing_validation is not None
    assert comparison.standard_validation.train_frame_ids == comparison.object_releasing_validation.train_frame_ids
    assert comparison.standard_validation.test_frame_ids == comparison.object_releasing_validation.test_frame_ids
    assert comparison.train_frame_ids == comparison.standard_validation.train_frame_ids
    assert comparison.test_frame_ids == comparison.standard_validation.test_frame_ids


def test_comparison_excludes_non_full_board_frames_from_both_arms():
    pattern = _chessboard_pattern()
    dataset = _full_board_dataset(pattern, n_frames=20)

    # 부분 검출 프레임을 하나 섞는다 (마지막 포인트 하나가 빠짐 - full-board 아님).
    full_ids = expected_object_releasing_ids(pattern)
    obj_pts = expected_object_releasing_object_points(pattern).astype(np.float32)
    partial_ids = full_ids[:-1]
    partial_obj = obj_pts[:-1]
    proj, _ = cv2.projectPoints(partial_obj, np.zeros(3), np.array([0.0, 0.0, 0.6]), TRUE_K, TRUE_D)
    det = DetectionResult(
        image_id="partial",
        success=True,
        corners=proj.reshape(-1, 1, 2).astype(np.float32),
        object_points=partial_obj.reshape(-1, 1, 3),
        ids=partial_ids.reshape(-1, 1),
        num_corners=len(partial_ids),
    )
    partial_frame = Frame(
        image_info=ImageInfo(image_id="partial", path="-", width=W, height=H),
        detection=det,
        status=FrameStatus.DETECTED,
    )
    dataset.frames.append(partial_frame)
    camera_config = _camera_config()

    comparison = orv.compare_standard_vs_object_releasing_brown(dataset, camera_config, pattern, seed=3)

    assert comparison.success, comparison.error_message
    assert "partial" not in comparison.train_frame_ids
    assert "partial" not in comparison.test_frame_ids
    assert "partial" not in comparison.eligible_frame_ids


def test_comparison_intrinsics_delta_matches_manual_diff():
    pattern = _chessboard_pattern()
    dataset = _full_board_dataset(pattern, n_frames=20)
    camera_config = _camera_config()

    comparison = orv.compare_standard_vs_object_releasing_brown(dataset, camera_config, pattern, seed=3)
    assert comparison.success, comparison.error_message

    sr = comparison.standard_result
    rr = comparison.object_releasing_result
    assert sr is not None and rr is not None

    expected_fx = float(rr.camera_matrix[0, 0] - sr.camera_matrix[0, 0])
    assert comparison.intrinsics_delta["fx"] == pytest.approx(expected_fx)

    expected_k1 = float(np.asarray(rr.distortion).reshape(-1)[0] - np.asarray(sr.distortion).reshape(-1)[0])
    assert comparison.intrinsics_delta["k1"] == pytest.approx(expected_k1)


def test_comparison_never_produces_a_winner_verdict():
    pattern = _chessboard_pattern()
    dataset = _full_board_dataset(pattern, n_frames=20)
    camera_config = _camera_config()

    comparison = orv.compare_standard_vs_object_releasing_brown(dataset, camera_config, pattern, seed=3)
    assert comparison.success

    table = orv.format_standard_vs_object_releasing_table(comparison)
    combined = " ".join(comparison.warnings).lower() + " " + table.lower()

    for banned in ("is better", "more accurate", "더 정확", "더 낫"):
        assert banned not in combined


def test_build_warnings_flags_overfitting_and_large_refinement():
    from calibration.types import CalibrationResult, CameraModelType, ValidationResult

    pattern = _chessboard_pattern()
    standard_result = CalibrationResult(model_name=CameraModelType.BROWN_CONRADY, rms_error=1.0, success=True)
    ro_result = CalibrationResult(
        model_name=CameraModelType.BROWN_CONRADY,
        rms_error=0.5,  # train RMS improved by 50%
        success=True,
        target_geometry_refinement={"max_displacement": 10 * pattern.square_size},  # way over 2% threshold
    )
    standard_validation = ValidationResult(test_rms=1.2, success=True)
    ro_validation = orv.ObjectReleasingValidationResult(test_rms=1.2, success=True)  # no hold-out improvement

    warnings = orv._build_warnings(standard_result, standard_validation, ro_result, ro_validation, pattern)

    assert any("overfitting" in w.lower() for w in warnings)
    assert any("geometry refinement" in w.lower() for w in warnings)


def test_build_warnings_empty_when_nothing_suspicious():
    from calibration.types import CalibrationResult, CameraModelType, ValidationResult

    pattern = _chessboard_pattern()
    standard_result = CalibrationResult(model_name=CameraModelType.BROWN_CONRADY, rms_error=1.0, success=True)
    ro_result = CalibrationResult(
        model_name=CameraModelType.BROWN_CONRADY,
        rms_error=0.95,
        success=True,
        target_geometry_refinement={"max_displacement": 0.001 * pattern.square_size},
    )
    standard_validation = ValidationResult(test_rms=1.1, success=True)
    ro_validation = orv.ObjectReleasingValidationResult(test_rms=0.9, success=True)  # holdout improved too

    warnings = orv._build_warnings(standard_result, standard_validation, ro_result, ro_validation, pattern)

    assert warnings == []
