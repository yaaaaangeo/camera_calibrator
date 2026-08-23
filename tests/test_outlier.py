"""
tests/test_outlier.py
==========================

설계 문서 9번 - 이상치 관리. MAD 기반 threshold 계산이 통계적으로 맞는지,
그리고 "파일을 삭제하지 않고 비활성화만 한다"는 원칙이 실제로 지켜지는지
확인한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.models.common import collect_calibration_inputs
from calibration.outlier import (
    apply_corner_outlier_removal,
    apply_outlier_removal,
    compute_mad_threshold,
    compute_per_corner_errors,
    format_corner_outlier_before_after,
    format_corner_outlier_summary,
    format_outlier_before_after,
    recalibrate_with_corner_outlier_pruning,
    recalibrate_with_outlier_pruning,
    recommend_corner_outliers,
    restore_frame,
)
from calibration.types import CameraModelType, Dataset, DetectionResult, Frame, FrameStatus, ImageInfo


def test_mad_threshold_ignores_extreme_outlier():
    """median+k*MAD는 평균/표준편차 기반 방식과 달리 극단값 하나에 민감하게
    안 흔들려야 한다 - 이게 MAD를 쓰는 이유(설계 문서 9번).
    """
    normal_errors = [0.3, 0.31, 0.29, 0.32, 0.28, 0.30]
    threshold_without_outlier = compute_mad_threshold(normal_errors)

    with_outlier = normal_errors + [50.0]  # 극단적인 이상치 하나 추가
    threshold_with_outlier = compute_mad_threshold(with_outlier)

    # MAD는 median 기반이라 극단값 하나에 크게 안 흔들려야 한다
    assert abs(threshold_with_outlier - threshold_without_outlier) < 1.0, (
        "이상치 하나에 threshold가 너무 크게 흔들림 - MAD가 아니라 표준편차처럼 동작하는 듯"
    )
    # 그리고 이상치 자체는 threshold를 넘어야 한다
    assert 50.0 > threshold_with_outlier


def test_mad_threshold_empty_input_returns_zero():
    assert compute_mad_threshold([]) == 0.0


def _make_frame(image_id: str) -> Frame:
    info = ImageInfo(image_id=image_id, path=f"/fake/{image_id}.jpg", width=100, height=100)
    det = DetectionResult(image_id=image_id, success=True, num_corners=10)
    return Frame(image_info=info, detection=det, status=FrameStatus.DETECTED)


def test_apply_outlier_removal_does_not_delete_file_only_disables():
    """설계 문서 9번 핵심 원칙: '파일 자체를 삭제하지 않는다. 메타데이터에서
    비활성화 처리한다.' Frame.image_info.path는 그대로 남아있고, status만
    바뀌어야 한다.
    """
    frame = _make_frame("img001")
    dataset = Dataset(frames=[frame])
    original_path = frame.image_info.path

    apply_outlier_removal(dataset, ["img001"], reason="high_reprojection_error")

    assert frame.status == FrameStatus.DISABLED_OUTLIER
    assert frame.image_info.path == original_path, "파일 경로가 그대로 남아있어야 함 (파일 삭제 금지 원칙)"
    assert frame not in dataset.enabled_frames


def test_restore_frame_reverses_outlier_removal():
    frame = _make_frame("img002")
    dataset = Dataset(frames=[frame])
    apply_outlier_removal(dataset, ["img002"], reason="test")
    assert frame.status == FrameStatus.DISABLED_OUTLIER

    restore_frame(dataset, "img002")
    assert frame.status == FrameStatus.DETECTED
    assert frame in dataset.enabled_frames


# ---------------------------------------------------------------------------
# 설계 문서 16번 - "왜 제거됐는지 기록" (per-frame 상세 사유)
# ---------------------------------------------------------------------------

def test_apply_outlier_removal_uses_per_frame_reasons_when_given():
    f1, f2 = _make_frame("img001"), _make_frame("img002")
    dataset = Dataset(frames=[f1, f2])
    reasons = {
        "img001": "high_reprojection_error (auto): frame RMS=1.200px > threshold=0.800px",
        "img002": "high_reprojection_error (auto): frame RMS=2.500px > threshold=0.800px",
    }
    apply_outlier_removal(dataset, ["img001", "img002"], reasons=reasons)
    assert f1.disabled_reason == reasons["img001"]
    assert f2.disabled_reason == reasons["img002"]


def test_apply_outlier_removal_falls_back_to_generic_reason_when_missing():
    frame = _make_frame("img003")
    dataset = Dataset(frames=[frame])
    apply_outlier_removal(dataset, ["img003"], reason="fallback_reason", reasons={"other_id": "x"})
    assert frame.disabled_reason == "fallback_reason"


def test_recalibrate_with_outlier_pruning_records_detailed_reason(synthetic_dataset, camera_config):
    """recalibrate_with_outlier_pruning()이 실제로 프레임을 제거할 때, 그
    사유 문자열에 실제 오차값과 threshold가 숫자로 들어가야 한다."""
    import copy
    dataset = copy.deepcopy(synthetic_dataset)
    _, outlier_result = recalibrate_with_outlier_pruning(
        dataset, camera_config, CameraModelType.PINHOLE, max_iterations=2,
    )
    for frame_id in outlier_result.removed_frame_ids:
        frame = next(f for f in dataset.frames if f.image_info.image_id == frame_id)
        assert "threshold=" in frame.disabled_reason
        assert "RMS=" in frame.disabled_reason


# ---------------------------------------------------------------------------
# 설계 문서 17번 - Outlier 제거 전후 효과 측정 (RMS/P95/파라미터)
# ---------------------------------------------------------------------------

def test_outlier_result_before_after_populated(synthetic_dataset, camera_config):
    import copy
    dataset = copy.deepcopy(synthetic_dataset)
    _, outlier_result = recalibrate_with_outlier_pruning(
        dataset, camera_config, CameraModelType.PINHOLE, max_iterations=2,
    )
    assert outlier_result.rms_before is not None
    assert outlier_result.camera_matrix_before is not None
    assert outlier_result.camera_matrix_after is not None
    assert outlier_result.p95_before is not None


def test_format_outlier_before_after_includes_key_metrics(synthetic_dataset, camera_config):
    import copy
    dataset = copy.deepcopy(synthetic_dataset)
    _, outlier_result = recalibrate_with_outlier_pruning(
        dataset, camera_config, CameraModelType.PINHOLE, max_iterations=3, k=0.5,
    )
    text = format_outlier_before_after(outlier_result)
    if outlier_result.removed_frame_ids:
        for label in ("RMSE", "P95", "fx", "fy"):
            assert label in text
    else:
        assert "없습니다" in text


def test_format_outlier_before_after_handles_no_removal():
    from calibration.types import OutlierResult
    text = format_outlier_before_after(OutlierResult(threshold_used=0.0))
    assert "없습니다" in text


# ---------------------------------------------------------------------------
# 설계 문서 16번 - Corner-level Outlier Detection
# ---------------------------------------------------------------------------

W, H = 1920, 1080
TRUE_K = np.array([[1000.0, 0, W / 2], [0, 1000.0, H / 2], [0, 0, 1]])
ZERO_D = np.zeros(5)


def _grid_frame(image_id: str, rvec, tvec, corrupt_index: int | None = None, jitter_std: float = 0.08) -> Frame:
    objp = np.zeros((6 * 5, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:5].T.reshape(-1, 2) * 0.04
    projected, _ = cv2.projectPoints(objp, rvec, tvec, TRUE_K, ZERO_D)
    projected = projected.reshape(-1, 2)
    if jitter_std > 0:
        # 완벽한(노이즈 0) 합성 좌표는 median/MAD가 거의 0으로 붕괴해 아주 작은
        # 수치 오차조차 "이상치"로 잡히는 병적인 경우를 만든다 - 실제 검출
        # 데이터에는 항상 약간의 픽셀 단위 잡음이 있으므로, 현실적인 정도의
        # 잡음(기본 0.08px)을 더해 통계 검정이 정상적인 조건에서 동작하게 한다.
        rng = np.random.default_rng(abs(hash(image_id)) % (2**32))
        projected = projected + rng.normal(0, jitter_std, size=projected.shape)
    if corrupt_index is not None:
        projected[corrupt_index] += np.array([40.0, -35.0])  # 명백한 이상치 코너 하나
    info = ImageInfo(image_id=image_id, path="-", width=W, height=H)
    det = DetectionResult(
        image_id=image_id, success=True,
        corners=projected.reshape(-1, 1, 2).astype(np.float32),
        object_points=objp.reshape(-1, 1, 3), num_corners=objp.shape[0],
    )
    return Frame(image_info=info, detection=det, status=FrameStatus.DETECTED)


def _build_corner_outlier_dataset(n_good: int = 12) -> Dataset:
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n_good):
        rvec = (rng.random(3) - 0.5) * 0.4
        tvec = np.array([(rng.random() - 0.5) * 0.2, (rng.random() - 0.5) * 0.2, 0.6 + rng.random() * 0.2])
        frames.append(_grid_frame(f"good_{i:02d}", rvec, tvec))
    # 코너 하나가 명백히 튀는 프레임을 하나 추가
    frames.append(_grid_frame("bad_corner", np.zeros(3), np.array([0.0, 0.0, 0.7]), corrupt_index=5))
    return Dataset(frames=frames)


class TestComputePerCornerErrors:
    def test_zero_for_perfect_projection(self):
        rvec, tvec = np.zeros(3), np.array([0.0, 0.0, 0.7])
        frame = _grid_frame("f0", rvec, tvec, jitter_std=0.0)
        errors = compute_per_corner_errors(frame, rvec, tvec, TRUE_K, ZERO_D, CameraModelType.PINHOLE)
        assert np.allclose(errors, 0.0, atol=1e-3)

    def test_detects_corrupted_corner(self):
        rvec, tvec = np.zeros(3), np.array([0.0, 0.0, 0.7])
        frame = _grid_frame("f1", rvec, tvec, corrupt_index=5, jitter_std=0.0)
        errors = compute_per_corner_errors(frame, rvec, tvec, TRUE_K, ZERO_D, CameraModelType.PINHOLE)
        assert errors[5] > 30.0
        assert np.all(np.delete(errors, 5) < 1.0)

    def test_empty_for_missing_detection(self):
        info = ImageInfo(image_id="empty", path="-", width=W, height=H)
        frame = Frame(image_info=info, status=FrameStatus.DETECTED)
        errors = compute_per_corner_errors(frame, np.zeros(3), np.zeros(3), TRUE_K, ZERO_D, CameraModelType.PINHOLE)
        assert errors.size == 0


class TestRecommendCornerOutliers:
    def test_flags_only_the_corrupted_corner(self):
        rvec, tvec = np.zeros(3), np.array([0.0, 0.0, 0.7])
        good = _grid_frame("good", rvec, tvec, jitter_std=0.0)
        bad = _grid_frame("bad", rvec, tvec, corrupt_index=5, jitter_std=0.0)
        rvecs = [rvec, rvec]
        tvecs = [tvec, tvec]
        candidates, threshold = recommend_corner_outliers(
            [good, bad], rvecs, tvecs, TRUE_K, ZERO_D, CameraModelType.PINHOLE
        )
        assert "bad" in candidates
        assert candidates["bad"] == [5]
        assert "good" not in candidates

    def test_no_outliers_when_all_clean(self):
        rvec, tvec = np.zeros(3), np.array([0.0, 0.0, 0.7])
        frames = [_grid_frame(f"f{i}", rvec, tvec, jitter_std=0.0) for i in range(3)]
        candidates, _ = recommend_corner_outliers(
            frames, [rvec] * 3, [tvec] * 3, TRUE_K, ZERO_D, CameraModelType.PINHOLE
        )
        assert candidates == {}

    def test_empty_input_returns_empty(self):
        candidates, threshold = recommend_corner_outliers([], [], [], TRUE_K, ZERO_D, CameraModelType.PINHOLE)
        assert candidates == {}
        assert threshold == 0.0


class TestApplyCornerOutlierRemoval:
    def test_updates_excluded_indices(self):
        frame = _make_frame("img001")
        dataset = Dataset(frames=[frame])
        apply_corner_outlier_removal(dataset, {"img001": [2, 5]})
        assert frame.detection.excluded_corner_indices == [2, 5]

    def test_is_union_not_overwrite(self):
        frame = _make_frame("img001")
        dataset = Dataset(frames=[frame])
        apply_corner_outlier_removal(dataset, {"img001": [2, 5]})
        apply_corner_outlier_removal(dataset, {"img001": [5, 7]})
        assert frame.detection.excluded_corner_indices == [2, 5, 7]

    def test_ignores_unknown_frame_id(self):
        frame = _make_frame("img001")
        dataset = Dataset(frames=[frame])
        apply_corner_outlier_removal(dataset, {"nonexistent": [0]})
        assert frame.detection.excluded_corner_indices == []

    def test_frame_status_untouched(self):
        """corner-level 제거는 프레임 status를 바꾸지 않는다 - 프레임 단위
        제거(apply_outlier_removal)와의 핵심 차이."""
        frame = _make_frame("img001")
        dataset = Dataset(frames=[frame])
        apply_corner_outlier_removal(dataset, {"img001": [0]})
        assert frame.status == FrameStatus.DETECTED
        assert frame in dataset.enabled_frames


class TestCollectCalibrationInputsRespectsExclusion:
    def test_excluded_corners_are_removed_from_calibration_input(self):
        rvec, tvec = np.zeros(3), np.array([0.0, 0.0, 0.7])
        frame = _grid_frame("f0", rvec, tvec)
        frame.detection.excluded_corner_indices = [0, 1, 2]
        dataset = Dataset(frames=[frame])
        usable_frames, object_points, image_points = collect_calibration_inputs(dataset)
        assert len(usable_frames) == 1
        assert object_points[0].shape[0] == frame.detection.num_corners - 3
        assert image_points[0].shape[0] == frame.detection.num_corners - 3

    def test_frame_dropped_entirely_if_too_few_corners_remain(self):
        rvec, tvec = np.zeros(3), np.array([0.0, 0.0, 0.7])
        frame = _grid_frame("f0", rvec, tvec)
        # 30개 코너 중 27개를 빼면 3개만 남음 (MIN_CORNERS_PER_FRAME=4보다 적음)
        frame.detection.excluded_corner_indices = list(range(27))
        dataset = Dataset(frames=[frame])
        usable_frames, _, _ = collect_calibration_inputs(dataset)
        assert usable_frames == []


class TestRecalibrateWithCornerOutlierPruning:
    def test_removes_the_bad_corner_not_the_whole_frame(self):
        dataset = _build_corner_outlier_dataset()
        result, corner_outlier_result = recalibrate_with_corner_outlier_pruning(
            dataset, camera_config=_camera_config_for(dataset), model=CameraModelType.PINHOLE,
            max_iterations=3,
        )
        assert result.success
        assert "bad_corner" in corner_outlier_result.removed_corners
        bad_frame = next(f for f in dataset.frames if f.image_info.image_id == "bad_corner")
        # 프레임은 여전히 활성 상태여야 한다 - 코너 하나만 빠졌지 프레임 전체가
        # 빠진 게 아니므로 (핵심 검증 포인트).
        assert bad_frame.status != FrameStatus.DISABLED_OUTLIER
        assert 5 in bad_frame.detection.excluded_corner_indices

    def test_before_after_fields_populated_when_corners_removed(self):
        """설계 문서 17번 - corner-level 제거에도 RMSE/P95/parameter
        before-after가 프레임 단위 버전과 동일하게 채워져야 한다."""
        dataset = _build_corner_outlier_dataset()
        _, corner_outlier_result = recalibrate_with_corner_outlier_pruning(
            dataset, camera_config=_camera_config_for(dataset), model=CameraModelType.PINHOLE,
            max_iterations=3,
        )
        assert corner_outlier_result.removed_corners  # 전제조건: 실제로 뭔가 제거됨
        assert corner_outlier_result.rms_before is not None
        assert corner_outlier_result.rms_after is not None
        assert corner_outlier_result.p95_before is not None
        assert corner_outlier_result.p95_after is not None
        assert corner_outlier_result.camera_matrix_before is not None
        assert corner_outlier_result.camera_matrix_after is not None
        assert corner_outlier_result.distortion_before is not None
        assert corner_outlier_result.distortion_after is not None
        # 이상치 하나를 제거했으니 RMS/P95는 개선(감소)돼야 한다
        assert corner_outlier_result.rms_after <= corner_outlier_result.rms_before
        assert corner_outlier_result.p95_after <= corner_outlier_result.p95_before

    def test_format_corner_outlier_before_after_includes_key_metrics(self):
        dataset = _build_corner_outlier_dataset()
        _, corner_outlier_result = recalibrate_with_corner_outlier_pruning(
            dataset, camera_config=_camera_config_for(dataset), model=CameraModelType.PINHOLE,
            max_iterations=3,
        )
        text = format_corner_outlier_before_after(corner_outlier_result)
        for label in ("RMSE", "P95", "fx", "fy"):
            assert label in text

    def test_format_corner_outlier_before_after_handles_no_removal(self):
        from calibration.types import CornerOutlierResult
        text = format_corner_outlier_before_after(CornerOutlierResult(threshold_used=0.0))
        assert "없습니다" in text

    def test_minimal_change_when_data_is_clean(self):
        """MAD 기반 threshold는 통계적 절차라, 완벽히 이상적인(노이즈가 거의
        0에 수렴하는) 합성 데이터에서는 median/MAD 자체가 극도로 작아져
        아주 미세한 수치 오차조차 threshold를 넘을 수 있다 - 그래서 "정확히
        0개"가 아니라 "극히 일부만" 제거되는지를 확인한다(전체 코너 수 대비
        비율로 - 실제 데이터에서 기대하는 동작과 같은 성격의 검증).
        """
        rvec, tvec = np.zeros(3), np.array([0.0, 0.0, 0.7])
        frames = [_grid_frame(f"f{i}", rvec + i * 0.01, tvec) for i in range(10)]
        dataset = Dataset(frames=frames)
        _, corner_outlier_result = recalibrate_with_corner_outlier_pruning(
            dataset, camera_config=_camera_config_for(dataset), model=CameraModelType.PINHOLE,
        )
        total_corners = sum(f.detection.num_corners for f in frames)
        assert corner_outlier_result.total_corners_removed < total_corners * 0.1

    def test_format_corner_outlier_summary_no_crash(self):
        dataset = _build_corner_outlier_dataset()
        _, corner_outlier_result = recalibrate_with_corner_outlier_pruning(
            dataset, camera_config=_camera_config_for(dataset), model=CameraModelType.PINHOLE,
        )
        text = format_corner_outlier_summary(corner_outlier_result)
        assert len(text) > 0

    def test_format_handles_no_outliers(self):
        from calibration.types import CornerOutlierResult
        text = format_corner_outlier_summary(CornerOutlierResult(threshold_used=0.0))
        assert "없습니다" in text


def _camera_config_for(dataset: Dataset):
    from calibration.types import CameraConfig
    f = dataset.frames[0]
    return CameraConfig(width=f.image_info.width, height=f.image_info.height)
