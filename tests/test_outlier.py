"""
tests/test_outlier.py
==========================

설계 문서 9번 - 이상치 관리. MAD 기반 threshold 계산이 통계적으로 맞는지,
그리고 "파일을 삭제하지 않고 비활성화만 한다"는 원칙이 실제로 지켜지는지
확인한다.
"""

from __future__ import annotations

from calibration.outlier import compute_mad_threshold, apply_outlier_removal, restore_frame
from calibration.types import Dataset, DetectionResult, Frame, FrameStatus, ImageInfo


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
