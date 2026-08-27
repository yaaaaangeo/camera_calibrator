"""
tests/test_radial_profile.py
=================================

설계 문서 4번 - Edge Error Map (Radial Error Profile). 왜곡을 무시한 Pinhole
모델로 캘리브레이션하면 렌즈 외곽에서 오차가 커져야 한다는 물리적으로
당연한 경향을, 실제 cv2.calibrateCamera로 검증한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.radial_profile import (
    compute_radial_error_bands,
    compute_radial_error_profile,
    format_radial_bands,
    format_radial_curve,
    format_radial_profile,
    radial_error_curve,
)
from calibration.types import CameraModelType, DetectionResult, Frame, FrameStatus, ImageInfo

W, H = 1920, 1080
TRUE_K = np.array([[1000.0, 0, W / 2], [0, 1000.0, H / 2], [0, 0, 1]])
TRUE_D = np.array([-0.35, 0.15, 0.0, 0.0, 0.0])


def _build_synthetic_pinhole_mismatch_frames():
    """실제로는 왜곡이 있는데, 그걸 무시하고 캘리브레이션한 프레임/결과를 만든다."""
    objp = np.zeros((8 * 6, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * 0.04

    rng = np.random.default_rng(0)
    frames = []
    for i in range(15):
        rvec = (rng.random(3) - 0.5) * 0.6
        tvec = np.array([(rng.random() - 0.5) * 0.3, (rng.random() - 0.5) * 0.3, 0.5 + rng.random() * 0.3])
        projected, _ = cv2.projectPoints(objp, rvec, tvec, TRUE_K, TRUE_D)
        projected = projected.reshape(-1, 2)
        if np.any(projected < 0) or np.any(projected[:, 0] > W) or np.any(projected[:, 1] > H):
            continue
        info = ImageInfo(image_id=f"synth_{i:02d}", path="-", width=W, height=H)
        det = DetectionResult(
            image_id=f"synth_{i:02d}", success=True,
            corners=projected.reshape(-1, 1, 2).astype(np.float32),
            object_points=objp.reshape(-1, 1, 3), num_corners=objp.shape[0],
        )
        frames.append(Frame(image_info=info, detection=det, status=FrameStatus.DETECTED))

    assert len(frames) >= 8, "합성 뷰가 너무 적게 생성됨 - 파라미터 조정 필요"

    object_points_list = [f.detection.object_points for f in frames]
    image_points_list = [f.detection.corners for f in frames]
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    rms, K_est, D_est, rvecs_est, tvecs_est = cv2.calibrateCamera(
        object_points_list, image_points_list, (W, H), None, None, flags=flags
    )
    return frames, K_est, D_est, list(rvecs_est), list(tvecs_est)


def test_radial_profile_increases_toward_edge_under_model_mismatch():
    """왜곡을 무시한 모델은 중심보다 외곽에서 오차가 커야 한다 - 물리적으로
    당연한 방향성이므로, 이게 안 맞으면 계산 로직에 버그가 있다는 뜻이다.
    """
    frames, K_est, D_est, rvecs_est, tvecs_est = _build_synthetic_pinhole_mismatch_frames()

    profile = compute_radial_error_profile(
        frames, rvecs_est, tvecs_est, K_est, D_est, (W, H), CameraModelType.PINHOLE, num_bins=6
    )

    valid_bins = [b for b in profile.bins if b.mean_error is not None and b.num_points >= 3]
    assert len(valid_bins) >= 2, "비교할 수 있을 만큼 충분한 bin이 없음"

    assert valid_bins[-1].mean_error > valid_bins[0].mean_error, (
        "왜곡을 무시한 모델에서는 외곽 구간 평균 오차가 중심 구간보다 커야 함"
    )


def test_radial_profile_fisheye_path_no_crash():
    """cv2.fisheye.projectPoints 경로도 예외 없이 동작해야 한다."""
    frames, K_est, D_est, rvecs_est, tvecs_est = _build_synthetic_pinhole_mismatch_frames()
    D4 = np.zeros((4, 1))

    profile = compute_radial_error_profile(
        frames[:5],
        [r.reshape(3, 1).astype(np.float64) for r in rvecs_est[:5]],
        [t.reshape(3, 1).astype(np.float64) for t in tvecs_est[:5]],
        K_est.astype(np.float64), D4, (W, H), CameraModelType.FISHEYE, num_bins=4,
    )
    assert len(profile.bins) == 4


def test_radial_profile_empty_input_returns_empty():
    profile = compute_radial_error_profile([], [], [], TRUE_K, TRUE_D, (W, H), CameraModelType.PINHOLE)
    assert profile.bins == []


def test_format_radial_profile_no_crash():
    frames, K_est, D_est, rvecs_est, tvecs_est = _build_synthetic_pinhole_mismatch_frames()
    profile = compute_radial_error_profile(
        frames, rvecs_est, tvecs_est, K_est, D_est, (W, H), CameraModelType.PINHOLE, num_bins=6
    )
    text = format_radial_profile(profile)
    assert len(text) > 0


# ---------------------------------------------------------------------------
# 설계 문서 14번 - Radial Error 분석 강화 (Center~Corner 6단계 대역)
# ---------------------------------------------------------------------------

def test_bins_have_full_statistics():
    """compute_radial_error_profile()의 각 bin이 mean/median/rms/p95/max를
    전부 채워야 한다 - 이제 mean_error 하나만 계산하던 시절과 다르다."""
    frames, K_est, D_est, rvecs_est, tvecs_est = _build_synthetic_pinhole_mismatch_frames()
    profile = compute_radial_error_profile(
        frames, rvecs_est, tvecs_est, K_est, D_est, (W, H), CameraModelType.PINHOLE, num_bins=6
    )
    populated = [b for b in profile.bins if b.num_points > 0]
    assert populated
    for b in populated:
        assert b.mean_error is not None
        assert b.median_error is not None
        assert b.rms_error is not None
        assert b.p95_error is not None
        assert b.max_error is not None
        # RMS는 항상 mean 이상이어야 한다 (RMS >= |mean| 부등식, 오차가 모두 양수이므로)
        assert b.rms_error >= b.mean_error - 1e-9


def test_compute_radial_error_bands_has_six_named_bands():
    frames, K_est, D_est, rvecs_est, tvecs_est = _build_synthetic_pinhole_mismatch_frames()
    bands = compute_radial_error_bands(frames, rvecs_est, tvecs_est, K_est, D_est, (W, H), CameraModelType.PINHOLE)
    assert len(bands.bins) == 6
    assert [b.label for b in bands.bins] == ["Center", "Inner", "Middle", "Outer", "Edge", "Corner"]


def test_radial_error_bands_boundaries_are_contiguous_and_increasing():
    frames, K_est, D_est, rvecs_est, tvecs_est = _build_synthetic_pinhole_mismatch_frames()
    bands = compute_radial_error_bands(frames, rvecs_est, tvecs_est, K_est, D_est, (W, H), CameraModelType.PINHOLE)
    for prev, cur in zip(bands.bins, bands.bins[1:]):
        assert prev.radius_max == cur.radius_min
    assert bands.bins[0].radius_min == 0.0
    assert bands.bins[-1].radius_max == bands.max_radius


def test_radial_error_bands_empty_input_returns_empty():
    bands = compute_radial_error_bands([], [], [], TRUE_K, TRUE_D, (W, H), CameraModelType.PINHOLE)
    assert bands.bins == []


def test_format_radial_bands_includes_all_band_names():
    frames, K_est, D_est, rvecs_est, tvecs_est = _build_synthetic_pinhole_mismatch_frames()
    bands = compute_radial_error_bands(frames, rvecs_est, tvecs_est, K_est, D_est, (W, H), CameraModelType.PINHOLE)
    text = format_radial_bands(bands)
    for label in ("Center", "Inner", "Middle", "Outer", "Edge", "Corner", "Mean", "Median", "RMS", "P95", "Max"):
        assert label in text


def test_format_radial_bands_handles_empty():
    text = format_radial_bands(compute_radial_error_bands([], [], [], TRUE_K, TRUE_D, (W, H), CameraModelType.PINHOLE))
    assert "없습니다" in text


def test_radial_error_curve_returns_points_sorted_by_radius():
    frames, K_est, D_est, rvecs_est, tvecs_est = _build_synthetic_pinhole_mismatch_frames()
    profile = compute_radial_error_profile(
        frames, rvecs_est, tvecs_est, K_est, D_est, (W, H), CameraModelType.PINHOLE, num_bins=6
    )
    points = radial_error_curve(profile)
    assert points
    radii = [r for r, _ in points]
    assert radii == sorted(radii)


def test_radial_error_curve_supports_alternate_metrics():
    frames, K_est, D_est, rvecs_est, tvecs_est = _build_synthetic_pinhole_mismatch_frames()
    profile = compute_radial_error_profile(
        frames, rvecs_est, tvecs_est, K_est, D_est, (W, H), CameraModelType.PINHOLE, num_bins=6
    )
    p95_points = radial_error_curve(profile, metric="p95_error")
    mean_points = radial_error_curve(profile, metric="mean_error")
    assert len(p95_points) == len(mean_points)
    # P95는 정의상 mean보다 항상 크거나 같아야 한다
    for (_, p95v), (_, meanv) in zip(p95_points, mean_points):
        assert p95v >= meanv - 1e-9


def test_format_radial_curve_no_crash():
    frames, K_est, D_est, rvecs_est, tvecs_est = _build_synthetic_pinhole_mismatch_frames()
    profile = compute_radial_error_profile(
        frames, rvecs_est, tvecs_est, K_est, D_est, (W, H), CameraModelType.PINHOLE, num_bins=6
    )
    text = format_radial_curve(profile)
    assert "Radius -> Error Curve" in text


def test_format_radial_curve_handles_empty():
    from calibration.types import RadialErrorProfile
    text = format_radial_curve(RadialErrorProfile(bins=[], max_radius=0.0))
    assert "없습니다" in text
