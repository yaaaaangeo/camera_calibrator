"""
tests/test_spatial_error_map.py
======================================

설계 문서 13번 - Spatial Error Map 강화 ("residual direction(X/Y 방향)" heatmap).
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.spatial_error_map import (
    _direction_arrow,
    compute_spatial_error_map,
    format_spatial_error_map,
    has_systematic_direction_bias,
)
from calibration.types import (
    CameraModelType,
    DetectionResult,
    Frame,
    FrameStatus,
    ImageInfo,
    SpatialErrorCell,
    SpatialErrorMap,
)

W, H = 1920, 1080
TRUE_K = np.array([[1000.0, 0, W / 2], [0, 1000.0, H / 2], [0, 0, 1]])
TRUE_D = np.array([-0.35, 0.15, 0.0, 0.0, 0.0])


def _build_synthetic_pinhole_mismatch_frames():
    """test_radial_profile.py와 동일한 패턴: 왜곡을 무시한 모델로 캘리브레이션한
    프레임/결과를 만든다 - 외곽에서 체계적으로 바깥쪽을 향하는 재투영 오차가
    생겨야 하므로, 방향 감지 로직을 검증하기에 적합한 시나리오다.
    """
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


class TestComputeSpatialErrorMap:
    def test_returns_full_grid_even_with_no_data(self):
        smap = compute_spatial_error_map([], [], [], TRUE_K, TRUE_D, (W, H), CameraModelType.PINHOLE)
        assert len(smap.cells) == 16  # 기본 4x4
        assert all(c.num_points == 0 for c in smap.cells)

    def test_custom_grid_size(self):
        smap = compute_spatial_error_map(
            [], [], [], TRUE_K, TRUE_D, (W, H), CameraModelType.PINHOLE, rows=2, cols=2
        )
        assert len(smap.cells) == 4

    def test_cells_have_rms_and_direction_when_data_present(self):
        frames, K_est, D_est, rvecs_est, tvecs_est = _build_synthetic_pinhole_mismatch_frames()
        smap = compute_spatial_error_map(
            frames, rvecs_est, tvecs_est, K_est, D_est, (W, H), CameraModelType.PINHOLE
        )
        populated = [c for c in smap.cells if c.num_points > 0]
        assert len(populated) > 0
        for c in populated:
            assert c.rms is not None and c.rms >= 0
            assert c.p95 is not None
            assert c.mean_dx is not None and c.mean_dy is not None
            assert c.direction_deg is not None

    def test_fisheye_path_no_crash(self):
        frames, K_est, D_est, rvecs_est, tvecs_est = _build_synthetic_pinhole_mismatch_frames()
        D4 = np.zeros((4, 1))
        smap = compute_spatial_error_map(
            frames[:5],
            [r.reshape(3, 1).astype(np.float64) for r in rvecs_est[:5]],
            [t.reshape(3, 1).astype(np.float64) for t in tvecs_est[:5]],
            K_est.astype(np.float64), D4, (W, H), CameraModelType.FISHEYE,
        )
        assert len(smap.cells) == 16

    def test_points_bucketed_into_correct_cell(self):
        """검출 좌표가 이미지 좌상단 사분면에만 있으면, 그 사분면에 해당하는
        셀(2x2 그리드에서 row=0,col=0)만 포인트를 가져야 한다."""
        objp = np.array([[0, 0, 0]], dtype=np.float32)
        detected = np.array([[100.0, 100.0]], dtype=np.float32)
        info = ImageInfo(image_id="f0", path="-", width=W, height=H)
        det = DetectionResult(
            image_id="f0", success=True,
            corners=detected.reshape(-1, 1, 2), object_points=objp.reshape(-1, 1, 3), num_corners=1,
        )
        frame = Frame(image_info=info, detection=det, status=FrameStatus.DETECTED)

        # cx=cy=0, fx=fy=1로 두면 object_point(0,0,0)이 tvec 위치 그대로 투영된다.
        K = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1]])
        D = np.zeros(5)
        rvec = np.zeros(3)
        tvec = np.array([100.0, 100.0, 1.0])  # 투영 결과가 (100,100)에 오도록

        smap = compute_spatial_error_map(
            [frame], [rvec], [tvec], K, D, (W, H), CameraModelType.PINHOLE, rows=2, cols=2
        )
        top_left = next(c for c in smap.cells if c.row == 0 and c.col == 0)
        other_cells = [c for c in smap.cells if not (c.row == 0 and c.col == 0)]
        assert top_left.num_points == 1
        assert all(c.num_points == 0 for c in other_cells)


class TestDirectionArrow:
    def test_zero_degrees_points_right(self):
        assert _direction_arrow(0.0) == "\u2192"

    def test_ninety_degrees_points_down(self):
        assert _direction_arrow(90.0) == "\u2193"

    def test_wraps_around_360(self):
        assert _direction_arrow(0.0) == _direction_arrow(360.0)

    def test_negative_angle_points_left_ish(self):
        assert _direction_arrow(180.0) == "\u2190"


class TestSystematicBiasDetection:
    def test_consistent_direction_across_cells_is_detected(self):
        cells = [
            SpatialErrorCell(row=r, col=c, num_points=10, rms=0.5, p95=0.8, mean_dx=1.0, mean_dy=0.1)
            for r in range(2) for c in range(2)
        ]
        smap = SpatialErrorMap(cells=cells, rows=2, cols=2)
        assert has_systematic_direction_bias(smap) is True

    def test_random_directions_are_not_flagged(self):
        directions = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
        cells = [
            SpatialErrorCell(row=r, col=c, num_points=10, rms=0.5, p95=0.8, mean_dx=dx, mean_dy=dy)
            for (r, c), (dx, dy) in zip([(0, 0), (0, 1), (1, 0), (1, 1)], directions)
        ]
        smap = SpatialErrorMap(cells=cells, rows=2, cols=2)
        assert has_systematic_direction_bias(smap) is False

    def test_too_few_populated_cells_returns_false(self):
        cells = [SpatialErrorCell(row=0, col=0, num_points=10, mean_dx=1.0, mean_dy=0.0)]
        smap = SpatialErrorMap(cells=cells, rows=2, cols=2)
        assert has_systematic_direction_bias(smap, min_cells=4) is False

    def test_empty_map_returns_false(self):
        assert has_systematic_direction_bias(SpatialErrorMap(cells=[], rows=4, cols=4)) is False


class TestFormatting:
    def test_format_handles_empty(self):
        text = format_spatial_error_map(SpatialErrorMap(cells=[]))
        assert "없습니다" in text

    def test_format_includes_grid_and_explanation(self):
        frames, K_est, D_est, rvecs_est, tvecs_est = _build_synthetic_pinhole_mismatch_frames()
        smap = compute_spatial_error_map(
            frames, rvecs_est, tvecs_est, K_est, D_est, (W, H), CameraModelType.PINHOLE
        )
        text = format_spatial_error_map(smap)
        assert "Spatial Error Map" in text
        assert "카메라 모델" in text

    def test_format_marks_empty_cells_as_na(self):
        smap = compute_spatial_error_map([], [], [], TRUE_K, TRUE_D, (W, H), CameraModelType.PINHOLE)
        text = format_spatial_error_map(smap)
        assert "N/A" in text
