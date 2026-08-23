"""
tests/test_quality_pose_distribution.py
==========================================

설계 문서 5번(분포 확장)/6번(Pose Diversity 강화) - X/Y위치, board 면적,
yaw/pitch/roll, 거리 분포 계산이 올바른지 확인한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.quality import (
    _estimate_rough_pose,
    compute_pose_distribution_stats,
    format_pose_distribution_stats,
)
from calibration.types import (
    CameraConfig,
    Dataset,
    DetectionResult,
    Frame,
    FrameStatus,
    ImageInfo,
)

W, H = 1920, 1080


def _flat_board_points(n_side: int = 6) -> np.ndarray:
    """z=0 평면 위 n_side x n_side 격자점 (보드 코너 근사), 미터 단위."""
    xs, ys = np.meshgrid(np.linspace(0, 0.2, n_side), np.linspace(0, 0.15, n_side))
    pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(n_side * n_side)], axis=1)
    return pts.astype(np.float32)


def _make_frame(image_id: str, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray) -> Frame | None:
    obj = _flat_board_points()
    proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, np.zeros(5))
    proj = proj.reshape(-1, 2)
    if np.any(proj < 0) or np.any(proj[:, 0] > W) or np.any(proj[:, 1] > H):
        return None
    cx, cy = float(proj[:, 0].mean()), float(proj[:, 1].mean())
    area_ratio = float(cv2.contourArea(cv2.convexHull(proj.astype(np.float32))) / (W * H))
    info = ImageInfo(image_id=image_id, path="-", width=W, height=H)
    det = DetectionResult(
        image_id=image_id, success=True,
        corners=proj.reshape(-1, 1, 2).astype(np.float32),
        object_points=obj.reshape(-1, 1, 3),
        ids=np.arange(len(obj), dtype=np.int32).reshape(-1, 1),
        num_corners=len(obj),
        board_area_ratio=area_ratio,
        board_center_px=(cx, cy),
        board_tilt_deg=0.0,
    )
    return Frame(image_info=info, detection=det, status=FrameStatus.DETECTED)


def _make_dataset(n_frames: int = 12, vary_pose: bool = True) -> Dataset:
    K = np.array([[1400.0, 0, W / 2], [0, 1400.0, H / 2], [0, 0, 1]])
    rng = np.random.default_rng(1)
    frames: list[Frame] = []
    attempts = 0
    while len(frames) < n_frames and attempts < n_frames * 20:
        attempts += 1
        if vary_pose:
            rvec = (rng.random(3) - 0.5) * 0.8
            tvec = np.array([
                (rng.random() - 0.5) * 0.3,
                (rng.random() - 0.5) * 0.3,
                0.6 + rng.random() * 0.6,
            ])
        else:
            rvec = np.array([0.05, 0.05, 0.0])
            tvec = np.array([0.0, 0.0, 0.8])
        frame = _make_frame(f"f{len(frames):02d}", rvec, tvec, K)
        if frame is not None:
            frames.append(frame)
    return Dataset(frames=frames)


class TestEstimateRoughPose:
    def test_returns_none_for_too_few_points(self):
        obj = _flat_board_points()[:2]
        img = obj[:, :2]
        assert _estimate_rough_pose(obj, img, (W, H)) is None

    def test_frontal_board_has_small_pitch_and_yaw(self):
        """정면(회전 거의 없음)으로 찍은 보드는 yaw/pitch가 0에 가까워야 한다."""
        K = np.array([[1400.0, 0, W / 2], [0, 1400.0, H / 2], [0, 0, 1]])
        obj = _flat_board_points()
        rvec = np.array([0.0, 0.0, 0.0])
        tvec = np.array([0.0, 0.0, 0.8])
        proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, np.zeros(5))
        pose = _estimate_rough_pose(obj, proj.reshape(-1, 2), (W, H))
        assert pose is not None
        yaw, pitch, roll, distance = pose
        assert abs(yaw) < 5.0
        assert abs(pitch) < 5.0
        assert distance > 0


class TestComputePoseDistributionStats:
    def test_varied_poses_have_high_coverage(self):
        dataset = _make_dataset(n_frames=15, vary_pose=True)
        camera_config = CameraConfig(width=W, height=H)
        stats = compute_pose_distribution_stats(dataset, camera_config)

        assert stats.x_position.sample_count > 0
        assert stats.yaw.sample_count > 0
        # 다양한 자세로 찍었으니 최소 하나 이상의 축은 coverage가 어느 정도 있어야 함
        assert any(
            s.coverage_score > 0.05
            for s in (stats.x_position, stats.y_position, stats.yaw, stats.pitch, stats.distance)
        )

    def test_identical_poses_have_zero_coverage(self):
        """전부 똑같은 자세로 찍으면(vary_pose=False) 대부분의 분산이 0에 가까워야 한다."""
        dataset = _make_dataset(n_frames=10, vary_pose=False)
        camera_config = CameraConfig(width=W, height=H)
        stats = compute_pose_distribution_stats(dataset, camera_config)
        assert stats.x_position.coverage_score < 0.2
        assert stats.y_position.coverage_score < 0.2

    def test_empty_dataset_returns_zero_stats(self):
        dataset = Dataset(frames=[])
        camera_config = CameraConfig(width=W, height=H)
        stats = compute_pose_distribution_stats(dataset, camera_config)
        assert stats.x_position.sample_count == 0
        assert stats.x_position.coverage_score == 0.0

    def test_format_does_not_crash_on_empty(self):
        stats = compute_pose_distribution_stats(Dataset(frames=[]), CameraConfig(width=W, height=H))
        text = format_pose_distribution_stats(stats)
        assert "N/A" in text

    def test_format_includes_all_axes(self):
        dataset = _make_dataset(n_frames=10, vary_pose=True)
        stats = compute_pose_distribution_stats(dataset, CameraConfig(width=W, height=H))
        text = format_pose_distribution_stats(stats)
        for label in ("X Position", "Y Position", "Board Area", "Yaw", "Pitch", "Roll", "Distance"):
            assert label in text
