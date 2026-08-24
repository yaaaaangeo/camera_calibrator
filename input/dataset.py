"""
input/dataset.py

Ties camera + lidar + extrinsic together into a single EvaluationDataset,
handling timestamp synchronization (nearest-neighbor within a tolerance)
and providing the contiguous time-block split needed by M3 (Hold-out
Consistency), per the Input Loader Spec (v0.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from input.camera import CameraModel, CameraFrame
from input.lidar import LidarModel, LidarFrame
from input.extrinsic import ExtrinsicModel


@dataclass
class SyncConfig:
    max_time_diff_ms: float = 50.0
    drop_unmatched: bool = True


@dataclass
class SyncedFrame:
    index: int
    timestamp: float
    camera_frame: CameraFrame
    lidar_frame: LidarFrame
    time_diff_ms: float


@dataclass
class SyncStats:
    num_camera_frames: int
    num_lidar_frames: int
    num_matched: int
    num_camera_dropped: int
    num_lidar_dropped: int
    mean_time_diff_ms: float
    max_time_diff_ms: float


@dataclass
class EvaluationDataset:
    camera: CameraModel
    lidar: LidarModel
    extrinsic: ExtrinsicModel
    sync_config: SyncConfig
    frames: list[SyncedFrame] = field(default_factory=list)
    sync_stats: Optional[SyncStats] = None
    warnings: list[str] = field(default_factory=list)

    def time_blocks(self, n: int) -> list[list[SyncedFrame]]:
        """
        Split self.frames into n contiguous, roughly-equal time blocks, in
        frame order (frames are already time-sorted by construction).
        Used by M3 (Hold-out Consistency). Per spec: contiguous blocks only
        -- no random shuffling, to preserve temporal structure.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if not self.frames:
            return [[] for _ in range(n)]

        total = len(self.frames)
        # np.array_split gives contiguous, near-equal-size chunks (some may
        # differ by 1 frame when total isn't divisible by n) -- exactly the
        # "자동 균등 분할" behavior specified.
        indices = np.array_split(np.arange(total), n)
        return [[self.frames[i] for i in block] for block in indices]


def _nearest_neighbor_sync(
    camera_frames: list[CameraFrame],
    lidar_frames: list[LidarFrame],
    max_time_diff_ms: float,
) -> tuple[list[SyncedFrame], SyncStats]:
    """
    Greedy nearest-neighbor matching: for each camera frame (assumed the
    typically-higher-rate or reference sensor), find the closest lidar frame
    in time. A lidar frame can only be matched once (first-come, closest
    camera frame claims it) to avoid duplicate matches when camera rate >>
    lidar rate.
    """
    cam_ts = np.array([f.timestamp for f in camera_frames])
    lid_ts = np.array([f.timestamp for f in lidar_frames])

    order = np.argsort(cam_ts)
    lidar_used = np.zeros(len(lidar_frames), dtype=bool)

    matches: list[SyncedFrame] = []
    diffs_ms: list[float] = []

    for idx in order:
        c_ts = cam_ts[idx]
        # nearest available (unused) lidar frame
        candidate_diffs = np.abs(lid_ts - c_ts)
        candidate_diffs_masked = np.where(lidar_used, np.inf, candidate_diffs)
        if candidate_diffs_masked.size == 0 or np.all(np.isinf(candidate_diffs_masked)):
            continue
        best_j = int(np.argmin(candidate_diffs_masked))
        diff_ms = float(candidate_diffs_masked[best_j]) * 1000.0

        if diff_ms <= max_time_diff_ms:
            lidar_used[best_j] = True
            matches.append(SyncedFrame(
                index=len(matches),
                timestamp=float(c_ts),
                camera_frame=camera_frames[idx],
                lidar_frame=lidar_frames[best_j],
                time_diff_ms=diff_ms,
            ))
            diffs_ms.append(diff_ms)
        # else: no lidar frame close enough -> this camera frame is dropped

    # re-sort matches by timestamp (matching order above followed cam_ts
    # sort already, but re-assert + fix up `index` for clarity/determinism)
    matches.sort(key=lambda m: m.timestamp)
    for i, m in enumerate(matches):
        m.index = i

    stats = SyncStats(
        num_camera_frames=len(camera_frames),
        num_lidar_frames=len(lidar_frames),
        num_matched=len(matches),
        num_camera_dropped=len(camera_frames) - len(matches),
        num_lidar_dropped=len(lidar_frames) - len(matches),
        mean_time_diff_ms=float(np.mean(diffs_ms)) if diffs_ms else float("nan"),
        max_time_diff_ms=float(np.max(diffs_ms)) if diffs_ms else float("nan"),
    )
    return matches, stats


def build_dataset(
    camera: CameraModel,
    camera_frames: list[CameraFrame],
    lidar: LidarModel,
    lidar_frames: list[LidarFrame],
    extrinsic: ExtrinsicModel,
    sync_config: Optional[SyncConfig] = None,
) -> EvaluationDataset:
    """
    Construct an EvaluationDataset by time-synchronizing camera and lidar
    frame sequences. Drops unmatched frames (per SyncConfig.drop_unmatched;
    the only currently-implemented behavior is drop=True) and always
    reports how many were dropped -- per spec, dropped-frame counts are a
    data-quality signal and must never be silently discarded.
    """
    sync_config = sync_config or SyncConfig()
    if not sync_config.drop_unmatched:
        raise NotImplementedError(
            "drop_unmatched=False (e.g. interpolation-based sync) is not "
            "implemented in this pass."
        )

    matches, stats = _nearest_neighbor_sync(
        camera_frames, lidar_frames, sync_config.max_time_diff_ms
    )

    warnings: list[str] = []
    if stats.num_matched == 0:
        warnings.append(
            "No camera-lidar frame pairs matched within max_time_diff_ms="
            f"{sync_config.max_time_diff_ms}. Check that both sensors' "
            "timestamps are in the same clock/epoch, or increase max_time_diff_ms."
        )
    else:
        drop_ratio = 1.0 - stats.num_matched / max(stats.num_camera_frames, 1)
        if drop_ratio > 0.5:
            warnings.append(
                f"Over half of camera frames ({stats.num_camera_dropped}/"
                f"{stats.num_camera_frames}) had no matching lidar frame within "
                f"{sync_config.max_time_diff_ms}ms. Sync quality is poor; "
                f"evaluation results may not be representative."
            )

    return EvaluationDataset(
        camera=camera,
        lidar=lidar,
        extrinsic=extrinsic,
        sync_config=sync_config,
        frames=matches,
        sync_stats=stats,
        warnings=warnings,
    )
