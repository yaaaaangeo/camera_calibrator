"""
evaluation/multiframe_consistency.py

M4. Multi-frame Consistency (see evaluation_metric_spec.md v0.4).

Measures whether a FIXED, existing T_CL produces a stable per-frame error
across the whole frame sequence -- i.e. whether the calibration is reliable
frame-to-frame, or whether specific frames blow up (momentary misalignment,
outlier scenes, sync glitches, etc).

Unlike M3 (which pools frames into contiguous TIME BLOCKS to check
generalization across time windows), M4 evaluates EACH FRAME INDEPENDENTLY
and looks at the distribution of per-frame error directly -- this is what
catches a single bad frame that a block-level average would smooth over.

Pipeline:
  1. For every synced frame, run M2 (edge_alignment) independently:
       E_i = M2(T_fixed, Frame_i)
  2. Collect the valid per-frame mean_px values.
  3. Aggregate Mean/STD/P95/Max across frames.
  4. Flag outlier frames: mean_px_i > outlier_multiplier * median(all mean_px)
     (per spec: 5x median, independent of floor(Z)).
  5. Classify STD against the sensor-relative floor(Z) using the STD
     multiplier scheme (1x / 3x, same scheme M3 uses).

Failure conditions (per spec):
  - fewer than min_frames total frames (default 30, per spec's open item
    "통계적 유의성 위해 몇 프레임 이상 필요한지, 예: 최소 30")
  - fewer than 2 valid (non-FAIL) frames (can't compute STD)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from evaluation.edge_alignment import evaluate_edge_alignment, EdgeAlignmentResult
from input.dataset import EvaluationDataset
from quality.noise_floor import (
    LidarSensorSpecForFloor,
    classify,
    STD_GOOD_MULTIPLIER,
    STD_WARNING_MULTIPLIER,
)


DEFAULT_MIN_FRAMES = 30
DEFAULT_OUTLIER_MULTIPLIER = 5.0


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    mean_px: float
    num_edge_points: int
    representative_depth_m: float
    floor_px: float
    classification: str    # GOOD | WARNING | BAD | FAIL (per-frame M2 classification)
    is_outlier: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class MultiFrameConsistencyResult:
    frame_results: list[FrameResult]
    num_frames_total: int
    num_valid_frames: int
    num_failed_frames: int
    num_outlier_frames: int
    outlier_frame_indices: list[int]
    mean_across_frames_px: float
    std_across_frames_px: float
    median_across_frames_px: float
    p95_across_frames_px: float
    max_across_frames_px: float
    floor_px: float
    classification: str    # GOOD | WARNING | BAD | FAIL
    warnings: list[str] = field(default_factory=list)


def evaluate_multiframe_consistency(
    dataset: EvaluationDataset,
    lidar_spec: LidarSensorSpecForFloor,
    min_frames: int = DEFAULT_MIN_FRAMES,
    outlier_multiplier: float = DEFAULT_OUTLIER_MULTIPLIER,
    edge_alignment_kwargs: Optional[dict] = None,
) -> MultiFrameConsistencyResult:
    """
    Compute M4 Multi-frame Consistency: run M2 independently on every synced
    frame in the dataset (using the fixed dataset.extrinsic.T_CL), then
    report the Mean/STD/P95/Max of per-frame error and flag outlier frames.
    """
    edge_alignment_kwargs = edge_alignment_kwargs or {}
    warnings: list[str] = []

    total_frames = len(dataset.frames)
    if total_frames < min_frames:
        warnings.append(
            f"Dataset has {total_frames} frame(s), below min_frames={min_frames}. "
            f"Multi-frame Consistency requires enough frames for the STD/P95 "
            f"statistics to be meaningful."
        )
        return _fail_result(total_frames, warnings)

    frame_results: list[FrameResult] = []
    for sf in dataset.frames:
        image = sf.camera_frame.load()
        points = sf.lidar_frame.load()
        result: EdgeAlignmentResult = evaluate_edge_alignment(
            image=image, points_lidar=points, T_CL=dataset.extrinsic.T_CL,
            camera=dataset.camera, lidar_spec=lidar_spec, **edge_alignment_kwargs,
        )
        frame_results.append(FrameResult(
            frame_index=sf.index,
            timestamp=sf.timestamp,
            mean_px=result.mean_px,
            num_edge_points=result.num_edge_points,
            representative_depth_m=result.representative_depth_m,
            floor_px=result.floor_px,
            classification=result.classification,
            warnings=list(result.warnings),
        ))

    valid = [f for f in frame_results if f.classification != "FAIL"]
    failed = [f for f in frame_results if f.classification == "FAIL"]

    if len(failed) > 0:
        warnings.append(f"{len(failed)}/{total_frames} frame(s) failed M2 evaluation and were excluded.")

    if len(valid) < 2:
        warnings.append(
            f"Only {len(valid)} valid frame(s) after excluding failures; "
            f"need >= 2 to compute STD. Cannot assess consistency."
        )
        return _fail_result(total_frames, warnings, frame_results=frame_results,
                             num_failed=len(failed))

    frame_means = np.array([f.mean_px for f in valid])
    median_px = float(np.median(frame_means))

    # Flag outliers: per spec, 5x median, independent of floor(Z). Guard
    # against median ~= 0 (near-perfect calibration) where the multiplier
    # rule would flag essentially any nonzero noise as an outlier -- in
    # that degenerate case we fall back to an absolute epsilon instead.
    if median_px > 1e-6:
        outlier_threshold = outlier_multiplier * median_px
    else:
        outlier_threshold = max(outlier_multiplier * 0.1, 0.5)  # px, conservative floor
        warnings.append(
            "Median per-frame error is ~0px; outlier detection fell back to an "
            "absolute threshold since the multiplier rule is degenerate at median=0."
        )

    for f in valid:
        f.is_outlier = f.mean_px > outlier_threshold

    outliers = [f for f in valid if f.is_outlier]

    mean_across = float(np.mean(frame_means))
    std_across = float(np.std(frame_means, ddof=1))
    p95_across = float(np.percentile(frame_means, 95))
    max_across = float(np.max(frame_means))

    floor_px = float(np.median([f.floor_px for f in valid]))

    overall_classification = classify(std_across, floor_px, STD_GOOD_MULTIPLIER, STD_WARNING_MULTIPLIER)

    if outliers:
        warnings.append(
            f"{len(outliers)} outlier frame(s) detected (error > {outlier_multiplier}x median): "
            f"frame indices {[f.frame_index for f in outliers]}."
        )

    return MultiFrameConsistencyResult(
        frame_results=frame_results,
        num_frames_total=total_frames,
        num_valid_frames=len(valid),
        num_failed_frames=len(failed),
        num_outlier_frames=len(outliers),
        outlier_frame_indices=[f.frame_index for f in outliers],
        mean_across_frames_px=mean_across,
        std_across_frames_px=std_across,
        median_across_frames_px=median_px,
        p95_across_frames_px=p95_across,
        max_across_frames_px=max_across,
        floor_px=floor_px,
        classification=overall_classification,
        warnings=warnings,
    )


def _fail_result(
    total_frames: int,
    warnings: list[str],
    frame_results: Optional[list[FrameResult]] = None,
    num_failed: int = 0,
) -> MultiFrameConsistencyResult:
    return MultiFrameConsistencyResult(
        frame_results=frame_results or [],
        num_frames_total=total_frames,
        num_valid_frames=0,
        num_failed_frames=num_failed,
        num_outlier_frames=0,
        outlier_frame_indices=[],
        mean_across_frames_px=float("nan"),
        std_across_frames_px=float("nan"),
        median_across_frames_px=float("nan"),
        p95_across_frames_px=float("nan"),
        max_across_frames_px=float("nan"),
        floor_px=float("nan"),
        classification="FAIL",
        warnings=warnings,
    )
