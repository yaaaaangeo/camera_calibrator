"""
evaluation/holdout_consistency.py

M3. Hold-out Consistency (see evaluation_metric_spec.md v0.4).

Measures whether a FIXED, existing T_CL performs consistently across
different contiguous time blocks of the dataset -- i.e. whether the
calibration is generalizing across the whole sequence rather than being
"accidentally okay" only in the scene/time window it happens to fit.

Pipeline:
  1. Split the synced dataset into N contiguous time blocks
     (EvaluationDataset.time_blocks -- no random shuffling, per spec).
  2. For each block, run M2 (edge_alignment) independently per frame,
     pool all edge-point errors across the block's frames, and compute one
     aggregate M2 result for that block.
  3. Collect each block's mean_px into a distribution across blocks; compute
     Mean/STD/range across blocks.
  4. Classify STD against the sensor-relative floor(Z), using the STD
     multiplier scheme (1x / 3x, per spec) -- NOT the M2 2x/5x scheme,
     since this is measuring spread, not per-point offset.

Failure conditions (per spec):
  - fewer than 3 valid blocks (statistically meaningless)
  - a block with fewer frames than min_frames_per_block is excluded
    (with a warning) rather than silently included
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from evaluation.edge_alignment import evaluate_edge_alignment, EdgeAlignmentResult
from input.camera import CameraModel
from input.dataset import EvaluationDataset, SyncedFrame
from quality.noise_floor import (
    LidarSensorSpecForFloor,
    resolve_floor_inputs,
    compute_floor,
    classify,
    STD_GOOD_MULTIPLIER,
    STD_WARNING_MULTIPLIER,
    M2_GOOD_MULTIPLIER,
    M2_WARNING_MULTIPLIER,
)


MIN_VALID_BLOCKS = 3


@dataclass
class BlockResult:
    block_index: int
    frame_indices: list[int]
    num_frames_total: int
    num_frames_valid: int   # frames where per-frame M2 did not FAIL
    num_frames_failed: int
    mean_px: float
    median_px: float
    p95_px: float
    num_edge_points: int
    representative_depth_m: float
    floor_px: float
    classification: str    # GOOD | WARNING | BAD | EXCLUDED | FAIL
    warnings: list[str] = field(default_factory=list)


@dataclass
class HoldoutConsistencyResult:
    block_results: list[BlockResult]
    num_valid_blocks: int
    block_means_px: list[float]
    mean_across_blocks_px: float
    std_across_blocks_px: float
    range_px: float
    floor_px: float
    classification: str    # GOOD | WARNING | BAD | FAIL
    warnings: list[str] = field(default_factory=list)


def _evaluate_block(
    block_index: int,
    frames: list[SyncedFrame],
    T_CL: np.ndarray,
    camera: CameraModel,
    lidar_spec: LidarSensorSpecForFloor,
    min_frames_per_block: int,
    edge_alignment_kwargs: dict,
) -> BlockResult:
    frame_indices = [f.index for f in frames]

    if len(frames) < min_frames_per_block:
        return BlockResult(
            block_index=block_index,
            frame_indices=frame_indices,
            num_frames_total=len(frames),
            num_frames_valid=0,
            num_frames_failed=0,
            mean_px=float("nan"), median_px=float("nan"), p95_px=float("nan"),
            num_edge_points=0, representative_depth_m=float("nan"), floor_px=float("nan"),
            classification="EXCLUDED",
            warnings=[
                f"Block {block_index} has {len(frames)} frames, below "
                f"min_frames_per_block={min_frames_per_block}; excluded from aggregation."
            ],
        )

    per_frame_results: list[EdgeAlignmentResult] = []
    for sf in frames:
        image = sf.camera_frame.load()
        points = sf.lidar_frame.load()
        result = evaluate_edge_alignment(
            image=image, points_lidar=points, T_CL=T_CL,
            camera=camera, lidar_spec=lidar_spec, **edge_alignment_kwargs,
        )
        per_frame_results.append(result)

    valid_results = [r for r in per_frame_results if r.classification != "FAIL"]
    num_failed = len(per_frame_results) - len(valid_results)

    if not valid_results:
        return BlockResult(
            block_index=block_index,
            frame_indices=frame_indices,
            num_frames_total=len(frames),
            num_frames_valid=0,
            num_frames_failed=num_failed,
            mean_px=float("nan"), median_px=float("nan"), p95_px=float("nan"),
            num_edge_points=0, representative_depth_m=float("nan"), floor_px=float("nan"),
            classification="FAIL",
            warnings=[f"Block {block_index}: all {len(frames)} frames failed M2 evaluation."],
        )

    pooled_errors = np.concatenate([r.edge_point_errors_px for r in valid_results])
    pooled_depth = float(np.median([r.representative_depth_m for r in valid_results]))
    num_edge_points = int(sum(r.num_edge_points for r in valid_results))

    floor_inputs = resolve_floor_inputs(
        fx_px=camera.intrinsics.fx, T_CL=T_CL, lidar_spec=lidar_spec,
        edge_localization_floor_px=camera.edge_localization_floor_px,
    )
    floor_px = compute_floor(floor_inputs, pooled_depth)

    mean_px = float(np.mean(pooled_errors))
    block_classification = classify(mean_px, floor_px, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER)

    warnings = list(floor_inputs.fallback_warnings)
    if num_failed > 0:
        warnings.append(f"Block {block_index}: {num_failed}/{len(frames)} frames failed M2 and were excluded from pooling.")

    return BlockResult(
        block_index=block_index,
        frame_indices=frame_indices,
        num_frames_total=len(frames),
        num_frames_valid=len(valid_results),
        num_frames_failed=num_failed,
        mean_px=mean_px,
        median_px=float(np.median(pooled_errors)),
        p95_px=float(np.percentile(pooled_errors, 95)),
        num_edge_points=num_edge_points,
        representative_depth_m=pooled_depth,
        floor_px=floor_px,
        classification=block_classification,
        warnings=warnings,
    )


def evaluate_holdout_consistency(
    dataset: EvaluationDataset,
    lidar_spec: LidarSensorSpecForFloor,
    n_blocks: int = 4,
    min_frames_per_block: int = 30,
    edge_alignment_kwargs: Optional[dict] = None,
) -> HoldoutConsistencyResult:
    """
    Compute M3 Hold-out Consistency: split dataset.frames into n_blocks
    contiguous time blocks, run M2 independently (pooled) per block, and
    report the Mean/STD/range of block-level error across blocks.

    dataset.extrinsic.T_CL is treated as fixed and applied identically to
    every block -- this is the whole point of M3 (does the SAME calibration
    generalize, not "what's the best T for each block").
    """
    edge_alignment_kwargs = edge_alignment_kwargs or {}
    warnings: list[str] = []

    raw_blocks = dataset.time_blocks(n_blocks)

    block_results = [
        _evaluate_block(
            block_index=i, frames=block, T_CL=dataset.extrinsic.T_CL,
            camera=dataset.camera, lidar_spec=lidar_spec,
            min_frames_per_block=min_frames_per_block,
            edge_alignment_kwargs=edge_alignment_kwargs,
        )
        for i, block in enumerate(raw_blocks)
    ]

    for b in block_results:
        warnings.extend(f"[block {b.block_index}] {w}" for w in b.warnings)

    valid_blocks = [b for b in block_results if b.classification in ("GOOD", "WARNING", "BAD")]

    if len(valid_blocks) < MIN_VALID_BLOCKS:
        warnings.append(
            f"Only {len(valid_blocks)} valid block(s) (need >= {MIN_VALID_BLOCKS}) -- "
            f"Hold-out Consistency is not statistically meaningful with this few blocks. "
            f"Consider a longer dataset, fewer n_blocks, or a smaller min_frames_per_block."
        )
        return HoldoutConsistencyResult(
            block_results=block_results,
            num_valid_blocks=len(valid_blocks),
            block_means_px=[b.mean_px for b in valid_blocks],
            mean_across_blocks_px=float("nan"),
            std_across_blocks_px=float("nan"),
            range_px=float("nan"),
            floor_px=float("nan"),
            classification="FAIL",
            warnings=warnings,
        )

    block_means = [b.mean_px for b in valid_blocks]
    mean_across = float(np.mean(block_means))
    std_across = float(np.std(block_means, ddof=1))  # sample std across blocks
    range_across = float(np.max(block_means) - np.min(block_means))

    # Representative floor for STD classification: median of the valid
    # blocks' own floor(Z) values (each already reflects that block's
    # representative depth).
    floor_px = float(np.median([b.floor_px for b in valid_blocks]))

    overall_classification = classify(std_across, floor_px, STD_GOOD_MULTIPLIER, STD_WARNING_MULTIPLIER)

    return HoldoutConsistencyResult(
        block_results=block_results,
        num_valid_blocks=len(valid_blocks),
        block_means_px=block_means,
        mean_across_blocks_px=mean_across,
        std_across_blocks_px=std_across,
        range_px=range_across,
        floor_px=floor_px,
        classification=overall_classification,
        warnings=warnings,
    )
