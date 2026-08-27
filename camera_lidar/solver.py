"""
camera_calibrator.camera_lidar.solver
========================================

Closed-form rigid-transform (Kabsch/Umeyama SVD) solve between matched
LiDAR and camera circle-center point sets, plus residual statistics.

Independent implementation (see camera_lidar/types.py module docstring for
license/provenance notes) -- this is standard point-set registration
(Kabsch 1976 / Umeyama 1991), not a port of upstream's
pcl::registration::TransformationEstimationSVD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResidualStats:
    rmse: float
    mean: float
    median: float
    p95: float
    max: float
    per_point: np.ndarray


def solve_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve for (R, t) minimizing sum ||R @ source_i + t - target_i||^2 --
    the closed-form Kabsch/Umeyama SVD solve. source/target are (N, 3),
    N >= 3, with source[i] corresponding to target[i]."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError(
            f"source/target must both be (N>=3, 3) with matching shape, "
            f"got {source.shape}, {target.shape}"
        )

    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid

    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.diag([1.0, 1.0, d])
    R = Vt.T @ correction @ U.T
    t = tgt_centroid - R @ src_centroid
    return R, t


def compute_residuals(source: np.ndarray, target: np.ndarray, R: np.ndarray, t: np.ndarray) -> ResidualStats:
    """Per-point Euclidean error after applying (R, t) to source, plus
    RMSE/mean/median/p95/max summary stats (same units as the input points,
    i.e. meters for LiDAR/camera circle centers)."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    predicted = (R @ source.T).T + t
    errors = np.linalg.norm(predicted - target, axis=1)
    return ResidualStats(
        rmse=float(np.sqrt(np.mean(errors ** 2))),
        mean=float(np.mean(errors)),
        median=float(np.median(errors)),
        p95=float(np.percentile(errors, 95)),
        max=float(np.max(errors)),
        per_point=errors,
    )
