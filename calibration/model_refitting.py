"""
camera_calibrator.calibration.model_refitting
=============================================

Approximate an OpenCV rational 8-coefficient pinhole model with a standard
OpenCV 5-coefficient pinhole model by fitting projections over the image FOV.

This is model refitting, not parameter truncation: D8[:5] is used only as a
baseline and optimizer initialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np


RefitMode = Literal["distortion_only", "full"]
WeightMode = Literal["uniform", "edge"]


@dataclass
class RefitErrorStats:
    rmse_px: float = 0.0
    mean_px: float = 0.0
    median_px: float = 0.0
    p95_px: float = 0.0
    p99_px: float = 0.0
    max_px: float = 0.0


@dataclass
class RefitOptimizationInfo:
    success: bool
    iterations: int
    cost: float
    message: str = ""


@dataclass
class ModelRefitResult:
    K_original: np.ndarray
    D_original: np.ndarray
    K_refitted: np.ndarray
    D_refitted: np.ndarray
    K_naive: np.ndarray
    D_naive: np.ndarray
    optimization: RefitOptimizationInfo
    error: RefitErrorStats
    naive_error: RefitErrorStats
    region_error: dict[str, RefitErrorStats] = field(default_factory=dict)
    naive_region_error: dict[str, RefitErrorStats] = field(default_factory=dict)
    sample_pixels: np.ndarray | None = None
    reference_pixels: np.ndarray | None = None
    refitted_pixels: np.ndarray | None = None
    naive_pixels: np.ndarray | None = None
    normalized_radius: np.ndarray | None = None

    @property
    def improvement_rmse_pct(self) -> float | None:
        if self.naive_error.rmse_px <= 0:
            return None
        return (self.naive_error.rmse_px - self.error.rmse_px) / self.naive_error.rmse_px * 100.0


def _as_camera_matrix(K) -> np.ndarray:
    arr = np.asarray(K, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"K shape must be (3, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("K contains NaN or Inf.")
    if arr[0, 0] <= 0 or arr[1, 1] <= 0:
        raise ValueError("fx and fy must be positive.")
    return arr.copy()


def _as_distortion8(D) -> np.ndarray:
    arr = np.asarray(D, dtype=np.float64).reshape(-1)
    if arr.size < 8:
        raise ValueError(f"D must contain at least 8 coefficients, got {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("D contains NaN or Inf.")
    return arr[:8].copy()


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive (width, height), got {image_size}.")
    return width, height


def sample_image_points(
    image_size: tuple[int, int],
    grid_size: tuple[int, int] = (80, 50),
    margin_ratio: float = 0.0,
) -> np.ndarray:
    """Dense image-wide pixel sampling, returned as (N, 2) [u, v]."""
    width, height = _validate_image_size(image_size)
    gx, gy = int(grid_size[0]), int(grid_size[1])
    if gx < 2 or gy < 2:
        raise ValueError("grid_size must be at least (2, 2).")
    mx = width * float(margin_ratio)
    my = height * float(margin_ratio)
    xs = np.linspace(mx, width - 1 - mx, gx, dtype=np.float64)
    ys = np.linspace(my, height - 1 - my, gy, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def pixels_to_normalized_rays(pixels: np.ndarray, K: np.ndarray) -> np.ndarray:
    pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (pts[:, 0] - cx) / fx
    y = (pts[:, 1] - cy) / fy
    return np.column_stack([x, y, np.ones_like(x)])


def project_rational_pinhole(
    rays: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """Project normalized rays using OpenCV's pinhole/rational distortion order."""
    rays = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
    normalized = rays[:, :2] / rays[:, 2:3]
    object_points = np.column_stack([normalized, np.ones(len(normalized))]).reshape(-1, 1, 3)
    projected, _ = cv2.projectPoints(
        object_points,
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.asarray(K, dtype=np.float64),
        np.asarray(D, dtype=np.float64).reshape(-1, 1),
    )
    return np.asarray(projected, dtype=np.float64).reshape(-1, 2)


def _radius_for_pixels(pixels: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = _validate_image_size(image_size)
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float64)
    max_radius = float(np.hypot(center[0], center[1]))
    return np.linalg.norm(np.asarray(pixels, dtype=np.float64) - center, axis=1) / max(max_radius, 1e-12)


def _weights(radius: np.ndarray, mode: WeightMode, edge_weight: float) -> np.ndarray:
    if mode == "uniform":
        return np.ones_like(radius, dtype=np.float64)
    if mode == "edge":
        return 1.0 + (float(edge_weight) - 1.0) * np.clip(radius, 0.0, 1.0) ** 2
    raise ValueError(f"Unknown weight mode: {mode}")


def compute_refit_error_stats(errors_px: np.ndarray) -> RefitErrorStats:
    arr = np.asarray(errors_px, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return RefitErrorStats()
    return RefitErrorStats(
        rmse_px=float(np.sqrt(np.mean(arr ** 2))),
        mean_px=float(np.mean(arr)),
        median_px=float(np.median(arr)),
        p95_px=float(np.percentile(arr, 95)),
        p99_px=float(np.percentile(arr, 99)),
        max_px=float(np.max(arr)),
    )


def _region_stats(errors_px: np.ndarray, radius: np.ndarray) -> dict[str, RefitErrorStats]:
    regions = {
        "center": radius < 0.33,
        "middle": (radius >= 0.33) & (radius < 0.66),
        "edge": radius >= 0.66,
    }
    return {name: compute_refit_error_stats(errors_px[mask]) for name, mask in regions.items()}


def _pack_params(K: np.ndarray, D5: np.ndarray, optimize_intrinsics: bool) -> np.ndarray:
    d = np.asarray(D5, dtype=np.float64).reshape(5)
    if not optimize_intrinsics:
        return d.copy()
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *d], dtype=np.float64)


def _unpack_params(params: np.ndarray, K_base: np.ndarray, optimize_intrinsics: bool) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(params, dtype=np.float64)
    K = np.asarray(K_base, dtype=np.float64).copy()
    if optimize_intrinsics:
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = p[:4]
        D5 = p[4:9]
    else:
        D5 = p[:5]
    return K, np.asarray(D5, dtype=np.float64).reshape(5)


def _bounds(K: np.ndarray, image_size: tuple[int, int], optimize_intrinsics: bool) -> tuple[np.ndarray, np.ndarray]:
    if not optimize_intrinsics:
        return np.full(5, -np.inf), np.full(5, np.inf)
    width, height = _validate_image_size(image_size)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    lower = np.array([
        max(1e-6, fx * 0.5),
        max(1e-6, fy * 0.5),
        -0.25 * width,
        -0.25 * height,
        -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,
    ], dtype=np.float64)
    upper = np.array([
        fx * 1.5,
        fy * 1.5,
        width * 1.25,
        height * 1.25,
        np.inf, np.inf, np.inf, np.inf, np.inf,
    ], dtype=np.float64)
    return lower, upper


def _residual_regularization(
    params: np.ndarray,
    initial: np.ndarray,
    optimize_intrinsics: bool,
    regularization: float,
) -> np.ndarray:
    if not optimize_intrinsics or regularization <= 0:
        return np.array([], dtype=np.float64)
    denom = np.maximum(np.abs(initial[:4]), 1.0)
    return np.sqrt(float(regularization)) * (params[:4] - initial[:4]) / denom


def _evaluate(K: np.ndarray, D: np.ndarray, rays: np.ndarray, reference_pixels: np.ndarray, radius: np.ndarray) -> tuple[RefitErrorStats, dict[str, RefitErrorStats], np.ndarray, np.ndarray]:
    pixels = project_rational_pinhole(rays, K, D)
    diff = pixels - reference_pixels
    err = np.linalg.norm(diff, axis=1)
    return compute_refit_error_stats(err), _region_stats(err, radius), pixels, err


def refit_extended_pinhole_to_pinhole(
    K,
    D,
    image_size: tuple[int, int],
    *,
    optimize_intrinsics: bool = True,
    mode: RefitMode | None = None,
    grid_size: tuple[int, int] = (80, 50),
    edge_weighting: bool = False,
    edge_weight: float = 2.0,
    loss: str = "linear",
    regularization: float = 1e-3,
    max_nfev: int = 300,
) -> ModelRefitResult:
    """Fit a 5-coefficient OpenCV pinhole model to an 8-coefficient rational model."""
    try:
        from scipy.optimize import least_squares
    except ImportError as e:
        raise ImportError("Model Refitting requires scipy. Install it with: pip install scipy") from e

    if mode is not None:
        optimize_intrinsics = mode == "full"
        if mode not in ("full", "distortion_only"):
            raise ValueError(f"Unknown refit mode: {mode}")

    K8 = _as_camera_matrix(K)
    D8 = _as_distortion8(D)
    image_size = _validate_image_size(image_size)

    sample_pixels = sample_image_points(image_size, grid_size=grid_size)
    rays = pixels_to_normalized_rays(sample_pixels, K8)
    radius = _radius_for_pixels(sample_pixels, image_size)
    reference_pixels = project_rational_pinhole(rays, K8, D8)

    D_naive = D8[:5].copy()
    K_naive = K8.copy()
    initial = _pack_params(K_naive, D_naive, optimize_intrinsics)
    weight = np.repeat(np.sqrt(_weights(radius, "edge" if edge_weighting else "uniform", edge_weight)), 2)

    def residual(params: np.ndarray) -> np.ndarray:
        K5, D5 = _unpack_params(params, K8, optimize_intrinsics)
        candidate = project_rational_pinhole(rays, K5, D5)
        pixel_residual = (candidate - reference_pixels).reshape(-1) * weight
        reg = _residual_regularization(params, initial, optimize_intrinsics, regularization)
        return np.concatenate([pixel_residual, reg]) if reg.size else pixel_residual

    lower, upper = _bounds(K8, image_size, optimize_intrinsics)
    opt = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss=loss,
        max_nfev=max_nfev,
    )
    K_refit, D_refit = _unpack_params(opt.x, K8, optimize_intrinsics)

    error, region_error, refitted_pixels, _ = _evaluate(K_refit, D_refit, rays, reference_pixels, radius)
    naive_error, naive_region_error, naive_pixels, _ = _evaluate(K_naive, D_naive, rays, reference_pixels, radius)

    return ModelRefitResult(
        K_original=K8,
        D_original=D8,
        K_refitted=K_refit,
        D_refitted=D_refit.reshape(5, 1),
        K_naive=K_naive,
        D_naive=D_naive.reshape(5, 1),
        optimization=RefitOptimizationInfo(
            success=bool(opt.success),
            iterations=int(opt.nfev),
            cost=float(opt.cost),
            message=str(opt.message),
        ),
        error=error,
        naive_error=naive_error,
        region_error=region_error,
        naive_region_error=naive_region_error,
        sample_pixels=sample_pixels,
        reference_pixels=reference_pixels,
        refitted_pixels=refitted_pixels,
        naive_pixels=naive_pixels,
        normalized_radius=radius,
    )
