"""
Camera projection utilities: project 3D points (already expressed in the
camera frame) into 2D pixel coordinates, for both pinhole and fisheye
(equidistant) camera models, with distortion.

Wraps OpenCV's projectPoints / fisheye.projectPoints so the rest of the
codebase doesn't need to know OpenCV's exact array-shape conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ])


def plumb_bob_dist_coeffs(coeffs: dict) -> np.ndarray:
    """Assemble OpenCV-style [k1, k2, p1, p2, k3] from a coeffs dict.
    Missing keys default to 0.0. k3 is optional (5th param)."""
    k1 = coeffs.get("k1", 0.0)
    k2 = coeffs.get("k2", 0.0)
    p1 = coeffs.get("p1", 0.0)
    p2 = coeffs.get("p2", 0.0)
    k3 = coeffs.get("k3", 0.0)
    return np.array([k1, k2, p1, p2, k3], dtype=float)


def fisheye_dist_coeffs(coeffs: dict) -> np.ndarray:
    """Assemble OpenCV fisheye-style [k1, k2, k3, k4] from a coeffs dict."""
    k1 = coeffs.get("k1", 0.0)
    k2 = coeffs.get("k2", 0.0)
    k3 = coeffs.get("k3", 0.0)
    k4 = coeffs.get("k4", 0.0)
    return np.array([k1, k2, k3, k4], dtype=float)


def project_points_pinhole(points_cam: np.ndarray, K: np.ndarray, dist_coeffs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Project (N, 3) points already expressed in the camera frame into pixel
    coordinates using the standard (plumb-bob) pinhole model.

    Returns (N, 2) pixel coordinates. Does NOT filter points behind the
    camera (z <= 0) or outside the image bounds -- caller is responsible
    (see project_lidar_to_image, which does apply those filters).
    """
    points_cam = np.asarray(points_cam, dtype=np.float64)
    if points_cam.ndim != 2 or points_cam.shape[1] != 3:
        raise ValueError(f"points_cam must be (N, 3), got {points_cam.shape}")
    if points_cam.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)
    img_pts, _ = cv2.projectPoints(points_cam, rvec, tvec, K.astype(np.float64), dist_coeffs.astype(np.float64))
    return img_pts.reshape(-1, 2)


def project_points_fisheye(points_cam: np.ndarray, K: np.ndarray, dist_coeffs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Project (N, 3) points already expressed in the camera frame into pixel
    coordinates using OpenCV's fisheye (equidistant) model.
    """
    points_cam = np.asarray(points_cam, dtype=np.float64)
    if points_cam.ndim != 2 or points_cam.shape[1] != 3:
        raise ValueError(f"points_cam must be (N, 3), got {points_cam.shape}")
    if points_cam.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    if dist_coeffs is None:
        dist_coeffs = np.zeros(4)

    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)
    pts = points_cam.reshape(1, -1, 3)
    img_pts, _ = cv2.fisheye.projectPoints(
        pts, rvec, tvec, K.astype(np.float64), dist_coeffs.astype(np.float64)
    )
    return img_pts.reshape(-1, 2)


@dataclass
class ProjectionResult:
    """Result of projecting a LiDAR point cloud into an image."""
    pixels: np.ndarray          # (M, 2) pixel coords of VALID points only
    depths: np.ndarray          # (M,) camera-frame depth (z) of VALID points
    source_indices: np.ndarray  # (M,) indices into the original point array
    num_input_points: int
    num_valid_points: int


def project_lidar_to_image(
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray],
    image_width: int,
    image_height: int,
    camera_model: str = "pinhole",
    min_depth_m: float = 0.05,
) -> ProjectionResult:
    """
    Full pipeline: transform LiDAR points into the camera frame via T_CL,
    project into pixel coordinates, and filter to points that are (a) in
    front of the camera (depth > min_depth_m) and (b) within the image
    bounds.

    This is the shared entry point used by M0 (sanity gate) and M2 (edge
    alignment) so both stay consistent.
    """
    from geometry.transform import transform_points  # local import avoids cycle at module load

    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    n = points_lidar.shape[0]

    points_cam = transform_points(T_CL, points_lidar)
    depths = points_cam[:, 2]

    front_mask = depths > min_depth_m
    front_indices = np.nonzero(front_mask)[0]

    if front_indices.size == 0:
        return ProjectionResult(
            pixels=np.zeros((0, 2)),
            depths=np.zeros((0,)),
            source_indices=np.zeros((0,), dtype=int),
            num_input_points=n,
            num_valid_points=0,
        )

    points_cam_front = points_cam[front_indices]

    if camera_model == "pinhole":
        pixels = project_points_pinhole(points_cam_front, K, dist_coeffs)
    elif camera_model == "fisheye":
        pixels = project_points_fisheye(points_cam_front, K, dist_coeffs)
    else:
        raise ValueError(f"Unknown camera_model: {camera_model!r}")

    in_bounds = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < image_width) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < image_height) &
        np.isfinite(pixels).all(axis=1)
    )

    valid_local_idx = np.nonzero(in_bounds)[0]

    return ProjectionResult(
        pixels=pixels[valid_local_idx],
        depths=depths[front_indices[valid_local_idx]],
        source_indices=front_indices[valid_local_idx],
        num_input_points=n,
        num_valid_points=valid_local_idx.size,
    )
