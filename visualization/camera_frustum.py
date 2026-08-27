"""
visualization/camera_frustum.py

Renders a schematic 3D view of the camera-LiDAR rig itself: the LiDAR
origin, the camera's position + viewing frustum placed in the LiDAR
frame using the extrinsic under evaluation, and (optionally) a light
scatter of a frame's LiDAR points for scale/context.

Where M2's overlay and the error heatmap show HOW WELL the calibration
lines up, this view shows WHAT the calibration IS: the raw
translation/rotation numbers in a config turned into "here's where the
camera physically sits and which way it's pointing, relative to the
LiDAR" -- something a person can sanity-check at a glance (e.g. "the
camera is behind the LiDAR" or "pointing the wrong way" jumps out
immediately here, in a way six numbers in a YAML file don't).

Rig geometry doesn't change frame-to-frame the way M2's per-point errors
do, so this is meant primarily as a one-time summary image -- e.g. near
the top of the HTML report, next to the extrinsic metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: no display server needed, safe for CLI/report generation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from geometry.transform import transform_points, invert_transform


_BG = "#0D1117"
_SURFACE = "#161B22"
_TEXT = "#8B98AC"
_GRID = "#2A3244"
_LIDAR_COLOR = "#58A6FF"
_CAMERA_COLOR = "#F0883E"
_FRUSTUM_COLOR = "#F0883E"
_POINTS_COLOR = "#3FB950"
_AXIS_COLORS = ("#F85149", "#3FB950", "#58A6FF")  # X red, Y green, Z blue

_DEFAULT_MAX_POINTS = 8000
_DEFAULT_DEPTH_M = 8.0


@dataclass
class FrustumGeometry:
    """Camera position/orientation and viewing-frustum corners, all
    expressed in the LIDAR frame (so they can be plotted directly
    alongside raw LiDAR points without any further transform)."""
    camera_origin: np.ndarray   # (3,)
    camera_axes: np.ndarray     # (3, 3), columns = camera x/y/z axes expressed in the lidar frame
    far_corners: np.ndarray     # (4, 3): TL, TR, BR, BL, in the lidar frame
    depth_m: float
    hfov_deg: float
    vfov_deg: float
    baseline_m: float


def compute_frustum_geometry(
    T_CL: np.ndarray,
    K: np.ndarray,
    image_width: int,
    image_height: int,
    depth_m: float,
) -> FrustumGeometry:
    """
    Derive the camera's position, orientation, and a pyramid-shaped
    viewing frustum out to `depth_m`, all expressed in the LiDAR frame.

    Uses the intrinsics matrix's fx/fy to get the (undistorted,
    pinhole-approximated) horizontal/vertical field of view. A schematic
    frustum is meant to communicate "roughly where, and which way", not
    per-pixel distortion, so this approximation is intentional even for
    fisheye cameras.
    """
    T_CL = np.asarray(T_CL, dtype=float)
    T_LC = invert_transform(T_CL)  # camera_from_lidar -> lidar_from_camera

    camera_origin = T_LC[:3, 3]
    camera_axes = T_LC[:3, :3]  # columns are the camera's x/y/z axes, expressed in the lidar frame

    fx, fy = float(K[0, 0]), float(K[1, 1])
    hfov = 2.0 * np.arctan((image_width / 2.0) / fx)
    vfov = 2.0 * np.arctan((image_height / 2.0) / fy)

    half_w = depth_m * np.tan(hfov / 2.0)
    half_h = depth_m * np.tan(vfov / 2.0)

    # Corners in the CAMERA frame (x right, y down, z forward).
    corners_cam = np.array([
        [-half_w, -half_h, depth_m],  # top-left
        [ half_w, -half_h, depth_m],  # top-right
        [ half_w,  half_h, depth_m],  # bottom-right
        [-half_w,  half_h, depth_m],  # bottom-left
    ])
    far_corners = transform_points(T_LC, corners_cam)

    return FrustumGeometry(
        camera_origin=camera_origin,
        camera_axes=camera_axes,
        far_corners=far_corners,
        depth_m=depth_m,
        hfov_deg=float(np.degrees(hfov)),
        vfov_deg=float(np.degrees(vfov)),
        baseline_m=float(np.linalg.norm(camera_origin)),
    )


def auto_frustum_depth(points_lidar: Optional[np.ndarray], T_CL: np.ndarray, fallback: float = _DEFAULT_DEPTH_M) -> float:
    """Pick a frustum depth that roughly matches the scene scale: the
    75th percentile of in-front-of-camera point depths, if points are
    given, else a fixed fallback. Keeps the drawn frustum sized to the
    actual scene instead of always the same fixed length regardless of
    environment.

    Public (not module-private) because visualization.interactive_viewer
    shares this exact scene-scaling logic for its own frustum trace --
    both need the same depth so the static PNG and the interactive scene
    agree on frustum size."""
    if points_lidar is None or len(points_lidar) == 0:
        return fallback
    points_cam = transform_points(T_CL, np.asarray(points_lidar, dtype=float)[:, :3])
    depths = points_cam[:, 2]
    depths = depths[depths > 0.05]
    if depths.size == 0:
        return fallback
    return float(np.clip(np.percentile(depths, 75), 1.0, 200.0))


def _draw_axis_triad(ax, origin: np.ndarray, axes: np.ndarray, length: float) -> None:
    for i in range(3):
        vec = axes[:, i] * length
        ax.plot(
            [origin[0], origin[0] + vec[0]],
            [origin[1], origin[1] + vec[1]],
            [origin[2], origin[2] + vec[2]],
            color=_AXIS_COLORS[i], linewidth=1.6,
        )


def _draw_frustum(ax, geom: FrustumGeometry) -> None:
    origin = geom.camera_origin
    corners = geom.far_corners
    for corner in corners:
        ax.plot([origin[0], corner[0]], [origin[1], corner[1]], [origin[2], corner[2]],
                color=_FRUSTUM_COLOR, linewidth=1.0, alpha=0.8)
    loop = np.vstack([corners, corners[0:1]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=_FRUSTUM_COLOR, linewidth=1.3, alpha=0.9)
    face = Poly3DCollection([corners], alpha=0.12, facecolor=_FRUSTUM_COLOR, edgecolor="none")
    ax.add_collection3d(face)


def _set_equal_bounds(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 0.5) * 1.15
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_camera_frustum_png(
    T_CL: np.ndarray,
    K: np.ndarray,
    image_width: int,
    image_height: int,
    points_lidar: Optional[np.ndarray] = None,
    depth_m: Optional[float] = None,
    max_points: int = _DEFAULT_MAX_POINTS,
    dpi: int = 130,
    elev: float = 22.0,
    azim: float = -60.0,
    seed: int = 0,
) -> bytes:
    """
    Render the rig-geometry summary: LiDAR origin (with axis triad),
    camera position + orientation (axis triad) + viewing frustum, and
    (if provided) a light context scatter of LiDAR points, all in the
    LiDAR frame.

    depth_m: how far out to draw the frustum. If None, auto-picked from
    points_lidar's depth distribution (see auto_frustum_depth) so the frustum
    scales with the actual scene rather than a fixed size regardless of
    rig/environment.
    """
    if depth_m is None:
        depth_m = auto_frustum_depth(points_lidar, T_CL)

    geom = compute_frustum_geometry(T_CL, K, image_width, image_height, depth_m)

    fig = plt.figure(figsize=(7.5, 6.5), dpi=dpi)
    try:
        fig.patch.set_facecolor(_BG)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(_SURFACE)

        if points_lidar is not None and len(points_lidar) > 0:
            pts = np.asarray(points_lidar, dtype=float)[:, :3]
            if pts.shape[0] > max_points:
                rng = np.random.default_rng(seed)
                keep = rng.choice(pts.shape[0], size=max_points, replace=False)
                pts = pts[keep]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=_POINTS_COLOR, s=0.6, alpha=0.35,
                       marker=".", linewidths=0, label="LiDAR points", depthshade=False)

        axis_len = max(geom.depth_m * 0.15, 0.15)
        _draw_axis_triad(ax, origin=np.zeros(3), axes=np.eye(3), length=axis_len)
        ax.scatter([0], [0], [0], c=_LIDAR_COLOR, s=70, marker="o", label="LiDAR origin", depthshade=False)

        ax.scatter([geom.camera_origin[0]], [geom.camera_origin[1]], [geom.camera_origin[2]],
                   c=_CAMERA_COLOR, s=70, marker="^", label="Camera origin", depthshade=False)
        _draw_axis_triad(ax, origin=geom.camera_origin, axes=geom.camera_axes, length=axis_len)
        _draw_frustum(ax, geom)

        _set_equal_bounds(ax, np.vstack([geom.camera_origin.reshape(1, 3), geom.far_corners, np.zeros((1, 3))]))

        ax.set_xlabel("X (m)", color=_TEXT, fontsize=8, labelpad=4)
        ax.set_ylabel("Y (m)", color=_TEXT, fontsize=8, labelpad=4)
        ax.set_zlabel("Z (m)", color=_TEXT, fontsize=8, labelpad=4)
        ax.set_title(
            f"Camera-LiDAR rig geometry  (HFOV {geom.hfov_deg:.0f}\u00b0, baseline {geom.baseline_m:.2f} m)",
            color=_TEXT, fontsize=10,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(colors=_TEXT, labelsize=6)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor(_SURFACE)
            axis.pane.set_edgecolor(_GRID)
            axis.line.set_color(_GRID)
        ax.grid(color=_GRID)
        ax.legend(loc="upper left", fontsize=7, facecolor=_SURFACE, edgecolor=_GRID, labelcolor=_TEXT)

        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        return buf.getvalue()
    finally:
        # Always close, even if a plotting call above raises -- otherwise
        # the figure leaks in matplotlib's global pyplot state.
        plt.close(fig)


def render_camera_frustum_from_dataset(
    dataset,
    frame_index: Optional[int] = None,
    depth_m: Optional[float] = None,
    **render_kwargs,
) -> bytes:
    """
    Convenience wrapper mirroring the other visualization modules'
    *_from_result/*_from_frame helpers: takes an EvaluationDataset
    directly and renders using its extrinsic + camera intrinsics, with
    one representative frame's points as context. frame_index defaults
    to the temporally-middle frame, matching app.cli's headline-frame
    convention.
    """
    if not dataset.frames:
        raise ValueError("Dataset has no synced frames; nothing to render.")
    idx = frame_index if frame_index is not None else len(dataset.frames) // 2
    idx = max(0, min(idx, len(dataset.frames) - 1))
    points = dataset.frames[idx].lidar_frame.load()
    camera = dataset.camera
    return render_camera_frustum_png(
        dataset.extrinsic.T_CL, camera.K(), camera.width, camera.height,
        points_lidar=points, depth_m=depth_m, **render_kwargs,
    )
