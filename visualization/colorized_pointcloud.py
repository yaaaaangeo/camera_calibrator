"""
visualization/colorized_pointcloud.py

Renders the "colorized point cloud" fusion view: each LiDAR point is
projected into the camera image and painted with the RGB color sampled
from its landing pixel, producing the classic camera+LiDAR fusion
picture. This is a direct, intuitive readout of extrinsic quality --
if T_CL is off, points near object boundaries sample the WRONG side's
color (e.g. a car's LiDAR return picking up the road's gray instead of
the car's paint), which shows up as visible color bleed/smearing at
edges. That's the same kind of error M2 measures numerically in pixels;
this view just makes it visible at a glance instead of reading a number.

Reuses geometry.projection.project_lidar_to_image -- the exact same
function M0/M2 already use -- so "which points are valid" (in front of
the camera, inside the image bounds) never drifts out of sync with the
rest of the tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: no display server needed, safe for CLI/report generation
import matplotlib.pyplot as plt

from geometry.projection import project_lidar_to_image
from geometry.transform import transform_points


_BG = "#0D1117"
_SURFACE = "#161B22"
_TEXT = "#8B98AC"
_GRID = "#2A3244"

_DEFAULT_MAX_POINTS = 60_000


@dataclass
class ColorizedPointCloudResult:
    """Colorized subset of a LiDAR frame: points expressed in the CAMERA
    frame (so plotting doesn't need T_CL again), each with an RGB color
    sampled from the camera image at its projected pixel."""
    points_cam: np.ndarray       # (M, 3) points in the camera frame
    colors_rgb: np.ndarray       # (M, 3) uint8 RGB colors sampled from image
    pixels: np.ndarray           # (M, 2) source pixel coords (post-subsample)
    num_input_points: int
    num_valid_points: int        # points that passed projection (before subsample)
    num_colorized_points: int    # points actually returned (after subsample)
    points_lidar: Optional[np.ndarray] = None  # (M, 3) the SAME points, in the original LiDAR frame


def colorize_lidar_points(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray],
    image_width: int,
    image_height: int,
    camera_model: str = "pinhole",
    min_depth_m: float = 0.05,
    max_points: int = _DEFAULT_MAX_POINTS,
    seed: int = 0,
) -> ColorizedPointCloudResult:
    """
    Project points_lidar into `image` via T_CL and sample RGB at each
    valid point's pixel. Colors are returned as RGB (not OpenCV's BGR)
    since this feeds a matplotlib plot, not another OpenCV image.

    max_points: if more than this many points pass the projection filter,
    deterministically (fixed seed) subsample down to max_points -- keeps
    the render responsive and the embedded PNG a reasonable size on dense
    clouds. Does not affect which points count as "valid" in the first
    place, only how many of the valid ones get drawn.

    Raises ValueError if `image`'s actual pixel dimensions don't match
    image_width/image_height. Sampling a pixel color below indexes into
    `image` using coordinates bounds-checked against image_width/
    image_height, not against image.shape -- so a mismatch (e.g. a config
    width/height that doesn't match the actual image file) would
    otherwise surface as an IndexError from deep inside the fancy-index
    call below, which doesn't say what's actually wrong.
    """
    if image.ndim < 2 or image.shape[0] != image_height or image.shape[1] != image_width:
        actual = "x".join(str(d) for d in image.shape[:2][::-1]) if image.ndim >= 2 else str(image.shape)
        raise ValueError(
            f"image_width/image_height ({image_width}x{image_height}) don't match "
            f"the actual image shape ({actual}). Check the camera config's "
            f"width/height against the actual image file."
        )

    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    proj = project_lidar_to_image(
        points_lidar, T_CL, K, dist_coeffs, image_width, image_height,
        camera_model=camera_model, min_depth_m=min_depth_m,
    )

    if proj.num_valid_points == 0:
        return ColorizedPointCloudResult(
            points_cam=np.zeros((0, 3)),
            colors_rgb=np.zeros((0, 3), dtype=np.uint8),
            pixels=np.zeros((0, 2)),
            num_input_points=proj.num_input_points,
            num_valid_points=0,
            num_colorized_points=0,
            points_lidar=np.zeros((0, 3)),
        )

    points_cam_valid = transform_points(T_CL, points_lidar)[proj.source_indices]
    points_lidar_valid = points_lidar[proj.source_indices, :3]

    px = np.clip(proj.pixels[:, 0].astype(int), 0, image_width - 1)
    py = np.clip(proj.pixels[:, 1].astype(int), 0, image_height - 1)
    bgr = image[py, px]           # (M, 3) BGR uint8, image-array indexing
    rgb = np.ascontiguousarray(bgr[:, ::-1])  # BGR -> RGB

    n = points_cam_valid.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, size=max_points, replace=False)
        points_cam_valid = points_cam_valid[keep]
        points_lidar_valid = points_lidar_valid[keep]
        rgb = rgb[keep]
        px, py = px[keep], py[keep]

    return ColorizedPointCloudResult(
        points_cam=points_cam_valid,
        colors_rgb=rgb.astype(np.uint8),
        pixels=np.stack([px, py], axis=1),
        num_input_points=proj.num_input_points,
        num_valid_points=n,
        num_colorized_points=points_cam_valid.shape[0],
        points_lidar=points_lidar_valid,
    )


def render_colorized_pointcloud_png(
    result: ColorizedPointCloudResult,
    dpi: int = 130,
    elev: float = -15.0,
    azim: float = -75.0,
    point_size: float = 1.5,
) -> Optional[bytes]:
    """
    Render a two-panel PNG:
      left  -- a 3D scatter of the colorized cloud in the camera frame
      right -- a straight-down bird's-eye view (X vs depth Z)
    A single static 3D angle can hide a misalignment that jumps out
    immediately from directly above, so both are included by default.

    Returns None if there are no colorized points to plot (e.g. nothing
    from this frame landed inside the image).
    """
    if result.num_colorized_points == 0:
        return None

    pts = result.points_cam
    colors = result.colors_rgb.astype(np.float64) / 255.0

    fig = plt.figure(figsize=(10.5, 5), dpi=dpi)
    try:
        fig.patch.set_facecolor(_BG)

        # --- Panel 1: 3D scatter in the camera frame -----------------------
        # Camera-frame convention: +X right, +Y down, +Z forward (depth).
        # Plotted as (X, Z, -Y) so "forward" reads as depth-into-the-page and
        # "up" reads as up, instead of the raw down-positive Y axis.
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax3d.set_facecolor(_SURFACE)
        ax3d.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=colors, s=point_size, marker=".", linewidths=0)
        ax3d.set_xlabel("X (m)", color=_TEXT, fontsize=8, labelpad=2)
        ax3d.set_ylabel("Depth Z (m)", color=_TEXT, fontsize=8, labelpad=2)
        ax3d.set_zlabel("Up (m)", color=_TEXT, fontsize=8, labelpad=2)
        ax3d.set_title("Colorized point cloud (camera frame)", color=_TEXT, fontsize=10)
        ax3d.view_init(elev=elev, azim=azim)
        ax3d.tick_params(colors=_TEXT, labelsize=6)
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.pane.set_facecolor(_SURFACE)
            axis.pane.set_edgecolor(_GRID)
            axis.line.set_color(_GRID)
        ax3d.grid(color=_GRID)

        # --- Panel 2: bird's-eye view (X vs depth Z) ------------------------
        ax_bev = fig.add_subplot(1, 2, 2)
        ax_bev.set_facecolor(_SURFACE)
        ax_bev.scatter(pts[:, 0], pts[:, 2], c=colors, s=point_size + 0.5, marker=".", linewidths=0)
        ax_bev.invert_yaxis()  # near depth at bottom, far depth toward the top
        ax_bev.set_xlabel("X (m)", color=_TEXT, fontsize=8)
        ax_bev.set_ylabel("Depth Z (m)", color=_TEXT, fontsize=8)
        ax_bev.set_title("Bird's-eye view", color=_TEXT, fontsize=10)
        ax_bev.tick_params(colors=_TEXT, labelsize=7)
        ax_bev.grid(color=_GRID, linewidth=0.5)
        for spine in ax_bev.spines.values():
            spine.set_color(_GRID)
        ax_bev.set_aspect("equal", adjustable="datalim")

        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        return buf.getvalue()
    finally:
        # Always close, even if a plotting call above raises -- otherwise
        # the figure leaks in matplotlib's global pyplot state.
        plt.close(fig)


def render_colorized_pointcloud_from_frame(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    camera,
    max_points: int = _DEFAULT_MAX_POINTS,
    **render_kwargs,
) -> Optional[bytes]:
    """
    Convenience wrapper mirroring visualization.overlay's
    render_overlay_from_result: takes the raw frame data plus a
    CameraModel (as already used throughout the pipeline) and produces
    the PNG in one call.
    """
    result = colorize_lidar_points(
        image, points_lidar, T_CL,
        K=camera.K(), dist_coeffs=camera.dist_coeffs(),
        image_width=camera.width, image_height=camera.height,
        camera_model=camera.projection_model_name(),
        max_points=max_points,
    )
    return render_colorized_pointcloud_png(result, **render_kwargs)
