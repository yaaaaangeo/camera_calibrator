"""
visualization/bev_dual_panel.py

Renders a two-panel diagnostic: the camera image (left, edge points drawn
GOOD/WARNING/BAD exactly as visualization.overlay draws them) alongside a
bird's-eye view of those SAME edge points in the camera frame (right,
X vs depth Z), colored with the same per-point classification. Background
LiDAR points are shown faintly in the BEV panel for spatial context.

overlay.py answers "how far off is each point"; error_heatmap.py answers
"where in the FRAME is error concentrated"; this view answers "where in
SPACE (distance, left/right) is error concentrated" -- e.g. does error
grow with range (points a translation-scale issue toward), or is it worse
on one side (points toward a rotation/mounting-angle issue)? Seeing both
panels side by side, with the same points highlighted in each, makes that
correspondence direct instead of having to cross-reference two separate
images mentally.

Recomputes which LiDAR points are "edge points" and their 3D positions
using the exact same extraction evaluation.edge_alignment.evaluate_edge_alignment
already ran (so the caller must pass matching edge_radius_px /
depth_jump_threshold_m / min_neighbors -- see render_bev_dual_panel_from_result,
which takes them directly). EdgeAlignmentResult only retains 2D pixels +
errors for its own (image-space) visualization, not 3D positions, so this
is the one place that needs the LiDAR-frame point for each edge point.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # headless: no display server needed, safe for CLI/report generation
import matplotlib.pyplot as plt

from geometry.projection import project_lidar_to_image
from geometry.transform import transform_points
from evaluation.edge_alignment import extract_lidar_edge_points
from quality.noise_floor import classify, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER
from visualization.overlay import render_overlay


_BG = "#0D1117"
_SURFACE = "#161B22"
_TEXT = "#8B98AC"
_GRID = "#2A3244"
_GOOD = "#3FB950"
_WARNING = "#D29922"
_BAD = "#F85149"
_CONTEXT_COLOR = "#3B4252"

_DEFAULT_CONTEXT_MAX_POINTS = 20_000
_COLOR_HEX = {"GOOD": _GOOD, "WARNING": _WARNING, "BAD": _BAD}


def _recompute_edge_points_lidar(
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    camera,
    edge_radius_px: float,
    depth_jump_threshold_m: float,
    min_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Re-run the same projection + edge-discontinuity extraction M2 uses,
    to recover each edge point's 3D LiDAR-frame position (which
    EdgeAlignmentResult doesn't retain -- only 2D pixels/errors). Returns
    (edge_points_lidar (M, 3), edge_pixels (M, 2)), in the same order
    evaluate_edge_alignment would produce them in, given identical
    parameters.
    """
    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    projection = project_lidar_to_image(
        points_lidar, T_CL, camera.K(), camera.dist_coeffs(),
        camera.width, camera.height, camera_model=camera.projection_model_name(),
    )
    if projection.num_valid_points == 0:
        return np.zeros((0, 3)), np.zeros((0, 2))

    edge_mask = extract_lidar_edge_points(
        projection.pixels, projection.depths,
        radius_px=edge_radius_px, depth_jump_threshold_m=depth_jump_threshold_m,
        min_neighbors=min_neighbors,
    )
    edge_source_idx = projection.source_indices[edge_mask]
    edge_points_lidar = points_lidar[edge_source_idx, :3]
    edge_pixels = projection.pixels[edge_mask]
    return edge_points_lidar, edge_pixels


def render_bev_dual_panel(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    camera,
    edge_alignment_result,
    edge_radius_px: float = 3.0,
    depth_jump_threshold_m: float = 0.3,
    min_neighbors: int = 3,
    context_max_points: int = _DEFAULT_CONTEXT_MAX_POINTS,
    dpi: int = 130,
    point_size: float = 10.0,
    seed: int = 0,
) -> Optional[bytes]:
    """
    Render the camera-image + bird's-eye-view dual panel and return PNG
    bytes.

    edge_radius_px / depth_jump_threshold_m / min_neighbors MUST match
    the values passed to evaluate_edge_alignment(...) when producing
    edge_alignment_result, or the recomputed edge points won't correspond
    1:1 with edge_alignment_result.edge_point_errors_px. Prefer
    render_bev_dual_panel_from_result if you have the same edge_kwargs
    dict already in hand from the M2 call.

    Returns None if edge_alignment_result FAILed, or if recomputing edge
    points with the given parameters doesn't reproduce the same point
    count as edge_alignment_result (a mismatched-parameters guard, since
    a silent misalignment between the two would mislabel points).
    """
    if edge_alignment_result.classification == "FAIL" or edge_alignment_result.edge_point_pixels is None:
        return None

    edge_points_lidar, edge_pixels = _recompute_edge_points_lidar(
        points_lidar, T_CL, camera, edge_radius_px, depth_jump_threshold_m, min_neighbors,
    )
    errors_px = edge_alignment_result.edge_point_errors_px
    floor_px = edge_alignment_result.floor_px

    if edge_points_lidar.shape[0] != errors_px.shape[0]:
        return None
    if edge_points_lidar.shape[0] == 0:
        return None

    classifications = np.array([classify(float(e), floor_px, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER)
                                 for e in errors_px])
    point_colors = [_COLOR_HEX.get(c, "#AAAAAA") for c in classifications]

    # --- Left panel source: reuse overlay.py's exact per-point coloring --
    overlay_bgr = render_overlay(image, edge_pixels, errors_px, floor_px)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # --- BEV context: all points, transformed to the camera frame -------
    edge_points_cam = transform_points(T_CL, edge_points_lidar)
    context_points_cam = None
    if points_lidar is not None and len(points_lidar) > 0:
        pts_cam = transform_points(T_CL, np.asarray(points_lidar, dtype=np.float64)[:, :3])
        in_front = pts_cam[:, 2] > 0.05
        pts_cam = pts_cam[in_front]
        if pts_cam.shape[0] > context_max_points:
            rng = np.random.default_rng(seed)
            keep = rng.choice(pts_cam.shape[0], size=context_max_points, replace=False)
            pts_cam = pts_cam[keep]
        context_points_cam = pts_cam

    fig, (ax_img, ax_bev) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=dpi)
    try:
        fig.patch.set_facecolor(_BG)

        ax_img.set_facecolor(_SURFACE)
        ax_img.imshow(overlay_rgb)
        ax_img.set_title("Camera view", color=_TEXT, fontsize=10)
        ax_img.axis("off")

        ax_bev.set_facecolor(_SURFACE)
        if context_points_cam is not None and context_points_cam.shape[0] > 0:
            ax_bev.scatter(context_points_cam[:, 0], context_points_cam[:, 2], c=_CONTEXT_COLOR,
                            s=1.0, alpha=0.5, marker=".", linewidths=0, label="LiDAR points")
        ax_bev.scatter(edge_points_cam[:, 0], edge_points_cam[:, 2], c=point_colors,
                        s=point_size, marker="o", linewidths=0, label="Edge points")
        ax_bev.invert_yaxis()  # near depth at bottom, far depth toward the top
        ax_bev.set_xlabel("X (m)", color=_TEXT, fontsize=8)
        ax_bev.set_ylabel("Depth Z (m)", color=_TEXT, fontsize=8)
        ax_bev.set_title("Bird's-eye view (same edge points, colored to match)", color=_TEXT, fontsize=10)
        ax_bev.tick_params(colors=_TEXT, labelsize=7)
        ax_bev.grid(color=_GRID, linewidth=0.5)
        for spine in ax_bev.spines.values():
            spine.set_color(_GRID)
        ax_bev.set_aspect("equal", adjustable="datalim")
        ax_bev.legend(loc="upper right", fontsize=7, facecolor=_SURFACE, edgecolor=_GRID, labelcolor=_TEXT)

        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        return buf.getvalue()
    finally:
        # Always close, even if a plotting call above raises -- otherwise
        # the figure leaks in matplotlib's global pyplot state.
        plt.close(fig)


def render_bev_dual_panel_from_result(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    camera,
    edge_alignment_result,
    edge_kwargs: Optional[dict] = None,
    **render_kwargs,
) -> Optional[bytes]:
    """
    Convenience wrapper matching the call sites in app/cli.py: pass the
    same `edge_kwargs` dict already used for evaluate_edge_alignment(...)
    (e.g. {"depth_jump_threshold_m": ..., "edge_radius_px": ...}) so the
    recomputed edge points line up with edge_alignment_result by
    construction. Returns PNG bytes, or None if there's nothing to draw.
    """
    edge_kwargs = edge_kwargs or {}
    return render_bev_dual_panel(
        image, points_lidar, T_CL, camera, edge_alignment_result, **edge_kwargs, **render_kwargs,
    )
