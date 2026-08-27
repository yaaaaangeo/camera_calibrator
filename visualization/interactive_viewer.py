"""
visualization/interactive_viewer.py

Builds the data for an INTERACTIVE (rotate/zoom, not a static PNG) 3D
scene combining two things every other visualization module in this
package renders separately as flat images: the colorized point cloud
(visualization.colorized_pointcloud) and the camera frustum
(visualization.camera_frustum), in one Plotly scene expressed in the
LiDAR frame.

This module only produces plain-Python, JSON-serializable data (trace
dicts + layout + config) -- it doesn't know anything about HTML or the
plotly.js runtime itself. report/html.py is the one place that turns
this into an embedded <div>+<script>, using the vendored plotly.js gl3d
bundle (report/vendor/) so the resulting report.html stays a single,
offline-viewable file with no CDN dependency -- important for CI
artifacts that may be viewed with no network access.

Kept intentionally light on point count (colorize_max_points defaults far
lower than colorized_pointcloud.py's own default) since, unlike a PNG,
every point here is embedded as literal numbers in the HTML's JSON
payload -- more points means a bigger report.html, not just a bigger
render.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from visualization.colorized_pointcloud import colorize_lidar_points
from visualization.camera_frustum import compute_frustum_geometry, auto_frustum_depth


_BG = "#0D1117"
_SURFACE = "#161B22"
_TEXT = "#8B98AC"
_GRID = "#2A3244"
_LIDAR_HEX = "#58A6FF"
_CAMERA_HEX = "#F0883E"
_FRUSTUM_HEX = "#F0883E"

_DEFAULT_COLORIZE_MAX_POINTS = 6_000
_COORD_DECIMALS = 3  # mm-level precision is plenty for a rig-geometry sanity view


def _round_list(arr: np.ndarray, decimals: int = _COORD_DECIMALS) -> list:
    """Round then convert to a plain Python list. Every coordinate in the
    scene is embedded as literal JSON text (unlike a PNG's fixed-size
    pixel grid), so trimming float precision meaningfully shrinks
    report.html -- 6.146575342465753 costs far more JSON bytes than
    6.147 for no visually-perceptible difference at this scene's scale."""
    return np.round(arr, decimals).tolist()


def build_interactive_scene(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    camera,
    colorize_max_points: int = _DEFAULT_COLORIZE_MAX_POINTS,
    depth_m: Optional[float] = None,
) -> dict:
    """
    Build a Plotly scene dict: {"data": [...traces], "layout": {...},
    "config": {...}, "num_points": int}. All values are plain Python
    (lists/floats/strings), ready for json.dumps.

    Combines:
      - the colorized point cloud (each LiDAR point painted with the
        camera pixel color it projects onto), in the LiDAR frame
      - the LiDAR origin and camera origin (as markers)
      - the camera's viewing frustum out to `depth_m` (auto-picked from
        the point cloud's depth distribution if not given, matching
        visualization.camera_frustum's convention)
    """
    colorized = colorize_lidar_points(
        image, points_lidar, T_CL,
        K=camera.K(), dist_coeffs=camera.dist_coeffs(),
        image_width=camera.width, image_height=camera.height,
        camera_model=camera.projection_model_name(),
        max_points=colorize_max_points,
    )

    if depth_m is None:
        depth_m = auto_frustum_depth(points_lidar, T_CL)
    geom = compute_frustum_geometry(T_CL, camera.K(), camera.width, camera.height, depth_m)

    traces: list[dict] = []

    if colorized.num_colorized_points > 0:
        pts = colorized.points_lidar
        # Hex ("#rrggbb", 7 chars) instead of "rgb(r,g,b)" (up to 17 chars)
        # roughly halves the per-point color payload across thousands of
        # points -- adds up fast since this JSON is embedded verbatim in
        # report.html, not just held in memory.
        color_strs = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colorized.colors_rgb.tolist()]
        traces.append({
            "type": "scatter3d", "mode": "markers", "name": "Colorized LiDAR points",
            "x": _round_list(pts[:, 0]), "y": _round_list(pts[:, 1]), "z": _round_list(pts[:, 2]),
            "marker": {"size": 1.8, "color": color_strs, "opacity": 0.85},
            "hoverinfo": "skip",
        })

    traces.append({
        "type": "scatter3d", "mode": "markers", "name": "LiDAR origin",
        "x": [0.0], "y": [0.0], "z": [0.0],
        "marker": {"size": 5, "color": _LIDAR_HEX, "symbol": "circle"},
    })

    co = geom.camera_origin
    traces.append({
        "type": "scatter3d", "mode": "markers", "name": "Camera origin",
        "x": [round(float(co[0]), _COORD_DECIMALS)],
        "y": [round(float(co[1]), _COORD_DECIMALS)],
        "z": [round(float(co[2]), _COORD_DECIMALS)],
        "marker": {"size": 5, "color": _CAMERA_HEX, "symbol": "diamond"},
    })

    # Frustum edges: 4 lines from the camera origin to each far corner,
    # plus the closed loop connecting the corners. None-separated segments
    # let a single scatter3d "lines" trace draw multiple disjoint lines.
    corners = geom.far_corners
    edge_x, edge_y, edge_z = [], [], []
    for corner in corners:
        edge_x += [round(float(co[0]), _COORD_DECIMALS), round(float(corner[0]), _COORD_DECIMALS), None]
        edge_y += [round(float(co[1]), _COORD_DECIMALS), round(float(corner[1]), _COORD_DECIMALS), None]
        edge_z += [round(float(co[2]), _COORD_DECIMALS), round(float(corner[2]), _COORD_DECIMALS), None]
    loop = np.vstack([corners, corners[0:1]])
    edge_x += _round_list(loop[:, 0]) + [None]
    edge_y += _round_list(loop[:, 1]) + [None]
    edge_z += _round_list(loop[:, 2]) + [None]
    traces.append({
        "type": "scatter3d", "mode": "lines", "name": "Camera frustum",
        "x": edge_x, "y": edge_y, "z": edge_z,
        "line": {"color": _FRUSTUM_HEX, "width": 3},
        "hoverinfo": "skip",
    })

    # Translucent far face (two triangles over the 4 corners).
    traces.append({
        "type": "mesh3d", "name": "Frustum far plane",
        "x": _round_list(corners[:, 0]), "y": _round_list(corners[:, 1]), "z": _round_list(corners[:, 2]),
        "i": [0, 0], "j": [1, 2], "k": [2, 3],
        "color": _FRUSTUM_HEX, "opacity": 0.15, "showscale": False, "hoverinfo": "skip",
    })

    layout = {
        "scene": {
            "xaxis": {"title": {"text": "X (m)"}, "color": _TEXT, "gridcolor": _GRID, "backgroundcolor": _SURFACE},
            "yaxis": {"title": {"text": "Y (m)"}, "color": _TEXT, "gridcolor": _GRID, "backgroundcolor": _SURFACE},
            "zaxis": {"title": {"text": "Z (m)"}, "color": _TEXT, "gridcolor": _GRID, "backgroundcolor": _SURFACE},
            "aspectmode": "data",
            "bgcolor": _BG,
        },
        "paper_bgcolor": _BG,
        "font": {"color": _TEXT},
        "margin": {"l": 0, "r": 0, "t": 36, "b": 0},
        "showlegend": True,
        "legend": {"bgcolor": _SURFACE, "bordercolor": _GRID, "font": {"color": _TEXT}},
        "title": {
            "text": f"Interactive rig view \u00b7 HFOV {geom.hfov_deg:.0f}\u00b0, baseline {geom.baseline_m:.2f} m",
            "font": {"color": _TEXT},
        },
    }
    config = {"displaylogo": False, "responsive": True}

    return {"data": traces, "layout": layout, "config": config, "num_points": colorized.num_colorized_points}


def build_interactive_scene_from_dataset(
    dataset,
    frame_index: Optional[int] = None,
    colorize_max_points: int = _DEFAULT_COLORIZE_MAX_POINTS,
    depth_m: Optional[float] = None,
) -> dict:
    """
    Convenience wrapper mirroring visualization.camera_frustum.render_camera_frustum_from_dataset:
    takes an EvaluationDataset directly, using one representative frame's
    image+points. frame_index defaults to the temporally-middle frame,
    matching app.cli's headline-frame convention.
    """
    if not dataset.frames:
        raise ValueError("Dataset has no synced frames; nothing to build a scene from.")
    idx = frame_index if frame_index is not None else len(dataset.frames) // 2
    idx = max(0, min(idx, len(dataset.frames) - 1))
    frame = dataset.frames[idx]
    return build_interactive_scene(
        frame.camera_frame.load(), frame.lidar_frame.load(), dataset.extrinsic.T_CL, dataset.camera,
        colorize_max_points=colorize_max_points, depth_m=depth_m,
    )
