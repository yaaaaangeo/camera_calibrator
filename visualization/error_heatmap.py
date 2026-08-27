"""
visualization/error_heatmap.py

Renders a spatial error heatmap: the image is split into a grid, M2's
per-point pixel errors are aggregated within each cell, and each cell is
painted a translucent GOOD/WARNING/BAD-scaled color and blended over the
camera frame. visualization/overlay.py already shows every individual
point's error; this view answers a different question -- "WHERE in the
frame is the calibration weakest", not "how bad is each point" -- so a
person can spot spatial patterns (edges/corners only, one side of the
frame, a particular depth band) that point at a specific root cause
(e.g. lens distortion under-modeled, a small rotation offset that only
shows up far from the principal point) rather than a uniform miscalibration.

Reuses the exact color scheme and floor(Z)-relative thresholds from
visualization/overlay.py (quality.noise_floor.classify), so a cell's color
and a point's color in the other view mean the same thing -- just averaged
over an area instead of shown per-point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2

from quality.noise_floor import M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER
from visualization.overlay import _COLOR_BGR


_DEFAULT_GRID_ROWS = 6
_DEFAULT_GRID_COLS = 8
_DEFAULT_MIN_POINTS_PER_CELL = 3
_DEFAULT_ALPHA = 0.45


@dataclass
class ErrorGridResult:
    """Per-cell aggregation of M2 point errors over an image grid.
    mean_err_px / counts are (grid_rows, grid_cols); cells with fewer than
    min_points_per_cell points are NaN in mean_err_px (excluded rather
    than shown with a noisy single-point color)."""
    mean_err_px: np.ndarray   # (grid_rows, grid_cols), NaN where excluded
    counts: np.ndarray        # (grid_rows, grid_cols), int
    grid_rows: int
    grid_cols: int
    cell_height: float
    cell_width: float
    floor_px: float
    min_points_per_cell: int
    num_points: int
    num_populated_cells: int


def compute_error_grid(
    image_height: int,
    image_width: int,
    edge_pixels: np.ndarray,
    errors_px: np.ndarray,
    floor_px: float,
    grid_rows: int = _DEFAULT_GRID_ROWS,
    grid_cols: int = _DEFAULT_GRID_COLS,
    min_points_per_cell: int = _DEFAULT_MIN_POINTS_PER_CELL,
) -> ErrorGridResult:
    """
    Bin edge_pixels/errors_px into a (grid_rows, grid_cols) grid over the
    image and average the error within each cell. Points outside the
    image bounds are dropped (shouldn't normally occur -- M2 only keeps
    in-bounds points -- but guarded here since this function may be
    called standalone).
    """
    edge_pixels = np.asarray(edge_pixels, dtype=np.float64)
    errors_px = np.asarray(errors_px, dtype=np.float64)

    cell_h = image_height / grid_rows
    cell_w = image_width / grid_cols
    n_cells = grid_rows * grid_cols

    mean_grid = np.full((grid_rows, grid_cols), np.nan)
    counts = np.zeros((grid_rows, grid_cols), dtype=int)

    if edge_pixels.shape[0] > 0:
        in_bounds = (
            (edge_pixels[:, 0] >= 0) & (edge_pixels[:, 0] < image_width) &
            (edge_pixels[:, 1] >= 0) & (edge_pixels[:, 1] < image_height)
        )
        if in_bounds.any():
            px = edge_pixels[in_bounds]
            err = errors_px[in_bounds]
            col_idx = np.clip((px[:, 0] / cell_w).astype(int), 0, grid_cols - 1)
            row_idx = np.clip((px[:, 1] / cell_h).astype(int), 0, grid_rows - 1)
            flat_idx = row_idx * grid_cols + col_idx

            # Vectorized per-cell sum/count via np.bincount -- the same
            # vectorization pattern evaluation.edge_alignment's
            # extract_lidar_edge_points already uses (np.maximum.at/
            # np.minimum.at there; np.bincount here, same idea: replace a
            # Python-level loop that re-masks the full point array on
            # every iteration with one C-level reduction). Turns this
            # from O(grid_rows * grid_cols * N) into O(N + grid_rows *
            # grid_cols), which only matters at finer grid resolutions
            # than the default 6x8, but costs nothing extra either way.
            flat_counts = np.bincount(flat_idx, minlength=n_cells)
            flat_sums = np.bincount(flat_idx, weights=err, minlength=n_cells)

            counts = flat_counts.reshape(grid_rows, grid_cols)
            with np.errstate(invalid="ignore", divide="ignore"):
                flat_means = flat_sums / flat_counts
            flat_means[flat_counts < min_points_per_cell] = np.nan
            mean_grid = flat_means.reshape(grid_rows, grid_cols)

    return ErrorGridResult(
        mean_err_px=mean_grid,
        counts=counts,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        cell_height=cell_h,
        cell_width=cell_w,
        floor_px=floor_px,
        min_points_per_cell=min_points_per_cell,
        num_points=int(edge_pixels.shape[0]),
        num_populated_cells=int(np.count_nonzero(~np.isnan(mean_grid))),
    )


def _cell_color_bgr(ratio: float, good_mult: float, warning_mult: float) -> np.ndarray:
    """
    Continuous GOOD -> WARNING -> BAD color for a floor(Z)-normalized
    error ratio, using the exact same color stops as overlay.py's
    per-point classification (so a cell's color at the classification
    boundaries matches what a point would be colored there). Ratios
    beyond 2x the warning threshold clip to solid BAD -- there's no
    additional signal in distinguishing "very bad" from "extremely bad".
    """
    good = np.array(_COLOR_BGR["GOOD"], dtype=np.float64)
    warn = np.array(_COLOR_BGR["WARNING"], dtype=np.float64)
    bad = np.array(_COLOR_BGR["BAD"], dtype=np.float64)

    if ratio <= good_mult:
        return good
    if ratio <= warning_mult:
        t = (ratio - good_mult) / (warning_mult - good_mult)
        return good * (1 - t) + warn * t
    cap = warning_mult * 2.0
    t = min((ratio - warning_mult) / max(cap - warning_mult, 1e-9), 1.0)
    return warn * (1 - t) + bad * t


def render_error_heatmap(
    image: np.ndarray,
    grid: ErrorGridResult,
    alpha: float = _DEFAULT_ALPHA,
    good_mult: float = M2_GOOD_MULTIPLIER,
    warning_mult: float = M2_WARNING_MULTIPLIER,
    draw_grid_lines: bool = True,
) -> Optional[np.ndarray]:
    """
    Blend a translucent GOOD/WARNING/BAD-scaled color over each populated
    grid cell. Cells with too few points (NaN in grid.mean_err_px) are
    left untouched -- no color, no claim about calibration quality there.

    Returns None if there are no populated cells or floor_px is missing/
    non-positive (nothing meaningful to normalize against).
    """
    if grid.num_populated_cells == 0 or not grid.floor_px or grid.floor_px <= 0:
        return None

    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    overlay = canvas.copy()

    for r in range(grid.grid_rows):
        for c in range(grid.grid_cols):
            err = grid.mean_err_px[r, c]
            if np.isnan(err):
                continue
            ratio = err / grid.floor_px
            color = _cell_color_bgr(ratio, good_mult, warning_mult)
            x0, y0 = int(round(c * grid.cell_width)), int(round(r * grid.cell_height))
            x1, y1 = int(round((c + 1) * grid.cell_width)), int(round((r + 1) * grid.cell_height))
            cv2.rectangle(overlay, (x0, y0), (x1, y1), tuple(color.tolist()), thickness=-1)

    blended = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

    if draw_grid_lines:
        h, w = canvas.shape[:2]
        line_color = (60, 60, 60)
        for r in range(1, grid.grid_rows):
            y = int(round(r * grid.cell_height))
            cv2.line(blended, (0, y), (w, y), line_color, 1, lineType=cv2.LINE_AA)
        for c in range(1, grid.grid_cols):
            x = int(round(c * grid.cell_width))
            cv2.line(blended, (x, 0), (x, h), line_color, 1, lineType=cv2.LINE_AA)

    return blended


def render_error_heatmap_from_result(
    image: np.ndarray,
    edge_alignment_result,
    grid_rows: int = _DEFAULT_GRID_ROWS,
    grid_cols: int = _DEFAULT_GRID_COLS,
    min_points_per_cell: int = _DEFAULT_MIN_POINTS_PER_CELL,
    **render_kwargs,
) -> Optional[np.ndarray]:
    """
    Convenience wrapper mirroring visualization.overlay.render_overlay_from_result:
    takes an EdgeAlignmentResult directly. Returns None if the result
    FAILed (no per-point data), or if too few points land in any single
    cell to say anything meaningful about spatial pattern.
    """
    if edge_alignment_result.classification == "FAIL" or edge_alignment_result.edge_point_pixels is None:
        return None
    h, w = image.shape[:2]
    grid = compute_error_grid(
        h, w,
        edge_alignment_result.edge_point_pixels,
        edge_alignment_result.edge_point_errors_px,
        edge_alignment_result.floor_px,
        grid_rows=grid_rows, grid_cols=grid_cols, min_points_per_cell=min_points_per_cell,
    )
    return render_error_heatmap(image, grid, **render_kwargs)
