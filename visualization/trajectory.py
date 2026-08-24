"""
visualization/trajectory.py

Renders the M4 (Multi-frame Consistency) per-frame error trajectory as a
line chart: frame index on the x-axis, mean per-frame px error on the
y-axis, GOOD/WARNING/BAD bands shaded using the same floor(Z) multiplier
thresholds M4 classifies against, and outlier frames marked distinctly.
This is what makes a single bad frame visible at a glance, versus reading
a table of numbers.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless: no display server needed, safe for CLI/report generation
import matplotlib.pyplot as plt
import numpy as np

from quality.noise_floor import M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER


_GOOD = "#3FB950"
_WARNING = "#D29922"
_BAD = "#F85149"
_BG = "#0D1117"
_SURFACE = "#161B22"
_TEXT = "#8B98AC"
_GRID = "#2A3244"


def render_m4_trajectory_png(m4_result, dpi: int = 130) -> Optional[bytes]:
    """
    Render the M4 frame-by-frame error trajectory as a PNG (bytes). Returns
    None if m4_result has no frame data to plot (e.g. it FAILed outright).

    Bands are drawn using M2's per-point multipliers (2x/5x) applied to
    m4_result.floor_px, since the y-axis here is per-frame MEAN px error
    (an M2-shaped quantity, not the STD that M4's own classification is
    based on) -- the bands answer "was this frame's calibration accuracy
    good", which is a different question from "is the sequence stable"
    that M4's overall classification answers.
    """
    frames = [f for f in m4_result.frame_results if f.classification != "FAIL"]
    if not frames:
        return None

    indices = np.array([f.frame_index for f in frames])
    values = np.array([f.mean_px for f in frames])
    is_outlier = np.array([f.is_outlier for f in frames])
    floor_px = m4_result.floor_px

    fig, ax = plt.subplots(figsize=(8.2, 3.2), dpi=dpi)
    try:
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_SURFACE)

        if floor_px and np.isfinite(floor_px) and floor_px > 0:
            good_line = M2_GOOD_MULTIPLIER * floor_px
            warn_line = M2_WARNING_MULTIPLIER * floor_px
            y_max = max(values.max(), warn_line) * 1.15
            ax.axhspan(0, good_line, color=_GOOD, alpha=0.08, zorder=0)
            ax.axhspan(good_line, warn_line, color=_WARNING, alpha=0.08, zorder=0)
            ax.axhspan(warn_line, y_max, color=_BAD, alpha=0.08, zorder=0)
            ax.set_ylim(0, y_max)

        ax.plot(indices, values, color=_TEXT, linewidth=1.2, zorder=2, alpha=0.85)
        ax.scatter(indices[~is_outlier], values[~is_outlier], color="#7DD3FC", s=14, zorder=3,
                   label="frame")
        if is_outlier.any():
            ax.scatter(indices[is_outlier], values[is_outlier], color=_BAD, s=32, zorder=4,
                       marker="x", linewidths=2, label="outlier")
            ax.legend(loc="upper left", frameon=False, labelcolor=_TEXT, fontsize=8)

        ax.set_xlabel("frame index", color=_TEXT, fontsize=9)
        ax.set_ylabel("mean error (px)", color=_TEXT, fontsize=9)
        ax.tick_params(colors=_TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        return buf.getvalue()
    finally:
        # Always close, even if a plotting call above raises -- otherwise
        # the figure leaks in matplotlib's global pyplot state.
        plt.close(fig)
