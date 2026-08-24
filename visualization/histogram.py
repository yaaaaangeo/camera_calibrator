"""
visualization/histogram.py

Renders a histogram of M2's per-point alignment errors (px), with vertical
reference lines at the GOOD/WARNING floor(Z) multiplier boundaries -- shows
the actual distribution behind the summary mean/median/P95 numbers (e.g.
whether a "good" mean is hiding a long bad tail).
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import matplotlib
matplotlib.use("Agg")
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
_ACCENT = "#7DD3FC"


def render_error_histogram_png(
    errors_px: np.ndarray,
    floor_px: float,
    good_mult: float = M2_GOOD_MULTIPLIER,
    warning_mult: float = M2_WARNING_MULTIPLIER,
    dpi: int = 130,
    bins: int = 40,
) -> Optional[bytes]:
    """Render a histogram of per-point px errors as a PNG (bytes). Returns
    None if errors_px is empty."""
    errors_px = np.asarray(errors_px, dtype=float)
    errors_px = errors_px[np.isfinite(errors_px)]
    if errors_px.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(8.2, 3.0), dpi=dpi)
    try:
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_SURFACE)

        ax.hist(errors_px, bins=bins, color=_ACCENT, alpha=0.85, zorder=3)

        if floor_px and np.isfinite(floor_px) and floor_px > 0:
            good_line = good_mult * floor_px
            warn_line = warning_mult * floor_px
            ax.axvline(good_line, color=_GOOD, linestyle="--", linewidth=1.2, zorder=4,
                       label=f"GOOD boundary ({good_line:.2f}px)")
            ax.axvline(warn_line, color=_BAD, linestyle="--", linewidth=1.2, zorder=4,
                       label=f"BAD boundary ({warn_line:.2f}px)")
            ax.legend(loc="upper right", frameon=False, labelcolor=_TEXT, fontsize=8)

        ax.set_xlabel("per-point error (px)", color=_TEXT, fontsize=9)
        ax.set_ylabel("point count", color=_TEXT, fontsize=9)
        ax.tick_params(colors=_TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.grid(True, axis="y", color=_GRID, linewidth=0.5, alpha=0.6)
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        return buf.getvalue()
    finally:
        # Always close, even if a plotting call above raises -- otherwise
        # the figure leaks in matplotlib's global pyplot state (it keeps
        # every un-closed figure alive until the process exits).
        plt.close(fig)
