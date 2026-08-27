"""
report/html.py

Renders a report dict (from report/builder.py) as a single self-contained
static HTML document -- an "instrument panel" for an existing calibration:
one glanceable Overall Quality reading up top, then Geometry/Generalization/
Stability category tiles, then per-metric detail tables, then any warnings.

Design intent (see design tokens below): this is a measurement instrument's
readout, not a marketing page. Dark console background, semantic
GOOD/WARNING/BAD/FAIL color coding used consistently everywhere (badges,
the overall-score gauge ring, category tiles), and monospace type for every
numeric readout to reinforce that these are precise sensor measurements.
No JavaScript is required for the core report -- the overall-score ring
is pure CSS (conic-gradient), so the file works as a plain
double-clickable static document with no server and no network dependency
beyond optional web fonts (which degrade gracefully to system fonts if
unavailable). The one opt-in exception is the interactive 3D viewer: when
an interactive scene is supplied, this module inlines the vendored
plotly.js gl3d bundle (report/vendor/, no CDN) plus the scene's JSON data
directly into the document, so the file still works fully offline -- just
no longer JS-free in that case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from html import escape
from functools import lru_cache
import base64
import json
import os


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "GOOD": "#3FB950",
    "WARNING": "#D29922",
    "BAD": "#F85149",
    "FAIL": "#6E7681",
}

_CSS = """
:root {
  --bg: #0D1117;
  --surface: #161B22;
  --surface-alt: #1C2333;
  --border: #2A3244;
  --text-primary: #E6EDF3;
  --text-secondary: #8B98AC;
  --accent: #7DD3FC;
  --good: #3FB950;
  --warning: #D29922;
  --bad: #F85149;
  --fail: #6E7681;
  --font-display: 'Space Grotesk', 'Segoe UI', system-ui, sans-serif;
  --font-body: 'Inter', 'Segoe UI', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text-primary);
  font-family: var(--font-body);
  line-height: 1.5;
  padding: 0 0 4rem 0;
}

@media (prefers-reduced-motion: no-preference) {
  body { animation: fade-in 0.4s ease-out; }
}
@keyframes fade-in { from { opacity: 0; } to { opacity: 1; } }

.container {
  max-width: 960px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem;
}

header.report-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 0.5rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 1.25rem;
  margin-bottom: 2rem;
}

header.report-header h1 {
  font-family: var(--font-display);
  font-size: clamp(1.4rem, 2.5vw, 1.9rem);
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 0;
}

header.report-header .meta {
  font-family: var(--font-mono);
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.hero {
  display: flex;
  align-items: center;
  gap: 2.5rem;
  flex-wrap: wrap;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2rem;
  margin-bottom: 2rem;
}

.gauge {
  width: 148px;
  height: 148px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  background: conic-gradient(var(--gauge-color) calc(var(--gauge-pct) * 1%), var(--surface-alt) 0);
}

.gauge-inner {
  width: 116px;
  height: 116px;
  border-radius: 50%;
  background: var(--surface);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.gauge-inner .score {
  font-family: var(--font-mono);
  font-size: 2rem;
  font-weight: 700;
  line-height: 1;
}

.gauge-inner .score-label {
  font-size: 0.65rem;
  color: var(--text-secondary);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-top: 0.2rem;
}

.hero-text h2 {
  font-family: var(--font-display);
  font-size: 1.1rem;
  margin: 0 0 0.4rem 0;
  color: var(--text-secondary);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.hero-text .verdict {
  font-family: var(--font-display);
  font-size: 1.6rem;
  font-weight: 600;
  margin: 0;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.2rem 0.65rem;
  border-radius: 999px;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  border: 1px solid currentColor;
}
.badge::before {
  content: '';
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
}

.category-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
  margin-bottom: 2.5rem;
}

.category-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem;
  border-top: 3px solid var(--card-color, var(--border));
}

.category-card .cat-label {
  font-size: 0.75rem;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 0.3rem;
}

.category-card .cat-metric {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--text-secondary);
  margin-bottom: 0.6rem;
}

.category-card .cat-score {
  font-family: var(--font-mono);
  font-size: 1.9rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
}

section.metric-section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem 1.75rem;
  margin-bottom: 1.5rem;
}

section.metric-section h3 {
  font-family: var(--font-display);
  font-size: 1.05rem;
  margin: 0 0 0.2rem 0;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

section.metric-section .metric-subtitle {
  color: var(--text-secondary);
  font-size: 0.85rem;
  margin: 0 0 1rem 0;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 0.75rem;
  margin-bottom: 1rem;
}

.stat {
  background: var(--surface-alt);
  border-radius: 8px;
  padding: 0.6rem 0.8rem;
}

.stat .stat-label {
  font-size: 0.68rem;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.stat .stat-value {
  font-family: var(--font-mono);
  font-size: 1.1rem;
  font-weight: 600;
  margin-top: 0.15rem;
}

table.data-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--font-mono);
  font-size: 0.82rem;
  margin-top: 0.5rem;
}

table.data-table th, table.data-table td {
  text-align: left;
  padding: 0.45rem 0.6rem;
  border-bottom: 1px solid var(--border);
}

table.data-table th {
  color: var(--text-secondary);
  font-weight: 500;
  text-transform: uppercase;
  font-size: 0.68rem;
  letter-spacing: 0.05em;
}

.warnings-section {
  border-radius: 12px;
  overflow: hidden;
}

.warning-item {
  background: var(--surface);
  border-left: 3px solid var(--warning);
  padding: 0.7rem 1rem;
  margin-bottom: 0.5rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
  border-radius: 0 8px 8px 0;
}

.matrix {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 2px;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  max-width: 320px;
  margin-top: 0.4rem;
}
.matrix span {
  background: var(--surface-alt);
  padding: 0.3rem 0.4rem;
  border-radius: 4px;
  text-align: right;
}

footer {
  text-align: center;
  color: var(--text-secondary);
  font-size: 0.75rem;
  font-family: var(--font-mono);
  margin-top: 2.5rem;
}

.view-toggle {
  display: inline-flex;
  gap: 0.25rem;
  background: var(--surface-alt);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.2rem;
  margin-top: 0.6rem;
}
.view-toggle-btn {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--text-secondary);
  background: transparent;
  border: none;
  border-radius: 6px;
  padding: 0.35rem 0.8rem;
  cursor: pointer;
}
.view-toggle-btn.active {
  background: var(--surface);
  color: var(--text-primary);
}
.view-panel {
  display: none;
}
.view-panel.active {
  display: block;
}
"""

_FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&'
    'family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;600;700&display=swap" '
    'rel="stylesheet">'
)


def _color_for(classification: str) -> str:
    return _STATUS_COLORS.get(classification, _STATUS_COLORS["FAIL"])


def _img_tag(image_bytes: Optional[bytes], alt: str, mime: str = "image/png") -> str:
    """Embed an image (PNG or GIF) as a base64 data URI so the HTML report
    stays a single self-contained file (no sibling image files to lose
    track of when shared). Returns an empty string if image_bytes is
    None, so callers can unconditionally splice this into a template
    without an if/else."""
    if not image_bytes:
        return ""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return (
        f'<img src="data:{mime};base64,{b64}" alt="{escape(alt)}" '
        f'style="width:100%; border-radius:10px; margin-top:0.75rem; display:block;">'
    )


def _badge(classification: str) -> str:
    color = _color_for(classification)
    return f'<span class="badge" style="color:{color}">{escape(classification)}</span>'


def _fmt(value, unit: str = "", digits: int = 3) -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return f"{value}{unit}"
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return escape(str(value))


def _stat(label: str, value: str) -> str:
    return (
        f'<div class="stat"><div class="stat-label">{escape(label)}</div>'
        f'<div class="stat-value">{value}</div></div>'
    )


def _render_hero(quality: dict) -> str:
    score = quality["overall_score"]
    classification = quality["overall_classification"]
    color = _color_for(classification)
    pct = max(0.0, min(100.0, score)) if score is not None else 0.0
    score_display = f"{score:.1f}" if score is not None else "&mdash;"

    verdict_text = {
        "GOOD": "Calibration looks solid across all measured metrics.",
        "WARNING": "Calibration shows some inconsistency worth reviewing.",
        "BAD": "Calibration does not hold up well under evaluation.",
        "FAIL": "Not enough valid data to assess this calibration.",
    }.get(classification, "")

    return f"""
    <div class="hero">
      <div class="gauge" style="--gauge-pct:{pct}; --gauge-color:{color};">
        <div class="gauge-inner">
          <div class="score" style="color:{color}">{score_display}</div>
          <div class="score-label">/ 100</div>
        </div>
      </div>
      <div class="hero-text">
        <h2>Overall Calibration Quality</h2>
        <p class="verdict">{_badge(classification)}</p>
        <p style="color:var(--text-secondary); margin-top:0.6rem; max-width:32rem;">{escape(verdict_text)}</p>
      </div>
    </div>
    """


def _render_categories(quality: dict) -> str:
    labels = {
        "geometry": ("Geometry", "How precisely LiDAR structure lines up with image edges, right now."),
        "generalization": ("Generalization", "Whether this calibration holds up across different time windows."),
        "stability": ("Stability", "Whether error stays consistent frame-to-frame, or spikes unpredictably."),
    }
    cards = []
    for cat in quality["categories"]:
        name = cat["name"]
        title, subtitle = labels.get(name, (name.title(), ""))
        color = _color_for(cat["classification"])
        score_display = f"{cat['score']:.1f}" if cat["score"] is not None else "&mdash;"
        cards.append(f"""
        <div class="category-card" style="--card-color:{color}">
          <div class="cat-label">{escape(title)}</div>
          <div class="cat-metric">{escape(cat["metric"])} &middot; {escape(subtitle)}</div>
          <div class="cat-score" style="color:{color}">{score_display}</div>
          {_badge(cat["classification"])}
        </div>
        """)
    return f'<div class="category-grid">{"".join(cards)}</div>'


def _render_m2(
    m2: dict,
    overlay_png: Optional[bytes] = None,
    histogram_png: Optional[bytes] = None,
    colorized_pointcloud_png: Optional[bytes] = None,
    error_heatmap_png: Optional[bytes] = None,
    bev_dual_panel_png: Optional[bytes] = None,
) -> str:
    return f"""
    <section class="metric-section">
      <h3>M2 &middot; Edge Alignment {_badge(m2["classification"])}</h3>
      <p class="metric-subtitle">How closely projected LiDAR depth-discontinuity points land on actual image edges.</p>
      <div class="stat-grid">
        {_stat("Mean error", _fmt(m2["mean_px"], " px"))}
        {_stat("Median error", _fmt(m2["median_px"], " px"))}
        {_stat("P95 error", _fmt(m2["p95_px"], " px"))}
        {_stat("Max error", _fmt(m2["max_px"], " px"))}
        {_stat("Noise floor", _fmt(m2["floor_px"], " px"))}
        {_stat("Edge points", _fmt(m2["num_edge_points"]))}
      </div>
      {_img_tag(overlay_png, "Projected LiDAR edge points over the camera image, colored GOOD/WARNING/BAD")}
      {_img_tag(histogram_png, "Histogram of per-point alignment error")}
      {_render_warning_list(m2["warnings"])}
      <p class="metric-subtitle" style="margin-top:1.25rem;">Spatial error heatmap: image split into a grid, with each cell's average error shown as a translucent GOOD/WARNING/BAD color. Errors concentrated at edges/corners or one side of the frame point at a specific cause (e.g. distortion, a small rotation offset) rather than uniform miscalibration.</p>
      {_img_tag(error_heatmap_png, "Grid heatmap of spatially-aggregated alignment error, colored GOOD/WARNING/BAD")}
      <p class="metric-subtitle" style="margin-top:1.25rem;">Bird's-eye view: the same edge points shown in both the camera image and a top-down view (X vs depth), colored identically in each. Makes it easy to see whether error grows with distance or concentrates on one side.</p>
      {_img_tag(bev_dual_panel_png, "Camera image and bird's-eye view side by side, with the same edge points highlighted and colored to match in both")}
      <p class="metric-subtitle" style="margin-top:1.25rem;">Fused view: LiDAR points colorized by the camera pixel they project onto. Color bleed or smearing at object edges is a visual sign of extrinsic misalignment.</p>
      {_img_tag(colorized_pointcloud_png, "LiDAR point cloud colorized by projected camera pixel, shown from a 3D angle and bird's-eye view")}
    </section>
    """


def _render_m3(m3: dict) -> str:
    rows = "".join(
        f'<tr><td>{b["block_index"]}</td><td>{b["num_frames_valid"]}/{b["num_frames_total"]}</td>'
        f'<td>{_fmt(b["mean_px"], " px")}</td><td>{_fmt(b["p95_px"], " px")}</td>'
        f'<td>{_badge(b["classification"])}</td></tr>'
        for b in m3["blocks"]
    )
    table = f"""
    <table class="data-table">
      <thead><tr><th>Block</th><th>Frames</th><th>Mean</th><th>P95</th><th>Status</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """ if m3["blocks"] else "<p style='color:var(--text-secondary)'>No blocks evaluated.</p>"

    return f"""
    <section class="metric-section">
      <h3>M3 &middot; Hold-out Consistency {_badge(m3["classification"])}</h3>
      <p class="metric-subtitle">Whether this fixed calibration performs consistently across different contiguous time windows.</p>
      <div class="stat-grid">
        {_stat("Mean across blocks", _fmt(m3["mean_across_blocks_px"], " px"))}
        {_stat("STD across blocks", _fmt(m3["std_across_blocks_px"], " px"))}
        {_stat("Range", _fmt(m3["range_px"], " px"))}
        {_stat("Noise floor", _fmt(m3["floor_px"], " px"))}
        {_stat("Valid blocks", _fmt(m3["num_valid_blocks"]))}
      </div>
      {table}
      {_render_warning_list(m3["warnings"])}
    </section>
    """


def _render_m4(m4: dict, trajectory_png: Optional[bytes] = None) -> str:
    trajectory = m4["frame_trajectory"]
    outlier_idx = set(m4["outlier_frame_indices"])
    # Keep the table readable on long sequences: show all outliers plus a
    # bounded sample of the rest, rather than every frame.
    sample = [f for f in trajectory if f["frame_index"] in outlier_idx]
    non_outliers = [f for f in trajectory if f["frame_index"] not in outlier_idx]
    sample += non_outliers[:5]
    if len(non_outliers) > 5:
        sample += non_outliers[-5:]
    sample.sort(key=lambda f: f["frame_index"])

    rows = "".join(
        f'<tr style="{"color:var(--bad)" if f["is_outlier"] else ""}">'
        f'<td>{f["frame_index"]}</td><td>{_fmt(f["mean_px"], " px")}</td>'
        f'<td>{_badge(f["classification"])}</td>'
        f'<td>{"outlier" if f["is_outlier"] else ""}</td></tr>'
        for f in sample
    )
    note = (
        f"<p style='color:var(--text-secondary); font-size:0.8rem;'>"
        f"Showing {len(sample)} of {len(trajectory)} frames (all outliers, plus a sample).</p>"
        if len(sample) < len(trajectory) else ""
    )
    table = f"""
    <table class="data-table">
      <thead><tr><th>Frame</th><th>Mean</th><th>Status</th><th></th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    {note}
    """ if trajectory else "<p style='color:var(--text-secondary)'>No frames evaluated.</p>"

    return f"""
    <section class="metric-section">
      <h3>M4 &middot; Multi-frame Consistency {_badge(m4["classification"])}</h3>
      <p class="metric-subtitle">Whether error stays stable frame-to-frame, or specific frames spike.</p>
      <div class="stat-grid">
        {_stat("Mean", _fmt(m4["mean_across_frames_px"], " px"))}
        {_stat("STD", _fmt(m4["std_across_frames_px"], " px"))}
        {_stat("P95", _fmt(m4["p95_across_frames_px"], " px"))}
        {_stat("Max", _fmt(m4["max_across_frames_px"], " px"))}
        {_stat("Outlier frames", _fmt(m4["num_outlier_frames"]))}
        {_stat("Frames evaluated", f'{m4["num_valid_frames"]}/{m4["num_frames_total"]}')}
      </div>
      {table}
      {_img_tag(trajectory_png, "Per-frame error trajectory with outliers marked")}
      {_render_warning_list(m4["warnings"])}
    </section>
    """


def _render_sequence_gif(gif_bytes: Optional[bytes]) -> str:
    """Render the (opt-in, --sequence-gif) animated overlay section.
    Returns "" if no GIF was generated, so callers can splice this in
    unconditionally."""
    if not gif_bytes:
        return ""
    return f"""
    <section class="metric-section">
      <h3>Sequence Overlay</h3>
      <p class="metric-subtitle">M2's overlay, sampled across the sequence and animated -- shows whether alignment quality holds steady over time or drifts, rather than a single snapshot.</p>
      {_img_tag(gif_bytes, "Animated GIF of the M2 overlay across sampled frames in the sequence", mime="image/gif")}
    </section>
    """


def _render_warning_list(warnings: list[str]) -> str:
    if not warnings:
        return ""
    items = "".join(f'<div class="warning-item">{escape(w)}</div>' for w in warnings)
    return f'<div style="margin-top:1rem;">{items}</div>'


_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")


@lru_cache(maxsize=1)
def _load_plotly_js() -> str:
    """Read the vendored plotly.js gl3d bundle off disk (see
    report/vendor/README.md). Cached so repeated report generation in a
    single process only pays the ~1.7MB read once."""
    path = os.path.join(_VENDOR_DIR, "plotly-gl3d.min.js")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _render_rig_geometry(
    metadata: dict,
    frustum_png: Optional[bytes] = None,
    interactive_scene: Optional[dict] = None,
    div_id: str = "cam-lidar-interactive-3d",
) -> str:
    """
    Render the top-of-report "Rig Geometry" section. If both a static
    frustum PNG and interactive scene data are available, shows a
    Static/Interactive toggle (defaulting to Static, since it costs
    nothing to display) with the interactive Plotly scene lazily
    initialized on first switch -- not on page load -- so opening the
    report doesn't pay Plotly's render cost unless someone actually asks
    for it. Falls back to whichever single view is available, or "" if
    neither is.
    """
    if not frustum_png and not interactive_scene:
        return ""

    ext = metadata["extrinsic"]
    stat_grid = f"""
      <div class="stat-grid">
        {_stat("Baseline", _fmt(ext["baseline_m"], " m"))}
        {_stat("Parent &rarr; child", f'{escape(ext["parent"])} &rarr; {escape(ext["child"])}')}
      </div>
    """

    if frustum_png and interactive_scene:
        scene_json = json.dumps(interactive_scene, separators=(",", ":"))
        body = f"""
      <div class="view-toggle">
        <button type="button" class="view-toggle-btn active" data-target="rig-static-view">Static</button>
        <button type="button" class="view-toggle-btn" data-target="rig-interactive-view">Interactive</button>
      </div>
      <div id="rig-static-view" class="view-panel active">
        {_img_tag(frustum_png, "3D view of the camera's position and viewing frustum in the LiDAR coordinate frame")}
      </div>
      <div id="rig-interactive-view" class="view-panel">
        <div id="{div_id}" style="width:100%; height:520px; margin-top:0.75rem; border-radius:10px; overflow:hidden;"></div>
      </div>
      <script>
        (function() {{
          var scene = {scene_json};
          var rendered = false;
          var panels = document.querySelectorAll('#rig-geometry-section .view-panel');
          var buttons = document.querySelectorAll('#rig-geometry-section .view-toggle-btn');
          buttons.forEach(function(btn) {{
            btn.addEventListener('click', function() {{
              var target = btn.getAttribute('data-target');
              panels.forEach(function(p) {{ p.classList.toggle('active', p.id === target); }});
              buttons.forEach(function(b) {{ b.classList.toggle('active', b === btn); }});
              if (target === 'rig-interactive-view' && !rendered) {{
                Plotly.newPlot({div_id!r}, scene.data, scene.layout, scene.config);
                rendered = true;
              }}
            }});
          }});
        }})();
      </script>
    """
    elif interactive_scene:
        scene_json = json.dumps(interactive_scene, separators=(",", ":"))
        body = f"""
      <div id="{div_id}" style="width:100%; height:520px; margin-top:0.75rem; border-radius:10px; overflow:hidden;"></div>
      <script>
        (function() {{
          var scene = {scene_json};
          Plotly.newPlot({div_id!r}, scene.data, scene.layout, scene.config);
        }})();
      </script>
    """
    else:
        body = _img_tag(frustum_png, "3D view of the camera's position and viewing frustum in the LiDAR coordinate frame")

    return f"""
    <section class="metric-section" id="rig-geometry-section">
      <h3>Rig Geometry</h3>
      <p class="metric-subtitle">Camera position and viewing frustum placed in the LiDAR frame, from the extrinsic under evaluation &mdash; a physical sanity check that's easier to read at a glance than raw translation/rotation numbers. The interactive view also overlays the colorized point cloud; drag to orbit, scroll to zoom.</p>
      {stat_grid}
      {body}
    </section>
    """


def _render_metadata(metadata: dict) -> str:
    cam = metadata["camera"]
    T = metadata["extrinsic"]["T_CL"]
    matrix_cells = "".join(f"<span>{v:.4f}</span>" for row in T for v in row)
    return f"""
    <section class="metric-section">
      <h3>Configuration</h3>
      <div class="stat-grid">
        {_stat("Camera model", escape(cam["model"]))}
        {_stat("Resolution", f'{cam["width"]}&times;{cam["height"]}')}
        {_stat("fx / fy", f'{cam["fx"]:.1f} / {cam["fy"]:.1f}')}
        {_stat("LiDAR source", escape(metadata["lidar"]["source_kind"]))}
        {_stat("Baseline", _fmt(metadata["extrinsic"]["baseline_m"], " m"))}
        {_stat("Synced frames", _fmt(metadata["dataset"]["num_synced_frames"]))}
      </div>
      <div class="stat-label" style="margin-top:0.8rem; margin-bottom:0.2rem;">T_CL (camera_from_lidar)</div>
      <div class="matrix">{matrix_cells}</div>
    </section>
    """


def _render_advanced(advanced: Optional[dict]) -> str:
    if not advanced:
        return ""
    parts = []

    plane = advanced.get("plane_consistency")
    if plane:
        parts.append(f"""
        <section class="metric-section">
          <h3>Plane Consistency {_badge(plane["classification"])}</h3>
          <p class="metric-subtitle">Advanced diagnostic: how well the dominant flat surface (ground/wall) lines up with its image silhouette.</p>
          <div class="stat-grid">
            {_stat("Plane found", _fmt(plane["plane_found"]))}
            {_stat("Inlier ratio", f'{plane["inlier_ratio"]*100:.1f}%' if plane["inlier_ratio"] is not None else "&mdash;")}
            {_stat("Boundary points", _fmt(plane["num_boundary_points"]))}
            {_stat("Mean error", _fmt(plane["mean_px"], " px"))}
          </div>
          {_render_warning_list(plane["warnings"])}
        </section>
        """)

    perturbation = advanced.get("perturbation")
    if perturbation:
        best = perturbation.get("best_sample")
        best_str = (f'{best["axis"]} {best["direction"]}{best["delta"]} &rarr; {best["mean_px"]:.3f} px'
                    if best else "&mdash;")
        parts.append(f"""
        <section class="metric-section">
          <h3>Perturbation Sensitivity {_badge(perturbation["classification"])}</h3>
          <p class="metric-subtitle">Advanced diagnostic: does a small nudge to T_CL find a better alignment nearby?</p>
          <div class="stat-grid">
            {_stat("Baseline", _fmt(perturbation["baseline_mean_px"], " px"))}
            {_stat("At local minimum", _fmt(perturbation["is_local_minimum"]))}
            {_stat("Improvement margin", _fmt(perturbation["improvement_margin_px"], " px"))}
            {_stat("Best nudge", best_str)}
          </div>
          {_render_warning_list(perturbation["warnings"])}
        </section>
        """)

    drift = advanced.get("temporal_drift")
    if drift:
        parts.append(f"""
        <section class="metric-section">
          <h3>Temporal Drift {_badge(drift["classification"])}</h3>
          <p class="metric-subtitle">Advanced diagnostic: does per-frame error trend up or down over the sequence?</p>
          <div class="stat-grid">
            {_stat("Slope", _fmt(drift["slope_px_per_frame"], " px/frame", digits=5))}
            {_stat("Significant", _fmt(drift["is_statistically_significant"]))}
            {_stat("p-value", _fmt(drift["p_value"], digits=4))}
            {_stat("Total drift", _fmt(drift["total_drift_px"], " px"))}
          </div>
          {_render_warning_list(drift["warnings"])}
        </section>
        """)

    if not parts:
        return ""
    return '<h2 style="font-family:var(--font-display); font-size:1rem; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.06em; margin: 2rem 0 1rem;">Advanced Diagnostics</h2>' + "".join(parts)


def render_html_report(report: dict, visuals: Optional[dict] = None) -> str:
    """
    Render a full self-contained HTML document string from a report dict
    built by report/builder.py.build_report().

    visuals: optional dict of PNG bytes to embed as base64 data URIs (so
    the HTML stays a single shareable file). Recognized keys:
      "overlay_png"              -- from visualization.overlay.render_overlay(...)
      "histogram_png"            -- from visualization.histogram.render_error_histogram_png(...)
      "trajectory_png"           -- from visualization.trajectory.render_m4_trajectory_png(...)
      "colorized_pointcloud_png" -- from visualization.colorized_pointcloud.render_colorized_pointcloud_from_frame(...)
      "error_heatmap_png"        -- from visualization.error_heatmap.render_error_heatmap_from_result(...)
      "camera_frustum_png"       -- from visualization.camera_frustum.render_camera_frustum_from_dataset(...)
      "bev_dual_panel_png"       -- from visualization.bev_dual_panel.render_bev_dual_panel_from_result(...)
      "interactive_scene"        -- a dict from visualization.interactive_viewer.build_interactive_scene(...)
                                     (NOT bytes -- raw JSON-serializable scene data, embedded + rendered
                                     client-side via the vendored plotly.js gl3d bundle)
      "sequence_gif"             -- GIF bytes from visualization.sequence.render_sequence_gif(...)
                                     (opt-in via app.cli's --sequence-gif; embedded as image/gif, not image/png)
    Any missing/None key simply omits that image -- visualization is
    optional and the report renders fine without it (see report/json.py's
    counterpart: the JSON report never carries images, only this HTML one).
    """
    visuals = visuals or {}
    metadata = report["metadata"]
    quality = report["quality_score"]

    body = (
        _render_hero(quality)
        + _render_categories(quality)
        + _render_rig_geometry(metadata, visuals.get("camera_frustum_png"), visuals.get("interactive_scene"))
        + _render_m2(report["m2_edge_alignment"], visuals.get("overlay_png"), visuals.get("histogram_png"),
                     visuals.get("colorized_pointcloud_png"), visuals.get("error_heatmap_png"),
                     visuals.get("bev_dual_panel_png"))
        + _render_m3(report["m3_holdout_consistency"])
        + _render_m4(report["m4_multiframe_consistency"], visuals.get("trajectory_png"))
        + _render_sequence_gif(visuals.get("sequence_gif"))
        + _render_advanced(report.get("advanced"))
        + _render_metadata(metadata)
        + _render_warning_list(report.get("warnings", []))
    )

    plotly_script_tag = f"<script>{_load_plotly_js()}</script>" if visuals.get("interactive_scene") else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cam-LiDAR Calibration Quality Report</title>
{_FONT_LINKS}
<style>{_CSS}</style>
{plotly_script_tag}
</head>
<body>
  <div class="container">
    <header class="report-header">
      <h1>Cam&ndash;LiDAR Calibration Quality</h1>
      <div class="meta">generated {escape(metadata["generated_at"])} &middot; v{escape(metadata["tool_version"])}</div>
    </header>
    {body}
    <footer>GT-free calibration evaluation &middot; not a substitute for target-based validation</footer>
  </div>
</body>
</html>"""


def write_html_report(report: dict, path: str, visuals: Optional[dict] = None) -> None:
    Path(path).write_text(render_html_report(report, visuals), encoding="utf-8")
