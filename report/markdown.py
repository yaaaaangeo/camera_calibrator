"""
report/markdown.py

Renders a report dict as GitHub-flavored markdown, sized and formatted
for pasting directly into a PR comment (e.g. via `gh pr comment --body-file`
or the GitHub API) -- the CI-facing counterpart to report/html.py's
full HTML report. Deliberately compact: a PR comment is skimmed, not
read like a document, so this leads with the one number that matters
(overall quality) and keeps everything else to a small table.

app/cli.py's --format github-comment writes this to stdout instead of
the normal console summary; report.json/report.html are still written
to disk exactly as usual either way.
"""

from __future__ import annotations

from typing import Optional


_EMOJI = {"GOOD": "\u2705", "WARNING": "\u26a0\ufe0f", "BAD": "\u274c", "FAIL": "\U0001F6AB"}
_CATEGORY_LABELS = {"geometry": "Geometry (M2)", "generalization": "Generalization (M3)", "stability": "Stability (M4)"}


def _emoji(classification: Optional[str]) -> str:
    return _EMOJI.get(classification, "\u2753")  # question mark for unknown/missing


def _fmt_score(score: Optional[float]) -> str:
    return f"{score:.1f}" if score is not None else "n/a"


def _fmt_delta(delta: Optional[float]) -> str:
    if delta is None:
        return ""
    sign = "+" if delta >= 0 else ""
    return f" ({sign}{delta:.1f})"


def render_github_comment(report: dict, diff: Optional[dict] = None) -> str:
    """
    Render `report` (and optionally a report/diff.py `compute_report_diff`
    result comparing it against a previous run) as GitHub-flavored
    markdown.
    """
    q = report["quality_score"]
    m0 = report.get("m0_sanity_gate")
    metadata = report["metadata"]

    overall_cls = q["overall_classification"]
    overall_score_str = _fmt_score(q["overall_score"])
    num_valid = q["num_valid_categories"]
    num_total = len(q["categories"])
    partial_note = f" ({num_valid}/{num_total} categories)" if num_valid < num_total else ""

    lines = [
        f"## {_emoji(overall_cls)} Cam-LiDAR Calibration Quality: {overall_cls} ({overall_score_str}/100){partial_note}",
        "",
    ]

    if m0 is not None:
        m0_emoji = "\u2705" if m0["passed"] else "\u274c"
        m0_label = "PASS" if m0["passed"] else "FAIL"
        lines.append(f"M0 Sanity Gate: {m0_emoji} {m0_label}")
        lines.append("")

    header = "| Category | Score | Status |"
    sep = "|---|---|---|"
    rows = []
    diff_categories = diff["categories"] if diff else {}
    for cat in q["categories"]:
        name = cat["name"]
        label = _CATEGORY_LABELS.get(name, name)
        score_str = _fmt_score(cat["score"])
        delta_str = _fmt_delta(diff_categories.get(name, {}).get("delta_score")) if diff else ""
        rows.append(f"| {label} | {score_str}{delta_str} | {_emoji(cat['classification'])} {cat['classification']} |")

    if diff:
        header = "| Category | Score | Δ vs previous | Status |"
        sep = "|---|---|---|---|"
        rows = []
        for cat in q["categories"]:
            name = cat["name"]
            label = _CATEGORY_LABELS.get(name, name)
            score_str = _fmt_score(cat["score"])
            d = diff_categories.get(name, {})
            delta_str = _fmt_delta(d.get("delta_score")).strip() or "n/a"
            regressed_marker = " \u26a0\ufe0f" if d.get("regressed") else ""
            rows.append(f"| {label} | {score_str} | {delta_str}{regressed_marker} | {_emoji(cat['classification'])} {cat['classification']} |")

    lines.append(header)
    lines.append(sep)
    lines.extend(rows)
    lines.append("")

    if diff:
        o = diff["overall"]
        regressed_marker = " \u26a0\ufe0f **regressed**" if o["regressed"] else ""
        lines.append(
            f"Overall vs. previous run: {_fmt_score(o['old_score'])} \u2192 {_fmt_score(o['new_score'])}"
            f"{_fmt_delta(o['delta_score'])}  ({o['old_classification']} \u2192 {o['new_classification']}){regressed_marker}"
        )
        lines.append("")

    n_warnings = len(report.get("warnings", []))
    if n_warnings:
        lines.append(f"<details><summary>{n_warnings} warning(s)</summary>\n")
        for w in report["warnings"]:
            lines.append(f"- {w}")
        lines.append("\n</details>")
        lines.append("")

    lines.append(f"*cam-lidar-eval v{metadata['tool_version']} &middot; generated {metadata['generated_at']}*")

    return "\n".join(lines)
