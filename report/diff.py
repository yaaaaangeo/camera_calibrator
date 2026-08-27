"""
report/diff.py

Compares two report dicts (as produced by report/builder.py -- typically
loaded back from a previous run's report.json) and computes what
changed: overall score/classification delta, and per-category
(geometry/generalization/stability) score/classification deltas.

Built for the "did this calibration change get better or worse than the
last CI run" question. app/cli.py's --compare-to flag is the primary
consumer, feeding the result into both the console summary and the
--format github-comment markdown output; --fail-on-regression uses
any_regressed to decide the process exit code.

Deliberately dict-in/dict-out (matching report/builder.py's own
plain-dict convention) rather than re-parsing dataclasses, since a
loaded report.json IS the plain-dict shape already -- no reason to
round-trip through anything richer.
"""

from __future__ import annotations

from typing import Optional


_CLASSIFICATION_RANK = {"GOOD": 0, "WARNING": 1, "BAD": 2, "FAIL": 3}


def _rank(classification: Optional[str]) -> int:
    """Higher = worse. Unknown/missing classifications are treated as
    worst-case (FAIL-equivalent) so a malformed/missing side of the
    comparison can't silently suppress a real regression."""
    return _CLASSIFICATION_RANK.get(classification, _CLASSIFICATION_RANK["FAIL"])


def _safe_sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def _category_entry(report: dict, name: str) -> Optional[dict]:
    for cat in report.get("quality_score", {}).get("categories", []):
        if cat.get("name") == name:
            return cat
    return None


def _level_diff(old_score, old_cls, new_score, new_cls) -> dict:
    old_r, new_r = _rank(old_cls), _rank(new_cls)
    same_rank_worse_score = (
        new_r == old_r and new_score is not None and old_score is not None and new_score < old_score
    )
    regressed = (new_r > old_r) or same_rank_worse_score
    return {
        "old_score": old_score, "new_score": new_score, "delta_score": _safe_sub(new_score, old_score),
        "old_classification": old_cls, "new_classification": new_cls,
        "regressed": bool(regressed),
    }


def compute_report_diff(old_report: dict, new_report: dict) -> dict:
    """
    Returns:
      {
        "overall": {old_score, new_score, delta_score,
                    old_classification, new_classification, regressed},
        "categories": {"geometry": {...same shape...}, "generalization": {...}, "stability": {...}},
        "any_regressed": bool,
      }

    "regressed" at a given level means classification got strictly worse
    (e.g. GOOD -> WARNING), OR classification held steady but the score
    dropped. A category missing on either side is treated as
    FAIL-equivalent for ranking (its score/classification come through
    as None rather than being silently skipped), so a category that
    newly failed -- or a category that used to fail and now doesn't --
    is still visible in the diff.
    """
    old_q = old_report.get("quality_score", {})
    new_q = new_report.get("quality_score", {})

    overall = _level_diff(
        old_q.get("overall_score"), old_q.get("overall_classification"),
        new_q.get("overall_score"), new_q.get("overall_classification"),
    )

    categories = {}
    any_regressed = overall["regressed"]
    for name in ("geometry", "generalization", "stability"):
        old_cat = _category_entry(old_report, name)
        new_cat = _category_entry(new_report, name)
        cat_diff = _level_diff(
            old_cat.get("score") if old_cat else None, old_cat.get("classification") if old_cat else None,
            new_cat.get("score") if new_cat else None, new_cat.get("classification") if new_cat else None,
        )
        categories[name] = cat_diff
        any_regressed = any_regressed or cat_diff["regressed"]

    return {"overall": overall, "categories": categories, "any_regressed": any_regressed}


def _fmt_score(score: Optional[float]) -> str:
    return f"{score:.1f}" if score is not None else "n/a"


def _fmt_delta(delta: Optional[float]) -> str:
    return f"{delta:+.1f}" if delta is not None else "n/a"


_CATEGORY_LABELS = {"geometry": "Geometry (M2)", "generalization": "Generalization (M3)", "stability": "Stability (M4)"}


def render_diff_console(diff: dict) -> str:
    """Plain-text summary for the CLI's console output, appended after
    the normal score summary when --compare-to is used."""
    o = diff["overall"]
    lines = ["", "  vs. previous run:"]
    lines.append(
        f"    Overall          : {_fmt_score(o['old_score'])} -> {_fmt_score(o['new_score'])} "
        f"({_fmt_delta(o['delta_score'])})  [{o['old_classification']} -> {o['new_classification']}]"
        f"{'  ** REGRESSED **' if o['regressed'] else ''}"
    )
    for name, c in diff["categories"].items():
        label = _CATEGORY_LABELS.get(name, name)
        lines.append(
            f"    {label:<17}: {_fmt_score(c['old_score'])} -> {_fmt_score(c['new_score'])} "
            f"({_fmt_delta(c['delta_score'])})  [{c['old_classification']} -> {c['new_classification']}]"
            f"{'  ** REGRESSED **' if c['regressed'] else ''}"
        )
    return "\n".join(lines)
