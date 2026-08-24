"""
report/json.py

Serializes a report dict (from report/builder.py) to JSON.

Enforces strict JSON validity: Python's json module allows NaN/Infinity as
non-standard extensions by default, which most JSON parsers (including
JavaScript's JSON.parse) reject. report/builder.py already sanitizes NaN/Inf
to None at the point each value is produced, but to_json_string still runs
with allow_nan=False as a hard backstop -- if a NaN slips through some path
that forgot to sanitize, this raises loudly instead of silently emitting
invalid JSON.
"""

from __future__ import annotations

import json as _json
from pathlib import Path


def to_json_string(report: dict, indent: int = 2) -> str:
    """Serialize a report dict to a JSON string. Raises ValueError if the
    report contains any non-finite float (NaN/Inf) that wasn't sanitized
    upstream -- see report/builder.py's _num()."""
    try:
        return _json.dumps(report, indent=indent, allow_nan=False, ensure_ascii=False)
    except ValueError as e:
        raise ValueError(
            f"Report contains a non-JSON-safe value (likely an un-sanitized "
            f"NaN/Infinity float). This indicates a bug in report/builder.py's "
            f"sanitization, not a valid evaluation result. Original error: {e}"
        ) from e


def write_json_report(report: dict, path: str) -> None:
    """Write a report dict to `path` as JSON."""
    content = to_json_string(report)
    Path(path).write_text(content, encoding="utf-8")
