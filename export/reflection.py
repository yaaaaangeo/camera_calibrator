from __future__ import annotations

import dataclasses
import html
from pathlib import Path

import yaml

from calibration.json_utils import json_safe
from calibration.windshield.reflection.types import ReflectionDatasetResult, ReflectionEvaluationResult


def export_reflection_yaml(result: ReflectionEvaluationResult | ReflectionDatasetResult, path: str) -> str:
    payload = {
        "reflection_evaluation": json_safe(dataclasses.asdict(result)),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return path


def export_reflection_html(result: ReflectionEvaluationResult | ReflectionDatasetResult, path: str) -> str:
    payload = json_safe(dataclasses.asdict(result))
    title = "Reflection Dataset Evaluation" if isinstance(result, ReflectionDatasetResult) else "Reflection Evaluation"
    rows = _summary_rows(payload)
    body_rows = "\n".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in rows
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1 {{ font-size: 22px; margin: 0 0 16px; }}
    table {{ border-collapse: collapse; min-width: 460px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f7fa; width: 220px; }}
    .note {{ margin-top: 16px; color: #52606d; max-width: 720px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <table>{body_rows}</table>
  <p class="note">Photometric windshield reflection metrics are separate from geometry calibration. Severity is experimental and requires real-world threshold validation.</p>
</body>
</html>
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(document)
    return path


def _summary_rows(payload: dict) -> list[tuple[str, str]]:
    keys = [
        "mode",
        "metric_version",
        "pair_id",
        "mean_strength",
        "median_strength",
        "p95_strength",
        "coverage",
        "coverage_threshold",
        "severity_score",
        "worst_pair_id",
        "success",
        "warning_message",
        "error_message",
    ]
    rows = []
    for key in keys:
        if key in payload and payload[key] is not None:
            rows.append((key, _format_value(payload[key])))
    return rows


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
