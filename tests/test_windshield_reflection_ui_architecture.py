from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_WORKSPACE = _ROOT / "ui" / "windshield_workspace.py"
_WORKER = _ROOT / "ui" / "reflection_worker.py"


def test_reflection_worker_exists_and_uses_evaluator_only():
    source = _WORKER.read_text(encoding="utf-8")

    assert "class ReflectionEvaluationWorker" in source
    assert "evaluate_reflection_dataset" in source
    assert "WindshieldCalibrationWorker" not in source
    assert "calibrate_" not in source


def test_reflection_tab_is_separate_from_geometry_comparison():
    source = _WORKSPACE.read_text(encoding="utf-8")

    assert "⑤ Reflection" in source
    assert "_build_reflection_tab" in source
    assert "ReflectionEvaluationWorker" in source
    assert "_build_comparison_tab" in source


def test_reflection_ui_keeps_raw_metric_labels_and_no_reference_wording():
    source = _WORKSPACE.read_text(encoding="utf-8")

    for label in (
        "Mean",
        "P95",
        "Coverage",
        "Bottom Mean",
        "Contrast Retention",
        "Edge Retention",
        "Saturation",
        "Glare",
    ):
        assert label in source
    assert "No-reference likelihood only" in source
    assert "GOOD / BAD" not in source


def test_reflection_export_state_is_separate_project_field():
    source = _WORKSPACE.read_text(encoding="utf-8")

    assert "_reflection_results" in source
    assert "reflection_results" in source
    assert "windshield_results, self._reflection_results" in source
