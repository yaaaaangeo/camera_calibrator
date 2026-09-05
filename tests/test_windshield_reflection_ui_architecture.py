from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_WORKSPACE = _ROOT / "ui" / "windshield_workspace.py"
_WORKER = _ROOT / "ui" / "reflection_worker.py"
_SUPPRESSION_WORKER = _ROOT / "ui" / "reflection_suppression_worker.py"


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
        "Reflection Mean",
        "Reflection P95",
        "Reflection Coverage",
        "Reflection Likelihood",
        "Bottom Mean",
        "Contrast Retention",
        "Edge Retention",
        "Saturation Coverage",
        "Glare Coverage",
        "Glare Strength",
    ):
        assert label in source
    assert "no-reference heuristic, not ground truth" in source
    assert "GOOD / BAD" not in source


def test_reflection_export_state_is_separate_project_field():
    source = _WORKSPACE.read_text(encoding="utf-8")

    assert "_reflection_results" in source
    assert "reflection_results" in source
    assert "windshield_results, self._reflection_results" in source


# ---------------------------------------------------------------------------
# STEP 7 - Reflection Suppression은 Evaluation과 별도 sub-tab/worker다
# (사용자 스펙 0/49/57/58번).
# ---------------------------------------------------------------------------

def test_suppression_worker_exists_and_is_separate_from_evaluation_worker():
    assert _SUPPRESSION_WORKER.exists(), "ui/reflection_suppression_worker.py가 존재하지 않습니다."
    source = _SUPPRESSION_WORKER.read_text(encoding="utf-8")
    assert "class ReflectionSuppressionWorker" in source
    assert "suppress_reflection" in source
    assert "ReflectionEvaluationWorker" not in source
    assert "evaluate_reflection_dataset" not in source
    assert "WindshieldCalibrationWorker" not in source


def test_reflection_tab_has_separate_evaluation_and_suppression_subtabs():
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "_build_reflection_evaluation_subtab" in source
    assert "_build_reflection_suppression_subtab" in source
    assert '"Evaluation"' in source
    assert '"Suppression"' in source


def test_suppression_subtab_has_required_visualization_labels():
    """사용자 스펙 51번 - Original/Predicted Reflection/Reflection Mask/
    Suppressed 4개 시각화가 전부 있어야 한다."""
    source = _WORKSPACE.read_text(encoding="utf-8")
    for label in ("suppression_original_image_label", "suppression_reflection_image_label",
                  "suppression_alpha_image_label", "suppression_output_image_label"):
        assert label in source


def test_suppression_subtab_has_mode_presets_and_before_after_metrics():
    source = _WORKSPACE.read_text(encoding="utf-8")
    for widget in ("suppression_mode_conservative_radio", "suppression_mode_standard_radio", "suppression_mode_strong_radio"):
        assert widget in source
    assert "suppression_metrics_table" in source
    assert "Reflection Mean" in source and "Edge Retention" in source and "Over-suppression" in source


def test_suppression_ui_never_calls_training_functions_directly():
    """사용자 스펙 59번 - GUI 안에서 training loop를 직접 돌리지 않는다."""
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "train_suppression_model" not in source
    worker_source = _SUPPRESSION_WORKER.read_text(encoding="utf-8")
    assert "train_suppression_model" not in worker_source
