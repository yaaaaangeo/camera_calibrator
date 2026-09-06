from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_WORKSPACE = _ROOT / "ui" / "windshield_workspace.py"
_EVAL_WORKER = _ROOT / "ui" / "ghost_evaluation_worker.py"
_SUPPRESSION_WORKER = _ROOT / "ui" / "ghost_suppression_worker.py"
_REFLECTION_WORKER = _ROOT / "ui" / "reflection_worker.py"
_REFLECTION_SUPPRESSION_WORKER = _ROOT / "ui" / "reflection_suppression_worker.py"


def test_ghost_evaluation_worker_exists_and_is_separate_from_reflection():
    assert _EVAL_WORKER.exists(), "ui/ghost_evaluation_worker.py가 존재하지 않습니다."
    source = _EVAL_WORKER.read_text(encoding="utf-8")
    assert "class GhostEvaluationWorker" in source
    assert "evaluate_ghost_" in source
    assert "ReflectionEvaluationWorker" not in source
    assert "evaluate_reflection_dataset" not in source
    assert "WindshieldCalibrationWorker" not in source


def test_ghost_suppression_worker_exists_and_is_separate_from_reflection():
    assert _SUPPRESSION_WORKER.exists(), "ui/ghost_suppression_worker.py가 존재하지 않습니다."
    source = _SUPPRESSION_WORKER.read_text(encoding="utf-8")
    assert "class GhostSuppressionWorker" in source
    assert "suppress_ghost" in source
    assert "ReflectionSuppressionWorker" not in source
    assert "suppress_reflection" not in source
    assert "load_suppression_model" not in source


def test_reflection_workers_do_not_reference_ghost():
    reflection_source = _REFLECTION_WORKER.read_text(encoding="utf-8")
    reflection_suppression_source = _REFLECTION_SUPPRESSION_WORKER.read_text(encoding="utf-8")
    assert "Ghost" not in reflection_source
    assert "Ghost" not in reflection_suppression_source
    assert "ghost" not in reflection_source
    assert "ghost" not in reflection_suppression_source


def test_ghost_tab_is_separate_from_reflection_tab():
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "⑥ Ghost" in source
    assert "_build_ghost_tab" in source
    assert "GhostEvaluationWorker" in source
    assert "GhostSuppressionWorker" in source
    assert "⑤ Reflection" in source
    assert "_build_reflection_tab" in source


def test_ghost_tab_has_separate_evaluation_and_suppression_subtabs():
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "_build_ghost_evaluation_subtab" in source
    assert "_build_ghost_suppression_subtab" in source


def test_ghost_ui_never_merges_displacement_and_strength_into_one_map():
    """사용자 스펙 - "Displacement와 Strength를 한 map에 억지로 합치지
    않는다": vector field(displacement)와 heatmap(strength)는 별도의 위젯
    (테이블)이어야 한다."""
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "ghost_vector_field_table" in source
    assert "ghost_strength_heatmap_table" in source
    assert "ghost_vector_field_table = ghost_strength_heatmap_table" not in source


def test_ghost_ui_never_labels_general_mode_as_ground_truth():
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "Ghost Ground Truth" not in source
    assert "no-reference heuristic, not ground truth" in source


def test_ghost_export_state_is_separate_project_field():
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "_ghost_results" in source
    assert "ghost_results" in source
    assert "self._reflection_results,\n            self._ghost_results,\n        )" in source or (
        "self._reflection_results" in source and "self._ghost_results" in source
    )


def test_ghost_suppression_worker_never_imports_reflection_suppression_module():
    """사용자 스펙 - Ghost 제거에 Reflection Suppression 모델을 사용하지
    않는다: worker 파일이 reflection_suppression 모듈을 실제로 import하지
    않아야 한다(설명 docstring에서 그 이름을 언급하는 것은 허용)."""
    source = _SUPPRESSION_WORKER.read_text(encoding="utf-8")
    assert "import calibration.windshield.reflection_suppression" not in source
    assert "from calibration.windshield.reflection_suppression" not in source
