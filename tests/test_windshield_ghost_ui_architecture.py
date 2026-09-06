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


# ===========================================================================
# STEP 8 stabilization - Edge Target UI/Worker 완전 연결 (섹션 2)
# ===========================================================================

def test_edge_target_mode_is_no_longer_rejected_in_ui():
    """예전에는 Edge Target 선택 시 "API로 직접 호출하세요"라며 실행을
    막았다 - stabilization 이후에는 이 차단 코드가 사라져야 한다."""
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "API로 직접 호출하세요" not in source
    assert "evaluate_ghost_edge_target()을 스크립트" not in source


def test_ui_provides_edge_axis_selection():
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "ghost_edge_axis_combo" in source
    assert "Auto" in source


def test_evaluation_worker_uses_shared_dispatcher_not_manual_branching():
    """Evaluation Worker가 point/general만 처리하고 edge_target을 조용히
    point_source처럼 잘못 처리하던 문제(사용자 스펙 2-D번)를 막기 위해,
    이제는 공용 dispatcher(`evaluate_ghost_image`)를 통해 실행해야 한다."""
    source = _EVAL_WORKER.read_text(encoding="utf-8")
    assert "evaluate_ghost_image" in source


def test_evaluation_worker_reads_images_inside_worker_not_ui_thread():
    """이미지 파일 로딩(cv2.imread)이 Qt main thread가 아니라 Worker.run()
    이 호출하는 파이프라인 안에서 일어나야 한다(사용자 스펙 2-E번, Qt main
    thread에는 파일 선택/설정 읽기만 남긴다). Worker.run()은 파일 경로
    리스트를 그대로 `evaluate_ghost_dataset_from_paths()`(Qt 비의존 순수
    함수)에 위임하고, 그 함수가 실제 `cv2.imread`를 수행한다."""
    worker_source = _EVAL_WORKER.read_text(encoding="utf-8")
    assert "def run(self)" in worker_source
    run_body = worker_source.split("def run(self)")[1]
    assert "evaluate_ghost_dataset_from_paths(" in run_body

    evaluator_source = (_ROOT / "calibration" / "windshield" / "ghost" / "evaluator.py").read_text(encoding="utf-8")
    pipeline_body = evaluator_source.split("def evaluate_ghost_dataset_from_paths(")[1]
    assert "cv2.imread" in pipeline_body

    # UI 핸들러(_on_run_ghost_evaluation)는 이미지를 직접 읽지 않고 경로만 넘긴다.
    workspace_source = _WORKSPACE.read_text(encoding="utf-8")
    handler_body = workspace_source.split("def _on_run_ghost_evaluation(self)")[1].split("\n    def ")[0]
    assert "cv2.imread" not in handler_body


# ===========================================================================
# STEP 8 stabilization - Multi-frame Dataset (섹션 3)
# ===========================================================================

def test_ui_supports_dataset_directory_input():
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "_on_load_ghost_dataset_directory" in source
    assert "Load Dataset Directory" in source


def test_evaluation_worker_reports_per_frame_progress():
    source = _EVAL_WORKER.read_text(encoding="utf-8")
    assert "progress.emit" in source


def test_fit_from_evaluation_uses_dataset_wide_fit_not_first_frame_only():
    """사용자 스펙 3-G번 - "반드시 per_frame[0]를 사용하지 않는다": UI
    fit 핸들러가 `fit_ghost_field_from_dataset`를 쓰고, `per_frame[0]`
    기반의 예전 `fit_ghost_field_from_spatial_map` 직접 호출은 사라져야
    한다."""
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "fit_ghost_field_from_dataset" in source
    fit_fn_body = source.split("def _on_fit_ghost_model_from_evaluation")[1].split("\n    def ")[0]
    assert "fit_ghost_field_from_dataset(" in fit_fn_body
    # 예전에는 `frame = self._ghost_result.per_frame[0]`을 fit에 직접 썼다 -
    # 지금은 dataset 전체를 fit_ghost_field_from_dataset()에 통째로 넘겨야
    # 하므로, per_frame[0]을 fit 대상으로 꺼내 쓰는 코드가 없어야 한다
    # (docstring 설명 문장은 허용 - 실제 대입/색인 코드만 금지).
    assert ".per_frame[0]" not in fit_fn_body


# ===========================================================================
# STEP 8 stabilization - ghost_models .ccproj persistence (섹션 4)
# ===========================================================================

def test_ghost_models_state_is_tracked_separately():
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "self._ghost_models" in source
    assert "dict[str, GhostField]" in source


def test_ghost_models_round_trips_through_import_and_export_state():
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert 'getattr(project, "ghost_models"' in source
    assert "self._ghost_models" in source.split("def export_state")[1]


def test_main_window_saves_ghost_models_into_calibration_project():
    main_window = _ROOT / "ui" / "main_window.py"
    source = main_window.read_text(encoding="utf-8")
    assert "ghost_models" in source
    assert "ghost_models=ghost_models" in source


def test_ghost_models_registered_on_fit_and_load_not_only_yaml_export():
    """사용자 스펙 4-F번 - YAML로 저장하지 않아도 fit/load된 모델이
    project 안에 남아 있어야 한다: fit/load 핸들러가 직접
    `self._ghost_models[...] = field`를 해야 한다."""
    source = _WORKSPACE.read_text(encoding="utf-8")
    fit_section = source.split("def _on_fit_ghost_model_from_evaluation")[1].split("def _on_save_ghost_model")[0]
    load_section = source.split("def _on_load_ghost_suppression_model")[1].split("def _on_fit_ghost_model_from_evaluation")[0]
    assert "self._ghost_models[" in fit_section
    assert "self._ghost_models[" in load_section


# ===========================================================================
# STEP 8 stabilization - Likelihood / Visualization (섹션 6)
# ===========================================================================

def test_general_likelihood_has_dedicated_ui_label_and_hides_irrelevant_panels():
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "ghost_likelihood_label" in source
    assert "is_general" in source
    assert "setVisible(is_general)" in source


def test_ui_renders_real_overlay_and_vector_field_and_heatmap_images():
    """숫자 표뿐 아니라 실제 이미지 기반 시각화(Overlay/Vector Field/
    Heatmap)가 있어야 한다(사용자 스펙 6-C/6-D/6-E번)."""
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "render_ghost_point_overlay" in source
    assert "render_vector_field_image" in source
    assert "render_strength_heatmap_image" in source
    assert "ghost_overlay_image_label" in source
    assert "ghost_vector_field_image_label" in source
    assert "ghost_strength_heatmap_image_label" in source


def test_ui_does_not_reimplement_ghost_analysis_for_visualization():
    """사용자 스펙 6-F번 - UI가 별도 분석 알고리즘을 수행하면 안 된다:
    시각화 렌더러는 calibration 패키지(순수 NumPy/OpenCV)에 있고, UI는
    그 결과를 QLabel에 표시만 해야 한다."""
    viz_module = _ROOT / "calibration" / "windshield" / "ghost" / "visualization.py"
    assert viz_module.exists()
    source = viz_module.read_text(encoding="utf-8")
    assert "import PySide6" not in source
    assert "from PySide6" not in source


def test_point_source_and_edge_target_metrics_are_shown_in_separate_tables():
    """사용자 스펙 2-F번 - Point Source의 dx/dy와 Edge의 scalar offset을
    같은 표에 억지로 보여주지 않는다."""
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "ghost_edge_metrics_table" in source
    assert "ghost_metrics_table" in source
    assert "ghost_edge_metrics_table" != "ghost_metrics_table"


# ===========================================================================
# STEP 8 stabilization - Suppression Detail Retention (섹션 7)
# ===========================================================================

def test_suppression_ui_shows_detail_retention_and_over_suppression_metrics():
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "Edge Retention" in source
    assert "Over-Suppression Score" in source
    assert "edge_retention" in source
    assert "over_suppression_score" in source


# ===========================================================================
# STEP 8 stabilization - Suppression evaluator mode dispatch (섹션 8)
# ===========================================================================

def test_suppression_worker_uses_shared_dispatcher_not_hardcoded_point_source():
    source = _SUPPRESSION_WORKER.read_text(encoding="utf-8")
    assert "evaluate_ghost_image(" in source
    assert "evaluate_ghost_point_source(" not in source  # 실제 호출(괄호 포함)만 금지 - docstring 설명은 허용


# ===========================================================================
# CI - Ghost 전용 GitHub Actions workflow (섹션 5)
# ===========================================================================

def test_ghost_ci_workflow_exists_and_runs_ghost_tests():
    workflow = _ROOT / ".github" / "workflows" / "ghost-tests.yml"
    assert workflow.exists(), ".github/workflows/ghost-tests.yml이 존재하지 않습니다."
    source = workflow.read_text(encoding="utf-8")
    assert "test_windshield_ghost.py" in source
    assert "test_windshield_ghost_suppression.py" in source
    assert "test_windshield_ghost_ui_architecture.py" in source


def test_ghost_ci_workflow_is_separate_from_other_workflows():
    """사용자 스펙 5-D번 - 기존 workflow를 Ghost 때문에 수정할 필요
    없음: reflection/neural workflow 파일에 Ghost 테스트를 억지로 넣지
    않는다."""
    for name in ("reflection-tests.yml", "reflection-suppression-tests.yml", "neural-tests.yml"):
        path = _ROOT / ".github" / "workflows" / name
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        assert "test_windshield_ghost" not in source


# ===========================================================================
# STEP 8 semantic/safety fix 1 - General Likelihood dataset UI
# ===========================================================================

def test_general_likelihood_ui_shows_dataset_mean_median_p95_not_mean_strength():
    """General mode label이 `dataset_result.mean_strength`가 아니라
    `mean_ghost_likelihood`/`median_ghost_likelihood`/`p95_ghost_likelihood`
    를 써야 한다(사용자 스펙 1번, "Likelihood != Strength")."""
    source = _WORKSPACE.read_text(encoding="utf-8")
    likelihood_section = source.split("if is_general:")[1].split("\n\n        values = [")[0]
    assert "dataset_result.mean_ghost_likelihood" in likelihood_section
    assert "dataset_result.median_ghost_likelihood" in likelihood_section
    assert "dataset_result.p95_ghost_likelihood" in likelihood_section
    assert "dataset_result.mean_strength" not in likelihood_section


def test_forbidden_likelihood_naming_never_used():
    source = _WORKSPACE.read_text(encoding="utf-8")
    for forbidden in ("Ghost Accuracy", "Ghost Probability", "Ghost Confidence", "Ghost Ground Truth Score"):
        assert forbidden not in source


# ===========================================================================
# STEP 8 semantic fix 2 - General Suppression UI shows Likelihood, not Strength
# ===========================================================================

def test_suppression_ui_shows_likelihood_reduction_for_general_mode():
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "Likelihood Reduction" in source
    assert "evaln.likelihood_reduction" in source
    assert "before.ghost_likelihood" in source
    assert "after.ghost_likelihood" in source


def test_suppression_worker_delegates_mode_split_to_pure_function():
    """Mode별 strength_reduction/likelihood_reduction 분기 로직은
    `build_suppression_evaluation()`(Qt 비의존)에 있고, worker는 그 함수를
    호출하기만 해야 한다 - PySide6 없이 로직을 직접 테스트할 수 있게
    한다."""
    source = _SUPPRESSION_WORKER.read_text(encoding="utf-8")
    assert "build_suppression_evaluation(" in source


# ===========================================================================
# STEP 8 semantic/safety fix 3 - GhostField Fit guard (UI)
# ===========================================================================

def test_ghost_fit_button_is_disabled_by_default_and_gated_by_mode():
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "self.ghost_fit_button" in source
    assert "self.ghost_fit_button.setEnabled(False)" in source
    assert "self.ghost_fit_button.setEnabled(is_point_source)" in source


def test_ghost_fit_button_has_explanatory_tooltip():
    source = _WORKSPACE.read_text(encoding="utf-8")
    assert "setToolTip" in source
    assert "2D main/ghost point displacement" in source


def test_fit_handler_defensively_rejects_non_point_source_mode():
    source = _WORKSPACE.read_text(encoding="utf-8")
    handler_body = source.split("def _on_fit_ghost_model_from_evaluation(self)")[1].split("\n    def ")[0]
    assert 'self._ghost_result.mode != "point_source"' in handler_body


# ===========================================================================
# STEP 8 semantic/safety fix 4 - Resolution gate UI wiring
# ===========================================================================

def test_evaluation_worker_calls_pure_dataset_from_paths_pipeline():
    source = _EVAL_WORKER.read_text(encoding="utf-8")
    assert "evaluate_ghost_dataset_from_paths" in source
