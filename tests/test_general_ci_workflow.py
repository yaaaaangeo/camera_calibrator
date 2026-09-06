"""
tests/test_general_ci_workflow.py
======================================

Phase D-4 - Unified General CI(.github/workflows/ci.yml) 검증.

기존 기능별 워크플로우(ghost/reflection/reflection-suppression/neural)는
그대로 두고, 이 워크플로우는 "핵심 회귀가 통과하는가"만 빠르게 확인한다.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_CI_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"


def test_general_ci_workflow_exists():
    assert _CI_WORKFLOW.exists(), ".github/workflows/ci.yml이 존재하지 않습니다."


def test_general_ci_workflow_is_valid_yaml():
    with open(_CI_WORKFLOW, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "jobs" in data


def test_general_ci_workflow_runs_fast_core_regression():
    source = _CI_WORKFLOW.read_text(encoding="utf-8")
    assert 'pytest -q -m "not slow"' in source


def test_general_ci_workflow_uses_python_matrix_310_and_311():
    with open(_CI_WORKFLOW, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    job = next(iter(data["jobs"].values()))
    versions = job["strategy"]["matrix"]["python-version"]
    assert "3.10" in versions
    assert "3.11" in versions


def test_general_ci_workflow_does_not_install_neural_extra():
    """이 워크플로우는 torch를 요구하지 않는다 - Neural 전용 회귀는
    neural-tests.yml이 계속 담당한다(사용자 스펙 - 기존 기능별 CI를
    대체하지 않는다)."""
    executable_lines = [
        line for line in _CI_WORKFLOW.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("#")
    ]
    executable_text = "\n".join(executable_lines).lower()
    assert "[neural]" not in executable_text
    assert "torch" not in executable_text


def test_general_ci_workflow_does_not_duplicate_or_replace_feature_specific_workflows():
    """기존 ghost/reflection/reflection-suppression/neural 워크플로우 파일은
    이 라운드에서 건드리지 않는다(존재 여부만 확인 - 내용은 각자 파일의
    기존 테스트가 이미 검증한다)."""
    for name in (
        "ghost-tests.yml",
        "reflection-tests.yml",
        "reflection-suppression-tests.yml",
        "neural-tests.yml",
    ):
        assert (_ROOT / ".github" / "workflows" / name).exists()
