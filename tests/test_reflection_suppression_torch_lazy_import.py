"""
tests/test_reflection_suppression_torch_lazy_import.py
==============================================================

STEP 7 - Reflection Suppression은 STEP 5 Neural Residual과 동일한 lazy
PyTorch import 원칙을 따른다(사용자 스펙 12/13번): Reflection Evaluation
(STEP 6)이 PyTorch dependency를 갖게 만들지 않는다.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(code: str, timeout: float = 60.0, env_overrides: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, cwd=str(_PROJECT_ROOT), timeout=timeout, env=env,
    )


def _assert_clean_ok(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout, f"stdout={result.stdout!r} stderr={result.stderr!r}"


def test_torch_free_submodules_do_not_load_torch():
    result = _run("""
        import sys
        assert "torch" not in sys.modules
        import calibration.windshield.reflection_suppression.config
        import calibration.windshield.reflection_suppression.types
        import calibration.windshield.reflection_suppression.synthetic
        import calibration.windshield.reflection_suppression.dataset
        assert "torch" not in sys.modules, "these submodules must not import torch"
        print("OK")
    """)
    _assert_clean_ok(result)


def test_package_import_does_not_load_torch():
    result = _run("""
        import sys
        assert "torch" not in sys.modules
        import calibration.windshield.reflection_suppression as rs
        assert "torch" not in sys.modules, "package __init__ import must not eagerly import torch"
        print("OK")
    """)
    _assert_clean_ok(result)


def test_reflection_evaluation_step6_never_imports_torch():
    """STEP 6 evaluator는 STEP 7 때문에 torch dependency를 갖게 되면 안 된다
    (사용자 스펙 13번, "STEP 7 때문에 STEP 6가 Torch dependency를 가지게
    만들지 않는다")."""
    result = _run("""
        import sys
        assert "torch" not in sys.modules
        import calibration.windshield.reflection.evaluator
        import calibration.windshield.reflection.alignment
        import calibration.windshield.reflection.metrics
        import calibration.windshield.reflection.types
        assert "torch" not in sys.modules
        print("OK")
    """)
    _assert_clean_ok(result)


def test_calling_require_torch_loads_torch_lazily():
    pytest.importorskip("torch")
    result = _run(
        """
        import sys
        assert "torch" not in sys.modules
        from calibration.windshield.reflection_suppression.model import _require_torch
        assert "torch" not in sys.modules
        _require_torch()
        assert "torch" in sys.modules
        print("OK")
        """,
        env_overrides={"KMP_DUPLICATE_LIB_OK": "TRUE"},
    )
    _assert_clean_ok(result)


def test_windshield_workspace_ui_does_not_import_reflection_suppression_at_top_level():
    """PySide6 유무와 무관하게 항상 실행 가능한 정적 검사 - UI가
    `calibration.windshield.reflection_suppression`을 top-level에서
    import하면 안 된다(항상 method 본문 안에서만, 사용자 스펙 12번)."""
    source = (_PROJECT_ROOT / "ui" / "windshield_workspace.py").read_text(encoding="utf-8")
    top_level_lines = []
    for line in source.splitlines():
        if line.startswith(("class ", "def ")):
            break
        top_level_lines.append(line)
    assert not any("reflection_suppression" in line for line in top_level_lines)
    assert "reflection_suppression" in source, "expected lazy (in-method) references somewhere"


def test_reflection_suppression_worker_only_imports_runtime_inside_run():
    source = (_PROJECT_ROOT / "ui" / "reflection_suppression_worker.py").read_text(encoding="utf-8")
    top_level_import_lines = []
    for line in source.splitlines():
        if line.startswith(("class ", "def ")):
            break
        if line.startswith(("import ", "from ")):
            top_level_import_lines.append(line)
    assert not any("reflection_suppression" in line for line in top_level_import_lines)
    assert "class ReflectionSuppressionWorker" in source
    assert "def run(self)" in source


def test_other_reflection_evaluation_still_works_when_torch_is_unavailable():
    """PyTorch가 아예 없는 환경을 흉내내도 STEP 6 Reflection Evaluation은
    정상 동작해야 하고, Suppression만 명확한 ImportError를 내야 한다."""
    code = """
        import sys
        sys.modules["torch"] = None
        sys.modules["torch.nn"] = None
        sys.modules["torch.optim"] = None

        import numpy as np
        from calibration.windshield.reflection.evaluator import evaluate_reflection
        from calibration.windshield.reflection.types import ReflectionEvaluationConfig

        rng = np.random.default_rng(0)
        normal = (rng.uniform(0, 255, size=(64, 64, 3))).astype(np.uint8)
        reference = normal.copy()
        result = evaluate_reflection(normal, reference, ReflectionEvaluationConfig(mode="reference"))
        assert result.success, result.error_message

        from calibration.windshield.reflection_suppression.model import build_model
        try:
            build_model()
            print("UNEXPECTED_SUCCESS")
        except ImportError as e:
            assert "PyTorch" in str(e)
            print("OK")
    """
    result = _run(code, timeout=120.0)
    _assert_clean_ok(result)
