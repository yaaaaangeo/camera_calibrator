"""
tests/test_neural_torch_lazy_import.py
==============================================

STEP 5 안정화 라운드 - 항목 1(진짜 lazy torch import) + 항목 5-G(Torch 없는
환경 시뮬레이션) 전용 회귀 테스트.

핵심 계약:

    import calibration.windshield.neural_config   -> torch 절대 로드 안 됨
    import calibration.windshield.neural_residual -> torch 절대 로드 안 됨
    import calibration.windshield.validation/projection -> torch 절대 로드 안 됨
    import ui.windshield_workspace                -> torch 절대 로드 안 됨

    Neural 함수를 실제로 호출(_require_torch() 등)해야만 그 순간 torch가 로드됨.

    PyTorch가 아예 없는 환경을 흉내내도(sys.modules에 None을 심어 import를
    막는 표준 기법) Grid/RBF/Spline 등 다른 모델은 여전히 정상 동작해야
    하고, Neural만 명확한 ImportError를 낸다.

매 검증을 별도의 완전히 새로운 interpreter(subprocess)에서 수행한다 -
현재 프로세스는 이미 이전 테스트에서 torch를 로드했을 수 있어서
`sys.modules` 검사가 무의미해지기 때문이다.
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
    """새 subprocess에서 코드를 실행한다. 기본적으로 `KMP_DUPLICATE_LIB_OK`는
    부모 프로세스(이 테스트 스위트를 실행하는 셸)에 어떻게 설정돼 있든 이
    subprocess에서는 제거한다 - "라이브러리가 이 값을 자동으로 설정하지
    않는다"는 계약을 부모 셸의 우연한 환경변수 설정과 무관하게 깨끗하게
    검증하기 위함이다. 실제로 torch를 로드해야 하는 테스트는
    `env_overrides={"KMP_DUPLICATE_LIB_OK": "TRUE"}`를 명시적으로 넘긴다."""
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


def test_neural_config_import_does_not_load_torch():
    result = _run("""
        import sys
        assert "torch" not in sys.modules
        import calibration.windshield.neural_config
        assert "torch" not in sys.modules, "neural_config must not import torch"
        print("OK")
    """)
    _assert_clean_ok(result)


def test_neural_residual_module_import_does_not_load_torch():
    result = _run("""
        import sys
        assert "torch" not in sys.modules
        import calibration.windshield.neural_residual
        assert "torch" not in sys.modules, "neural_residual MODULE import must not eagerly import torch"
        print("OK")
    """)
    _assert_clean_ok(result)


def test_windshield_validation_and_projection_import_does_not_load_torch():
    result = _run("""
        import sys
        assert "torch" not in sys.modules
        import calibration.windshield.validation
        import calibration.windshield.projection
        assert "torch" not in sys.modules
        print("OK")
    """)
    _assert_clean_ok(result)


def test_no_module_auto_sets_kmp_duplicate_lib_ok_env_var():
    result = _run("""
        import os
        assert "KMP_DUPLICATE_LIB_OK" not in os.environ
        import calibration.windshield.neural_config
        import calibration.windshield.neural_residual
        import calibration.windshield.validation
        import calibration.windshield.projection
        assert "KMP_DUPLICATE_LIB_OK" not in os.environ, "library must not auto-set this global env var"
        print("OK")
    """)
    _assert_clean_ok(result)


def test_calling_require_torch_loads_torch_lazily():
    """실제로 torch가 설치된 이 환경에서는, `_require_torch()`를 호출하는
    순간에만 torch가 로드돼야 한다(그 전까지는 로드되지 않는다)."""
    pytest.importorskip("torch")
    result = _run(
        """
        import sys
        assert "torch" not in sys.modules
        from calibration.windshield.neural_residual import _require_torch
        assert "torch" not in sys.modules, "importing the function itself must not load torch"
        _require_torch()
        assert "torch" in sys.modules, "torch should be loaded now"
        print("OK")
        """,
        # 이 테스트는 실제로 torch를 로드한다 - 이 프로젝트의 Windows/Anaconda
        # 환경에서 numpy(MKL)와 OpenMP 런타임이 충돌할 수 있어(모듈 docstring
        # 참고) 명시적으로 완화 플래그를 켠 채 실행한다(라이브러리 자체가
        # 자동 설정하지 않는다는 계약과는 별개 - 여기서는 "사용자가 명시적으로
        # 설정"하는 시나리오를 그대로 흉내낸다).
        env_overrides={"KMP_DUPLICATE_LIB_OK": "TRUE"},
    )
    _assert_clean_ok(result)


def test_ui_windshield_workspace_import_does_not_load_torch():
    """PySide6이 이 환경에서 import 가능한 경우에만 실제로 검증한다 - 이
    sandbox는 PySide6 자체가 별개의 DLL 문제로 깨져 있을 수 있으므로, 그
    경우는 skip한다(Neural/torch와 무관한 환경 제약)."""
    result = _run("""
        import sys
        try:
            import PySide6.QtWidgets  # noqa: F401
        except ImportError:
            print("SKIP_NO_PYSIDE6")
            sys.exit(0)
        assert "torch" not in sys.modules
        import ui.windshield_workspace
        assert "torch" not in sys.modules, "ui.windshield_workspace import must not eagerly import torch"
        print("OK")
    """)
    if "SKIP_NO_PYSIDE6" in result.stdout:
        pytest.skip("PySide6 is not importable in this environment (unrelated DLL issue) - cannot verify live.")
    _assert_clean_ok(result)


def test_ui_windshield_workspace_source_does_not_reference_neural_residual_module():
    """PySide6 유무와 무관하게 항상 실행 가능한 정적 검사(소스 텍스트 수준) -
    UI가 `calibration.windshield.neural_residual`을 import하는 코드를 다시
    추가하지 않는지 회귀 방지한다."""
    source = (_PROJECT_ROOT / "ui" / "windshield_workspace.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "windshield.neural_residual" not in stripped, f"UI must not import neural_residual: {line!r}"


def test_worker_does_not_import_neural_residual_at_module_top_level():
    """`ui/windshield_worker.py`도 top-level에서 neural_residual을 import하면
    안 된다 - import는 항상 method=="neural" 분기 안(함수 본문)에서만
    일어나야 한다."""
    source = (_PROJECT_ROOT / "ui" / "windshield_worker.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    # 모듈 최상단 import 블록(첫 클래스/함수 정의 전)만 검사한다.
    top_level_lines = []
    for line in lines:
        if line.startswith(("class ", "def ")):
            break
        top_level_lines.append(line)
    for line in top_level_lines:
        assert "neural_residual" not in line, f"top-level import of neural_residual found: {line!r}"
    assert "neural_residual" in source, "expected a lazy (in-function) import of neural_residual somewhere"


# ---------------------------------------------------------------------------
# 항목 5-G - PyTorch가 없는 환경을 흉내낸다(sys.modules에 None을 심어
# import가 항상 ImportError를 내게 만드는 표준 기법). Grid/RBF/Spline은
# 여전히 동작해야 하고, Neural만 명확한 ImportError를 내야 한다.
# ---------------------------------------------------------------------------

def test_other_models_work_and_neural_fails_cleanly_when_torch_is_unavailable():
    code = """
        import sys
        sys.modules["torch"] = None  # 이후 'import torch'는 항상 ImportError
        sys.modules["torch.nn"] = None
        sys.modules["torch.optim"] = None

        # 1) Neural이 아닌 다른 Windshield 모델은 torch 없이도 정상 동작해야 한다.
        import numpy as np
        from calibration.types import CameraModelType
        from calibration.validation import split_train_test
        from calibration.windshield.base import WindshieldConfig, WindshieldModelType
        from calibration.windshield.residual_ray import calibrate_residual_ray
        from calibration.windshield.residual_rbf import calibrate_residual_rbf
        from tests._windshield_test_utils import (
            build_synthetic_residual_ray_dataset,
            default_camera_config,
            default_camera_matrix_distortion,
            default_residual_delta_fn,
        )

        K, D = default_camera_matrix_distortion()
        dataset = build_synthetic_residual_ray_dataset(K, D, default_residual_delta_fn(K))
        camera_config = default_camera_config()
        train_ids = [f.image_info.image_id for f in dataset.frames[:-2]]
        test_ids = [f.image_info.image_id for f in dataset.frames[-2:]]
        config = WindshieldConfig(
            base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D,
            windshield_model=WindshieldModelType.RESIDUAL_RAY,
        )
        grid_result = calibrate_residual_ray(dataset, config, camera_config, train_ids, test_ids)
        assert grid_result.success, grid_result.error_message

        rbf_config = WindshieldConfig(
            base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D,
            windshield_model=WindshieldModelType.RESIDUAL_RAY,
            residual_ray_hint={"method": "rbf", "rbf_num_centers": 8.0, "rbf_smoothing": 1e-4},
        )
        rbf_result = calibrate_residual_rbf(dataset, rbf_config, camera_config, train_ids, test_ids)
        assert rbf_result.success, rbf_result.error_message

        # 2) Neural은 torch가 없으므로 명확한 ImportError를 내야 한다(다른
        #    예외 타입/조용한 실패가 아니라).
        from calibration.windshield.neural_residual import calibrate_neural_residual
        neural_config = WindshieldConfig(
            base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D,
            windshield_model=WindshieldModelType.RESIDUAL_RAY,
            residual_ray_hint={"method": "neural"},
        )
        try:
            calibrate_neural_residual(dataset, neural_config, camera_config, train_ids, test_ids)
            print("UNEXPECTED_SUCCESS")
        except ImportError as e:
            assert "PyTorch" in str(e)
            print("OK")
    """
    result = _run(code, timeout=120.0)
    _assert_clean_ok(result)
