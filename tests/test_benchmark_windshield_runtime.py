"""
tests/test_benchmark_windshield_runtime.py
===============================================

Phase C-3 - scripts/benchmark_windshield_runtime.py 배선 확인.

전체 기본 스케일(최대 10,000포인트 x Spline/Neural의 root-solve/MLP
forward)을 도는 실제 벤치마크는 몇 분 이상 걸릴 수 있어 이 회귀 스위트의
목적(빠른 배선 확인)에 맞지 않는다 - 여기서는 각 모델 builder가 정상적으로
`WindshieldModel`을 만들고, `benchmark_model()`이 아주 작은 스케일(n=1,
repeats=1)에서 예외 없이 세 티어(Scalar/Batch/LUT) 모두를 계산하는지만
확인한다. 실제 처리량 수치 자체(예: "LUT가 몇 배 빠른가")는 이 테스트의
관심사가 아니다 - 그건 스크립트를 직접 실행해서 보고해야 하는 실측치다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "benchmark_windshield_runtime.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("benchmark_windshield_runtime", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # dataclass() resolves postponed (`from __future__ import annotations`)
    # type hints via sys.modules[cls.__module__] - the module must already be
    # registered there before exec_module() runs its @dataclass decorators.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return _load_script_module()


def test_script_has_valid_syntax_and_imports_cleanly(bench):
    assert hasattr(bench, "main")
    assert hasattr(bench, "benchmark_model")
    assert hasattr(bench, "MODEL_BUILDERS")


@pytest.mark.parametrize("name", ["Baseline", "Spherical", "Residual Grid", "Residual RBF"])
def test_fast_model_builders_produce_a_windshield_model(bench, name):
    from calibration.windshield.base import WindshieldModel

    model = bench.MODEL_BUILDERS[name]()
    assert isinstance(model, WindshieldModel)


def test_neural_builder_returns_none_or_a_model_never_raises(bench):
    """torch가 없으면 None(SKIPPED로 보고됨), 있으면 정상 모델이어야 한다 -
    어느 쪽이든 예외를 내면 안 된다(사용자 스펙 - Jetson에서 torch가
    없어도 스크립트 전체가 죽으면 안 됨)."""
    from calibration.windshield.base import WindshieldModel

    model = bench.MODEL_BUILDERS["Neural Residual"]()
    assert model is None or isinstance(model, WindshieldModel)


def test_benchmark_model_runs_all_three_tiers_at_tiny_scale(bench):
    model = bench.MODEL_BUILDERS["Baseline"]()
    results = bench.benchmark_model("Baseline", model, (1,), repeats=1)
    assert set(results.keys()) == {1}
    tiers = results[1]
    for key in (
        "scalar_project", "batch_project", "lut_project",
        "scalar_unproject", "batch_unproject", "lut_unproject",
    ):
        assert key in tiers
        assert tiers[key].n_points >= 1
        assert tiers[key].mean_ms >= 0.0


@pytest.mark.slow
def test_benchmark_model_runs_for_spline_at_tiny_scale(bench):
    """Spline은 root-solve 두 번(inner+outer)이 코너마다 필요해 다른
    모델보다 훨씬 느리다(모듈 자체 docstring 참고) - n=1로만 배선을
    확인한다."""
    model = bench.MODEL_BUILDERS["Spline"]()
    results = bench.benchmark_model("Spline", model, (1,), repeats=1)
    assert results[1]["scalar_project"].n_points == 1
