"""
tests/test_windshield_worker_architecture.py
=================================================

Windshield calibration이 더 이상 Qt UI 스레드에서 동기 실행되지 않는지를
"구조적으로" 확인한다(사용자 스펙 Test 7). ui/*.py는 PySide6에 의존하지만,
이 파일은 소스 텍스트만 읽고 문자열/AST 검사만 하므로 PySide6가 설치돼
있지 않거나(이 sandbox처럼 DLL 로드가 깨져 있어도) 항상 실행할 수 있다 -
importorskip으로 건너뛰지 않는다.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKSPACE_PATH = _REPO_ROOT / "ui" / "windshield_workspace.py"
_WORKER_PATH = _REPO_ROOT / "ui" / "windshield_worker.py"


def _calls(text: str, func_name: str) -> bool:
    """text 안에 func_name(...)에 대한 실제 "호출"이 있는지 검사한다.

    단순 substring 검사(`func_name + "("` in text)는 `_on_run_windshield_
    calibration`처럼 func_name을 접미어로 포함하는 다른 식별자(메서드 이름
    자체)까지 false positive로 잡아버린다 - 앞에 단어 문자(word character,
    밑줄 포함)가 없을 때만 "진짜 호출"로 인정한다."""
    return re.search(rf"(?<!\w){re.escape(func_name)}\(", text) is not None


def _extract_function_source(source: str, func_name: str) -> str:
    """단순 들여쓰기 기반 추출 - 이 파일에서 함수/메서드 하나의 본문만
    필요하므로 AST 대신 텍스트로 충분하다."""
    lines = source.splitlines()
    start = None
    indent = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"def {func_name}("):
            start = i
            indent = len(line) - len(line.lstrip())
            continue
        if start is not None and line.strip() and (len(line) - len(line.lstrip())) <= indent and i > start:
            return "\n".join(lines[start:i])
    if start is not None:
        return "\n".join(lines[start:])
    raise AssertionError(f"function {func_name} not found")


def test_windshield_worker_file_exists_with_expected_class_and_signals():
    assert _WORKER_PATH.exists(), "ui/windshield_worker.py가 존재하지 않습니다."
    source = _WORKER_PATH.read_text(encoding="utf-8")
    assert "class WindshieldCalibrationWorker" in source
    for signal_name in ("progress", "result_ready", "not_implemented", "error", "finished"):
        assert f"{signal_name} = Signal(" in source, f"{signal_name} signal이 없습니다."
    assert "def run(self)" in source


def test_windshield_worker_does_not_import_qt_widgets():
    """Worker는 순수 data object만 다뤄야 한다 - QWidget 계열을 참조하면
    (moveToThread 이후 다른 스레드에서 위젯을 건드릴 위험이 생기므로) 안 된다."""
    source = _WORKER_PATH.read_text(encoding="utf-8")
    assert "QWidget" not in source
    assert "QMessageBox" not in source


def test_on_run_windshield_calibration_does_not_call_calibration_functions_directly():
    """_on_run_windshield_calibration()의 본문 자체에는 run_windshield_calibration(
    이나 run_residual_ray_calibration_with_diagnostics( 호출이 없어야 한다 -
    실제 계산은 WindshieldCalibrationWorker.run() 안(별도 QThread)에서만
    일어나야 하고, UI 핸들러는 워커를 만들어 thread.start()만 해야 한다."""
    source = _WORKSPACE_PATH.read_text(encoding="utf-8")
    handler_source = _extract_function_source(source, "_on_run_windshield_calibration")

    assert not _calls(handler_source, "run_windshield_calibration")
    assert not _calls(handler_source, "run_residual_ray_calibration_with_diagnostics")
    assert "WindshieldCalibrationWorker(" in handler_source
    assert "run_worker_in_thread(" in handler_source
    assert "thread.start()" in handler_source


def test_on_run_windshield_calibration_disables_run_button_and_shows_running_status():
    source = _WORKSPACE_PATH.read_text(encoding="utf-8")
    handler_source = _extract_function_source(source, "_on_run_windshield_calibration")
    assert "self.run_button.setEnabled(False)" in handler_source
    assert "Running" in handler_source
    assert "self.run_button.setEnabled(True)" in source  # thread.finished 콜백 쪽에 있음


def test_worker_run_dispatches_neural_when_method_is_neural():
    """사용자 스펙 5-D번 - method=="neural"이면 run_neural_residual_
    calibration_with_diagnostics(가 실제로 호출돼야 하고, 그 import는
    run() 본문 안(lazy, method=="neural" 분기)에만 있어야 한다(모듈
    top-level에 있으면 PyTorch가 없는 환경에서 워커 자체를 import하지
    못하게 된다)."""
    source = _WORKER_PATH.read_text(encoding="utf-8")
    run_source = _extract_function_source(source, "run")
    assert 'method == "neural"' in run_source
    assert _calls(run_source, "run_neural_residual_calibration_with_diagnostics")
    assert "from calibration.windshield.neural_residual import run_neural_residual_calibration_with_diagnostics" in run_source

    top_level_lines = []
    for line in source.splitlines():
        if line.startswith(("class ", "def ")):
            break
        top_level_lines.append(line)
    assert not any("neural_residual" in line for line in top_level_lines), (
        "neural_residual must only be imported lazily inside run(), not at module top-level"
    )


def test_worker_run_dispatches_grid_rbf_neural_spline_to_distinct_functions():
    """Grid/RBF/Neural/Spline이 서로 다른 함수로 정확히 분기되는지 - 어느
    하나가 다른 하나로 잘못 dispatch되지 않는다는 것을 최소한 "그 함수
    이름이 run() 본문에 전부 등장한다"는 수준에서 확인한다(세부 분기
    로직의 정확성은 calibrate_neural_residual/calibrate_residual_rbf 등
    backend 테스트가 이미 검증한다 - 여기서는 worker가 그 4개 함수를 모두
    실제로 참조하는지만 구조적으로 본다)."""
    source = _WORKER_PATH.read_text(encoding="utf-8")
    run_source = _extract_function_source(source, "run")
    assert _calls(run_source, "run_residual_ray_calibration_with_diagnostics")
    assert _calls(run_source, "run_residual_rbf_calibration_with_diagnostics")
    assert _calls(run_source, "run_neural_residual_calibration_with_diagnostics")
    assert _calls(run_source, "run_spline_calibration_with_diagnostics")


def test_workspace_module_never_calls_heavy_calibration_functions_at_module_scope():
    """run_windshield_calibration/run_residual_ray_calibration_with_diagnostics에
    대한 실제 호출(함수 정의가 아니라 '이름(' 형태)이 windshield_worker.py를
    제외한 ui/windshield_workspace.py 안 어디에도 없어야 한다 - 이 두 계산은
    오직 ui/windshield_worker.py::WindshieldCalibrationWorker.run() 안에서만
    호출돼야 한다."""
    source = _WORKSPACE_PATH.read_text(encoding="utf-8")
    assert not _calls(source, "run_windshield_calibration")
    assert not _calls(source, "run_residual_ray_calibration_with_diagnostics")

    worker_source = _WORKER_PATH.read_text(encoding="utf-8")
    assert _calls(worker_source, "run_windshield_calibration")
    assert _calls(worker_source, "run_residual_ray_calibration_with_diagnostics")
