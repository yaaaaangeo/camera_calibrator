"""
tests/test_dependency_consistency.py
=========================================

Priority 6 안정화 - requirements.txt / pyproject.toml의 desktop dependency
정책이 서로 어긋나지 않는지 확인한다.

발견된 실제 버그: `requirements.txt`에 `PySide6`가 통째로 빠져 있었다 -
README는 "방법 A(requirements.txt)로도 GUI가 설치된다"고 안내하지만,
실제로는 `pip install -r requirements.txt` 후 `python -m app.main`을
실행하면 PySide6 ImportError로 즉시 실패했다. `scipy`도 두 파일이 서로
다른 최소 버전(1.10 vs 1.11)을 선언하고 있었다.

Jetson profile(`requirements-jetson*.txt`)은 이 테스트의 대상이 아니다 -
데스크톱과 다른 정책(고정 버전 pin, PySide6 minor 버전 등)을 의도적으로
쓴다(README 12.2/JETSON.md 참고).
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

_REQ_LINE_RE = re.compile(r"^([A-Za-z0-9_.\-]+)\s*(>=|==)\s*([0-9][0-9A-Za-z.\-]*)")
_QUOTED_ENTRY_RE = re.compile(r'"([^"]+)"')


def _parse_requirements(path: Path) -> dict[str, tuple[str, str]]:
    pkgs: dict[str, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _REQ_LINE_RE.match(line)
        if m:
            pkgs[m.group(1).lower()] = (m.group(2), m.group(3))
    return pkgs


def _parse_pyproject_dependencies() -> dict[str, tuple[str, str]]:
    """`[project] dependencies = [...]` 블록만 텍스트로 뽑아 파싱한다 -
    `tomllib`은 Python 3.11+ 표준 라이브러리라 이 프로젝트가 지원하는
    Python 3.10에서는 쓸 수 없고, `tomli`도 dev 의존성에 없어서 새 패키지를
    추가하지 않기 위해 이 프로젝트의 단순한 `dependencies = [...]` 포맷만
    직접 스캔한다(범용 TOML 파서가 아님 - 이 파일의 실제 포맷에 맞춘
    최소 구현)."""
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    assert match, "pyproject.toml에서 dependencies = [...] 블록을 찾지 못했습니다."
    pkgs: dict[str, tuple[str, str]] = {}
    for entry in _QUOTED_ENTRY_RE.findall(match.group(1)):
        m = _REQ_LINE_RE.match(entry.strip())
        if m:
            pkgs[m.group(1).lower()] = (m.group(2), m.group(3))
    return pkgs


def test_requirements_txt_and_pyproject_declare_pyside6():
    """GUI 의존성(PySide6)이 두 설치 경로 모두에 존재해야 한다 - 하나라도
    빠지면 그 경로로 설치한 사용자는 GUI를 아예 못 띄운다."""
    req = _parse_requirements(_ROOT / "requirements.txt")
    pyproject = _parse_pyproject_dependencies()
    assert "pyside6" in req, "requirements.txt에 PySide6가 없습니다 - GUI가 설치되지 않습니다."
    assert "pyside6" in pyproject, "pyproject.toml dependencies에 PySide6가 없습니다."


def test_requirements_txt_and_pyproject_agree_on_shared_package_versions():
    """두 파일 모두에 등장하는 패키지는 최소 버전이 서로 달라서는 안 된다
    (예: scipy>=1.10 vs scipy>=1.11처럼 서로 다른 최소 버전을 요구하면
    어느 설치 경로를 쓰냐에 따라 실제로 다른 dependency graph가 만들어진다)."""
    req = _parse_requirements(_ROOT / "requirements.txt")
    pyproject = _parse_pyproject_dependencies()
    shared = set(req) & set(pyproject)
    assert shared, "두 파일 사이에 공유되는 패키지가 하나도 없습니다 - 파싱 로직을 확인하세요."
    mismatches = {
        name: (req[name], pyproject[name])
        for name in shared
        if req[name] != pyproject[name]
    }
    assert not mismatches, f"requirements.txt와 pyproject.toml의 최소 버전이 다릅니다: {mismatches}"
