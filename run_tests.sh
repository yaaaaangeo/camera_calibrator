#!/usr/bin/env bash
# Runs the full pytest suite (tests/test_*.py) via the real pytest runner.
#
# Phase A-2 안정화 - 이전 버전은 `python3 "$f"`로 각 테스트 파일을
# standalone script처럼 실행했지만, 이 저장소의 테스트 파일은 전부 pytest
# 함수/fixture(assert, pytest.approx, pytest.raises, tmp_path 등) 기반이고
# `if __name__ == "__main__":` 블록이 하나도 없다 - 즉 `python3 file.py`는
# 모듈을 import만 하고 실제 테스트 함수를 단 하나도 호출하지 않은 채 항상
# exit code 0으로 끝났다("0 passed, 0 failed"를 조용히 성공으로 보고).
# pytest가 파일을 직접 수집/실행하도록 고친다.
set -euo pipefail

cd "$(dirname "$0")"

python -m pytest -q
