#!/usr/bin/env bash
# Fast regression: same pytest suite as run_tests.sh, excluding tests marked
# `slow`(pytest.ini에 이미 등록된 marker - 합성 이미지 렌더링 + Standard
# 4-model 계산이 포함된 느린 통합 테스트). 빠른 wiring 확인용.
set -euo pipefail

cd "$(dirname "$0")"

python -m pytest -q -m "not slow"
