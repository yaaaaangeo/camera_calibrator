#!/usr/bin/env bash
# 테스트 실행 (./install.sh --dev 필요).
#   ./run_tests.sh               빠른 테스트 (slow 제외)
#   ./run_tests.sh --all         전체
#   ./run_tests.sh tests/test_cli.py -v   pytest 인자 그대로 전달 (intrinsic/ 기준 경로)
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${root}/intrinsic"

# ROS 환경이 source 된 셸에서는 ROS 의 pytest 플러그인(launch_testing)이 끼어들어
# 수집 단계에서 실패한다. 이 프로젝트는 외부 플러그인을 쓰지 않으므로 자동 로드를 끈다.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

if [[ "${1:-}" == "--all" ]]; then
    shift
    exec "${root}/venv/bin/python" -m pytest -q -p no:cacheprovider "$@"
fi
exec "${root}/venv/bin/python" -m pytest -q -p no:cacheprovider -m "not slow" "$@"
