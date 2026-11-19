#!/usr/bin/env bash
# Calibration Tool 실행.
#   ./run.sh                      GUI
#   ./run.sh -v --log-file x.log  상세 로그 (app/main.py 참고)
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${root}/venv/bin/python"

if [[ ! -x "${python}" ]]; then
    echo "venv 가 없습니다. 먼저 ./install.sh 를 실행하세요." >&2
    exit 1
fi

# install.sh 가 CycloneDDS 를 소스 빌드한 경우(휠이 없는 aarch64 등)
dds_home="${root}/venv/opt/cyclonedds"
if [[ -d "${dds_home}/lib" ]]; then
    export CYCLONEDDS_HOME="${dds_home}"
    export LD_LIBRARY_PATH="${dds_home}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cd "${root}/intrinsic"
exec "${python}" -m app.main "$@"
