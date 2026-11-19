#!/usr/bin/env bash
# Calibration Tool 명령줄 (GUI 없이).
#   ./cli.sh intrinsic --help       카메라 내부파라미터
#   ./cli.sh camera-lidar --help    카메라·LiDAR 외부파라미터
#   ./cli.sh automation --help      자동화 (run / live / publish ...)
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${root}/venv/bin/python"
if [[ ! -x "${python}" ]]; then
    echo "venv 가 없습니다. 먼저 ./install.sh 를 실행하세요." >&2
    exit 1
fi
dds_home="${root}/venv/opt/cyclonedds"
if [[ -d "${dds_home}/lib" ]]; then
    export CYCLONEDDS_HOME="${dds_home}"
    export LD_LIBRARY_PATH="${dds_home}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
cd "${root}/intrinsic"
exec "${python}" -m app.tool "$@"
