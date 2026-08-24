#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${project_dir}/.venv-jetson"

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "ERROR: 이 설치 스크립트는 JetPack aarch64 전용입니다 (현재: $(uname -m))." >&2
    exit 2
fi

if [[ ! -r /etc/os-release ]]; then
    echo "ERROR: /etc/os-release를 읽을 수 없습니다." >&2
    exit 2
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
    echo "ERROR: JetPack 6의 Ubuntu 22.04만 지원합니다 (현재: ${ID:-?} ${VERSION_ID:-?})." >&2
    exit 2
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
    echo "ERROR: 먼저 'source /opt/ros/humble/setup.bash'를 실행하세요." >&2
    exit 2
fi

if [[ "${ROS_DISTRO}" != "humble" ]]; then
    echo "ERROR: JetPack 6 패키지는 ROS 2 Humble 기준입니다 (현재: ${ROS_DISTRO})." >&2
    exit 2
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3-venv \
    ros-humble-rclpy \
    ros-humble-sensor-msgs \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libdbus-1-3 \
    libfontconfig1 \
    libx11-xcb1 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libxcb1 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxi6 \
    libxrender1

# rclpy/sensor_msgs는 apt로 설치되므로 system-site-packages가 필수다.
python3 -m venv --system-site-packages "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
"${venv_dir}/bin/python" -m pip install -r "${project_dir}/requirements-jetson.txt"

# pyproject의 일반 데스크톱 OpenCV dependency가 다시 설치되지 않게 한다.
"${venv_dir}/bin/python" -m pip install --no-deps -e "${project_dir}"

"${venv_dir}/bin/python" "${project_dir}/scripts/jetson_preflight.py"

echo
echo "설치 완료. 실행할 때마다 다음 순서로 시작하세요:"
echo "  source /opt/ros/humble/setup.bash"
echo "  source ${venv_dir}/bin/activate"
echo "  camera-calibrator --verbose"
