#!/usr/bin/env bash
# JetPack 5.1.2 / Jetson AGX Orin Developer Kit 64GB 전용 설치 스크립트
# (Jetson Linux R35.4.x, Ubuntu 20.04, aarch64, ROS 1 Noetic).
#
# scripts/install_jetson.sh(JetPack 6.2.1 / Ubuntu 22.04 / ROS 2 Humble)와
# 짝을 이루지만, 이 스크립트는 venv를 --system-site-packages로 만들지
# 않는다. ROS 1 Noetic의 apt 패키지(rospy/sensor_msgs/cv_bridge)는 Ubuntu
# 20.04 기본 Python 3.8용으로 빌드되어 있고, 이 프로젝트는 JetPack 5.1.2
# 에서도 Python 3.10을 요구한다(requirements-jetson-jp512.txt의 numpy>=1.26/
# PySide6 6.5.3 wheel 때문에 3.8로 낮추지 않음) - 특히 cv_bridge는 컴파일된
# Boost.Python 확장이라, 3.8용으로 빌드된 .so를 3.10 인터프리터가 아예 로드할
# 수 없다(단순 경고가 아니라 ImportError). 그래서 README(방법 E)가 이미
# 명시한 대로 ROS1 live-topic 연동은 이 venv 밖(ROS 환경을 source한 별도
# 실행)에서 다룬다 - 이 스크립트는 그 결정을 코드로 옮긴 것뿐이며, ROS
# 패키지를 이 venv 안에 절대 설치하지 않는다.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${project_dir}/.venv-jetson-jp512"

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
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "20.04" ]]; then
    echo "ERROR: JetPack 5.1.2의 Ubuntu 20.04만 지원합니다 (현재: ${ID:-?} ${VERSION_ID:-?})." >&2
    exit 2
fi

if ! command -v python3.10 >/dev/null 2>&1; then
    echo "ERROR: python3.10을 찾을 수 없습니다." >&2
    echo "  Ubuntu 20.04 기본 Python은 3.8이지만 이 프로젝트는 JetPack 5.1.2에서도" >&2
    echo "  3.10을 요구합니다(numpy>=1.26 / PySide6 6.5.3 wheel 요구사항 때문에" >&2
    echo "  3.8로 낮추지 않습니다). 예: deadsnakes PPA로 설치할 수 있습니다 -" >&2
    echo "    sudo add-apt-repository ppa:deadsnakes/ppa" >&2
    echo "    sudo apt-get update && sudo apt-get install -y python3.10 python3.10-venv" >&2
    echo "  이 스크립트는 apt 저장소를 임의로 추가하지 않습니다 - 직접 설치한 뒤" >&2
    echo "  다시 실행하세요." >&2
    exit 2
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3.10-venv \
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

# ROS1 패키지와의 ABI 불일치 때문에 --system-site-packages를 쓰지 않는다
# (위 헤더 설명 참조) - 순수 격리된 venv.
python3.10 -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
"${venv_dir}/bin/python" -m pip install -r "${project_dir}/requirements-jetson-jp512.txt"

# pyproject의 일반 데스크톱 OpenCV dependency가 다시 설치되지 않게 한다.
"${venv_dir}/bin/python" -m pip install --no-deps -e "${project_dir}"

"${venv_dir}/bin/python" "${project_dir}/scripts/jetson_jp512_preflight.py"

echo
echo "설치 완료. 실행할 때마다 다음 순서로 시작하세요:"
echo "  source ${venv_dir}/bin/activate"
echo "  python -m app.main"
echo
echo "ROS1 Noetic live topic을 사용하려면(rospy/sensor_msgs/cv_bridge는 이"
echo "venv에 설치되지 않았습니다 - apt/ROS 환경에서 관리, README 방법 E 참고):"
echo "  source /opt/ros/noetic/setup.bash"
