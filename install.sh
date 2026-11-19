#!/usr/bin/env bash
# Calibration Tool 설치 - 이 스크립트 하나로 실행 환경 전체를 만든다.
#
#   ./install.sh              실행 환경 (시스템 패키지 + venv + Python 패키지)
#   ./install.sh --dev        + 테스트 도구(pytest)
#   ./install.sh --no-system  apt 시스템 패키지 설치를 건너뜀 (sudo 권한이 없을 때)
#
# 하는 일
#   1) (Ubuntu/Debian) 빠진 시스템 패키지만 apt 로 설치 - sudo 비밀번호를 물을 수 있음
#        python3 / python3-venv, Qt(xcb·OpenGL) 런타임 라이브러리, 한글·이모지 폰트
#   2) 이 폴더에 venv/ 생성 (깨진 venv 가 있으면 다시 만듦)
#   3) requirements.txt 설치
#   4) cyclonedds: 휠이 없는 플랫폼(aarch64 등)이면 CycloneDDS C 라이브러리를
#      venv/opt/cyclonedds 에 소스 빌드한 뒤 설치
#   5) 설치 결과 확인 (import, Qt 플랫폼 플러그인 의존 라이브러리)
# 여러 번 실행해도 안전하다 - 이미 있는 것은 건너뛴다.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${root}"

dev=0
system=1
for arg in "$@"; do
    case "${arg}" in
        --dev) dev=1 ;;
        --no-system) system=0 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "알 수 없는 옵션: ${arg} (./install.sh --help)" >&2; exit 2 ;;
    esac
done

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[주의] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[실패] %s\033[0m\n' "$*" >&2; exit 1; }

warnings=()

# ---------------------------------------------------------------- 시스템 패키지
have_apt=0
if [[ ${system} -eq 1 ]] && command -v apt-get >/dev/null 2>&1 && command -v dpkg-query >/dev/null 2>&1; then
    have_apt=1
fi

as_root() {
    if [[ ${EUID} -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        die "관리자 권한이 필요합니다 (sudo 없음): $*"
    fi
}

apt_updated=0
apt_install() {  # 설치 안 된 패키지만 골라 설치
    local missing=()
    local pkg
    for pkg in "$@"; do
        if ! dpkg-query -W -f='${Status}' "${pkg}" 2>/dev/null | grep -q "install ok installed"; then
            missing+=("${pkg}")
        fi
    done
    [[ ${#missing[@]} -eq 0 ]] && return 0
    echo "apt 설치: ${missing[*]}"
    if [[ ${apt_updated} -eq 0 ]]; then
        as_root apt-get update
        apt_updated=1
    fi
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
}

# PySide6(Qt 6) xcb 플랫폼 플러그인 + OpenGL(3D 포인트클라우드 뷰) + 한글·이모지(버튼 아이콘) 폰트.
# Ubuntu 데스크톱에는 대부분 이미 있고, libxcb-cursor0 가 자주 빠져 있다.
qt_runtime_pkgs=(
    ca-certificates git procps
    libgl1 libegl1 libopengl0 libgl1-mesa-dri libglib2.0-0
    libfontconfig1 libfreetype6 libdbus-1-3
    libx11-6 libx11-xcb1 libxext6 libxrender1 libxi6 libsm6 libice6
    libxkbcommon0 libxkbcommon-x11-0
    libxcb1 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0
    libxcb-render0 libxcb-render-util0 libxcb-shape0 libxcb-shm0 libxcb-sync1
    libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1
    fonts-noto-cjk fonts-noto-color-emoji
)

find_python() {
    local candidate
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "${candidate}" >/dev/null 2>&1 &&
            "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

if [[ ${have_apt} -eq 1 ]]; then
    step "시스템 패키지 확인 (빠진 것만 apt 로 설치)"
    python_cmd="$(find_python || true)"
    if [[ -z "${python_cmd}" ]]; then
        apt_install python3
        python_cmd="$(find_python || true)"
    fi
    [[ -n "${python_cmd}" ]] || die "Python 3.10 이상을 찾지 못했습니다. (Ubuntu 22.04 이상 권장)"
    py_ver="$("${python_cmd}" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    # venv 모듈(ensurepip) - Ubuntu 는 python3.X-venv 로 따로 나뉘어 있다.
    apt_install "python${py_ver}-venv" "${qt_runtime_pkgs[@]}"
    echo "시스템 패키지 OK"
else
    python_cmd="$(find_python || true)"
    [[ -n "${python_cmd}" ]] || die "Python 3.10 이상이 필요합니다."
    if [[ ${system} -eq 1 ]]; then
        warn "apt 가 없는 시스템입니다. Qt 런타임 라이브러리(libxcb-cursor 등)와 python venv 모듈은 직접 설치하세요."
    fi
fi

# ---------------------------------------------------------------- venv
step "Python 가상환경 (venv/) - ${python_cmd} $("${python_cmd}" -c 'import platform; print(platform.python_version())')"
venv_ok() {
    [[ -x venv/bin/python ]] && venv/bin/python -c 'import sys, pip' >/dev/null 2>&1
}
if [[ -d venv ]] && ! venv_ok; then
    echo "기존 venv/ 가 깨져 있어 다시 만듭니다 (pip 없음 또는 다른 PC 에서 복사됨)."
    rm -rf venv
fi
if [[ ! -d venv ]]; then
    if ! "${python_cmd}" -m venv venv; then
        rm -rf venv
        die "venv 생성 실패. Ubuntu: sudo apt install python${py_ver:-3}-venv 후 다시 실행하세요."
    fi
fi
venv_ok || die "venv/ 에 pip 가 없습니다."
py=venv/bin/python
"${py}" -m pip install --upgrade --disable-pip-version-check pip setuptools wheel

# ---------------------------------------------------------------- Python 패키지
step "Python 패키지 (requirements.txt)"
tmp_req="$(mktemp)"
trap 'rm -f "${tmp_req}"' EXIT
# cyclonedds 는 휠이 없을 때 소스 빌드가 필요하므로 아래에서 따로 설치한다.
grep -viE '^[[:space:]]*cyclonedds' requirements.txt > "${tmp_req}"
"${py}" -m pip install --disable-pip-version-check -r "${tmp_req}"

# ---------------------------------------------------------------- cyclonedds
step "cyclonedds (ROS 2 토픽 직접 수신)"
dds_spec="$(grep -iE '^[[:space:]]*cyclonedds' requirements.txt | head -1 | sed 's/#.*//; s/[[:space:]]//g')"
dds_spec="${dds_spec:-cyclonedds}"
dds_home="${root}/venv/opt/cyclonedds"

build_cyclonedds_from_source() {
    # PyPI 에 이 플랫폼용 휠이 없을 때: C 라이브러리를 같은 버전 태그로 빌드한다.
    local version src
    version="$("${py}" -m pip index versions cyclonedds 2>/dev/null | sed -n 's/^cyclonedds (\(.*\))$/\1/p')"
    [[ -n "${version}" ]] || { warn "cyclonedds 버전 조회 실패"; return 1; }
    echo "휠이 없어 CycloneDDS ${version} 을 소스에서 빌드합니다 (수 분 소요)."
    if [[ ${have_apt} -eq 1 ]]; then
        apt_install build-essential cmake git "python${py_ver}-dev"
    fi
    for tool in cc cmake git; do
        command -v "${tool}" >/dev/null 2>&1 || { warn "${tool} 이 없어 빌드할 수 없습니다."; return 1; }
    done
    src="$(mktemp -d)"
    git clone --depth 1 --branch "${version}" https://github.com/eclipse-cyclonedds/cyclonedds.git "${src}/cyclonedds" &&
        cmake -S "${src}/cyclonedds" -B "${src}/build" -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="${dds_home}" -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
            -DBUILD_IDLC_TESTING=OFF -DENABLE_SSL=OFF -DENABLE_SECURITY=OFF &&
        cmake --build "${src}/build" --target install -j "$(nproc)" &&
        CYCLONEDDS_HOME="${dds_home}" "${py}" -m pip install --disable-pip-version-check \
            --no-binary cyclonedds "cyclonedds==${version}"
    local status=$?
    rm -rf "${src}"
    return ${status}
}

if "${py}" -c 'import cyclonedds.domain' >/dev/null 2>&1; then
    echo "이미 설치됨"
elif [[ "${CALIB_DDS_FROM_SOURCE:-0}" != 1 ]] &&
    "${py}" -m pip install --disable-pip-version-check --only-binary cyclonedds "${dds_spec}"; then
    :
elif build_cyclonedds_from_source; then
    :
else
    warnings+=("cyclonedds 설치 실패 - ROS 2 가 source 되지 않은 PC 에서 ROS 2 토픽 실시간 캡처를 못 합니다. (ROS 1, ROS 2+rclpy, bag 파일은 정상)")
fi

if [[ ${dev} -eq 1 ]]; then
    step "개발/테스트 도구 (requirements-dev.txt)"
    grep -vE '^[[:space:]]*-r' requirements-dev.txt > "${tmp_req}"
    "${py}" -m pip install --disable-pip-version-check -r "${tmp_req}"
fi

# ---------------------------------------------------------------- 확인
step "설치 확인"
[[ -d "${dds_home}/lib" ]] && export CYCLONEDDS_HOME="${dds_home}" LD_LIBRARY_PATH="${dds_home}/lib:${LD_LIBRARY_PATH:-}"
"${py}" - <<'PY' || die "필수 패키지 import 실패 (위 메시지 참고)"
import importlib
mods = ["PySide6.QtWidgets", "PySide6.QtOpenGLWidgets", "numpy", "scipy", "yaml", "cv2", "rosbags", "pyqtgraph", "OpenGL"]
for name in mods:
    importlib.import_module(name)
import cv2
assert hasattr(cv2, "aruco"), "cv2.aruco 없음 - opencv-contrib 가 아닌 opencv 가 섞여 설치됨"
print("  필수 패키지 OK  (OpenCV %s)" % cv2.__version__)
try:
    import cyclonedds.domain  # noqa: F401
    print("  cyclonedds OK")
except Exception as exc:  # 선택 기능
    print("  cyclonedds 없음:", exc)
PY

qxcb="$("${py}" -c 'import PySide6, pathlib; print(pathlib.Path(PySide6.__file__).parent / "Qt/plugins/platforms/libqxcb.so")')"
if [[ -f "${qxcb}" ]] && command -v ldd >/dev/null 2>&1; then
    missing_libs="$(ldd "${qxcb}" 2>/dev/null | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ')"
    if [[ -n "${missing_libs}" ]]; then
        warnings+=("Qt 화면 표시(xcb)에 필요한 라이브러리가 없습니다: ${missing_libs}")
    else
        echo "  Qt xcb 플랫폼 플러그인 의존 라이브러리 OK"
    fi
fi
QT_QPA_PLATFORM=offscreen "${py}" -c 'from PySide6.QtWidgets import QApplication; QApplication([])' \
    || die "Qt 초기화 실패"
echo "  Qt 초기화 OK"

if [[ ${#warnings[@]} -gt 0 ]]; then
    echo
    for message in "${warnings[@]}"; do warn "${message}"; done
fi
echo
echo "설치 완료. 실행: ./run.sh"
