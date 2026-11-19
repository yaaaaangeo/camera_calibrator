from __future__ import annotations

import os
import pytest
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_jetson_requirements_use_jammy_arm64_compatible_wheels():
    requirements = (ROOT / "requirements-jetson.txt").read_text(encoding="utf-8")

    assert "PySide6==6.7.3" in requirements
    assert "opencv-contrib-python-headless==4.10.0.84" in requirements
    assert "numpy==1.26.4" in requirements
    assert "\nopencv-contrib-python==" not in requirements


def test_jetson_installer_preserves_ros_apt_packages_and_avoids_core_deps():
    installer = (ROOT / "scripts" / "install_jetson.sh").read_text(encoding="utf-8")

    assert "--system-site-packages" in installer
    assert "--no-deps -e" in installer
    assert "source /opt/ros/humble/setup.bash" in installer


def test_jetson_installer_has_valid_bash_syntax():
    if shutil.which("bash") is None:
        pytest.skip("bash is not available on this test host")
    result = subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / "install_jetson.sh")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = ((result.stderr or "") + (result.stdout or "")).replace("\x00", "")
    if os.name == "nt" and "E_ACCESSDENIED" in output:
        pytest.skip("bash/WSL is present but inaccessible on this Windows test host")
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Phase C-2 - JetPack 5.1.2 install profile (scripts/install_jetson_jp512.sh).
# ---------------------------------------------------------------------------

def test_jp512_installer_avoids_system_site_packages_and_ros_pip_installs():
    """JetPack 5.1.2의 ROS1 Noetic apt 패키지(rospy/sensor_msgs/cv_bridge)는
    Ubuntu 20.04 기본 Python 3.8용으로 빌드되어 있어, 이 프로젝트가 요구하는
    Python 3.10 venv와 ABI가 맞지 않는다(cv_bridge는 컴파일된 확장이라 특히
    치명적) - JetPack 6 installer(install_jetson.sh)와 달리
    --system-site-packages를 쓰면 안 되고, ROS 패키지를 pip로 설치하려는
    시도가 있어서도 안 된다."""
    installer = (ROOT / "scripts" / "install_jetson_jp512.sh").read_text(encoding="utf-8")
    executable_lines = [line for line in installer.splitlines() if not line.strip().startswith("#")]
    executable_text = "\n".join(executable_lines)

    assert "--system-site-packages" not in executable_text
    assert "--no-deps -e" in installer
    assert "python3.10" in installer
    assert "requirements-jetson-jp512.txt" in installer
    assert "jetson_jp512_preflight.py" in installer
    for forbidden in ("pip install rospy", "pip install cv_bridge", "pip install sensor_msgs", "pip install torch"):
        assert forbidden not in installer


def test_jp512_installer_checks_ubuntu_20_04_and_aarch64():
    installer = (ROOT / "scripts" / "install_jetson_jp512.sh").read_text(encoding="utf-8")
    assert "aarch64" in installer
    assert '"20.04"' in installer


def test_jp512_installer_has_valid_bash_syntax():
    if shutil.which("bash") is None:
        pytest.skip("bash is not available on this test host")
    result = subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / "install_jetson_jp512.sh")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = ((result.stderr or "") + (result.stdout or "")).replace("\x00", "")
    if os.name == "nt" and "E_ACCESSDENIED" in output:
        pytest.skip("bash/WSL is present but inaccessible on this Windows test host")
    assert result.returncode == 0, result.stderr
