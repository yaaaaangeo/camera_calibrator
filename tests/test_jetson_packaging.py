from __future__ import annotations

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
    result = subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / "install_jetson.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
