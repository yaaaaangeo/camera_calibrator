from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements-jetson-jp512.txt"
PREFLIGHT = ROOT / "scripts" / "jetson_jp512_preflight.py"
INSTALLER = ROOT / "scripts" / "install_jetson_jp512.sh"


def _requirement_names() -> set[str]:
    names: set[str] = set()
    for line in REQ.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.split("==", 1)[0].lower())
    return names


def test_jetson_jp512_requirements_include_core_gui_dependencies():
    assert _requirement_names() == {
        "numpy",
        "scipy",
        "opencv-contrib-python-headless",
        "pyyaml",
        "shiboken6",
        "pyside6-essentials",
        "pyside6-addons",
        "pyside6",
    }


def test_jetson_jp512_requirements_keep_ros_out_of_pip_profile():
    forbidden = {"rosbag", "rospy", "sensor_msgs", "cv_bridge", "rclpy"}
    assert _requirement_names().isdisjoint(forbidden)


def test_jetson_jp512_requirements_do_not_keep_unused_desktop_dependencies():
    unused = {"matplotlib", "jsonschema"}
    assert _requirement_names().isdisjoint(unused)


def test_jetson_jp512_preflight_checks_required_capabilities():
    text = PREFLIGHT.read_text(encoding="utf-8")
    for needle in (
        "platform.machine()",
        "sys.version_info[:2] != (3, 10)",
        "_has_glibc_at_least(2, 31)",
        'hasattr(cv2, "aruco")',
        'hasattr(cv2, "findCirclesGrid")',
        'hasattr(cv2, "calibrateCameraRO")',
        'hasattr(cv2, "fisheye")',
        "import PySide6",
    ):
        assert needle in text


def test_jetson_jp512_installer_is_explicit_about_ros1_live_topic_limit():
    text = INSTALLER.read_text(encoding="utf-8")
    assert "Python 3.10 venv에서는 ROS1 Noetic" in text
    assert "live topic을 직접 지원하지 않습니다" in text
