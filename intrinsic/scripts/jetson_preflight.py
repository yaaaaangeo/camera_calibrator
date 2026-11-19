"""Fail-fast dependency and ROS visibility check for JetPack deployments."""

from __future__ import annotations

import platform
import sys


def main() -> int:
    failures: list[str] = []

    if platform.machine() != "aarch64":
        failures.append(f"architecture={platform.machine()} (aarch64 필요)")
    if sys.version_info[:2] != (3, 10):
        failures.append(f"python={platform.python_version()} (JetPack 6 기본 Python 3.10 필요)")

    try:
        import cv2
        if not hasattr(cv2, "aruco"):
            failures.append("cv2.aruco 없음 (opencv-contrib wheel 필요)")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"OpenCV import 실패: {exc}")

    try:
        import PySide6
        from PySide6.QtCore import qVersion
        _ = (PySide6.__version__, qVersion())
    except Exception as exc:  # noqa: BLE001
        failures.append(f"PySide6 import 실패: {exc}")

    try:
        import rclpy  # noqa: F401
        from sensor_msgs.msg import CompressedImage, Image  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        failures.append(
            f"ROS 2 Python import 실패: {exc} (source /opt/ros/humble/setup.bash 확인)"
        )

    if failures:
        print("Jetson preflight FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("Jetson preflight OK: aarch64 / Python / OpenCV contrib / PySide6 / ROS 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
