"""Fail-fast dependency check for JetPack 5.1.2 / Ubuntu 20.04 / Python 3.10."""

from __future__ import annotations

import platform
import sys


def _has_glibc_at_least(major: int, minor: int) -> bool:
    try:
        current = platform.libc_ver()[1]
        cur_major, cur_minor = (int(part) for part in current.split(".")[:2])
    except Exception:
        return False
    return (cur_major, cur_minor) >= (major, minor)


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    if platform.machine() != "aarch64":
        failures.append(f"architecture={platform.machine()} (aarch64 required)")
    if sys.version_info[:2] != (3, 10):
        failures.append(f"python={platform.python_version()} (Python 3.10 required)")
    if not _has_glibc_at_least(2, 31):
        failures.append(f"glibc={platform.libc_ver()[1] or 'unknown'} (glibc >= 2.31 required)")

    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import matplotlib  # noqa: F401
        import yaml  # noqa: F401
        import jsonschema  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        failures.append(f"scientific/runtime import failed: {exc}")

    try:
        import cv2

        if not hasattr(cv2, "aruco"):
            failures.append("cv2.aruco missing (opencv-contrib package required)")
        if not hasattr(cv2, "findCirclesGrid"):
            failures.append("cv2.findCirclesGrid missing")
        if not hasattr(cv2, "calibrateCameraRO"):
            failures.append("cv2.calibrateCameraRO missing")
        if not hasattr(cv2, "fisheye"):
            failures.append("cv2.fisheye missing")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"OpenCV import/capability check failed: {exc}")

    try:
        import PySide6
        from PySide6.QtCore import qVersion

        _ = (PySide6.__version__, qVersion())
    except Exception as exc:  # noqa: BLE001
        failures.append(f"PySide6 import failed: {exc}")

    try:
        import rospy  # noqa: F401
        from sensor_msgs.msg import CompressedImage, Image  # noqa: F401

        warnings.append("ROS1 Noetic Python packages are visible in the current environment.")
    except Exception:
        warnings.append("ROS1 Noetic Python packages not visible; source the ROS environment if live ROS is needed.")

    if failures:
        print("JetPack 5.1.2 preflight FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        for warning in warnings:
            print(f"  note: {warning}", file=sys.stderr)
        return 1

    print("JetPack 5.1.2 preflight OK: aarch64 / Python 3.10 / glibc / OpenCV contrib / PySide6")
    for warning in warnings:
        print(f"note: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
