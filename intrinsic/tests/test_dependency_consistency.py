"""루트 requirements.txt 가 두 기능(내부파라미터 + 카메라·LiDAR 룸)에 필요한 패키지를 모두 선언하는지 확인.

예전에 requirements.txt 에서 PySide6 가 빠져 `pip install -r requirements.txt` 후
GUI 가 ImportError 로 바로 죽은 적이 있다. 패키지를 하나로 합친 뒤에도 같은 실수가
없도록 import 이름이 아니라 배포 이름 기준으로 확인한다.
"""

from __future__ import annotations

import re
from pathlib import Path

_REQUIREMENTS = Path(__file__).resolve().parents[2] / "requirements.txt"

_REQUIRED = {
    # 공통 / 내부파라미터
    "pyside6", "numpy", "opencv-contrib-python-headless", "scipy", "pyyaml",
    # bag 읽기 (두 기능 모두)
    "rosbags",
    # 카메라·LiDAR 룸의 3D 뷰
    "pyqtgraph", "pyopengl",
    # 카메라·LiDAR 룸 실시간 캡처 (ROS 2 미설치 PC)
    "cyclonedds",
}


def _declared() -> set[str]:
    names = set()
    for line in _REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            names.add(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower())
    return names


def test_root_requirements_cover_both_tools():
    missing = _REQUIRED - _declared()
    assert not missing, f"requirements.txt 에 빠진 패키지: {sorted(missing)}"


def test_only_one_opencv_distribution():
    opencv = {n for n in _declared() if n.startswith("opencv")}
    assert opencv == {"opencv-contrib-python-headless"}, (
        "opencv-* 배포판은 하나만 선언해야 한다 (같은 cv2 모듈을 놓고 충돌): " + str(opencv)
    )
