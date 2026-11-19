"""
app.tool
========

Calibration Tool 명령줄의 단일 입구. 저장소 루트에서:

    ./cli.sh intrinsic    [옵션...]   카메라 내부파라미터 계산          (= python -m app.cli)
    ./cli.sh camera-lidar [옵션...]   카메라·LiDAR 외부파라미터 (bag)   (= gui/room_calib.py)
    ./cli.sh automation   [명령...]   자동화: run / check / live / factory  (= python -m automation)
    ./cli.sh gui          [옵션...]   GUI 실행                          (= ./run.sh)

각 명령 뒤의 옵션은 그대로 넘긴다 (`./cli.sh intrinsic --help`).
예전 진입점(python -m app.cli, python -m automation, gui/room_calib.py)도 그대로 동작한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXTRINSIC = Path(__file__).resolve().parents[2] / "extrinsic"


def _intrinsic(argv):
    from app.cli import main
    return main(argv)


def _camera_lidar(argv):
    if str(_EXTRINSIC) not in sys.path:
        sys.path.insert(0, str(_EXTRINSIC))
    from gui.room_calib import main
    return main(argv)


def _automation(argv):
    from automation.cli import main
    return main(argv)


def _gui(argv):
    sys.argv = [sys.argv[0], *argv]
    from app.main import main
    return main()


COMMANDS = {
    "intrinsic": (_intrinsic, "카메라 내부파라미터 계산 (이미지 폴더 / bag)"),
    "camera-lidar": (_camera_lidar, "카메라·LiDAR 외부파라미터 (캘리브레이션 룸 bag)"),
    "automation": (_automation, "자동화: run / check / live / factory"),
    "gui": (_gui, "GUI 실행"),
}
ALIASES = {"lidar": "camera-lidar", "extrinsic": "camera-lidar", "room": "camera-lidar", "auto": "automation"}


def usage() -> str:
    lines = ["사용법: ./cli.sh <명령> [옵션...]", "", "명령:"]
    lines += [f"  {name:<14} {desc}" for name, (_fn, desc) in COMMANDS.items()]
    lines += ["", "각 명령의 옵션: ./cli.sh <명령> --help"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    name = ALIASES.get(argv[0], argv[0])
    if name not in COMMANDS:
        print(f"알 수 없는 명령: {argv[0]}\n\n{usage()}", file=sys.stderr)
        return 2
    sys.argv[0] = f"./cli.sh {name}"  # 하위 명령 도움말에 이 이름이 나온다
    return int(COMMANDS[name][0](argv[1:]) or 0)


if __name__ == "__main__":
    sys.exit(main())
