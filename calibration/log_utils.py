"""
camera_calibrator.calibration.log_utils
============================================

로깅 설정을 한 곳에서 관리한다.

지금까지 이 프로젝트는 로그를 전혀 안 썼다 - `app/cli.py`의 사용자용
요약/표 출력을 빼면 나머지 모듈(특히 ros_live.py, rosbag_reader.py)은
문제가 생겨도 예외 메시지 말고는 아무 흔적이 안 남았다. README 5.2번이
스스로 인정하듯 실시간 ROS 구독은 "실제 ROS 환경에서만 재현되는" 문제가
있어서, 사용자가 버그를 재현해 캡처해 보낼 방법이 없으면 원인 파악이
사실상 불가능하다. 이 모듈은 그 문제를 풀기 위한 최소한의 장치:

  - `setup_logging()`을 앱 시작 시(CLI/GUI 진입점) 한 번 호출하면 이후
    모든 모듈의 `logging.getLogger(__name__)` 호출이 여기서 설정한 콘솔/
    파일 핸들러를 그대로 따른다 (표준 로깅 전파 - 각 모듈은 이 파일을
    몰라도 됨).
  - 콘솔 출력 레벨은 `--verbose`/`--quiet`로 조정 가능.
  - `--log-file`을 주면 콘솔 레벨과 무관하게 DEBUG 레벨 전체를 파일에
    남긴다 - "사용자 컴퓨터에서만 재현되는" 문제는 이 파일을 받는 게
    제일 빠르다.

주의: 이 모듈은 `app/cli.py`가 만드는 사용자용 진행상황/결과 출력(표,
요약)을 대체하지 않는다. 그런 출력은 "프로그램의 정상적인 결과물"이라
지금처럼 stdout에 그대로 남아야 한다 - 이 모듈이 다루는 건 "진단/디버깅용
흔적"이다.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def setup_logging(
    verbosity: int = 0,
    quiet: bool = False,
    log_file: str | None = None,
) -> None:
    """루트 로거를 설정한다. 앱 진입점(app/main.py, app/cli.py)에서 딱 한 번 호출.

    Args:
        verbosity: `-v`/`--verbose`를 준 횟수. 0=WARNING, 1=INFO, 2 이상=DEBUG.
        quiet: True면 콘솔 출력 레벨을 ERROR로 올린다 (파일 로그에는 영향 없음 -
            `--quiet`이어도 `--log-file`을 같이 주면 파일에는 전체가 남는다).
        log_file: 지정하면 이 경로에 DEBUG 레벨 전체를 남긴다. 상위 디렉터리가
            없으면 만든다.
    """
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # 실제 필터링은 각 핸들러의 레벨이 담당

    # 재호출(테스트, 여러 번 실행되는 장기 프로세스 등) 대비 - 핸들러 중복 방지
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console_level = logging.WARNING
    if verbosity >= 2:
        console_level = logging.DEBUG
    elif verbosity == 1:
        console_level = logging.INFO
    if quiet:
        console_level = logging.ERROR

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root.addHandler(file_handler)

    _CONFIGURED = True


def is_configured() -> bool:
    """setup_logging()이 이미 호출됐는지 (테스트/재진입 방지용)."""
    return _CONFIGURED
