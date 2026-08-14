"""
camera_calibrator.app.main
==============================

실행: python -m app.main  (camera_calibrator/ 디렉토리에서)

옵션 (Qt 자체 옵션과 섞여도 되도록 argparse 대신 간단히 직접 파싱한다):
    --verbose / -v   : 콘솔 진단 로그 상세도 (여러 번 줄수록 자세해짐: INFO -> DEBUG)
    --log-file PATH  : DEBUG 레벨 전체 로그를 이 파일에 남김 (버그 리포트용)

예: python -m app.main -v --log-file ~/camera_calibrator_debug.log
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from calibration.log_utils import setup_logging
from ui.main_window import MainWindow
from app.zen import print_zen_greeting


def _parse_logging_args(argv: list[str]) -> tuple[int, str | None]:
    """QApplication에 넘길 argv는 그대로 두고, 로깅 관련 옵션만 따로 뽑아낸다.

    GUI라 argparse로 --help까지 만들 필요는 없고(Qt 옵션과 충돌 위험도 있음),
    로깅 옵션 두 개만 가볍게 스캔한다.
    """
    verbosity = argv.count("-v") + argv.count("--verbose")
    log_file: str | None = None
    if "--log-file" in argv:
        idx = argv.index("--log-file")
        if idx + 1 < len(argv):
            log_file = argv[idx + 1]
    return verbosity, log_file


def main() -> int:
    verbosity, log_file = _parse_logging_args(sys.argv[1:])
    setup_logging(verbosity=verbosity, log_file=log_file)

    # 이스터에그: 캘리브레이션 로직과 완전히 무관하게, 앱이 뜰 때마다 터미널에
    # "The Zen of Camera Calibration" 중 한 줄을 무작위로 인사처럼 띄운다.
    print_zen_greeting()

    app = QApplication(sys.argv)
    app.setApplicationName("Camera Calibration Tool")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
