"""
camera_calibrator.app.main
==============================

실행: python -m app.main  (camera_calibrator/ 디렉토리에서)
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Camera Calibration Tool")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
