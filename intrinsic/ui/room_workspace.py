"""
camera_calibrator.ui.room_workspace
=======================================

카메라·LiDAR 캘리브레이션 화면(extrinsic/gui 의 RoomWindow)을 이 앱 안에 끼워 넣는 래퍼.

룸 툴 자체는 형제 폴더 extrinsic/gui 에 있고,
여기서는 그 RoomWindow 를 위젯으로 삽입하고 홈으로 돌아가는 버튼만 붙인다.

- 룸 패키지 위치: 환경변수 CAMERA_LIDAR_ROOT, 없으면 ../extrinsic
- import 는 카드를 처음 눌렀을 때 한 번만 한다. pyqtgraph/rosbags 등 룸 전용
  의존성이 없어도 내부파라미터 툴은 정상 동작하고, 룸 화면에 원인만 표시한다.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ui.theme import Theme

logger = logging.getLogger(__name__)

_DEFAULT_ROOM_ROOT = Path(__file__).resolve().parent.parent.parent / "extrinsic"


def room_package_root() -> Path:
    return Path(os.environ.get("CAMERA_LIDAR_ROOT", _DEFAULT_ROOM_ROOT)).expanduser().resolve()


def _import_room_window():
    root = room_package_root()
    if not (root / "gui" / "ui" / "room" / "window.py").is_file():
        raise FileNotFoundError(
            f"카메라·LiDAR 캘리브레이션 패키지를 찾을 수 없습니다: {root}\n"
            "환경변수 CAMERA_LIDAR_ROOT 로 extrinsic 폴더 경로를 지정하세요."
        )
    # 룸 쪽 cloud_view 가 pyqtgraph.opengl 을 쓰므로, 처음 import 되기 전에 Qt 바인딩을 고정한다.
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    if str(root) not in sys.path:
        sys.path.append(str(root))
    from gui.ui.room.window import RoomWindow  # noqa: PLC0415 - 지연 import 가 목적

    return RoomWindow


class RoomWorkspace(QWidget):
    back_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.room_window = None
        self._load_error: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        bar = QHBoxLayout()
        back = QPushButton("← Home")
        back.clicked.connect(self.back_requested.emit)
        bar.addWidget(back)
        title = QLabel("CAMERA–LiDAR")
        title.setStyleSheet("font-size: 15px; font-weight: 700;")
        bar.addWidget(title, stretch=1)
        root.addLayout(bar)

        self._body = QVBoxLayout()
        root.addLayout(self._body, stretch=1)

    @property
    def is_loaded(self) -> bool:
        return self.room_window is not None

    def ensure_loaded(self, vehicle: str = "") -> bool:
        """처음 열 때 RoomWindow 를 만든다. 실패하면 화면에 이유를 보여주고 False.

        `vehicle`: 홈 화면에서 고른 차량 - 그 차량으로 연다 (이미 열려 있으면 그 차량으로 바꾼다).
        """
        if self.room_window is not None:
            if vehicle and self.room_window.state.vehicle != vehicle:
                self.room_window.set_vehicle(vehicle)
            return True
        if self._load_error is not None:
            return False
        try:
            RoomWindow = _import_room_window()
            window = RoomWindow()
        except Exception as exc:  # noqa: BLE001 - 사용자에게 그대로 보여준다
            logger.exception("카메라·LiDAR 캘리브레이션 로드 실패")
            self._load_error = f"{type(exc).__name__}: {exc}"
            msg = QLabel(
                "카메라·LiDAR 캘리브레이션 화면을 불러오지 못했습니다.\n\n"
                f"{self._load_error}\n\n"
                "설치 확인: ./install.sh (requirements.txt 의 pyqtgraph, PyOpenGL, rosbags 등)\n\n"
                + traceback.format_exc(limit=3)
            )
            msg.setWordWrap(True)
            msg.setTextInteractionFlags(Qt.TextSelectableByMouse)
            msg.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            msg.setStyleSheet(f"color: {Theme.WARNING}; padding: 12px;")
            self._body.addWidget(msg)
            return False
        # QMainWindow 를 일반 위젯처럼 삽입한다 (자체 상태바/스텝 목록은 그대로 유지).
        window.setWindowFlags(Qt.Widget)
        self._body.addWidget(window)
        self.room_window = window
        # 결과 폴더(calib_result/<차량>/<세션>/)와 차량 프로필이 홈 화면에서 고른 차량으로 정해진다.
        if vehicle:
            window.set_vehicle(vehicle)
        return True

    def shutdown(self) -> None:
        """룸 페이지들의 백그라운드 QThread 와 실시간 토픽 구독 정리 (RoomWindow.closeEvent 와 동일)."""
        if self.room_window is None:
            return
        for page in self.room_window.pages:
            try:
                page.shutdown()
            except Exception:  # noqa: BLE001
                logger.exception("room page shutdown 실패")
