"""Room main window: the step list on the left, the active step on the right.

Above the step: which vehicle and session this is, and -- on the steps that are
done once per camera -- which camera is being looked at. Steps are switched by
clicking them in the list (no header or previous/next buttons).
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ui.theme import Theme

from gui.core import session
from gui.ui.room import RoomState
from gui.ui.room.pages_input import CameraStep, CamBoardsStep, DataStep
from gui.ui.room.pages_lidar import LidarBoardsStep
from gui.ui.room.pages_result import OverlayStep, SaveStep, SolveStep


class RoomWindow(QtWidgets.QMainWindow):
    vehicle_changed = QtCore.Signal(str)  # the app's home screen follows it (shared.vehicle)

    def __init__(self, vehicle: str | None = None):
        super().__init__()
        self.setWindowTitle("카메라·LiDAR 캘리브레이션")
        self.resize(1600, 980)
        self.state = RoomState()
        self.pages = [DataStep(self.state), LidarBoardsStep(self.state), CameraStep(self.state),
                      CamBoardsStep(self.state), SolveStep(self.state), OverlayStep(self.state),
                      SaveStep(self.state)]

        self.vehicle_label = QtWidgets.QLabel()
        self.vehicle_label.setToolTip("차량은 홈 화면에서 고릅니다")
        self.session_label = QtWidgets.QLabel()
        self.cam_label = QtWidgets.QLabel("카메라")
        self.cam_combo = QtWidgets.QComboBox()
        self.cam_combo.setMinimumWidth(260)
        self.cam_combo.currentIndexChanged.connect(self._camera_changed)
        self.nav = QtWidgets.QListWidget()
        self.nav.setMinimumWidth(170)
        self.nav.setMaximumWidth(320)
        self.nav.setSpacing(2)
        self.stack = QtWidgets.QStackedWidget()
        for page in self.pages:
            page.changed.connect(self._refresh_nav)
            page.cameras_changed.connect(self._fill_cameras)
            page.select_camera.connect(self._select_camera)
            self.stack.addWidget(page)
            self.nav.addItem(QtWidgets.QListWidgetItem())
        self.nav.currentRowChanged.connect(self._goto)


        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.vehicle_label)
        bar.addWidget(self.session_label)
        bar.addStretch(1)
        bar.addWidget(self.cam_label)
        bar.addWidget(self.cam_combo)

        right = QtWidgets.QVBoxLayout()
        right.addLayout(bar)
        right.addWidget(self.stack, 1)
        rw = QtWidgets.QWidget()
        rw.setLayout(right)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.nav)
        split.addWidget(rw)
        split.setStretchFactor(1, 1)
        split.setSizes([230, 1370])
        split.setChildrenCollapsible(False)
        self.setCentralWidget(split)

        self._timer = QtCore.QTimer(self, interval=1000)
        self._timer.timeout.connect(self._refresh_bar)
        self._timer.start()
        if vehicle:
            self.set_vehicle(vehicle)
        self.nav.setCurrentRow(0)
        self._refresh_nav()
        self._refresh_bar()

    # ------------------------------------------------------------- vehicle

    def set_vehicle(self, name: str):
        s = self.state
        name = session.safe_name(name)
        if s.vehicle and s.vehicle != name and s.cloud is not None:
            # Results of another vehicle must not be saved under this one.
            s.session_id = ""
        s.vehicle = name
        s.profile = session.load_profile(name)
        for p in self.pages:
            p.on_vehicle()
        self._refresh_bar()
        self._refresh_nav()
        self.vehicle_changed.emit(name)

    def _refresh_bar(self):
        s = self.state
        self.vehicle_label.setText(f"🚚  {s.vehicle or '차량 미선택 — 홈 화면에서 고르세요'}")
        self.vehicle_label.setStyleSheet("font-weight: 600;" if s.vehicle else f"color: {Theme.BAD}; font-weight: 600;")
        if s.session_id:
            self.session_label.setText(f"세션 <b>{session.session_label(s.session_id)}</b> · {s.session_dir}")
        else:
            self.session_label.setText("<i>캡처 전 — 세션은 캡처 시각으로 정해집니다</i>")

    # ------------------------------------------------------------- cameras

    def _fill_cameras(self):
        s = self.state
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        for c in s.cams:
            self.cam_combo.addItem(f"{c.name}   ({c.topic})")
        self.cam_combo.setCurrentIndex(min(s.cur, len(s.cams) - 1))
        self.cam_combo.blockSignals(False)
        self._refresh_bar()
        self._refresh_nav()
        self._update_cam_bar()

    def _select_camera(self, i: int):
        if 0 <= i < self.cam_combo.count():
            self.cam_combo.setCurrentIndex(i)

    def _camera_changed(self, i):
        if i < 0:
            return
        self.state.cur = i
        page = self.pages[self.nav.currentRow()]
        page.on_enter()
        self._refresh_nav()

    def _update_cam_bar(self):
        page = self.pages[max(self.nav.currentRow(), 0)]
        show = page.per_camera and len(self.state.cams) > 0
        self.cam_label.setVisible(show)
        self.cam_combo.setVisible(show)

    # ----------------------------------------------------------------- nav

    def closeEvent(self, e):
        for p in self.pages:
            p.shutdown()
        super().closeEvent(e)

    def _goto(self, row):
        if row < 0:
            return
        page = self.pages[row]
        self.stack.setCurrentIndex(row)
        self._update_cam_bar()
        page.on_enter()
        self._refresh_nav()

    def _refresh_nav(self):
        for i, page in enumerate(self.pages):
            mark = "✓" if page.is_complete() else "○"
            st = page.status_text()
            self.nav.item(i).setText(f" {mark}  {page.title}\n      {st}" if st else f" {mark}  {page.title}")
        self._refresh_bar()
