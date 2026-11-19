"""Calibration-room GUI: shared state, a background-task helper, an image canvas.

A step list on the left, each step inspectable before moving on: capture, find
the boards in the cloud (once, inside the ROI), then for every camera choose
the lens, find the boards in the image, pair and solve; look at the overlays,
save them all under the vehicle and the capture's date.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from shared.qt import TaskRunner, bgr_to_qpixmap
from ui.theme import Theme


@dataclass
class CamSlot:
    """One camera in the session: its pictures, lens, boards and answer."""

    topic: str
    name: str  # position, e.g. FRONT_LEFT -- the result folder
    images: list = field(default_factory=list)

    camera_name: str = ""
    camera: object = None
    ranking: list = field(default_factory=list)  # [(preset, boards, px)]
    ranked: bool = False  # ranking was run (an empty one then means: no preset fits)

    cam_boards: list = field(default_factory=list)
    cam_excluded: set = field(default_factory=set)
    detected: bool = False  # board detection was run on this camera

    result: object = None
    excluded_pairs: set = field(default_factory=set)
    solver: str = "auto"
    prior: object = None  # extrinsic_io.Extrinsic, for comparison and as a matching hint
    saved_to: str = ""
    direction: str = ""  # which way it faces (room.DIRECTIONS), to tell identical boards apart

    def image(self, k: int | None = None):
        if not self.images:
            return None
        return self.images[len(self.images) // 2 if k is None else min(k, len(self.images) - 1)]

    def clear_from(self, stage: str):
        if stage in ("camera",):
            self.cam_boards, self.cam_excluded, self.detected = [], set(), False
        if stage in ("camera", "cam_boards", "lidar_boards"):
            self.result, self.excluded_pairs, self.saved_to = None, set(), ""


# Per-camera fields that the steps read and write as if there were one camera:
# on the state they mean "the selected camera's".
_CAM_FIELDS = ("images", "camera_name", "camera", "cam_boards", "cam_excluded", "result",
               "excluded_pairs", "solver", "prior", "saved_to")


class RoomState:
    """Everything the steps hand each other. Later steps read, earlier ones write.

    One LiDAR capture, shared; any number of cameras (`cams`), one of them
    selected (`cur`). The per-camera fields (`camera`, `cam_boards`, `result`,
    ...) read and write the selected camera, so a step written for one camera
    works on whichever is selected.
    """

    def __init__(self):
        self.vehicle: str = ""
        self.profile = None  # session.VehicleProfile
        self.forward = None  # vehicle forward in the LiDAR frame, deg (room.solve_cameras)
        self.session_id: str = ""  # capture time, names the result folder

        self.bag: str = ""
        self.lidar_topics: list[str] = []
        self.at: float = 0.5
        self.sweeps: int = 1
        self.frames: int = 1

        self.cloud: np.ndarray | None = None
        self.single: np.ndarray | None = None
        self.intensity: np.ndarray | None = None
        self.n_sweeps: int = 0

        self.target_name: str = "원본 CAD"
        self.target = None
        self.layouts: tuple = ()

        self.roi: list = []  # FilterBox list; empty = whole cloud
        self.lidar_boards: list = []
        self.lidar_excluded: set = set()
        self.lidar_stats: dict = {}
        self.method: str = "occupancy"

        self.cams: list[CamSlot] = []
        self.cur: int = 0

        self.live = None  # LiveSession while subscribed
        self.snapshot_path: str = ""  # mcap written from the last capture

    # ---------------------------------------------------------- cameras

    @property
    def slot(self) -> CamSlot | None:
        if not self.cams:
            return None
        return self.cams[min(max(self.cur, 0), len(self.cams) - 1)]

    @property
    def camera_topic(self) -> str:
        return self.slot.topic if self.slot else ""

    def image(self, k: int | None = None):
        return self.slot.image(k) if self.slot else None

    def set_cameras(self, slots: list[CamSlot]):
        """A new capture: these cameras, nothing solved yet. Lens choices carry over by topic."""
        old = {c.topic: c for c in self.cams}
        for c in slots:
            o = old.get(c.topic)
            if o is not None and c.camera is None:
                c.camera_name, c.camera, c.prior = o.camera_name, o.camera, o.prior
            if o is not None and not c.direction:
                c.direction = o.direction
        self.cams = slots
        self.cur = 0

    # ------------------------------------------------------------ paths

    @property
    def session_dir(self):
        from gui.core import session

        if not self.vehicle or not self.session_id:
            return None
        return session.vehicle_dir(self.vehicle) / self.session_id

    def invalidate_from(self, stage: str, every_camera: bool = False):
        """Drop everything computed after `stage`, so nothing stale is shown.

        data          everything after the capture, every camera
        target        boards in both sensors and every result (the board changed)
        camera        the selected camera's boards and result (its lens changed)
        cam_boards    the selected camera's result
        lidar_boards  every camera's result (they all share the LiDAR boards)
        """
        if stage in ("data", "target"):
            self.lidar_boards, self.lidar_excluded, self.lidar_stats = [], set(), {}
            for c in self.cams:
                c.clear_from("camera")
            return
        if stage == "lidar_boards" or every_camera:
            for c in self.cams:
                c.clear_from(stage)
        elif self.slot is not None:
            self.slot.clear_from(stage)


def _cam_property(name):
    def get(self):
        s = self.slot
        if s is None:
            return [] if name in ("images", "cam_boards") else (set() if name in ("cam_excluded", "excluded_pairs") else
                                                                  ("" if name in ("camera_name", "saved_to") else
                                                                   ("auto" if name == "solver" else None)))
        return getattr(s, name)

    def put(self, value):
        if self.slot is None:
            raise RuntimeError(f"카메라가 없어 {name} 을(를) 정할 수 없습니다")
        setattr(self.slot, name, value)

    return property(get, put)


for _f in _CAM_FIELDS:
    setattr(RoomState, _f, _cam_property(_f))


class RoomStep(QtWidgets.QWidget):
    """A page of the room window; holds the shared room state."""

    changed = QtCore.Signal()
    cameras_changed = QtCore.Signal()  # the camera list itself changed (new capture, renamed)
    select_camera = QtCore.Signal(int)  # ask the window to select this camera
    title = "Step"
    subtitle = ""
    per_camera = False  # done once per camera: the window shows the camera selector

    def __init__(self, state: RoomState, parent=None):
        super().__init__(parent)
        self.state = state
        self._runner = TaskRunner(self)

    def is_complete(self) -> bool:
        return True

    def status_text(self) -> str:
        return ""

    def on_enter(self):
        pass

    def on_vehicle(self):
        """The vehicle (and so its profile) was chosen or changed."""

    def shutdown(self):
        self._runner.shutdown()

    # ---------------------------------------------------------- background

    def run_task(self, fn, on_done, on_fail=None, busy: QtWidgets.QWidget | None = None):
        """Run `fn()` off the UI thread, then `on_done(result)` back on it (shared.qt.TaskRunner)."""
        self._runner.run(fn, on_done, on_fail, busy)


class ImageCanvas(QtWidgets.QGraphicsView):
    """An image you can zoom with the wheel (towards the cursor) and drag around.

    Double-click fits it back to the window. The view keeps its zoom across
    updates, so changing an overlay setting does not throw away a close look.
    """

    clicked = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self._item = QtWidgets.QGraphicsPixmapItem()
        self._item.setTransformationMode(QtCore.Qt.SmoothTransformation)
        self.scene().addItem(self._item)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QtGui.QColor(Theme.CANVAS_BG))
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        self._fitted = True
        self._size = None
        self.setMinimumSize(400, 300)

    def set_image(self, bgr: np.ndarray | None):
        if bgr is None:
            self._item.setPixmap(QtGui.QPixmap())
            return
        h, w = bgr.shape[:2]
        self._item.setPixmap(bgr_to_qpixmap(bgr))
        self.scene().setSceneRect(0, 0, w, h)
        if self._fitted or self._size != (w, h):
            self.fit()
        self._size = (w, h)

    def fit(self):
        self.fitInView(self._item, QtCore.Qt.KeepAspectRatio)
        self._fitted = True

    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)
        self._fitted = False

    def mouseDoubleClickEvent(self, e):
        self.fit()

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.RightButton:
            p = self.mapToScene(e.position().toPoint())
            self.clicked.emit(p.x(), p.y())
        super().mousePressEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fitted:
            self.fit()


def table(headers: list[str]) -> QtWidgets.QTableWidget:
    t = QtWidgets.QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def cell(text, check: bool | None = None, color: str | None = None) -> QtWidgets.QTableWidgetItem:
    it = QtWidgets.QTableWidgetItem(str(text))
    if check is not None:
        it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
        it.setCheckState(QtCore.Qt.Checked if check else QtCore.Qt.Unchecked)
    if color:
        it.setForeground(QtGui.QColor(color))
    return it


def note(text: str) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet("color: palette(mid);")
    return lab


DIRECTION_TIP = ("카메라가 차량 기준으로 어느 쪽을 보는지. 보드가 모두 똑같아서, 보드가 1~2개만 보이는 카메라는 "
                 "이 방향으로 어느 보드인지 가립니다. 이름(front/left/rear/right)에서 짐작한 값이 먼저 들어갑니다.")


def direction_combo(value: str = "") -> QtWidgets.QComboBox:
    """A camera's facing: '' (unknown) or one of room.DIRECTIONS."""
    from gui.core.room import DIRECTIONS

    cb = QtWidgets.QComboBox()
    cb.addItem("방향 ?", "")
    for k in DIRECTIONS:
        cb.addItem(k, k)
    cb.setToolTip(DIRECTION_TIP)
    cb.setCurrentIndex(max(0, cb.findData(value or "")))
    return cb
