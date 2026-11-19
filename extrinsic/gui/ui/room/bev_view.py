"""Top-down view of the cloud with ROI boxes drawn, moved, resized and turned by mouse.

Screen layout: the LiDAR's +x (forward) points up and +y (left) points left,
as a map of the room seen from above with the vehicle facing up. The plot's
horizontal axis is therefore -y and its vertical axis x.

The cloud is shown as a raster (highest point per cell, coloured by height,
brightened by count), not as a million scatter points: it draws instantly and
reads like a floor plan, which is what placing an ROI needs.

A box's height range is not drawn here -- it is set with numbers next to the
view. What the mouse sets is the footprint: centre, size and yaw.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ui.theme import Theme

from gui.core.project import FilterBox

ROI_PEN = pg.mkPen((255, 190, 0), width=2)
ROI_SEL_PEN = pg.mkPen((255, 60, 60), width=3)
HANDLE_PEN = pg.mkPen(Theme.TEXT_PRIMARY, width=2)


def _turbo(v: np.ndarray) -> np.ndarray:
    """0..1 -> RGB uint8, a compact approximation of the turbo colormap."""
    import cv2

    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0, ::-1]
    return lut[np.clip((v * 255).astype(int), 0, 255)]


def box_to_rect(b: FilterBox):
    """FilterBox -> (pos, size, angle) of a pyqtgraph ROI in plot coordinates."""
    c = b.center()
    w, h = b.y_max - b.y_min, b.x_max - b.x_min  # plot width runs along -y, height along x
    a = np.radians(b.yaw)
    centre = np.array([-c[1], c[0]])
    # The ROI turns about its own origin (its first corner), so the corner is
    # the centre minus half the size turned by the angle.
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    corner = centre - rot @ np.array([w / 2, h / 2])
    return corner, (w, h), b.yaw


def rect_to_box(roi: pg.ROI, z_min: float, z_max: float) -> FilterBox:
    """The inverse of `box_to_rect`, from the ROI as the user left it."""
    w, h = roi.size()
    a = roi.angle()
    centre = roi.mapToParent(QtCore.QPointF(w / 2, h / 2))
    cx, cy = centre.y(), -centre.x()
    return FilterBox(cx - h / 2, cx + h / 2, cy - w / 2, cy + w / 2, z_min, z_max, yaw=float(a) % 360.0)


class _DrawBox(pg.ViewBox):
    """A ViewBox that, in draw mode, turns a left-drag into a new rectangle."""

    drawn = QtCore.Signal(object, object)  # start, end (plot coords)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.draw_mode = False
        self._rubber = None

    def mouseDragEvent(self, ev, axis=None):
        if not self.draw_mode or ev.button() != QtCore.Qt.LeftButton:
            return super().mouseDragEvent(ev, axis)
        ev.accept()
        p0, p1 = self.mapToView(ev.buttonDownPos()), self.mapToView(ev.pos())
        if self._rubber is None:
            self._rubber = QtWidgets.QGraphicsRectItem()
            self._rubber.setPen(pg.mkPen(Theme.TEXT_PRIMARY, width=1, style=QtCore.Qt.DashLine))
            self.addItem(self._rubber, ignoreBounds=True)
        self._rubber.setRect(QtCore.QRectF(p0, p1).normalized())
        if ev.isFinish():
            self.removeItem(self._rubber)
            self._rubber = None
            self.drawn.emit(np.array([p0.x(), p0.y()]), np.array([p1.x(), p1.y()]))


class BevView(pg.PlotWidget):
    roi_changed = QtCore.Signal(int)  # a box was moved / resized / turned by mouse
    roi_selected = QtCore.Signal(int)
    roi_drawn = QtCore.Signal(object)  # FilterBox footprint (z left 0)
    point_clicked = QtCore.Signal(float, float)  # LiDAR x, y

    def __init__(self, parent=None):
        self._vb = _DrawBox()
        super().__init__(parent, viewBox=self._vb)
        self.setBackground(Theme.BG_PRIMARY)
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setLabel("bottom", "← +y (왼쪽)          LiDAR y [m]          -y (오른쪽) →")
        self.setLabel("left", "LiDAR x [m] (앞 ↑)")
        self.getAxis("bottom").setTicks(None)
        self.image = pg.ImageItem()
        self.addItem(self.image)
        self.boards = pg.ScatterPlotItem(size=14, pen=pg.mkPen("k", width=1.5), symbol="s")
        self.addItem(self.boards)
        self.origin = pg.ScatterPlotItem([0], [0], size=16, symbol="t1", brush=pg.mkBrush(0, 200, 255))
        self.addItem(self.origin)
        self._labels: list = []
        self.rois: list[pg.ROI] = []
        self.sel = -1
        self._vb.drawn.connect(self._drawn)
        self.scene().sigMouseClicked.connect(self._clicked)
        # Bottom axis ticks show -u = y, so the numbers read as LiDAR y.
        self.getAxis("bottom").tickStrings = lambda values, scale, spacing: [f"{-v * scale:g}" for v in values]

    # ---------------------------------------------------------------- cloud

    def set_cloud(self, xyz: np.ndarray | None, cell: float = 0.05, reach: float = 40.0,
                  z_range: tuple[float, float] | None = None):
        if xyz is None or not len(xyz):
            self.image.clear()
            return
        xyz = np.asarray(xyz, np.float32)
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        xyz = xyz[r < reach]
        if z_range is not None:
            xyz = xyz[(xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1])]
        if not len(xyz):
            self.image.clear()
            return
        u, v = -xyz[:, 1], xyz[:, 0]
        u0, v0 = np.floor(u.min() / cell) * cell, np.floor(v.min() / cell) * cell
        iu = ((u - u0) / cell).astype(int)
        iv = ((v - v0) / cell).astype(int)
        W, H = iu.max() + 1, iv.max() + 1
        flat = iv * W + iu
        zmax = np.full(W * H, -np.inf, np.float32)
        np.maximum.at(zmax, flat, xyz[:, 2])
        count = np.bincount(flat, minlength=W * H).astype(np.float32)
        lo, hi = np.percentile(xyz[:, 2], [2, 98])
        has = count > 0
        norm = np.zeros(W * H, np.float32)
        norm[has] = (zmax[has] - lo) / max(hi - lo, 1e-3)
        rgb = _turbo(np.clip(norm, 0, 1)).astype(np.float32)
        bright = np.clip(0.35 + 0.65 * np.log1p(count) / np.log1p(np.percentile(count[has], 95)), 0, 1)
        rgb = (rgb * bright[:, None]).astype(np.uint8)
        bg = Theme.BG_PRIMARY.lstrip("#")
        rgb[~has] = tuple(int(bg[k:k + 2], 16) for k in (0, 2, 4))  # 빈 칸 = 화면 배경색
        img = rgb.reshape(H, W, 3)
        # ImageItem takes [x, y] = [column, row]
        self.image.setImage(np.transpose(img, (1, 0, 2)), autoLevels=False)
        self.image.setRect(QtCore.QRectF(u0, v0, W * cell, H * cell))

    def fit(self, reach: float = 15.0):
        self.setRange(xRange=(-reach, reach), yRange=(-reach * 0.4, reach * 1.2))

    # ----------------------------------------------------------------- ROIs

    def set_draw_mode(self, on: bool):
        self._vb.draw_mode = on
        self.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)

    def set_rois(self, boxes: list[FilterBox], selected: int = -1):
        for r in self.rois:
            self.removeItem(r)
        self.rois = []
        for k, b in enumerate(boxes):
            pos, size, ang = box_to_rect(b)
            roi = pg.ROI(pos, size, angle=ang, pen=ROI_PEN, handlePen=HANDLE_PEN, movable=True,
                         rotatable=True, resizable=True)
            roi.addScaleHandle([1, 1], [0, 0])
            roi.addScaleHandle([0, 0], [1, 1])
            roi.addScaleHandle([1, 0.5], [0, 0.5])
            roi.addScaleHandle([0.5, 1], [0.5, 0])
            roi.addRotateHandle([0, 1], [0.5, 0.5])
            roi.sigRegionChangeFinished.connect(lambda _r, k=k: self.roi_changed.emit(k))
            roi.sigClicked.connect(lambda _r, _e, k=k: self.select(k, emit=True))
            roi.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
            self.addItem(roi)
            self.rois.append(roi)
        self.select(selected)

    def select(self, k: int, emit: bool = False):
        self.sel = k
        for i, r in enumerate(self.rois):
            r.setPen(ROI_SEL_PEN if i == k else ROI_PEN)
        if emit:
            self.roi_selected.emit(k)

    def footprint(self, k: int, z_min: float, z_max: float) -> FilterBox:
        return rect_to_box(self.rois[k], z_min, z_max)

    def _drawn(self, p0, p1):
        (u0, v0), (u1, v1) = np.minimum(p0, p1), np.maximum(p0, p1)
        if u1 - u0 < 0.2 or v1 - v0 < 0.2:
            return
        self.roi_drawn.emit(FilterBox(v0, v1, -u1, -u0, 0.0, 0.0))

    def _clicked(self, ev):
        if ev.button() != QtCore.Qt.LeftButton or self._vb.draw_mode:
            return
        p = self._vb.mapSceneToView(ev.scenePos())
        self.point_clicked.emit(float(p.y()), float(-p.x()))

    # --------------------------------------------------------------- boards

    def set_boards(self, items: list[tuple[str, float, float, tuple]]):
        """(label, x, y, rgb) per board."""
        for t in self._labels:
            self.removeItem(t)
        self._labels = []
        if not items:
            self.boards.setData([], [])
            return
        self.boards.setData([-y for _, _, y, _ in items], [x for _, x, _, _ in items],
                            brush=[pg.mkBrush(*c) for *_, c in items])
        for label, x, y, c in items:
            t = pg.TextItem(label, color=c, anchor=(0.5, 1.4))
            t.setPos(-y, x)
            self.addItem(t)
            self._labels.append(t)
