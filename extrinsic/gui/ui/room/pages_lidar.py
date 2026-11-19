"""Step 2: find the boards in the still cloud -- inside the room's ROI -- once, for every camera.

The ROI is drawn on a top-down view: where in the room the boards stand. It
is the vehicle profile's, so the second session of a vehicle searches the
right place without being told. Without an ROI the whole cloud is searched,
which on a 128-channel sweep is ten times slower and finds the same boards.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtWidgets

from ui.theme import Theme

from gui.core import room
from gui.core.project import FilterBox
from gui.ui.cloud_view import CloudView
from gui.ui.room.bev_view import BevView
from gui.ui.color_controls import ColorControls
from gui.ui.room import RoomStep, cell, table

# Shown points are capped; detection always uses the whole cloud.
DISPLAY_POINTS = 1_500_000

BOX_RGBA = (1.0, 0.75, 0.1, 1.0)
SEL_RGBA = (1.0, 0.2, 0.2, 1.0)
OFF_RGBA = (0.5, 0.5, 0.5, 0.6)
ROI_RGBA = (0.2, 0.85, 1.0, 0.9)


def _box_lines(box) -> np.ndarray:
    c = np.asarray(box.corners(), np.float64)  # x outer, y middle, z inner
    q = c[[0, 4, 6, 2, 1, 5, 7, 3]]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return np.vstack([q[list(e)] for e in edges]).astype(np.float32)


class LidarBoardsStep(RoomStep):
    title = "2. LiDAR 보드"
    subtitle = "ROI 안에서 평면 탐색 → 구멍 4개 검출 · 모든 카메라가 함께 씀"

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self._shown_for = None
        self._disp = None
        self._disp_int = None
        self._box_items: list = []
        self._roi_items: list = []
        self._roi_sel = -1

        # ---- ROI
        self.roi_table = table(["ROI", "x m", "y m", "z m", "yaw °"])
        self.roi_table.setMaximumHeight(130)
        self.roi_table.itemSelectionChanged.connect(self._roi_table_pick)
        self.draw_btn = QtWidgets.QPushButton("✏  드래그로 ROI 그리기")
        self.draw_btn.setCheckable(True)
        self.draw_btn.toggled.connect(self._draw_mode)
        self.roi_del = QtWidgets.QPushButton("ROI 삭제")
        self.roi_del.clicked.connect(self.delete_roi)
        self.roi_spins = {}
        rg = QtWidgets.QGridLayout()
        for k, (key, lab) in enumerate((("x_min", "x 최소"), ("x_max", "x 최대"), ("y_min", "y 최소"), ("y_max", "y 최대"),
                                        ("z_min", "z 최소"), ("z_max", "z 최대"))):
            sp = QtWidgets.QDoubleSpinBox(minimum=-200, maximum=200, singleStep=0.1, decimals=2)
            sp.setSuffix(" m")
            sp.valueChanged.connect(self._roi_spun)
            self.roi_spins[key] = sp
            rg.addWidget(QtWidgets.QLabel(lab), k // 2, 2 * (k % 2))
            rg.addWidget(sp, k // 2, 2 * (k % 2) + 1)
        self.roi_yaw = QtWidgets.QDoubleSpinBox(minimum=-180, maximum=360, singleStep=1.0, decimals=1)
        self.roi_yaw.setSuffix(" °")
        self.roi_yaw.valueChanged.connect(self._roi_spun)
        rg.addWidget(QtWidgets.QLabel("yaw"), 3, 0)
        rg.addWidget(self.roi_yaw, 3, 1)
        self.bev_z = QtWidgets.QCheckBox("BEV 에 선택 ROI 높이 범위만")
        self.bev_z.toggled.connect(self._redraw_bev_cloud)
        rg.addWidget(self.bev_z, 3, 2, 1, 2)

        self.method = QtWidgets.QComboBox()
        self.method.addItems(["occupancy", "template", "experimental", "original", "ring", "boundary"])
        self.method.setToolTip("template = 보드 도면 전체를 점에 맞추는 방식. 점이 듬성한 보드(벨로다인, 멀리·높이 있는 보드)에 강함")
        self.voxels = QtWidgets.QLineEdit("0.03, 0.05")
        self.rmin = QtWidgets.QDoubleSpinBox(minimum=0.0, maximum=50.0, value=0.8, singleStep=0.5)
        self.rmax = QtWidgets.QDoubleSpinBox(minimum=1.0, maximum=200.0, value=25.0, singleStep=1.0)
        self.complete = QtWidgets.QCheckBox("구멍 3개면 직사각형으로 4번째 보완")
        self.complete.setChecked(True)
        self.template = QtWidgets.QCheckBox("구멍 검출이 실패하면 보드 모델 정합(template)으로 재시도")
        self.template.setChecked(True)
        self.search_btn = QtWidgets.QPushButton("▶  ROI 안에서 자동 탐색")
        self.search_btn.setMinimumHeight(34)
        self.search_btn.clicked.connect(self.search)
        self.pick_btn = QtWidgets.QPushButton("✚ 클릭으로 보드 추가")
        self.pick_btn.setCheckable(True)
        self.pick_btn.toggled.connect(self._pick_mode)
        self.del_btn = QtWidgets.QPushButton("선택 삭제")
        self.del_btn.clicked.connect(self.delete)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.table = table(["사용", "보드", "출처", "x", "y", "z", "거리 m", "반지름 mm", "평면 잔차 mm", "비고"])
        self.table.itemChanged.connect(self._toggled)
        self.table.itemSelectionChanged.connect(self._selected)

        self.bev = BevView()
        self.bev.roi_drawn.connect(self._roi_drawn)
        self.bev.roi_changed.connect(self._roi_moved)
        self.bev.roi_selected.connect(self._roi_select)
        self.bev.point_clicked.connect(self._bev_clicked)
        self.view = CloudView()
        self.view.point_picked.connect(self._picked)
        self.colors = ColorControls()
        self.colors.changed.connect(self.redraw_points)
        self.point_size = QtWidgets.QDoubleSpinBox(minimum=0.5, maximum=8.0, value=1.5, singleStep=0.5)
        self.point_size.valueChanged.connect(self.redraw_points)
        self.only_near = QtWidgets.QCheckBox("보드 박스 근처만")
        self.only_near.toggled.connect(self.redraw_points)
        views = QtWidgets.QHBoxLayout()
        for name, label in (("top", "Top"), ("front", "Front"), ("side", "Side"), ("iso", "Iso")):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _=False, n=name: self.view.apply_preset(n))
            views.addWidget(b)
        fit = QtWidgets.QPushButton("전체 보기")
        fit.clicked.connect(lambda: self.view.fit(self._disp))
        views.addWidget(fit)
        views.addWidget(QtWidgets.QLabel("점 크기"))
        views.addWidget(self.point_size)
        views.addWidget(self.only_near)
        views.addStretch(1)

        roi_box = QtWidgets.QGroupBox("ROI")
        rl = QtWidgets.QVBoxLayout(roi_box)
        rb = QtWidgets.QHBoxLayout()
        rb.addWidget(self.draw_btn)
        rb.addWidget(self.roi_del)
        rl.addLayout(rb)
        rl.addWidget(self.roi_table)
        rl.addLayout(rg)

        form = QtWidgets.QFormLayout()
        form.addRow("구멍 검출 방식", self.method)
        form.addRow("평면 탐색 복셀 (m, 쉼표)", self.voxels)
        form.addRow("거리 범위 최소", self.rmin)
        form.addRow("거리 범위 최대", self.rmax)
        adv = QtWidgets.QGroupBox("검출 설정")
        adv.setCheckable(True)
        adv.setChecked(False)
        advw = QtWidgets.QWidget()
        al = QtWidgets.QVBoxLayout(advw)
        al.setContentsMargins(0, 0, 0, 0)
        al.addLayout(form)
        al.addWidget(self.complete)
        al.addWidget(self.template)
        advw.setVisible(False)
        adv.toggled.connect(advw.setVisible)
        QtWidgets.QVBoxLayout(adv).addWidget(advw)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(roi_box)
        left.addWidget(adv)
        left.addWidget(self.search_btn)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.pick_btn)
        row.addWidget(self.del_btn)
        left.addLayout(row)
        left.addWidget(self.summary)
        left.addWidget(self.table, 1)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(640)

        right3d = QtWidgets.QVBoxLayout()
        right3d.setContentsMargins(0, 0, 0, 0)
        right3d.addWidget(self.colors)
        right3d.addLayout(views)
        right3d.addWidget(self.view, 1)
        w3d = QtWidgets.QWidget()
        w3d.setLayout(right3d)
        bevw = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bevw)
        bl.setContentsMargins(0, 0, 0, 0)
        fit_bev = QtWidgets.QPushButton("전체 보기")
        fit_bev.clicked.connect(lambda: self.bev.fit())
        bb = QtWidgets.QHBoxLayout()
        bb.addWidget(QtWidgets.QLabel("위에서 본 모습 — 색 = 높이, 밝기 = 점 수. ▲ = LiDAR 원점, ■ = 찾은 보드"))
        bb.addStretch(1)
        bb.addWidget(fit_bev)
        bl.addLayout(bb)
        bl.addWidget(self.bev, 1)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(bevw, "BEV · ROI")
        self.tabs.addTab(w3d, "3D")
        rw = self.tabs

        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(lw)
        lay.addWidget(rw, 1)

    # ------------------------------------------------------------- display

    def on_vehicle(self):
        prof = self.state.profile
        if prof is not None and prof.roi and not self.state.roi:
            self.state.roi = [FilterBox(**vars(b)) for b in prof.roi]
            self.fill_roi()

    def on_enter(self):
        s = self.state
        if s.cloud is not None and self._shown_for is not s.cloud:
            self._shown_for = s.cloud
            xyz = s.cloud
            inten = s.intensity
            if len(xyz) > DISPLAY_POINTS:
                pick = np.random.default_rng(0).choice(len(xyz), DISPLAY_POINTS, replace=False)
                xyz = xyz[pick]
                inten = None if inten is None else inten[pick]
            self._disp, self._disp_int = xyz, inten
            self.redraw_points()
            self.view.fit(self._disp[np.linalg.norm(self._disp, axis=1) < 15])
            self._redraw_bev_cloud()
            self.bev.fit()
        if not s.roi and s.profile is not None and s.profile.roi:
            self.on_vehicle()
        self.fill_roi()
        self.fill()
        self.draw_boards()
        # The ROI is known (from this vehicle's profile, or drawn earlier): search
        # straight away -- that is the point of having one.
        if s.cloud is not None and s.roi and not s.lidar_stats and s.target is not None and self.search_btn.isEnabled():
            self.search()
        elif s.cloud is not None and not s.roi and not s.lidar_stats:
            self.summary.setText("BEV 에서 보드들이 서 있는 곳에 ROI 를 그리고 '자동 탐색' 을 누르세요. "
                                 "ROI 없이 누르면 전체 클라우드를 탐색합니다.")

    def _redraw_bev_cloud(self, *_):
        s = self.state
        if s.cloud is None:
            return
        zr = None
        if self.bev_z.isChecked() and 0 <= self._roi_sel < len(s.roi):
            b = s.roi[self._roi_sel]
            zr = (b.z_min, b.z_max)
        self.bev.set_cloud(self._disp if self._disp is not None else s.cloud, z_range=zr)

    # ------------------------------------------------------------------ ROI

    def _default_z(self):
        """A height range for a new ROI: from just above the floor to the top of the cloud near the boards."""
        s = self.state
        if s.cloud is None or not len(s.cloud):
            return -2.0, 3.0
        z = s.cloud[np.linalg.norm(s.cloud[:, :2], axis=1) < 15, 2]
        if not len(z):
            return -2.0, 3.0
        lo, hi = np.percentile(z, [1, 99.5])
        return float(lo + 0.15), float(hi + 0.2)

    def fill_roi(self):
        s = self.state
        t = self.roi_table
        t.blockSignals(True)
        t.setRowCount(0)
        for k, b in enumerate(s.roi):
            t.insertRow(k)
            for j, v in enumerate([f"ROI{k + 1}", f"{b.x_min:.1f} ~ {b.x_max:.1f}", f"{b.y_min:.1f} ~ {b.y_max:.1f}",
                                   f"{b.z_min:.1f} ~ {b.z_max:.1f}", f"{b.yaw:.0f}"]):
                t.setItem(k, j, cell(v))
        if 0 <= self._roi_sel < len(s.roi):
            t.selectRow(self._roi_sel)
        t.blockSignals(False)
        self.bev.set_rois(s.roi, self._roi_sel)
        self._show_roi_spins()
        self._draw_roi_3d()

    def _show_roi_spins(self):
        s = self.state
        on = 0 <= self._roi_sel < len(s.roi)
        for sp in list(self.roi_spins.values()) + [self.roi_yaw]:
            sp.blockSignals(True)
            sp.setEnabled(on)
        if on:
            b = s.roi[self._roi_sel]
            for k, sp in self.roi_spins.items():
                sp.setValue(getattr(b, k))
            self.roi_yaw.setValue(b.yaw)
        for sp in list(self.roi_spins.values()) + [self.roi_yaw]:
            sp.blockSignals(False)

    def _draw_roi_3d(self):
        for it in self._roi_items:
            self.view.removeItem(it)
        self._roi_items = []
        for b in self.state.roi:
            it = gl.GLLinePlotItem(pos=_box_lines(b), color=ROI_RGBA, width=1.5, mode="lines", antialias=True)
            self.view.addItem(it)
            self._roi_items.append(it)

    def _roi_edited(self):
        """The ROI changed: what was found in the old one no longer answers for the new one."""
        s = self.state
        s.lidar_stats = {}
        if s.profile is not None:
            s.profile.roi = [FilterBox(**vars(b)) for b in s.roi]
        self.changed.emit()

    def _draw_mode(self, on):
        self.bev.set_draw_mode(on)
        if on:
            self.tabs.setCurrentIndex(0)
            self.summary.setText("BEV 에서 왼쪽 버튼으로 드래그해 ROI 를 그리세요")

    def _roi_drawn(self, box):
        zl, zh = self._default_z()
        box.z_min, box.z_max = zl, zh
        self.state.roi.append(box)
        self._roi_sel = len(self.state.roi) - 1
        self.draw_btn.setChecked(False)
        self.fill_roi()
        self._roi_edited()
        self.summary.setText(f"ROI{self._roi_sel + 1} 추가 — 높이 {zl:.1f}~{zh:.1f} m (필요하면 숫자로 조정)")

    def _roi_moved(self, k):
        s = self.state
        if not 0 <= k < len(s.roi):
            return
        old = s.roi[k]
        s.roi[k] = self.bev.footprint(k, old.z_min, old.z_max)
        self._roi_sel = k
        # Not a full refill: the ROI item being dragged must survive.
        self.roi_table.blockSignals(True)
        b = s.roi[k]
        for j, v in enumerate([f"ROI{k + 1}", f"{b.x_min:.1f} ~ {b.x_max:.1f}", f"{b.y_min:.1f} ~ {b.y_max:.1f}",
                               f"{b.z_min:.1f} ~ {b.z_max:.1f}", f"{b.yaw:.0f}"]):
            self.roi_table.setItem(k, j, cell(v))
        self.roi_table.selectRow(k)
        self.roi_table.blockSignals(False)
        self.bev.select(k)
        self._show_roi_spins()
        self._draw_roi_3d()
        self._roi_edited()

    def _roi_select(self, k):
        self._roi_sel = k
        self.roi_table.blockSignals(True)
        self.roi_table.selectRow(k)
        self.roi_table.blockSignals(False)
        self._show_roi_spins()
        if self.bev_z.isChecked():
            self._redraw_bev_cloud()

    def _roi_table_pick(self):
        rows = self.roi_table.selectionModel().selectedRows()
        if rows:
            self._roi_sel = rows[0].row()
            self.bev.select(self._roi_sel)
            self._show_roi_spins()
            if self.bev_z.isChecked():
                self._redraw_bev_cloud()

    def _roi_spun(self, *_):
        s = self.state
        k = self._roi_sel
        if not 0 <= k < len(s.roi):
            return
        v = {key: sp.value() for key, sp in self.roi_spins.items()}
        if v["x_max"] <= v["x_min"] or v["y_max"] <= v["y_min"] or v["z_max"] <= v["z_min"]:
            return
        s.roi[k] = FilterBox(**v, yaw=self.roi_yaw.value())
        self.fill_roi()
        if self.bev_z.isChecked():
            self._redraw_bev_cloud()
        self._roi_edited()

    def delete_roi(self):
        s = self.state
        if 0 <= self._roi_sel < len(s.roi):
            del s.roi[self._roi_sel]
            self._roi_sel = min(self._roi_sel, len(s.roi) - 1)
            self.fill_roi()
            self._roi_edited()

    def _bev_clicked(self, x, y):
        """A click on the BEV while adding by hand: try a board around the highest point there."""
        if not self.pick_btn.isChecked() or self.state.cloud is None:
            return
        c = self.state.cloud
        near = c[np.hypot(c[:, 0] - x, c[:, 1] - y) < 0.25]
        if not len(near):
            self.summary.setText("그 자리에 점이 없습니다")
            return
        self._picked(near[np.argsort(near[:, 2])[len(near) // 2]])

    def redraw_points(self, *_):
        if self._disp is None:
            return
        xyz, inten = self._disp, self._disp_int
        if self.only_near.isChecked() and self.state.lidar_boards:
            m = np.zeros(len(xyz), bool)
            for b in self.state.lidar_boards:
                m |= np.linalg.norm(xyz - b.centers.mean(axis=0), axis=1) < 1.5
            xyz = xyz[m]
            inten = None if inten is None else inten[m]
        used = self.view.set_points_styled(xyz, inten, self.colors.style, size=self.point_size.value())
        self.colors.report_range(used, inten is not None)

    def draw_boards(self):
        for it in self._box_items:
            self.view.removeItem(it)
        self._box_items = []
        s = self.state
        sel = self._selected_labels()
        holes, holes_sel = [], []
        for b in s.lidar_boards:
            col = SEL_RGBA if b.label in sel else (OFF_RGBA if b.label in s.lidar_excluded else BOX_RGBA)
            it = gl.GLLinePlotItem(pos=_box_lines(b.box), color=col, width=2.0, mode="lines", antialias=True)
            self.view.addItem(it)
            self._box_items.append(it)
            (holes_sel if b.label in sel else holes).append(b.centers)
        self.view.set_markers("holes", np.vstack(holes) if holes else None, (1.0, 0.0, 1.0, 1.0), 12.0)
        self.view.set_markers("holes_sel", np.vstack(holes_sel) if holes_sel else None, (1.0, 1.0, 0.0, 1.0), 16.0)
        items = []
        for b in s.lidar_boards:
            c = b.centers.mean(axis=0)
            col = (255, 60, 60) if b.label in sel else ((130, 130, 130) if b.label in s.lidar_excluded else (255, 0, 255))
            items.append((b.label, float(c[0]), float(c[1]), col))
        self.bev.set_boards(items)

    def fill(self):
        s = self.state
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for b in s.lidar_boards:
            r = self.table.rowCount()
            self.table.insertRow(r)
            c = b.centers.mean(axis=0)
            radii = ", ".join(f"{x * 1000:.0f}" for x in (b.det.radii or []))
            completed = getattr(b.det, "completed", False)
            templated = getattr(b.det, "templated", False) or b.det.method == "template"
            vals = [b.label, b.source, f"{c[0]:.2f}", f"{c[1]:.2f}", f"{c[2]:.2f}", f"{np.linalg.norm(c):.2f}",
                    radii, f"{b.det.plane_rms * 1000:.1f}",
                    "3+1 보완" if completed else ("모델 정합" if templated else "")]
            self.table.setItem(r, 0, cell("", check=b.label not in s.lidar_excluded))
            for k, v in enumerate(vals, start=1):
                self.table.setItem(r, k, cell(v, color=Theme.WARNING if (completed or templated) and k == 9 else None))
        self.table.blockSignals(False)
        st = s.lidar_stats
        if st:
            where = (f"ROI {len(s.roi)}개 ({st.get('searched', 0):,} / {st.get('points', 0):,}점)"
                     if s.roi else f"전체 ({st.get('searched', 0):,}점)")
            self.summary.setText(f"{where} · 평면 {st.get('patches', 0)}개 → 보드 크기 {st.get('sized', 0)}개 → "
                                 f"구멍 4개 {st.get('detected', 0)}개 · {st.get('seconds', 0):.1f} 초"
                                 f"  (수동·배치 포함 총 {len(s.lidar_boards)}개)")

    def _selected_labels(self):
        return {self.table.item(i.row(), 1).text() for i in self.table.selectionModel().selectedRows()}

    def _selected(self):
        sel = self._selected_labels()
        self.draw_boards()
        for b in self.state.lidar_boards:
            if b.label in sel:
                c = b.centers.mean(axis=0)
                self.view.focus_on((c[0] - 1, c[0] + 1, c[1] - 1, c[1] + 1, c[2] - 1, c[2] + 1), 1.5)
                break

    def _toggled(self, item):
        if item.column() != 0:
            return
        label = self.table.item(item.row(), 1).text()
        if item.checkState() == QtCore.Qt.Checked:
            self.state.lidar_excluded.discard(label)
        else:
            self.state.lidar_excluded.add(label)
        self.state.invalidate_from("lidar_boards")
        self.draw_boards()
        self.changed.emit()

    # --------------------------------------------------------------- actions

    def _params(self):
        try:
            vox = tuple(float(x) for x in self.voxels.text().split(",") if x.strip())
        except ValueError:
            vox = (0.03, 0.05)
        return dict(method=self.method.currentText(), voxels=vox or (0.03, 0.05),
                    min_range=self.rmin.value(), max_range=self.rmax.value(),
                    complete=self.complete.isChecked(), template=self.template.isChecked())

    def search(self):
        s = self.state
        if s.cloud is None or s.target is None:
            self.summary.setText("1단계에서 먼저 캡처하세요")
            return
        p = self._params()
        s.method = p["method"]
        roi = list(s.roi)
        self.summary.setText(f"ROI {len(roi)}개 안에서 탐색 중…" if roi else "전체 클라우드 탐색 중… (수십 초)")
        import time

        def work():
            t0 = time.time()
            boards, stats = room.find_lidar_boards(s.cloud, s.single, s.target, p["method"], p["max_range"],
                                                   p["min_range"], p["voxels"], p["complete"], p["template"],
                                                   roi=roi or None)
            stats["seconds"] = time.time() - t0
            return boards, stats

        def done(out):
            s.lidar_boards, s.lidar_stats = out
            if roi and s.profile is not None and s.vehicle:
                s.profile.roi = [FilterBox(**vars(b)) for b in roi]
                s.profile.save()
            s.lidar_excluded = set()
            s.invalidate_from("lidar_boards")
            self.fill()
            self.draw_boards()
            self.redraw_points()
            self.changed.emit()

        self.run_task(work, done, busy=self.search_btn)

    def _pick_mode(self, on):
        self.view.set_pick_mode(on)
        if on:
            self.draw_btn.setChecked(False)
        self.summary.setText("3D 뷰에서 보드 면 위의 점을, 또는 BEV 에서 보드 자리를 클릭하세요" if on else "")

    def _picked(self, xyz):
        if not self.pick_btn.isChecked():
            return
        self.pick_btn.setChecked(False)
        s = self.state
        p = self._params()
        n = sum(1 for b in s.lidar_boards if b.source == "수동")
        label = f"M{n}"
        self.summary.setText(f"클릭한 자리 {np.round(np.asarray(xyz), 2).tolist()} 에서 검출 중…")

        def work():
            return room.board_at(s.cloud, s.single, xyz, s.target, p["method"], p["complete"], label, p["template"])

        def done(out):
            board, reason = out
            if board is None:
                self.summary.setText(f"실패: {reason}")
                return
            if any(np.linalg.norm(b.centers.mean(0) - board.centers.mean(0)) < 0.2 for b in s.lidar_boards):
                self.summary.setText("이미 있는 보드입니다")
                return
            s.lidar_boards.append(board)
            s.invalidate_from("lidar_boards")
            self.summary.setText(f"{label} 추가됨")
            self.fill()
            self.draw_boards()
            self.changed.emit()

        self.run_task(work, done)

    def delete(self):
        sel = self._selected_labels()
        if not sel:
            return
        self.state.lidar_boards = [b for b in self.state.lidar_boards if b.label not in sel]
        self.state.invalidate_from("lidar_boards")
        self.fill()
        self.draw_boards()
        self.changed.emit()

    def active(self):
        return [b for b in self.state.lidar_boards if b.label not in self.state.lidar_excluded]

    def is_complete(self):
        return len(self.active()) > 0

    def status_text(self):
        return f"보드 {len(self.active())}개" if self.state.lidar_boards else "탐색 전"
