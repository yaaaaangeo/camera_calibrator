"""Steps 5-7: pair and solve, look at the overlay every which way, save."""

from __future__ import annotations


import cv2
import numpy as np
from PySide6 import QtCore, QtWidgets

from ui.theme import Theme

from gui.core import room
from gui.core import solve as S
from gui.core.project import RESULT_DIR
from gui.core.room_overlay import COLORMAPS, OverlayStyle, nudged, render
from gui.ui.room import ImageCanvas, RoomStep, cell, direction_combo, table


def _active(state):
    cams = [b for b in state.cam_boards if b.label not in state.cam_excluded]
    lidar = [b for b in state.lidar_boards if b.label not in state.lidar_excluded]
    return cams, lidar


def _pose_text(sol) -> str:
    from gui.core.session import pose_text

    return pose_text(sol)


def _job(slot) -> room.CameraJob:
    return room.CameraJob([b for b in slot.cam_boards if b.label not in slot.cam_excluded], slot.camera,
                          slot.prior.solution if slot.prior is not None else None, slot.direction)


def solve_camera(slot, lidar, cloud, single, target, method, place: bool, solver: str, forward=None):
    """Pair and solve one camera against the shared LiDAR boards.

    Returns (result, LiDAR boards placed from the extrinsic that are not in `lidar`).
    """
    out, _ = room.solve_cameras([_job(slot)], lidar, cloud, single, target, method, place, solver, forward)
    return out[0]


# ===================================================================== solve


class SolveStep(RoomStep):
    title = "5. 대응·풀이"
    subtitle = "카메라마다: 어느 LiDAR 보드가 어느 카메라 보드인지, 그리고 extrinsic"
    per_camera = True

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.place_chk = QtWidgets.QCheckBox("LiDAR 에서 못 찾은 카메라 보드를 extrinsic 위치에서 다시 찾기")
        self.place_chk.setChecked(True)
        self.solver = QtWidgets.QComboBox()
        self.solver.addItem("자동 (holdout 오차로 선택)", "auto")
        self.solver.addItem("Kabsch (모든 점 동일 가중)", "kabsch")
        self.solver.addItem("가중 (센서별 노이즈 공분산)", "weighted")
        self.solver.currentIndexChanged.connect(self.refit)
        self.all_btn = QtWidgets.QPushButton("▶  모든 카메라 풀기")
        self.all_btn.setMinimumHeight(34)
        self.all_btn.clicked.connect(self.run_all)
        self.run_btn = QtWidgets.QPushButton("이 카메라만 다시 풀기")
        self.run_btn.clicked.connect(self.run)
        self.overview = table(["카메라", "방향", "보드 쌍", "RMSE mm", "이미지 px", "holdout px", "상태"])
        self.overview.setMaximumHeight(170)
        self.overview.itemSelectionChanged.connect(self._overview_pick)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.table = table(["사용", "카메라=LiDAR", "출처", "거리 m", "기울기 °", "RMSE mm", "이미지 px",
                            "빼면 이동 mm", "빼면 회전 °", "비고"])
        self.table.itemChanged.connect(self._toggled)
        self.coverage = QtWidgets.QLabel()
        self.coverage.setWordWrap(True)
        self.report = QtWidgets.QPlainTextEdit(readOnly=True)
        from PySide6 import QtGui
        self.report.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.all_btn)
        top.addWidget(self.run_btn)
        top.addWidget(QtWidgets.QLabel("풀이 방식"))
        top.addWidget(self.solver)
        top.addStretch(1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.place_chk)
        lay.addWidget(self.overview)
        lay.addWidget(self.summary)
        lay.addWidget(self.table, 2)
        lay.addWidget(self.coverage)
        lay.addWidget(self.report, 1)

    def on_enter(self):
        s = self.state
        self.fill()
        ready = [c for c in s.cams if c.camera is not None and c.cam_boards]
        if ready and _active(s)[1] and all(c.result is None for c in ready) and self.all_btn.isEnabled():
            self.run_all()

    def _overview_pick(self):
        rows = self.overview.selectionModel().selectedRows()
        if rows and rows[0].row() != self.state.cur:
            self.select_camera.emit(rows[0].row())

    def _merge_placed(self, placed):
        # Boards placed from one camera's extrinsic are LiDAR boards like any
        # other: they belong in step 2's list, and every other camera may use them.
        known = {b.label for b in self.state.lidar_boards}
        for b in placed:
            if b.label not in known:
                self.state.lidar_boards.append(b)

    def run_all(self):
        s = self.state
        _, lidar = _active(s)
        slots = [c for c in s.cams if c.camera is not None and any(b.label not in c.cam_excluded for b in c.cam_boards)]
        if not slots or not lidar:
            self.summary.setText("2·4단계에서 LiDAR 보드와 카메라 보드를 먼저 찾으세요")
            return
        place, solver = self.place_chk.isChecked(), self.solver.currentData()
        self.summary.setText(f"카메라 {len(slots)}대 매칭 중…")
        cloud, single, target, method = s.cloud, s.single, s.target, s.method

        prof = s.profile
        known = prof.forward_deg if prof is not None else None

        def work():
            # Cameras that see several boards first: their directions fix the
            # vehicle's forward, which then tells a one-board camera which board
            # it sees. Each sees the boards the ones before it placed.
            out, forward = room.solve_cameras([_job(c) for c in slots], lidar, cloud, single, target,
                                              method, place, solver, known_forward=known)
            return [(c, res, placed) for c, (res, placed) in zip(slots, out)], forward

        def done(result):
            out, forward = result
            for c, res, placed in out:
                c.result, c.excluded_pairs, c.solver, c.saved_to = res, set(), solver, ""
                self._merge_placed(placed)
            s.forward = forward
            if forward is not None and prof is not None and s.vehicle:
                prof.forward_deg = forward
                prof.save()
            self.fill()
            self.changed.emit()

        self.run_task(work, done, busy=self.all_btn)

    def run(self):
        s = self.state
        slot = s.slot
        cams, lidar = _active(s)
        if slot is None or not cams or not lidar:
            self.summary.setText("2·4단계에서 양쪽 보드를 먼저 찾으세요")
            return
        place, solver = self.place_chk.isChecked(), self.solver.currentData()
        self.summary.setText("매칭 중…")
        cloud, single, target, method = s.cloud, s.single, s.target, s.method

        forward = s.forward if s.forward is not None else (s.profile.forward_deg if s.profile is not None else None)

        def work():
            return solve_camera(slot, lidar, cloud, single, target, method, place, solver, forward)

        def done(out):
            res, placed = out
            slot.result, slot.excluded_pairs, slot.solver, slot.saved_to = res, set(), solver, ""
            self._merge_placed(placed)
            self.fill()
            self.changed.emit()

        self.run_task(work, done, busy=self.run_btn)

    def refit(self, *_):
        s = self.state
        if s.result is None or not s.result.pairs:
            return
        room.finalize(s.result, s.camera, s.target, self.solver.currentData(), s.excluded_pairs)
        s.solver = self.solver.currentData()
        self.fill()
        self.changed.emit()

    def _toggled(self, item):
        if item.column() != 0:
            return
        sid = self.table.item(item.row(), 1).text()
        if item.checkState() == QtCore.Qt.Checked:
            self.state.excluded_pairs.discard(sid)
        else:
            self.state.excluded_pairs.add(sid)
        self.refit()

    def fill_overview(self):
        s = self.state
        t = self.overview
        t.blockSignals(True)
        t.setRowCount(0)
        for k, c in enumerate(s.cams):
            r = c.result
            if r is None:
                st = "보드 없음" if not c.cam_boards else ("내부 파라미터 없음" if c.camera is None else "풀이 전")
                vals, color = [c.name, "", "—", "—", "—", "—", st], Theme.TEXT_MUTED
            elif not r.ok or r.solution is None:
                vals, color = [c.name, "", "—", "—", "—", "—", "실패: " + "; ".join(r.notes)[:80]], Theme.BAD
            else:
                hk = min((v for v in r.holdout.values() if np.isfinite(v)), default=float("nan"))
                warn = r.image_rmse > 3.0 or len(r.scenes) < 3 or r.ambiguous
                st = "모호 — 방향 지정" if r.ambiguous else ("확인 필요" if warn else "OK")
                vals = [c.name, "", len(r.scenes), f"{r.solution.rmse * 1000:.1f}", f"{r.image_rmse:.2f}",
                        "—" if not np.isfinite(hk) else f"{hk:.2f}", st + (" · 저장됨" if c.saved_to else "")]
                color = Theme.BAD if r.ambiguous else Theme.WARNING if warn else Theme.GOOD
            t.insertRow(k)
            for j, v in enumerate(vals):
                t.setItem(k, j, cell(v, color=color if j == 6 else None))
            combo = direction_combo(c.direction)
            combo.currentIndexChanged.connect(lambda _i, c=c, cb=combo: self._set_direction(c, cb.currentData()))
            t.setCellWidget(k, 1, combo)
        if s.cams:
            t.selectRow(min(s.cur, len(s.cams) - 1))
        t.blockSignals(False)

    def _set_direction(self, slot, direction):
        """A camera's facing changed: remember it, and re-solve if that camera was solved."""
        from gui.core import session

        slot.direction = direction or ""
        prof = self.state.profile
        if prof is not None and self.state.vehicle:
            e = prof.cameras.get(slot.topic)
            prof.cameras[slot.topic] = session.CameraEntry(slot.name, e.preset if e else slot.camera_name, slot.direction)
            prof.save()
        if slot.result is not None and self.run_btn.isEnabled():
            k = self.state.cams.index(slot)
            if k != self.state.cur:
                self.select_camera.emit(k)
            QtCore.QTimer.singleShot(0, self.run)

    def fill(self):
        s = self.state
        self.fill_overview()
        res = s.result
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        if res is None:
            self.summary.setText("아직 풀지 않았습니다")
            self.report.clear()
            self.coverage.clear()
            self.table.blockSignals(False)
            return
        by_cam = {c.label: c for c in res.cams}
        sol = res.solution
        for i, j, _ in res.pairs:
            cb, lb = res.cams[j], res.lidar[i]
            sid = f"{cb.label}={lb.label}"
            loo = res.loo.get(sid, {})
            warn = bool(loo) and (loo["shift_mm"] > 30 or loo["rotation_deg"] > 0.5)
            used = sid not in s.excluded_pairs
            remark = []
            if getattr(lb.det, "completed", False):
                remark.append("3+1 보완")
            if getattr(lb.det, "templated", False) or lb.det.method == "template":
                remark.append("모델 정합")
            vals = [sid, lb.source, f"{cb.det.distance:.2f}", f"{cb.det.tilt_deg:.1f}",
                    f"{sol.per_scene_rmse.get(sid, float('nan')) * 1000:.1f}" if used and sol else "—",
                    f"{res.per_board_px.get(sid, float('nan')):.2f}" if used else "—",
                    f"{loo.get('shift_mm', float('nan')):.1f}", f"{loo.get('rotation_deg', float('nan')):.2f}",
                    ", ".join(remark)]
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, cell("", check=used))
            for k, v in enumerate(vals, start=1):
                self.table.setItem(r, k, cell(v, color=Theme.BAD if warn and k in (7, 8) else None))
        matched = {j for _, j, _ in res.pairs}
        for j, cb in enumerate(res.cams):
            if j not in matched:
                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 1, cell(f"{cb.label}=—", color=Theme.TEXT_MUTED))
                self.table.setItem(r, 9, cell("LiDAR 에서 못 찾음", color=Theme.TEXT_MUTED))
        self.table.blockSignals(False)

        if not res.ok or sol is None:
            self.summary.setText("풀지 못했습니다: " + "; ".join(res.notes))
            self.report.setPlainText(room.report(res))
            return
        hk, hw = res.holdout.get("kabsch", float("nan")), res.holdout.get("weighted", float("nan"))
        self.summary.setText(
            f"<b>보드 {len(res.scenes)}개</b> 사용 · RMSE <b>{sol.rmse * 1000:.1f} mm</b> · 이미지 <b>{res.image_rmse:.2f} px</b>"
            f" · 풀이 {'가중' if res.weighted else 'Kabsch'} (holdout px: Kabsch {hk:.2f}, 가중 {hw:.2f})<br>"
            + _pose_text(sol).replace("\n", "<br>")
            + (f"<br><span style='color:{Theme.BAD}'>" + " · ".join(res.notes) + "</span>" if res.notes else ""))
        cov = S.coverage(res.scenes)
        if cov:
            self.coverage.setText(
                f"커버리지 (카메라 좌표): 가로 {cov['x_range']:.2f} m · 세로 {cov['y_range']:.2f} m · 깊이 {cov['z_range']:.2f} m"
                f" · 기울기 {cov['tilt_min']:.0f}~{cov['tilt_max']:.0f}°  — 보드가 넓게 퍼지고 기울기가 다양할수록 회전이 잘 잡힙니다")
        self.report.setPlainText(room.report(res))

    def is_complete(self):
        return any(c.result is not None and c.result.ok for c in self.state.cams)

    def status_text(self):
        cams = self.state.cams
        done = [c for c in cams if c.result is not None]
        if not done:
            return "풀이 전"
        return " · ".join(f"{c.name} {c.result.image_rmse:.1f}px" if c.result.ok else f"{c.name} 실패" for c in done)


# =================================================================== overlay


class OverlayStep(RoomStep):
    title = "6. 오버레이 검증"
    subtitle = "카메라마다 투영 결과를 여러 방식으로 — 휠 확대, 드래그 이동, 더블클릭 맞춤"
    per_camera = True

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self._last = None
        self._timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self._timer.timeout.connect(self.render)
        self._blink = QtCore.QTimer(self, interval=600)
        self._blink.timeout.connect(self._blink_tick)
        self._blink_on = False

        def hook(w, sig):
            getattr(w, sig).connect(lambda *_: self._timer.start())
            return w

        self.color_by = hook(QtWidgets.QComboBox(), "currentIndexChanged")
        for key, label in (("range", "거리"), ("height", "높이 (LiDAR z)"), ("intensity", "Intensity"), ("flat", "단색")):
            self.color_by.addItem(label, key)
        self.cmap = hook(QtWidgets.QComboBox(), "currentIndexChanged")
        self.cmap.addItems(list(COLORMAPS))
        self.psize = hook(QtWidgets.QSpinBox(minimum=0, maximum=8, value=2), "valueChanged")
        self.dim = hook(QtWidgets.QSlider(QtCore.Qt.Horizontal, minimum=0, maximum=100, value=55), "valueChanged")
        self.alpha = hook(QtWidgets.QSlider(QtCore.Qt.Horizontal, minimum=10, maximum=100, value=100), "valueChanged")
        self.maxr = hook(QtWidgets.QDoubleSpinBox(minimum=1.0, maximum=200.0, value=15.0), "valueChanged")
        self.maxpts = hook(QtWidgets.QSpinBox(minimum=10_000, maximum=5_000_000, value=600_000, singleStep=100_000), "valueChanged")
        self.c_points = hook(QtWidgets.QCheckBox("점"), "toggled")
        self.c_holes = hook(QtWidgets.QCheckBox("구멍 (+ LiDAR, ○ 카메라)"), "toggled")
        self.c_markers = hook(QtWidgets.QCheckBox("마커 테두리"), "toggled")
        self.c_labels = hook(QtWidgets.QCheckBox("보드 이름"), "toggled")
        for c in (self.c_points, self.c_holes, self.c_markers, self.c_labels):
            c.setChecked(True)
        self.c_only = hook(QtWidgets.QCheckBox("보드 박스 안 점만"), "toggled")
        self.c_occl = hook(QtWidgets.QCheckBox("가려진 점 숨기기 (시점 차이)"), "toggled")
        self.c_occl.setChecked(True)
        self.c_edges = hook(QtWidgets.QCheckBox("에지 비교 (이미지 에지 흰색 vs LiDAR 깊이 경계)"), "toggled")
        self.edge_jump = hook(QtWidgets.QDoubleSpinBox(minimum=0.05, maximum=3.0, value=0.3, singleStep=0.05), "valueChanged")
        self.c_undist = hook(QtWidgets.QCheckBox("왜곡 제거한 이미지에 투영"), "toggled")
        self.c_undist.setToolTip("이미지를 렌즈 모델로 펴고, 점은 왜곡 없이 투영합니다.\n"
                                 "원본 투영과 같은 비교라 어긋남은 똑같이 보이지만, 직선이 곧게 보여 판단이 쉽습니다.")
        self.c_side = hook(QtWidgets.QCheckBox("원본과 나란히"), "toggled")
        self.undist_view = hook(QtWidgets.QComboBox(), "currentIndexChanged")
        self.undist_view.addItem("자르기 (빈 곳 없음)", "crop")
        self.undist_view.addItem("전체 화각 (가장자리 포함)", "full")
        self.c_live = QtWidgets.QCheckBox("실시간: 최신 카메라 프레임과 최신 스윕으로 갱신")
        self.c_live.setToolTip("1단계 '실시간 토픽'에서 구독 중일 때만 켜집니다")
        self.c_live.toggled.connect(self._live_toggle)
        self.live_sweeps = QtWidgets.QSpinBox(minimum=1, maximum=20, value=2)
        self.live_sweeps.setPrefix("스윕 ")
        self._live_timer = QtCore.QTimer(self, interval=330)
        self._live_timer.timeout.connect(lambda: self.isVisible() and self.render())
        self.c_blink = QtWidgets.QCheckBox("원본과 번갈아 보기 (깜빡임)")
        self.c_blink.toggled.connect(self._blink_toggle)
        self.frame = hook(QtWidgets.QSpinBox(minimum=0, maximum=0), "valueChanged")
        self.focus = hook(QtWidgets.QComboBox(), "currentIndexChanged")

        self.nudge = []
        grid = QtWidgets.QGridLayout()
        for k, (lab, unit, step) in enumerate((("rx", "°", 0.05), ("ry", "°", 0.05), ("rz", "°", 0.05),
                                               ("tx", "mm", 5.0), ("ty", "mm", 5.0), ("tz", "mm", 5.0))):
            sp = QtWidgets.QDoubleSpinBox(minimum=-1000, maximum=1000, singleStep=step, decimals=2 if unit == "°" else 0)
            sp.setSuffix(f" {unit}")
            sp.valueChanged.connect(lambda *_: self._timer.start())
            grid.addWidget(QtWidgets.QLabel(lab), k // 3, 2 * (k % 3))
            grid.addWidget(sp, k // 3, 2 * (k % 3) + 1)
            self.nudge.append(sp)
        reset = QtWidgets.QPushButton("Δ 초기화")
        reset.clicked.connect(lambda: [sp.setValue(0.0) for sp in self.nudge])
        self.apply_btn = QtWidgets.QPushButton("Δ 를 결과에 반영")
        self.apply_btn.clicked.connect(self.apply_nudge)

        self.edge_btn = QtWidgets.QPushButton("에지 정렬 검사 (더 잘 맞는 방향이 있는지)")
        self.edge_btn.clicked.connect(self.edge_check)
        self.edge_info = QtWidgets.QLabel()
        self.edge_info.setWordWrap(True)
        self.save_img = QtWidgets.QPushButton("이 화면 이미지로 저장…")
        self.save_img.clicked.connect(self.save_image)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        self.canvas = ImageCanvas()

        form = QtWidgets.QFormLayout()
        form.addRow("색 기준", self.color_by)
        form.addRow("컬러맵", self.cmap)
        form.addRow("점 크기 (px 반지름)", self.psize)
        form.addRow("사진 밝기", self.dim)
        form.addRow("점 불투명도", self.alpha)
        form.addRow("최대 거리 m", self.maxr)
        form.addRow("최대 점 수", self.maxpts)
        form.addRow("프레임", self.frame)
        form.addRow("보드 확대", self.focus)
        shows = QtWidgets.QGridLayout()
        for k, w in enumerate((self.c_points, self.c_holes, self.c_markers, self.c_labels, self.c_only,
                               self.c_occl)):
            shows.addWidget(w, k // 2, k % 2)
        und = QtWidgets.QHBoxLayout()
        und.addWidget(self.c_undist)
        und.addWidget(self.c_side)
        und.addWidget(self.undist_view)
        und.addStretch(1)
        shows.addLayout(und, 4, 0, 1, 2)

        left = QtWidgets.QVBoxLayout()
        left.addLayout(form)
        left.addLayout(shows)
        left.addWidget(self.c_edges)
        er = QtWidgets.QHBoxLayout()
        er.addWidget(QtWidgets.QLabel("  깊이 경계 문턱 m"))
        er.addWidget(self.edge_jump)
        er.addStretch(1)
        left.addLayout(er)
        left.addWidget(self.c_blink)
        lr = QtWidgets.QHBoxLayout()
        lr.addWidget(self.c_live)
        lr.addWidget(self.live_sweeps)
        lr.addStretch(1)
        left.addLayout(lr)
        box = QtWidgets.QGroupBox("extrinsic 미세조정 미리보기 (카메라 좌표계)")
        bl = QtWidgets.QVBoxLayout(box)
        bl.addLayout(grid)
        rr = QtWidgets.QHBoxLayout()
        rr.addWidget(reset)
        rr.addWidget(self.apply_btn)
        bl.addLayout(rr)
        left.addWidget(box)
        left.addWidget(self.edge_btn)
        left.addWidget(self.edge_info)
        left.addWidget(self.save_img)
        left.addWidget(self.info)
        left.addStretch(1)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(470)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(lw)
        lay.addWidget(self.canvas, 1)

    def on_enter(self):
        s = self.state
        self.frame.blockSignals(True)
        self.frame.setMaximum(max(len(s.images) - 1, 0))
        self.frame.setValue(len(s.images) // 2)
        self.frame.blockSignals(False)
        self.focus.blockSignals(True)
        keep = self.focus.currentText()
        self.focus.clear()
        self.focus.addItem("전체")
        if s.result is not None:
            self.focus.addItems([c.label for c in s.result.cams])
        if keep:
            self.focus.setCurrentText(keep)
        self.focus.blockSignals(False)
        self.render()

    def style(self) -> OverlayStyle:
        return OverlayStyle(
            show_points=self.c_points.isChecked(), color_by=self.color_by.currentData(),
            colormap=self.cmap.currentText(), point_px=self.psize.value(), dim=self.dim.value() / 100,
            alpha=self.alpha.value() / 100, max_range=self.maxr.value(), max_points=self.maxpts.value(),
            only_boards=self.c_only.isChecked(), hide_occluded=self.c_occl.isChecked(),
            show_holes=self.c_holes.isChecked(),
            show_markers=self.c_markers.isChecked(), show_labels=self.c_labels.isChecked(),
            edges=self.c_edges.isChecked(), edge_jump=self.edge_jump.value(),
            undistort=self.c_undist.isChecked(), side_by_side=self.c_side.isChecked(),
            undistort_view=self.undist_view.currentData())

    def delta(self):
        return np.array([sp.value() for sp in self.nudge])

    def _live_toggle(self, on):
        if on and self.state.live is None:
            self.c_live.setChecked(False)
            self.info.setText("실시간 구독 중이 아닙니다 — 1단계 '실시간 토픽' 탭에서 구독을 시작하세요")
            return
        (self._live_timer.start if on else self._live_timer.stop)()
        self.render()

    def render(self):
        s = self.state
        img = s.image(self.frame.value())
        cloud, inten = s.cloud, s.intensity
        live_on = self.c_live.isChecked() and s.live is not None
        if live_on:
            _, frame = s.live.latest_image(s.camera_topic)
            if frame is not None:
                img = frame
                cloud, inten = s.live.latest_cloud(self.live_sweeps.value())
        if img is None:
            return
        res = s.result
        if res is None or not res.ok:
            self.info.setText(f"{s.slot.name if s.slot else ''}: 5단계에서 먼저 풀어주세요 — 지금은 사진만 보입니다")
            self.canvas.set_image(img)
            return
        focus = None if self.focus.currentText() in ("", "전체") else self.focus.currentText()
        out, info = render(img, s.camera, res, cloud, inten, self.style(), self.delta(), focus)
        self._last = out
        d = self.delta()
        extra = f" · Δ 적용 중 {np.round(d, 2).tolist()}" if np.any(d) else ""
        rng = f" · 색 범위 {info['range'][0]:.2f}~{info['range'][1]:.2f}" if info["range"] else ""
        self.info.setText(("● 실시간 · " if live_on else "") + f"투영된 점 {info['n_points']:,}개{rng}{extra}\n"
                          f"이미지 오차 {res.image_rmse:.2f} px (보드 {len(res.scenes)}개)")
        if not self._blink_on:
            self.canvas.set_image(out)

    def edge_check(self):
        from gui.core.room_overlay import edge_alignment

        s = self.state
        if s.result is None or not s.result.ok:
            return
        img = s.image(self.frame.value())
        sol = nudged(s.result.solution, self.delta())
        self.edge_info.setText("검사 중… (10~30초)")

        def work():
            return edge_alignment(img, s.camera, sol, s.cloud)

        def done(out):
            base, best, best_score = out
            big = np.any(np.abs(best[:3]) > 0.1) or np.any(np.abs(best[3:]) > 15)
            verdict = (f"<span style='color:{Theme.BAD}'>더 잘 맞는 방향이 있습니다 — 아래 Δ 를 미리보기로 넣어 확인해 보세요</span>"
                       if big and best_score < base - 0.15 else
                       f"<span style='color:{Theme.GOOD}'>현재 결과가 에지 기준으로도 최적점 근처입니다</span>")
            self.edge_info.setText(
                f"LiDAR 깊이 경계 → 이미지 에지 평균 거리 <b>{base:.2f} px</b><br>"
                f"최적 Δ rx {best[0]:+.3f}° ry {best[1]:+.3f}° rz {best[2]:+.3f}° · "
                f"t {best[3]:+.0f} {best[4]:+.0f} {best[5]:+.0f} mm → {best_score:.2f} px<br>{verdict}")
            self._edge_best = best

        self.run_task(work, done, busy=self.edge_btn)

    def _blink_toggle(self, on):
        if on:
            self._blink.start()
        else:
            self._blink.stop()
            self._blink_on = False
            if self._last is not None:
                self.canvas.set_image(self._last)

    def _blink_tick(self):
        s = self.state
        self._blink_on = not self._blink_on
        if self._blink_on:
            from dataclasses import replace as dc_replace

            img = s.image(self.frame.value())
            if self.c_live.isChecked() and s.live is not None:
                live_img = s.live.latest_image(s.camera_topic)[1]
                img = live_img if live_img is not None else img
            focus = None if self.focus.currentText() in ("", "전체") else self.focus.currentText()
            # The bare photo, framed exactly as the overlay is -- same view,
            # same crop, side by side if that is on -- so the two swap in place.
            bare = dc_replace(self.style(), show_points=False, show_holes=False, show_markers=False,
                              show_labels=False, edges=False, dim=1.0)
            out, _ = render(img, s.camera, s.result, None, None, bare, None, focus)
            self.canvas.set_image(out)
        elif self._last is not None:
            self.canvas.set_image(self._last)

    def apply_nudge(self):
        s = self.state
        d = self.delta()
        if s.result is None or not s.result.ok or not np.any(d):
            return
        if QtWidgets.QMessageBox.question(
                self, "수동 보정 반영",
                f"Δ {np.round(d, 2).tolist()} 를 결과 extrinsic 에 반영할까요?\n"
                "5단계에서 다시 풀면 사라집니다.") != QtWidgets.QMessageBox.Yes:
            return
        s.result.solution = nudged(s.result.solution, d)
        s.result.image_rmse, s.result.per_board_px = S.image_rmse(s.result.solution, s.result.scenes, s.camera)
        s.result.notes.append(f"수동 보정 Δ {np.round(d, 2).tolist()} (rx,ry,rz °, tx,ty,tz mm)")
        for sp in self.nudge:
            sp.blockSignals(True)
            sp.setValue(0.0)
            sp.blockSignals(False)
        self.render()
        self.changed.emit()

    def save_image(self):
        if self._last is None:
            return
        s = self.state
        start = (s.session_dir / f"overlay_{s.slot.name}.png") if s.session_dir is not None and s.slot else RESULT_DIR / "overlay.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "이미지 저장", str(start), "PNG (*.png)")
        if path:
            cv2.imwrite(path, self._last)
            self.info.setText(f"저장: {path}")

    def is_complete(self):
        return any(c.result is not None and c.result.ok for c in self.state.cams)

    def status_text(self):
        return "확인 가능" if self.is_complete() else "풀이 필요"


# ====================================================================== save


class SaveStep(RoomStep):
    title = "7. 저장"
    subtitle = "모든 카메라를 차량 / 세션(캡처 시각) 폴더에"

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.where = QtWidgets.QLabel()
        self.where.setWordWrap(True)
        self.where.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.note_edit = QtWidgets.QLineEdit()
        self.note_edit.setPlaceholderText("메모 (선택) — 예: 보드 3번 위치 옮긴 뒤, 타이어 공기압 조정 후")
        self.cams = table(["저장", "카메라", "토픽", "보드 쌍", "이미지 px", "폴더"])
        self.files = {}
        for key, label in (("extrinsic", "extrinsic.txt — FAST-LIVO2 형식"),
                           ("yaml", "result.yaml — R, t, 4x4 양방향, 보드별 수치, 차량·날짜"),
                           ("boards", "boards.yaml — 보드별 구멍 중심 (LiDAR / 카메라)"),
                           ("overlay", "overlay.png — 기본 스타일 오버레이"),
                           ("report", "report.txt · report.html — 보고서")):
            c = QtWidgets.QCheckBox(label)
            c.setChecked(True)
            self.files[key] = c
        self.save_btn = QtWidgets.QPushButton("💾  체크한 카메라 모두 저장")
        self.save_btn.setMinimumHeight(36)
        self.save_btn.clicked.connect(self.save)
        self.open_btn = QtWidgets.QPushButton("폴더 열기")
        self.open_btn.clicked.connect(self.open_dir)
        self.git_btn = QtWidgets.QPushButton("⬆  Git 업로드")
        self.git_btn.setToolTip("저장한 카메라 결과를 결과 전용 Git 저장소에 올립니다 (기본: extrinsic.txt, result.yaml). 대화상자의 ⚙ 설정에서 바꿀 수 있습니다")
        self.git_btn.clicked.connect(self.git_upload)
        self.preview = QtWidgets.QPlainTextEdit(readOnly=True)
        from PySide6 import QtGui
        self.preview.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.where)
        lay.addWidget(self.note_edit)
        lay.addWidget(self.cams, 1)
        fl = QtWidgets.QGridLayout()
        for k, c in enumerate(self.files.values()):
            fl.addWidget(c, k // 2, k % 2)
        lay.addLayout(fl)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.save_btn, 1)
        row.addWidget(self.open_btn)
        row.addWidget(self.git_btn)
        lay.addLayout(row)
        lay.addWidget(self.status)
        lay.addWidget(self.preview, 1)

    def on_enter(self):
        s = self.state
        d = s.session_dir
        from gui.core import session

        if d is None:
            self.where.setText(f"<span style='color:{Theme.BAD}'>차량을 고르고 캡처하면 저장 위치가 정해집니다</span>")
        else:
            self.where.setText(f"<b>차량</b> {s.vehicle} · <b>세션</b> {session.session_label(s.session_id)} "
                               f"(캡처 시각)<br><b>저장 위치</b> {d}")
        t = self.cams
        t.setRowCount(0)
        for k, c in enumerate(s.cams):
            ok = c.result is not None and c.result.ok
            t.insertRow(k)
            t.setItem(k, 0, cell("", check=ok))
            if not ok:
                t.item(k, 0).setFlags(QtCore.Qt.ItemIsEnabled)
            vals = [c.name, c.topic, len(c.result.scenes) if ok else "—", f"{c.result.image_rmse:.2f}" if ok else "풀이 없음",
                    str(d / c.name) if d is not None else "—"]
            for j, v in enumerate(vals, start=1):
                t.setItem(k, j, cell(v, color=None if ok else Theme.TEXT_MUTED))
        self.git_btn.setEnabled(d is not None and any(c.saved_to for c in s.cams))
        if s.result is not None and s.result.ok:
            h, w = s.images[0].shape[:2]
            self.preview.setPlainText(f"[{s.slot.name}]\n" + S.to_fast_livo2(s.result.solution, s.camera, w, h)
                                      + "\n" + _pose_text(s.result.solution))
        else:
            self.preview.setPlainText("5단계에서 먼저 풀어주세요")

    def open_dir(self):
        d = self.state.session_dir
        if d is not None and d.exists():
            _open_folder(d)

    def save(self):
        from gui.core import session

        s = self.state
        d = s.session_dir
        if d is None:
            self.status.setText("차량과 캡처가 있어야 저장할 수 있습니다")
            return
        chosen = [s.cams[r] for r in range(self.cams.rowCount())
                  if self.cams.item(r, 0).checkState() == QtCore.Qt.Checked
                  and s.cams[r].result is not None and s.cams[r].result.ok]
        if not chosen:
            self.status.setText("저장할 결과가 없습니다")
            return
        files = [k for k, c in self.files.items() if c.isChecked()]
        written, summary = [], []
        for c in chosen:
            meta = {
                "vehicle": s.vehicle, "session": s.session_id, "date": session.session_label(s.session_id),
                "camera_name": c.name, "camera_topic": c.topic, "source": s.bag,
                "lidar_topics": list(s.lidar_topics), "sweeps": s.n_sweeps, "frames": len(c.images),
                "target": s.target_name,
            }
            out = d / c.name
            names = session.save_camera(out, res=c.result, camera=c.camera, camera_preset=c.camera_name,
                                        image=c.image(), cloud=s.cloud, intensity=s.intensity, meta=meta,
                                        files=files, excluded=c.excluded_pairs)
            c.saved_to = str(out)
            written.append(f"{c.name}: {', '.join(names)}")
            r = c.result
            summary.append({"name": c.name, "topic": c.topic, "preset": c.camera_name, "ok": True,
                            "boards": len(r.scenes), "rmse_m": round(float(r.solution.rmse), 5),
                            "image_rmse_px": round(float(r.image_rmse), 3),
                            "T_camera_lidar": np.round(r.solution.matrix, 9).tolist()})
        session.write_summary(d, vehicle=s.vehicle, sid=s.session_id, source=s.bag, lidar_topics=s.lidar_topics,
                              n_sweeps=s.n_sweeps, roi=s.roi, target=s.target_name, cameras=summary,
                              note=self.note_edit.text().strip())
        if s.profile is not None:
            s.profile.lidar_topics = list(s.lidar_topics)
            for c in s.cams:
                s.profile.cameras[c.topic] = session.CameraEntry(c.name, c.camera_name, c.direction)
                if c.camera_name:
                    s.profile.lenses[c.name] = c.camera_name  # Camera Intrinsic 목록과 같은 렌즈로
            s.profile.roi = list(s.roi) or s.profile.roi
            s.profile.target = s.target_name
            s.profile.save()
        self.status.setText(f"저장: {d}<br>" + "<br>".join(written) + "<br>session.yaml · 차량 프로필 갱신")
        self.on_enter()
        self.changed.emit()

    def git_upload(self):
        s = self.state
        saved = [c for c in s.cams if c.saved_to]
        if s.session_dir is None or not saved:
            self.status.setText("먼저 저장하세요 — 저장된 결과만 올립니다")
            return
        try:  # 공용 업로드 모듈은 메인 앱(intrinsic/) 쪽에 있다
            from export import git_upload
            from ui.git_upload_dialog import GitUploadDialog
        except ImportError as e:
            self.status.setText(f"Git 업로드 모듈을 불러오지 못했습니다: {e}")
            return
        unsure = [c.name for c in saved if not room.reliable(c.result) or any("모호" in n for n in c.result.notes)]
        if unsure and QtWidgets.QMessageBox.question(
                self, "Git 업로드",
                "5단계에서 '확인 필요'였던 카메라가 저장돼 있습니다:\n  " + ", ".join(unsure)
                + "\n\n보드가 3개 미만이거나 대응이 모호하면 결과가 틀릴 수 있습니다. 그래도 올릴까요?") \
                != QtWidgets.QMessageBox.Yes:
            return
        names = [c.name for c in saved]
        GitUploadDialog(lambda config: git_upload.plan_extrinsic(s.session_dir, s.vehicle, names, config),
                        "Git 업로드 - CAMERA–LiDAR", self).exec()

    def is_complete(self):
        return any(c.saved_to for c in self.state.cams)

    def status_text(self):
        n = sum(bool(c.saved_to) for c in self.state.cams)
        return f"{n}대 저장됨" if n else "저장 전"


def _open_folder(path):
    from PySide6 import QtGui

    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
