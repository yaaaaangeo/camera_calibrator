"""Steps 1-3: load a capture, choose the camera, find the boards in the image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtWidgets

from ui.theme import Theme

from gui.core import camera_ops, presets, room
from gui.core.project import Target
from gui.core.room_overlay import BOARD_COLORS
from gui.ui.room import ImageCanvas, RoomStep, cell, direction_combo, table


# ====================================================================== data


def bag_format(path: str) -> str:
    """What kind of recording this is, as far as the files say."""
    p = Path(path)
    meta = (p / "metadata.yaml") if p.is_dir() else (p.parent / "metadata.yaml")
    if p.suffix == ".bag":
        return "ROS 1 bag"
    storage, version, distro = "", "", ""
    if meta.exists():
        import yaml

        info = (yaml.safe_load(meta.read_text()) or {}).get("rosbag2_bagfile_information", {})
        storage, version, distro = info.get("storage_identifier", ""), info.get("version", ""), info.get("ros_distro", "")
    if not storage:
        storage = {".mcap": "mcap", ".db3": "sqlite3"}.get(p.suffix, "?")
    return f"ROS 2 rosbag2 ({storage}" + (f", metadata v{version}" if version else "") + (f", {distro}" if distro else "") + ")"


def mosaic(images: list, names: list[str], width: int = 1920) -> np.ndarray | None:
    """Several cameras' pictures in one, labelled, for a glance at what was captured."""
    images = [(im, n) for im, n in zip(images, names) if im is not None]
    if not images:
        return None
    cols = int(np.ceil(np.sqrt(len(images))))
    w = width // cols
    tiles = []
    for im, n in images:
        h = int(im.shape[0] * w / im.shape[1])
        t = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        cv2.putText(t, n, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
        cv2.putText(t, n, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        tiles.append(t)
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(len(tiles) // cols)]
    return np.vstack(rows)


class DataStep(RoomStep):
    title = "1. 캡처"
    subtitle = "실시간 토픽에서 한 순간을 캡처 — 또는 bag(ROS1) · rosbag2(db3/mcap) 파일"

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.bag_edit = QtWidgets.QLineEdit()
        self.bag_edit.setReadOnly(True)
        browse = QtWidgets.QPushButton("파일…")
        browse.setToolTip(".bag (ROS 1), .db3 / .mcap (ROS 2)")
        browse.clicked.connect(self.browse)
        browse_dir = QtWidgets.QPushButton("rosbag2 폴더…")
        browse_dir.setToolTip("metadata.yaml 이 있는 rosbag2 폴더 (db3 / mcap) — 세션의 capture/ 폴더도 됩니다")
        browse_dir.clicked.connect(self.browse_dir)
        self.format_label = QtWidgets.QLabel()

        self.lidar_list = QtWidgets.QListWidget()
        self.lidar_list.setMaximumHeight(110)
        self.cam_table = table(["포함", "카메라 토픽", "이름", "방향"])
        self.cam_table.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)
        self.at_spin = QtWidgets.QDoubleSpinBox(minimum=0.0, maximum=1.0, singleStep=0.05, value=0.5)
        self.sweeps_spin = QtWidgets.QSpinBox(minimum=1, maximum=500, value=1)
        self.frames_spin = QtWidgets.QSpinBox(minimum=1, maximum=100, value=1)
        self.load_btn = QtWidgets.QPushButton("▶  불러오기")
        self.load_btn.setMinimumHeight(34)
        self.load_btn.clicked.connect(self.load)
        self.summary = QtWidgets.QLabel("bag 을 고르세요")
        self.summary.setWordWrap(True)
        self.stats = table(["토픽", "메시지", "Hz", "종류"])
        self.preview = ImageCanvas()

        form = QtWidgets.QFormLayout()
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.bag_edit, 1)
        row.addWidget(browse)
        row.addWidget(browse_dir)
        form.addRow("bag", row)
        form.addRow("형식", self.format_label)
        form.addRow("LiDAR 토픽 (여러 개 체크 = 합침)", self.lidar_list)
        form.addRow("bag 내 위치 (0~1)", self.at_spin)
        form.addRow("LiDAR 스윕", self.sweeps_spin)
        form.addRow("카메라 프레임", self.frames_spin)

        left = QtWidgets.QVBoxLayout()
        left.addLayout(form)
        left.addWidget(QtWidgets.QLabel("카메라 (체크한 것을 한꺼번에 캘리브레이션, 이름은 결과 폴더명)"))
        left.addWidget(self.cam_table, 1)
        left.addWidget(self.load_btn)
        left.addWidget(self.summary)
        left.addWidget(self.stats, 1)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(560)

        bag_tab = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(bag_tab)
        bl.addWidget(lw)
        bl.addWidget(self.preview, 1)

        from gui.ui.room.live_panel import LivePanel

        self.live = LivePanel(self)
        self.live.captured.connect(self._captured)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.live, "실시간 토픽 (ROS 1 / ROS 2)")
        self.tabs.addTab(bag_tab, "bag / rosbag2 파일")
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.tabs)

    # ------------------------------------------------------- shared helpers

    def ask_vehicle(self) -> bool:
        """차량은 홈 화면에서 고른다 - 여기서는 골랐는지만 본다."""
        return bool(self.state.vehicle)

    def default_target(self):
        from gui.core.project import Target

        t = {p.name: p.target for p in presets.load_targets()}
        prof = self.state.profile
        return t.get(prof.target if prof is not None and prof.target else "원본 CAD") or Target()

    def adopt_cameras(self, slots: list):
        """A new capture's cameras: lens presets and priors from the profile, then remember them."""
        from gui.core import session

        s = self.state
        prof = s.profile
        if s.target is None:
            # The LiDAR step comes before any camera step and needs the board's size.
            s.target = self.default_target()
            s.target_name = prof.target if prof is not None and prof.target else "원본 CAD"
        cams = {p.name: p.camera for p in presets.load()}
        prev = session.previous_extrinsics(s.vehicle) if s.vehicle else {}
        for c in slots:
            lens = prof.lens_for(c.topic, c.name) if prof is not None else ""
            if c.camera is None and lens in cams:
                c.camera_name, c.camera = lens, cams[lens]
            if c.prior is None and c.name in prev:
                c.prior = prev[c.name][0]
            if not c.direction:
                c.direction = prof.camera_direction(c.topic, c.name) if prof is not None else room.guess_direction(c.name)
        s.set_cameras(slots)
        if prof is not None:
            prof.lidar_topics = list(s.lidar_topics)
            for c in s.cams:
                old = prof.cameras.get(c.topic)
                prof.cameras[c.topic] = session.CameraEntry(c.name, c.camera_name or (old.preset if old else ""),
                                                            c.direction)
            if s.vehicle:
                prof.save()
        self.cameras_changed.emit()

    def on_vehicle(self):
        self.live.on_vehicle()
        if self.bag_edit.text():
            self.set_bag(self.bag_edit.text())

    def _captured(self):
        s = self.state
        self.summary.setText(f"실시간 캡처: 스윕 {s.n_sweeps}개 · {len(s.cloud):,}점 · 카메라 {len(s.cams)}대")
        self.preview.set_image(mosaic([c.image() for c in s.cams], [c.name for c in s.cams]))
        self.changed.emit()

    def shutdown(self):
        self.live.shutdown()
        super().shutdown()

    # ------------------------------------------------------------ bag files

    def browse_dir(self):
        start = str(Path(self.bag_edit.text()).parent) if self.bag_edit.text() else str(Path.home())
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "rosbag2 폴더 선택", start)
        if path:
            self.set_bag(path)

    def browse(self):
        start = str(Path(self.bag_edit.text()).parent) if self.bag_edit.text() else str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "bag 선택", start, "bag (*.bag *.db3 *.mcap);;모두 (*)")
        if path:
            self.set_bag(path)

    def set_bag(self, path: str):
        from gui.core import session
        from gui.core.bag_reader import inspect

        try:
            info = inspect(path)
        except Exception as e:  # noqa: BLE001
            self.summary.setText(f"열 수 없습니다: {e}")
            return
        self.tabs.setCurrentIndex(1)
        prof = self.state.profile
        self.bag_edit.setText(path)
        self.format_label.setText(bag_format(path))
        self.lidar_list.clear()
        known = set(prof.lidar_topics) if prof is not None and prof.lidar_topics else None
        for t in info.by_kind("lidar"):
            it = QtWidgets.QListWidgetItem(t.name)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if known is None or t.name in known else QtCore.Qt.Unchecked)
            self.lidar_list.addItem(it)
        self.cam_table.setRowCount(0)
        for t in info.by_kind("image"):
            r = self.cam_table.rowCount()
            self.cam_table.insertRow(r)
            self.cam_table.setItem(r, 0, cell("", check=True))
            self.cam_table.setItem(r, 1, cell(t.name))
            name = prof.camera_name(t.name) if prof is not None else session.camera_name_from_topic(t.name)
            it = QtWidgets.QTableWidgetItem(name)
            self.cam_table.setItem(r, 2, it)
            direction = prof.camera_direction(t.name, name) if prof is not None else room.guess_direction(name or t.name)
            self.cam_table.setCellWidget(r, 3, direction_combo(direction))
        self.stats.setRowCount(0)
        for t in info.topics:
            if t.kind not in ("lidar", "image"):
                continue
            r = self.stats.rowCount()
            self.stats.insertRow(r)
            for k, v in enumerate([t.name, t.count, f"{t.hz:.1f}", t.kind]):
                self.stats.setItem(r, k, cell(v))
        self.summary.setText(f"{Path(path).name}: {info.duration:.1f} s. 토픽을 고르고 불러오세요.")

    def load(self):
        from gui.core import session

        topics = [self.lidar_list.item(i).text() for i in range(self.lidar_list.count())
                  if self.lidar_list.item(i).checkState() == QtCore.Qt.Checked]
        rows = [r for r in range(self.cam_table.rowCount())
                if self.cam_table.item(r, 0).checkState() == QtCore.Qt.Checked]
        cams = [(self.cam_table.item(r, 1).text(), session.safe_name(self.cam_table.item(r, 2).text())) for r in rows]
        facing = {self.cam_table.item(r, 1).text(): self.cam_table.cellWidget(r, 3).currentData() or "" for r in rows}
        if not self.bag_edit.text() or not topics or not cams:
            self.summary.setText("bag, LiDAR 토픽(1개 이상), 카메라 토픽(1개 이상)을 고르세요")
            return
        names = [n for _, n in cams]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            self.summary.setText(f"카메라 이름이 겹칩니다: {', '.join(sorted(dup))}")
            return
        s = self.state
        if not s.vehicle and not self.ask_vehicle():
            self.summary.setText("홈 화면에서 차량을 먼저 고르세요")
            return
        s.bag, s.lidar_topics = self.bag_edit.text(), topics
        s.at, s.sweeps, s.frames = self.at_spin.value(), self.sweeps_spin.value(), self.frames_spin.value()
        self.summary.setText("읽는 중…")

        def work():
            return room.load_capture_multi(s.bag, s.lidar_topics, [t for t, _ in cams], s.at, s.sweeps, s.frames)

        def done(cap):
            from gui.ui.room import CamSlot

            s.cloud, s.single, s.intensity, s.n_sweeps = cap.cloud, cap.single, cap.intensity, cap.n_sweeps
            # A recording is dated by when it was recorded, not when it was read.
            s.session_id = session.session_id(cap.stamp_ns)
            self.adopt_cameras([CamSlot(topic=t, name=n, images=cap.images.get(t, []), direction=facing.get(t, ""))
                                for t, n in cams])
            s.invalidate_from("data")
            z = s.cloud[:, 2] if len(s.cloud) else np.zeros(1)
            per = " · ".join(f"{c.name} {len(c.images)}장" for c in s.cams)
            self.summary.setText(
                f"LiDAR: 스윕 {s.n_sweeps}개 · {len(s.cloud):,}점 · z {np.percentile(z, 1):.1f}~{np.percentile(z, 99):.1f} m"
                f" · intensity {'있음' if s.intensity is not None else '없음'}\n카메라: {per}")
            self.preview.set_image(mosaic([c.image() for c in s.cams], [c.name for c in s.cams]))
            self.changed.emit()

        self.run_task(work, done, busy=self.load_btn)

    def is_complete(self):
        return self.state.cloud is not None and any(c.images for c in self.state.cams)

    def status_text(self):
        s = self.state
        if not self.is_complete():
            if s.live is not None:
                return "실시간 구독 중 · 캡처 전"
            return "캡처 전"
        src = "실시간 " if s.bag.startswith("live:") else ""
        return f"{src}{len(s.cloud) / 1e6:.2f}M점 · 카메라 {len(s.cams)}대"


# ==================================================================== camera


class CameraStep(RoomStep):
    title = "3. 카메라"
    subtitle = "카메라마다 내부 파라미터 — 보드 재투영 오차로 고르기"
    per_camera = True

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.cameras = {p.name: p.camera for p in presets.load()}
        self.targets = {p.name: p.target for p in presets.load_targets()}

        self.overview = table(["카메라", "토픽", "해상도", "프리셋", "보드", "재투영 px"])
        self.overview.itemSelectionChanged.connect(self._overview_pick)
        self.all_btn = QtWidgets.QPushButton("▶  모든 카메라 자동 순위")
        self.all_btn.setMinimumHeight(32)
        self.all_btn.clicked.connect(self.rank_all)
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(list(self.cameras))
        self.combo.currentTextChanged.connect(self.choose)
        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.addItems(list(self.targets) or ["원본 CAD"])
        self.target_combo.setCurrentText("원본 CAD")
        self.target_combo.currentTextChanged.connect(self.choose_target)
        self.rank_btn = QtWidgets.QPushButton("이 카메라만 다시 비교")
        self.rank_btn.clicked.connect(self.rank)
        self.import_btn = QtWidgets.QPushButton("파일에서 렌즈 불러오기…")
        self.import_btn.setToolTip("Camera Intrinsic 결과(camera_<model>.yaml) 나 OpenCV YAML·ROS camera_info·Kalibr 파일을\n"
                                   "렌즈 프리셋으로 등록하고 이 카메라에 씁니다.")
        self.import_btn.clicked.connect(self.import_lens)
        self.rank_table = table(["프리셋", "모델", "보드", "재투영 중앙값 px"])
        self.rank_table.itemSelectionChanged.connect(self._pick_ranked)
        self.params = QtWidgets.QLabel()
        self.params.setWordWrap(True)
        self.params.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems(["원본", "왜곡 보정 (native)", "왜곡 보정 (full)"])
        self.view_combo.currentIndexChanged.connect(self.redraw)
        self.canvas = ImageCanvas()

        form = QtWidgets.QFormLayout()
        form.addRow("보드 규격 (모든 카메라)", self.target_combo)
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(self.combo, 1)
        preset_row.addWidget(self.import_btn)
        form.addRow("이 카메라 프리셋", preset_row)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.all_btn)
        left.addWidget(self.overview, 1)
        left.addLayout(form)
        left.addWidget(self.params)
        left.addWidget(self.rank_btn)
        left.addWidget(self.rank_table, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("보기"))
        row.addWidget(self.view_combo, 1)
        left.addLayout(row)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(560)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(lw)
        lay.addWidget(self.canvas, 1)

    def import_lens(self):
        """Camera Intrinsic 결과 파일 등을 렌즈 프리셋으로 등록하고 지금 카메라에 쓴다."""
        from pathlib import Path

        from shared import camera as shared_camera

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "렌즈(내부 파라미터) 파일", str(Path.home()),
            "캘리브레이션 파일 (*.yaml *.yml *.json);;모든 파일 (*)")
        if not path:
            return
        try:
            lens = shared_camera.read_lens_file(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "렌즈 불러오기 실패", f"{path}\n\n{e}")
            return
        slot = self.state.slot
        image = slot.image() if slot is not None else None
        if image is not None and lens.width and lens.height and (lens.width, lens.height) != image.shape[1::-1]:
            h, w = image.shape[:2]
            if QtWidgets.QMessageBox.question(
                    self, "해상도가 다릅니다",
                    f"파일은 {lens.width}x{lens.height}, 이 카메라 영상은 {w}x{h} 입니다.\n그래도 등록할까요?") \
                    != QtWidgets.QMessageBox.Yes:
                return
        name, ok = QtWidgets.QInputDialog.getText(self, "렌즈 프리셋 이름", "프리셋 이름 (같은 이름이 있으면 바꿉니다):",
                                                  text=Path(path).parent.parent.name + "_" + Path(path).stem
                                                  if Path(path).parent.name == "intrinsic" else Path(path).stem)
        if not ok or not name.strip():
            return
        shared_camera.add_lens_preset(name.strip(), lens, f"{Path(path).name} 에서 불러옴")
        self.cameras = {p.name: p.camera for p in presets.load()}
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(list(self.cameras))
        self.combo.blockSignals(False)
        self.combo.setCurrentText(name.strip())  # choose() 가 이 카메라에 적용한다
        self.choose(name.strip())

    def on_vehicle(self):
        prof = self.state.profile
        if prof is not None and prof.target and prof.target in self.targets:
            self.target_combo.setCurrentText(prof.target)

    def on_enter(self):
        s = self.state
        if s.target is None:
            self.choose_target()
        self.fill_overview()
        self.show_slot()
        # Rank on first visit rather than silently taking the first preset in
        # the list -- that is a different lens on this rig, and everything
        # after this step would quietly inherit it.
        if s.cams and any(not c.ranked and c.camera is None for c in s.cams) and self.all_btn.isEnabled():
            self.rank_all()

    # ---------------------------------------------------------- overview

    def fill_overview(self):
        s = self.state
        t = self.overview
        t.blockSignals(True)
        t.setRowCount(0)
        for k, c in enumerate(s.cams):
            img = c.image()
            size = f"{img.shape[1]}x{img.shape[0]}" if img is not None else "—"
            top = next((r for r in c.ranking if r[0] == c.camera_name), None)
            vals = [c.name, c.topic, size, c.camera_name or "미선택",
                    "—" if top is None else top[1], "—" if top is None or not np.isfinite(top[2]) else f"{top[2]:.2f}"]
            t.insertRow(k)
            for j, v in enumerate(vals):
                t.setItem(k, j, cell(v, color=Theme.BAD if j == 3 and not c.camera_name else None))
        if s.cams:
            t.selectRow(min(s.cur, len(s.cams) - 1))
        t.blockSignals(False)

    def _overview_pick(self):
        rows = self.overview.selectionModel().selectedRows()
        if rows and rows[0].row() != self.state.cur:
            self.select_camera.emit(rows[0].row())

    def show_slot(self):
        """Put the selected camera's choice and ranking on screen, without changing anything."""
        s = self.state
        slot = s.slot
        self.combo.blockSignals(True)
        if slot is not None and slot.camera_name:
            self.combo.setCurrentText(slot.camera_name)
        self.combo.blockSignals(False)
        self.fill_ranking(slot.ranking if slot is not None else [])
        self.describe()
        self.redraw()

    # ------------------------------------------------------------ choose

    def choose_target(self, *_):
        s = self.state
        name = self.target_combo.currentText()
        if s.target is not None and s.target_name == name:
            return
        s.target_name = name
        s.target = self.targets.get(name, Target())
        if s.profile is not None:
            s.profile.target = name
        s.invalidate_from("target")
        for c in s.cams:
            c.ranking, c.ranked = [], False
        self.changed.emit()

    def choose(self, *_):
        s = self.state
        name = self.combo.currentText()
        if not name or s.slot is None:
            return
        if s.camera_name != name:
            s.camera_name, s.camera = name, self.cameras[name]
            s.invalidate_from("camera")
            if s.profile is not None and s.camera_topic in s.profile.cameras:
                s.profile.cameras[s.camera_topic].preset = name
        self.describe()
        self.fill_overview()
        self.redraw()
        self.changed.emit()

    def describe(self):
        s = self.state
        c, t = s.camera, s.target or Target()
        if c is None:
            self.params.setText("프리셋을 고르세요")
            return
        img = s.image()
        pp_warn = ""
        if img is not None:
            ih, iw = img.shape[:2]
            off = room.principal_point_offset(c, iw, ih)
            if off > room.PRINCIPAL_POINT_TOL:
                pp_warn = (f"<br><span style='color:{Theme.BAD}'><b>⚠ 이 프리셋은 이 카메라용이 아닐 가능성이 큽니다</b>: "
                           f"주점 ({c.cx:.0f}, {c.cy:.0f}) 이 {iw}x{ih} 이미지 중앙에서 {off:.0%} 벗어나 있습니다. "
                           f"보드는 맞아 보여도 extrinsic 이동량이 틀어져 다른 거리의 물체가 어긋납니다.</span>")
        self.params.setText(
            f"<b>{c.model_label}</b>  fx {c.fx:.1f} fy {c.fy:.1f} cx {c.cx:.1f} cy {c.cy:.1f}<br>"
            f"dist {np.round(c.dist(), 4).tolist()}<br>"
            f"보드: 마커 {t.marker_size * 1000:.0f} mm, 마커 중심 ±{t.delta_width_qr_center:.3f}×±{t.delta_height_qr_center:.3f} m, "
            f"구멍 {t.delta_width_circles:.3f}×{t.delta_height_circles:.3f} m, r {t.circle_radius * 1000:.0f} mm" + pp_warn)

    # -------------------------------------------------------------- rank

    def _rank_one(self, img, target, layouts):
        h, w = img.shape[:2]
        return room.rank_cameras(img, self.cameras, target, layouts, w, h)

    def rank_all(self):
        s = self.state
        slots = [c for c in s.cams if c.image() is not None]
        if not slots:
            return
        target, layouts = s.target or Target(), s.layouts or (room.STOCK_LAYOUT,)
        self.params.setText(f"카메라 {len(slots)}대 비교 중…")

        def work():
            return [(c, self._rank_one(c.image(), target, layouts)) for c in slots]

        def done(out):
            for c, ranking in out:
                self._apply_ranking(c, ranking)
            self.fill_overview()
            self.show_slot()
            self.changed.emit()

        self.run_task(work, done, busy=self.all_btn)

    def rank(self):
        s = self.state
        slot = s.slot
        if slot is None or slot.image() is None:
            return
        target, layouts = s.target or Target(), s.layouts or (room.STOCK_LAYOUT,)

        def work():
            return self._rank_one(slot.image(), target, layouts)

        def done(ranking):
            self._apply_ranking(slot, ranking, force=True)
            self.fill_overview()
            self.show_slot()
            self.changed.emit()

        self.run_task(work, done, busy=self.rank_btn)

    def _apply_ranking(self, slot, ranking, force: bool = False):
        """Keep the ranking; take its winner unless the camera already has a lens that found boards."""
        slot.ranking, slot.ranked = ranking, True
        if not ranking or ranking[0][1] == 0:
            return
        mine = next((r for r in ranking if r[0] == slot.camera_name), None)
        if force or slot.camera is None or mine is None or mine[1] < ranking[0][1]:
            if slot.camera_name != ranking[0][0]:
                slot.camera_name, slot.camera = ranking[0][0], self.cameras[ranking[0][0]]
                slot.clear_from("camera")
                prof = self.state.profile
                if prof is not None and slot.topic in prof.cameras:
                    prof.cameras[slot.topic].preset = slot.camera_name

    def fill_ranking(self, ranking):
        self.rank_table.blockSignals(True)
        self.rank_table.setRowCount(0)
        for name, n, rms in ranking:
            r = self.rank_table.rowCount()
            self.rank_table.insertRow(r)
            for k, v in enumerate([name, self.cameras[name].model_label, n,
                                   "—" if not np.isfinite(rms) else f"{rms:.2f}"]):
                self.rank_table.setItem(r, k, cell(v, color=Theme.ACCENT if r == 0 else None))
        self.rank_table.blockSignals(False)
        s = self.state
        img = s.image()
        if img is not None and s.slot is not None and s.slot.ranked and not ranking:
            h, w = img.shape[:2]
            self.params.setText(
                f"<span style='color:{Theme.BAD}'><b>이 카메라({w}x{h})에 맞는 내부 파라미터 프리셋이 없습니다.</b><br>"
                "주점(cx, cy)이 이미지 중앙에서 10% 넘게 벗어난 프리셋은 다른 카메라·해상도용이라 제외했습니다. "
                "다른 카메라 값으로 풀면 보드는 맞아 보여도 extrinsic 이동량이 틀어져, 보드와 다른 거리의 물체가 어긋납니다. "
                "이 카메라의 내부 캘리브레이션 값을 camera_intrinsics.yaml 에 추가하세요.</span>")

    def _pick_ranked(self):
        rows = self.rank_table.selectionModel().selectedRows()
        if rows:
            self.combo.setCurrentText(self.rank_table.item(rows[0].row(), 0).text())

    def redraw(self, *_):
        s = self.state
        img = s.image()
        if img is None or s.camera is None:
            self.canvas.set_image(img)
            return
        mode = self.view_combo.currentIndex()
        if mode == 1:
            img = camera_ops.undistort_image(s.camera, img, "native")
        elif mode == 2:
            img = camera_ops.undistort_image(s.camera, img, "full")
        self.canvas.set_image(img)

    def is_complete(self):
        return bool(self.state.cams) and all(c.camera is not None for c in self.state.cams)

    def status_text(self):
        cams = self.state.cams
        if not cams:
            return "캡처 필요"
        return f"{sum(c.camera is not None for c in cams)}/{len(cams)} 카메라"


# =============================================================== camera boards


class CamBoardsStep(RoomStep):
    title = "4. 카메라 보드"
    subtitle = "카메라마다 마커를 보드별로 묶고 PnP 로 확인"
    per_camera = True

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.layout_edit = QtWidgets.QLineEdit()
        self.layout_edit.setPlaceholderText("비우면 모든 보드 1,2,4,3   (예: 1,2,4,3;5,6,8,7)")
        self.tol_spin = QtWidgets.QDoubleSpinBox(minimum=0.3, maximum=5.0, singleStep=0.1, value=1.2)
        self.rms_spin = QtWidgets.QDoubleSpinBox(minimum=0.5, maximum=20.0, singleStep=0.5, value=4.0)
        self.min_spin = QtWidgets.QSpinBox(minimum=3, maximum=4, value=3)
        self.all_btn = QtWidgets.QPushButton("▶  모든 카메라 보드 검출")
        self.all_btn.setMinimumHeight(34)
        self.all_btn.clicked.connect(self.detect_all)
        self.detect_btn = QtWidgets.QPushButton("이 카메라만 다시 검출")
        self.detect_btn.clicked.connect(self.detect)
        self.overview = QtWidgets.QLabel()
        self.overview.setWordWrap(True)
        self.summary = QtWidgets.QLabel()
        self.table = table(["사용", "보드", "마커", "거리 m", "기울기 °", "재투영 px", "마커 px", "프레임", "흩어짐 mm"])
        self.table.itemChanged.connect(self._toggled)
        self.table.itemSelectionChanged.connect(self.redraw)
        self.show_all = QtWidgets.QCheckBox("검출된 모든 마커 id 표시 (묶이지 않은 것 포함)")
        self.show_all.setChecked(True)
        self.show_all.toggled.connect(self.redraw)
        self.frame_spin = QtWidgets.QSpinBox(minimum=0, maximum=0)
        self.frame_spin.valueChanged.connect(self.redraw)
        self.canvas = ImageCanvas()

        form = QtWidgets.QFormLayout()
        form.addRow("마커 id 배치", self.layout_edit)
        form.addRow("이웃 마커 허용 (마커 폭 배수)", self.tol_spin)
        form.addRow("PnP 재투영 상한 px", self.rms_spin)
        form.addRow("보드당 최소 마커", self.min_spin)
        left = QtWidgets.QVBoxLayout()
        left.addLayout(form)
        left.addWidget(self.all_btn)
        left.addWidget(self.overview)
        left.addWidget(self.detect_btn)
        left.addWidget(self.summary)
        left.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("표시 프레임"))
        row.addWidget(self.frame_spin)
        row.addStretch(1)
        left.addLayout(row)
        left.addWidget(self.show_all)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(620)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(lw)
        lay.addWidget(self.canvas, 1)

    def on_enter(self):
        s = self.state
        self.frame_spin.blockSignals(True)
        self.frame_spin.setMaximum(max(len(s.images) - 1, 0))
        self.frame_spin.setValue(len(s.images) // 2)
        self.frame_spin.blockSignals(False)
        self.fill()
        self.redraw()
        if s.cams and any(c.camera is not None and not c.detected for c in s.cams) and self.all_btn.isEnabled():
            self.detect_all()

    def _params(self):
        """(layouts, detector kwargs) or None with the reason shown."""
        from gui.room_calib import parse_layouts

        try:
            layouts = parse_layouts(self.layout_edit.text().strip() or None)
        except SystemExit as e:
            self.summary.setText(str(e))
            return None
        if self.state.profile is not None:
            self.state.profile.layouts = self.layout_edit.text().strip()
        return layouts, dict(tol=self.tol_spin.value(), max_rms_px=self.rms_spin.value(),
                             min_markers=self.min_spin.value())

    def on_vehicle(self):
        prof = self.state.profile
        if prof is not None and prof.layouts:
            self.layout_edit.setText(prof.layouts)

    def _store(self, slot, boards):
        slot.cam_boards, slot.cam_excluded, slot.detected = boards, set(), True
        slot.clear_from("cam_boards")

    def detect_all(self):
        s = self.state
        slots = [c for c in s.cams if c.camera is not None and c.images]
        if not slots:
            self.summary.setText("3단계에서 카메라 내부 파라미터를 먼저 고르세요")
            return
        p = self._params()
        if p is None:
            return
        s.layouts, kw = p
        target = s.target
        self.overview.setText(f"카메라 {len(slots)}대 검출 중…")

        def work():
            return [(c, room.detect_boards_multi(c.images, c.camera, target, s.layouts, **kw)) for c in slots]

        def done(out):
            for c, boards in out:
                self._store(c, boards)
            self.fill()
            self.redraw()
            self.changed.emit()

        self.run_task(work, done, busy=self.all_btn)

    def detect(self):
        s = self.state
        slot = s.slot
        if slot is None or slot.camera is None or not slot.images:
            self.summary.setText("1·3단계를 먼저 마치세요")
            return
        p = self._params()
        if p is None:
            return
        s.layouts, kw = p
        target = s.target

        def work():
            return room.detect_boards_multi(slot.images, slot.camera, target, s.layouts, **kw)

        def done(boards):
            self._store(slot, boards)
            self.fill()
            self.redraw()
            self.changed.emit()

        self.run_task(work, done, busy=self.detect_btn)

    def fill(self):
        s = self.state
        self.overview.setText(" · ".join(
            f"<b>{c.name}</b> {len([b for b in c.cam_boards if b.label not in c.cam_excluded])}"
            if c.detected else f"<b>{c.name}</b> —" for c in s.cams))
        if s.slot is not None and s.slot.detected:
            used = sum(b.det.n_markers for b in s.cam_boards)
            self.summary.setText(f"{s.slot.name}: 마커 {used}개를 보드 {len(s.cam_boards)}개로 묶음")
        else:
            self.summary.setText("")
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for b in s.cam_boards:
            r = self.table.rowCount()
            self.table.insertRow(r)
            d = b.det
            warn = Theme.BAD if d.reproj_rms > 2.0 or d.marker_px < 40 else None
            vals = [b.label, d.n_markers, f"{d.distance:.2f}", f"{d.tilt_deg:.1f}", f"{d.reproj_rms:.2f}",
                    f"{d.marker_px:.0f}", b.frames, f"{b.spread_mm:.1f}"]
            self.table.setItem(r, 0, cell("", check=b.label not in s.cam_excluded))
            for k, v in enumerate(vals, start=1):
                self.table.setItem(r, k, cell(v, color=warn if k in (5, 6) else None))
        self.table.blockSignals(False)

    def _toggled(self, item):
        if item.column() != 0:
            return
        label = self.table.item(item.row(), 1).text()
        if item.checkState() == QtCore.Qt.Checked:
            self.state.cam_excluded.discard(label)
        else:
            self.state.cam_excluded.add(label)
        self.state.slot.clear_from("cam_boards")
        self.fill()
        self.redraw()
        self.changed.emit()

    def redraw(self, *_):
        from gui.core.detect_camera import detect_markers

        s = self.state
        img = s.image(self.frame_spin.value())
        if img is None or s.camera is None:
            self.canvas.set_image(img)
            return
        out = img.copy()
        if self.show_all.isChecked():
            ids, corners = detect_markers(img, small=True)
            for i, c in zip(ids, corners):
                cv2.polylines(out, [c.astype(np.int32)], True, (160, 160, 160), 1)
                cv2.putText(out, str(i), tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
        sel = {self.table.item(i.row(), 1).text() for i in self.table.selectionModel().selectedRows()}
        for k, b in enumerate(s.cam_boards):
            col = BOARD_COLORS[k % len(BOARD_COLORS)]
            off = b.label in s.cam_excluded
            thick = 5 if b.label in sel else 2
            if off:
                col = (90, 90, 90)
            for pts in b.det.image_corners:
                cv2.polylines(out, [pts.astype(np.int32)], True, col, thick)
            hull = cv2.convexHull(np.vstack(b.det.image_corners).astype(np.int32))
            cv2.polylines(out, [hull], True, col, 1 if not sel or b.label not in sel else 3)
            for (u, v) in camera_ops.project_points(s.camera, b.det.centers):
                cv2.circle(out, (int(u), int(v)), 10, col, 2)
            camera_ops.draw_axes(out, s.camera, b.det.rvec, b.det.tvec, 0.25, 2)
            u, v = b.centre_px.astype(int)
            txt = f"{b.label} {b.det.distance:.1f}m {b.det.reproj_rms:.1f}px"
            cv2.putText(out, txt, (u - 90, v - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 6)
            cv2.putText(out, txt, (u - 90, v - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        self.canvas.set_image(out)

    def active(self):
        return [b for b in self.state.cam_boards if b.label not in self.state.cam_excluded]

    def is_complete(self):
        cams = self.state.cams
        return any(any(b.label not in c.cam_excluded for b in c.cam_boards) for c in cams)

    def status_text(self):
        cams = self.state.cams
        if not cams:
            return "캡처 필요"
        if not any(c.detected for c in cams):
            return "검출 전"
        return " · ".join(f"{c.name} {len([b for b in c.cam_boards if b.label not in c.cam_excluded])}" for c in cams)
