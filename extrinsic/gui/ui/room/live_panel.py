"""Live input: find topics on whatever ROS is out there, watch every camera, capture one moment.

Every camera topic is subscribed and shown side by side, because a topic name
rarely says where a camera points: the cameras that see the room's front are
picked by their pictures (and by how many markers each one sees), named, and
calibrated together from one capture.

The capture is the point where live stops mattering: from there the steps run
exactly as they do on a bag. By default it is one LiDAR sweep and one picture
per camera -- the room stands still, so more only averages noise, and the
option to stack more is there for that.

Before the capture the panel answers the questions that decide whether it will
be any good -- are the boards in view, is everything still, does the extrinsic
the vehicle already has (or last session's) still line up -- and after a
solve, whether the new one lines up better.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ui.theme import Theme

from shared.qt import bgr_to_qpixmap

from gui.core import live, presets, room, session
from gui.core.room_overlay import OverlayStyle, render
from gui.ui.room import CamSlot, ImageCanvas, cell, direction_combo, table

VIEW_MODES = [("plain", "영상만"), ("markers", "마커 검출"), ("boards", "보드 검출 (내부 파라미터)"),
              ("project", "점 투영 (extrinsic)")]
EXT_SOURCES = [("prior", "기존 extrinsic"), ("result", "이번 결과"), ("blink", "번갈아 (기존 ↔ 결과)")]
THUMB_W = 320


def _bgr_to_pixmap(img: np.ndarray) -> QtGui.QPixmap:
    return bgr_to_qpixmap(img)


class CamTile(QtWidgets.QFrame):
    """One camera in the grid: its picture, whether it is in, and what it is called."""

    clicked = QtCore.Signal(str)

    def __init__(self, topic: live.LiveTopic, name: str, direction: str = "", parent=None):
        super().__init__(parent)
        self.topic = topic
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setLineWidth(2)
        self.pic = QtWidgets.QLabel("대기 중…")
        self.pic.setAlignment(QtCore.Qt.AlignCenter)
        self.pic.setFixedSize(THUMB_W, int(THUMB_W * 9 / 16))
        self.pic.setStyleSheet(f"background: {Theme.BG_ELEVATED}; color: {Theme.TEXT_SECONDARY};")
        self.pic.setCursor(QtCore.Qt.PointingHandCursor)
        self.use = QtWidgets.QCheckBox("포함")
        self.use.setToolTip("이번 캘리브레이션에 이 카메라를 넣습니다")
        self.name = QtWidgets.QLineEdit(name)
        self.name.setPlaceholderText("위치 이름 (예: FRONT_LEFT)")
        self.dir = direction_combo(direction)
        self.info = QtWidgets.QLabel(topic.name)
        self.info.setStyleSheet("color: #888;")
        self.info.setWordWrap(True)
        self.markers = 0
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.use)
        row.addWidget(self.name, 1)
        row.addWidget(self.dir)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.pic)
        lay.addLayout(row)
        lay.addWidget(self.info)
        self.set_selected(False)

    def mousePressEvent(self, e):
        self.clicked.emit(self.topic.key)
        super().mousePressEvent(e)

    def set_selected(self, on: bool):
        self.setStyleSheet(f"CamTile {{ border: 3px solid {Theme.ACCENT}; }}" if on else "CamTile { border: 3px solid transparent; }")

    def set_picture(self, img: np.ndarray | None, markers: list | None, hz: float):
        if img is None:
            return
        h, w = img.shape[:2]
        s = THUMB_W / w
        th = cv2.resize(img, (THUMB_W, int(h * s)), interpolation=cv2.INTER_AREA)
        if markers is not None:
            self.markers = len(markers)
            for c in markers:
                cv2.polylines(th, [(c * s).astype(np.int32)], True, (0, 230, 0), 2)
        # cv2.putText draws ASCII only (Hangul comes out as '?').
        tag = f"markers {self.markers}" if markers is not None or self.markers else ""
        if tag:
            cv2.putText(th, tag, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
            cv2.putText(th, tag, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if self.markers else (0, 200, 255), 2)
        self.pic.setFixedHeight(th.shape[0])
        self.pic.setPixmap(_bgr_to_pixmap(th))
        self.info.setText(f"{self.topic.name} · {w}x{h} · {hz:.1f} Hz")


class LivePanel(QtWidgets.QWidget):
    """Lives inside step 1. Emits `captured` once a capture has been loaded into the state."""

    captured = QtCore.Signal()

    def __init__(self, step, parent=None):
        super().__init__(parent)
        self.step = step
        self.state = step.state
        self.session: live.LiveSession | None = None
        self._probe: live.Probe | None = None
        self._blink = False
        self._last_view = 0.0
        self._rr = 0  # round-robin index for thumbnail refresh
        self.tiles: dict[str, CamTile] = {}
        self.selected: str | None = None  # camera key shown large
        self.priors: dict[str, object] = {}  # camera topic -> extrinsic_io.Extrinsic
        self._prev = {}  # camera name -> (Extrinsic, session id), from the vehicle's last sessions
        self._files: list = []  # Extrinsics from a loaded file
        self.presets = {p.name: p.camera for p in presets.load()}

        # ---- connection
        self.master = QtWidgets.QLineEdit(live.ros1_master_uri())
        self.domain = QtWidgets.QSpinBox(minimum=0, maximum=232)
        import os
        self.domain.setValue(int(os.environ.get("ROS_DOMAIN_ID", "0") or 0))
        self.backend = QtWidgets.QComboBox()
        for key, label in (("auto", "자동 (rclpy 있으면 rclpy, 없으면 DDS)"), ("rclpy", "rclpy (ROS 2 설치 사용)"),
                           ("dds", "DDS 직접 (ROS 설치 불필요)"), ("off", "ROS 2 끄기")):
            self.backend.addItem(label, key)
        self.ros1_on = QtWidgets.QCheckBox("ROS 1 도 찾기")
        self.ros1_on.setChecked(True)
        self.probe_btn = QtWidgets.QPushButton("🔍  탐색")
        self.probe_btn.clicked.connect(self.probe)
        self.notes = QtWidgets.QLabel()
        self.notes.setWordWrap(True)

        self.topics = table(["구독", "토픽", "종류", "ROS", "백엔드", "pub", "reliable"])
        self.start_btn = QtWidgets.QPushButton("▶  구독 시작")
        self.start_btn.setCheckable(True)
        self.start_btn.toggled.connect(self.toggle)
        self.stats = table(["토픽", "버퍼", "Hz", "도착 지연 ms", "마지막 수신"])

        # ---- capture
        self.sweeps = QtWidgets.QSpinBox(minimum=0, maximum=20, value=0)
        self.sweeps.setSpecialValueText("자동")
        self.sweeps.setToolTip("자동: 보통 1개, 다운샘플된 점군(표본 간격 6 cm 이상)이면 10개.\n"
                               "회전식 LiDAR 는 늘려도 점 밀도는 늘지 않고 거리 노이즈만 평균됩니다")
        self.frames = QtWidgets.QSpinBox(minimum=1, maximum=10, value=1)
        self.frames.setToolTip("카메라마다 LiDAR 시각에 가장 가까운 프레임 수. 여러 장이면 보드 검출을 평균합니다")
        self.snap_btn = QtWidgets.QPushButton("📸  캡처 → 캘리브레이션에 사용")
        self.snap_btn.setMinimumHeight(38)
        self.snap_btn.setStyleSheet("font-weight: 600;")
        self.snap_btn.clicked.connect(self.snapshot)
        self.rec_chk = QtWidgets.QCheckBox("캡처를 세션 폴더에 mcap 으로도 저장 (오프라인 재현용)")
        self.rec_chk.setChecked(True)
        self.snap_info = QtWidgets.QLabel()
        self.snap_info.setWordWrap(True)

        # ---- camera grid
        self.grid_host = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_host)
        self.grid.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.grid_host)
        self.cols = QtWidgets.QSpinBox(minimum=1, maximum=6, value=3)
        self.cols.valueChanged.connect(self._layout_tiles)
        self.marker_chk = QtWidgets.QCheckBox("썸네일에 마커 검출")
        self.marker_chk.setChecked(True)
        self.all_btn = QtWidgets.QPushButton("마커 보이는 카메라만 포함")
        self.all_btn.clicked.connect(self._include_seeing)
        self.grid_info = QtWidgets.QLabel()

        # ---- large view
        self.mode = QtWidgets.QComboBox()
        for k, label in VIEW_MODES:
            self.mode.addItem(label, k)
        self.mode.setCurrentIndex(1)
        self.preset = QtWidgets.QComboBox()
        self.preset.addItem("(내부 파라미터 없음)", "")
        for n in self.presets:
            self.preset.addItem(n, n)
        self.preset.setToolTip("투영·보드 검출용. 차량 프로필에 저장된 값이 있으면 자동으로 골라집니다")
        self.preset.currentIndexChanged.connect(self._preset_changed)
        self.ext_src = QtWidgets.QComboBox()
        for k, label in EXT_SOURCES:
            self.ext_src.addItem(label, k)
        self.prior_btn = QtWidgets.QPushButton("extrinsic 파일…")
        self.prior_btn.setToolTip("차량 YAML(*_calibration.yaml), 이 도구의 result.yaml, FAST-LIVO2 extrinsic.txt")
        self.prior_btn.clicked.connect(self.load_prior)
        self.prior_combo = QtWidgets.QComboBox()
        self.prior_combo.setMinimumWidth(220)
        self.prior_combo.currentIndexChanged.connect(self._pick_prior)
        self.undist = QtWidgets.QCheckBox("왜곡 제거")
        self.view_sweeps = QtWidgets.QSpinBox(minimum=1, maximum=20, value=2)
        self.view_sweeps.setPrefix("스윕 ")
        self.view_hz = QtWidgets.QDoubleSpinBox(minimum=0.5, maximum=15.0, value=4.0, singleStep=0.5)
        self.view_hz.setSuffix(" Hz")
        self.view_info = QtWidgets.QLabel()
        self.view_info.setWordWrap(True)
        self.canvas = ImageCanvas()

        self._timer = QtCore.QTimer(self, interval=250)
        self._timer.timeout.connect(self.tick)
        self._blink_timer = QtCore.QTimer(self, interval=700)
        self._blink_timer.timeout.connect(lambda: setattr(self, "_blink", not self._blink))
        self._blink_timer.start()

        # ---- layout
        form = QtWidgets.QFormLayout()
        form.addRow("ROS 1 master", self.master)
        form.addRow("ROS 2 domain", self.domain)
        form.addRow("ROS 2 수신 방식", self.backend)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.ros1_on)
        row.addWidget(self.probe_btn)
        left = QtWidgets.QVBoxLayout()
        left.addLayout(form)
        left.addLayout(row)
        left.addWidget(self.notes)
        left.addWidget(self.topics, 2)
        left.addWidget(self.start_btn)
        left.addWidget(self.stats, 1)
        cap = QtWidgets.QGroupBox("캡처")
        cf = QtWidgets.QFormLayout(cap)
        cf.addRow("LiDAR 스윕", self.sweeps)
        cf.addRow("카메라 프레임", self.frames)
        cf.addRow(self.rec_chk)
        cf.addRow(self.snap_btn)
        cf.addRow(self.snap_info)
        left.addWidget(cap)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(520)

        gbar = QtWidgets.QHBoxLayout()
        gbar.addWidget(QtWidgets.QLabel("열"))
        gbar.addWidget(self.cols)
        gbar.addWidget(self.marker_chk)
        gbar.addWidget(self.all_btn)
        gbar.addWidget(self.grid_info, 1)
        gtop = QtWidgets.QVBoxLayout()
        gtop.setContentsMargins(0, 0, 0, 0)
        gtop.addLayout(gbar)
        gtop.addWidget(scroll, 1)
        gw = QtWidgets.QWidget()
        gw.setLayout(gtop)

        v1 = QtWidgets.QHBoxLayout()
        v1.addWidget(QtWidgets.QLabel("보기"))
        v1.addWidget(self.mode)
        v1.addWidget(QtWidgets.QLabel("내부"))
        v1.addWidget(self.preset, 1)
        v1.addWidget(self.undist)
        v2 = QtWidgets.QHBoxLayout()
        v2.addWidget(QtWidgets.QLabel("extrinsic"))
        v2.addWidget(self.ext_src)
        v2.addWidget(self.prior_combo, 1)
        v2.addWidget(self.prior_btn)
        v2.addWidget(self.view_sweeps)
        v2.addWidget(self.view_hz)
        bottom = QtWidgets.QVBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addLayout(v1)
        bottom.addLayout(v2)
        bottom.addWidget(self.view_info)
        bottom.addWidget(self.canvas, 1)
        bw = QtWidgets.QWidget()
        bw.setLayout(bottom)

        vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vsplit.addWidget(gw)
        vsplit.addWidget(bw)
        vsplit.setSizes([430, 520])

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(lw)
        lay.addWidget(vsplit, 1)

    # ------------------------------------------------------------- vehicle

    def on_vehicle(self):
        """Last session's answers become the priors, matched by camera name."""
        v = self.state.vehicle
        self._prev = session.previous_extrinsics(v) if v else {}
        for tile in self.tiles.values():
            tile.name.setText(self._default_name(tile.topic.name))
        self._fill_priors()

    def _default_name(self, topic: str) -> str:
        prof = self.state.profile
        return prof.camera_name(topic) if prof is not None else session.camera_name_from_topic(topic)

    # ----------------------------------------------------------------- probe

    def probe(self):
        self.stop()
        self.notes.setText("탐색 중…")
        uri = self.master.text().strip() if self.ros1_on.isChecked() else None
        backend, domain = self.backend.currentData(), self.domain.value()
        ros1 = self.ros1_on.isChecked()

        def work():
            pr = live.probe(timeout=2.0, ros1_uri=uri or None, domain_id=domain, ros2_backend=backend)
            if not ros1 and "ros1" in pr.backends:
                pr.backends.pop("ros1").close()
                pr.topics = [t for t in pr.topics if t.backend != "ros1"]
            return pr

        def done(pr):
            if self._probe is not None:
                for b in self._probe.backends.values():
                    b.close()
            self._probe = pr
            self.notes.setText("\n".join(pr.notes))
            self.fill_topics()

        self.step.run_task(work, done, busy=self.probe_btn)

    def fill_topics(self):
        t = self.topics
        t.setRowCount(0)
        prof = self.state.profile
        known_lidar = set(prof.lidar_topics) if prof is not None else set()
        # The profile's LiDAR topics, if any of them is on the network now. A
        # renamed topic (new driver, ROS 1 -> 2) must not leave every LiDAR unchecked.
        known_lidar &= {tp.name for tp in self._probe.topics if tp.kind == "lidar"}
        for tp in self._probe.topics:
            if not tp.kind:
                continue
            r = t.rowCount()
            t.insertRow(r)
            if tp.kind == "lidar":
                on = tp.name in known_lidar if known_lidar else ("merge" in tp.name or tp.name.endswith("points"))
            else:
                on = True
            t.setItem(r, 0, cell("", check=on))
            for k, v in enumerate([tp.name, "LiDAR" if tp.kind == "lidar" else "카메라", tp.flavor, tp.backend,
                                   tp.publishers, "예" if tp.reliable else "아니오"], start=1):
                t.setItem(r, k, cell(v))
            t.item(r, 1).setData(QtCore.Qt.UserRole, tp.key)

    def _checked(self):
        by_key = {tp.key: tp for tp in self._probe.topics}
        lidar, cams = [], []
        for r in range(self.topics.rowCount()):
            tp = by_key[self.topics.item(r, 1).data(QtCore.Qt.UserRole)]
            if self.topics.item(r, 0).checkState() != QtCore.Qt.Checked:
                continue
            (lidar if tp.kind == "lidar" else cams).append(tp)
        return lidar, cams

    # ------------------------------------------------------------- subscribe

    def toggle(self, on: bool):
        if not on:
            self.stop()
            return
        if self._probe is None:
            self.start_btn.setChecked(False)
            self.notes.setText("먼저 탐색하세요")
            return
        lidar, cams = self._checked()
        if not lidar or not cams:
            self.start_btn.setChecked(False)
            self.notes.setText("LiDAR 토픽 1개 이상과 카메라 토픽 1개 이상을 체크하세요")
            return
        try:
            self.session = live.LiveSession(self._probe)
            self.session.start(lidar, cams)
        except Exception as e:  # noqa: BLE001
            self.start_btn.setChecked(False)
            self.notes.setText(f"구독 실패: {e}")
            return
        self.state.live = self.session
        self._probe = None  # the session owns the backends now
        self.start_btn.setText("■  구독 정지")
        self._make_tiles(cams)
        self._timer.start()

    def _make_tiles(self, cams):
        for tile in self.tiles.values():
            tile.setParent(None)
        self.tiles = {}
        prof = self.state.profile
        known = set(prof.cameras) if prof is not None else set()
        for tp in cams:
            name = self._default_name(tp.name)
            direction = prof.camera_direction(tp.name, name) if prof is not None else room.guess_direction(name)
            tile = CamTile(tp, name, direction)
            tile.use.setChecked(tp.name in known)
            tile.clicked.connect(self._select)
            tile.use.toggled.connect(self._count_included)
            self.tiles[tp.key] = tile
        self._layout_tiles()
        if cams:
            self._select(cams[0].key)
        self._count_included()

    def _layout_tiles(self, *_):
        for tile in self.tiles.values():
            self.grid.removeWidget(tile)
        n = self.cols.value()
        for k, tile in enumerate(self.tiles.values()):
            self.grid.addWidget(tile, k // n, k % n)

    def _select(self, key: str):
        self.selected = key
        for k, tile in self.tiles.items():
            tile.set_selected(k == key)
        tile = self.tiles[key]
        prof = self.state.profile
        slot = self._slot_for(tile.topic.name)
        preset = (slot.camera_name if slot is not None and slot.camera_name
                  else (prof.lens_for(tile.topic.name) if prof is not None else ""))
        i = self.preset.findData(preset)
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(max(i, 0))
        self.preset.blockSignals(False)
        self._fill_priors()
        self._last_view = 0.0

    def _include_seeing(self):
        for tile in self.tiles.values():
            tile.use.setChecked(tile.markers > 0)

    def _count_included(self, *_):
        n = sum(t.use.isChecked() for t in self.tiles.values())
        self.grid_info.setText(f"카메라 {len(self.tiles)}대 중 <b>{n}대 포함</b>")

    def stop(self):
        self._timer.stop()
        if self.session is not None:
            self.session.stop()
            self.session = None
            self.state.live = None
        self.start_btn.blockSignals(True)
        self.start_btn.setChecked(False)
        self.start_btn.setText("▶  구독 시작")
        self.start_btn.blockSignals(False)

    # ------------------------------------------------------------------ tick

    def tick(self):
        s = self.session
        if s is None or not self.isVisible():
            return
        st = s.stats()
        self.stats.setRowCount(0)
        for key, v in st.items():
            r = self.stats.rowCount()
            self.stats.insertRow(r)
            stale = v.age_s > 2.0
            for k, val in enumerate([key.split(":", 1)[1], v.count, f"{v.hz:.1f}",
                                     "—" if np.isnan(v.stamp_lag_ms) else (
                                         f"{v.stamp_lag_ms:.0f}" if abs(v.stamp_lag_ms) < 60_000
                                         else f"{v.stamp_lag_ms / 3.6e6:+.1f} h (녹화 재생?)"),
                                     "—" if not np.isfinite(v.age_s) else f"{v.age_s:.1f}s 전"]):
                self.stats.setItem(r, k, cell(val, color=Theme.BAD if stale and k == 4 else None))
        # A topic that is listed but silent: say why, if the backend knows
        # (ROS 1: the robot's hostname does not resolve here, a firewall, ...).
        age = {k.split(":", 1)[1]: v.age_s for k, v in st.items()}
        problems = {t: p for t, p in s.problems().items() if age.get(t, float("inf")) > 2.0}
        text = "\n".join(f"⚠ {t}: {p}" for t, p in problems.items())
        if text != getattr(self, "_problem_text", ""):
            self._problem_text = text
            self.notes.setText(text)
        # One thumbnail per tick: decoding and marker detection on every camera
        # at once would stall the UI with eight cameras.
        keys = list(self.tiles)
        if keys:
            key = keys[self._rr % len(keys)]
            self._rr += 1
            _, img = s.latest_image(key)
            if img is not None:
                markers = None
                if self.marker_chk.isChecked():
                    from gui.core.detect_camera import detect_markers

                    scale = min(1.0, 1280 / img.shape[1])
                    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
                    _, corners = detect_markers(small)
                    markers = [c / scale for c in corners]
                self.tiles[key].set_picture(img, markers, st.get(key, live.TopicStats()).hz)
        if time.time() - self._last_view >= 1.0 / self.view_hz.value():
            self._last_view = time.time()
            self.draw_view()

    # ------------------------------------------------------------ large view

    def _slot_for(self, topic: str):
        return next((c for c in self.state.cams if c.topic == topic), None)

    def _camera(self):
        name = self.preset.currentData()
        return self.presets.get(name) if name else None

    def _preset_changed(self, *_):
        self._last_view = 0.0

    def _extrinsic(self, topic: str):
        """(solution, label) for the projection, or (None, reason)."""
        prior = self.priors.get(topic)
        slot = self._slot_for(topic)
        result = slot.result.solution if slot is not None and slot.result is not None and slot.result.ok else None
        src = self.ext_src.currentData()
        if src == "blink":
            src = "prior" if self._blink else "result"
        if src == "prior":
            return (prior.solution, f"기존: {prior.name}") if prior else (None, "기존 extrinsic 이 없습니다 — 파일을 불러오거나 지난 세션을 고르세요")
        return (result, "이번 결과") if result is not None else (None, "아직 이 카메라의 결과가 없습니다 (5단계)")

    def draw_view(self):
        s = self.session
        if s is None or self.selected not in self.tiles:
            return
        topic = self.tiles[self.selected].topic.name
        stamp, img = s.latest_image(self.selected)
        if img is None:
            self.view_info.setText("카메라 프레임 대기 중…")
            return
        mode = self.mode.currentData()
        cam = self._camera()
        t0 = time.time()
        info = ""
        if mode == "markers":
            from gui.core.detect_camera import detect_markers

            out = img.copy()
            ids, corners = detect_markers(img)
            for i, c in zip(ids, corners):
                cv2.polylines(out, [c.astype(np.int32)], True, (0, 230, 0), 3)
                cv2.putText(out, str(i), tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 0), 2)
            info = f"마커 {len(ids)}개"
        elif mode == "plain" or cam is None:
            out = img
            if cam is not None and self.undist.isChecked():
                bare = OverlayStyle(show_points=False, show_holes=False, show_markers=False, show_labels=False,
                                    dim=1.0, undistort=True)
                out, _ = render(img, cam, SimpleNamespace(solution=None, cams=[], pairs=[], lidar=[]), None, None, bare)
            if cam is None and mode != "plain":
                info = "내부 파라미터 프리셋을 고르면 보드 검출·투영이 켜집니다"
        else:
            tgt = self.state.target or self.step.default_target()
            boards = room.detect_boards(img, cam, tgt)
            sol, label = (None, "")
            if mode == "project":
                sol, label = self._extrinsic(topic)
            res = SimpleNamespace(solution=sol, cams=boards, pairs=[], lidar=[])
            cloud, inten = (s.latest_cloud(self.view_sweeps.value()) if sol is not None else (None, None))
            style = OverlayStyle(show_points=sol is not None, point_px=2, max_points=200_000, hide_occluded=True,
                                 undistort=self.undist.isChecked())
            out, _ = render(img, cam, res, cloud, inten, style)
            info = f"보드 {len(boards)}개 · " + (label if mode == "project" else "보드 검출")
            if mode == "project" and sol is None:
                info = label
        self.canvas.set_image(out)
        self.view_info.setText(f"<b>{self.tiles[self.selected].name.text()}</b> ({topic}) · {info} · "
                               f"렌더 {1000 * (time.time() - t0):.0f} ms")

    # ------------------------------------------------------------- extrinsic

    def _fill_priors(self):
        """The prior choices for the selected camera: last sessions' results, then a loaded file's."""
        self.prior_combo.blockSignals(True)
        self.prior_combo.clear()
        self.prior_combo.addItem("(없음)", None)
        for name, (ex, sid) in self._prev.items():
            self.prior_combo.addItem(f"지난 세션: {ex.name}", ex)
        for ex in self._files:
            self.prior_combo.addItem(f"{ex.name}  ({ex.source})", ex)
        topic = self.tiles[self.selected].topic.name if self.selected in self.tiles else None
        cur = self.priors.get(topic) if topic else None
        if cur is None and topic:
            # Same camera name as last time -> last time's answer.
            name = self.tiles[self.selected].name.text()
            if name in self._prev:
                cur = self._prev[name][0]
                self.priors[topic] = cur
        idx = 0
        for i in range(self.prior_combo.count()):
            if self.prior_combo.itemData(i) is cur and cur is not None:
                idx = i
        self.prior_combo.setCurrentIndex(idx)
        self.prior_combo.blockSignals(False)

    def load_prior(self):
        from gui.core import extrinsic_io

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "기존 extrinsic", str(Path.home()), "extrinsic (*.yaml *.yml *.txt);;모두 (*)")
        if not path:
            return
        try:
            items = extrinsic_io.load(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "불러오기 실패", str(e))
            return
        self._files = items
        self._fill_priors()
        if len(items) == 1:
            self.prior_combo.setCurrentIndex(self.prior_combo.count() - 1)
        self.ext_src.setCurrentIndex(0)
        self.mode.setCurrentIndex(3)

    def _pick_prior(self, i):
        if self.selected not in self.tiles:
            return
        topic = self.tiles[self.selected].topic.name
        ex = self.prior_combo.itemData(i)
        if ex is None:
            self.priors.pop(topic, None)
        else:
            self.priors[topic] = ex
        slot = self._slot_for(topic)
        if slot is not None:
            slot.prior = ex
        self._last_view = 0.0

    # --------------------------------------------------------------- capture

    def snapshot(self):
        s = self.session
        if s is None:
            self.snap_info.setText("구독 중이 아닙니다")
            return
        st = self.state
        if not st.vehicle and not self.step.ask_vehicle():
            self.snap_info.setText("홈 화면에서 차량을 먼저 고르세요")
            return
        chosen = [t for t in self.tiles.values() if t.use.isChecked()]
        if not chosen:
            self.snap_info.setText("'포함' 한 카메라가 없습니다")
            return
        names = [session.safe_name(t.name.text()) for t in chosen]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            self.snap_info.setText(f"<span style='color:{Theme.BAD}'>카메라 이름이 겹칩니다: {', '.join(sorted(dup))}</span>")
            return
        try:
            snap = s.snapshot(self.sweeps.value(), self.frames.value(), [t.topic for t in chosen])
        except Exception as e:  # noqa: BLE001
            self.snap_info.setText(str(e))
            return

        st.cloud, st.single, st.intensity, st.n_sweeps = snap.cloud, snap.single, snap.intensity, snap.n_sweeps
        st.bag = f"live:{','.join(t.name for t in s.lidar)}"
        st.lidar_topics = [t.name for t in s.lidar]
        st.sweeps, st.frames = self.sweeps.value(), self.frames.value()
        st.session_id = session.session_id()  # the wall clock: this is when the room was captured
        slots = []
        for tile, name in zip(chosen, names):
            tile.name.setText(name)
            slot = CamSlot(topic=tile.topic.name, name=name, images=snap.images.get(tile.topic.name, []))
            slot.prior = self.priors.get(tile.topic.name)
            slot.direction = tile.dir.currentData() or ""
            slots.append(slot)
        self.step.adopt_cameras(slots)
        st.invalidate_from("data")

        warn = []
        if snap.n_sweeps < self.sweeps.value():
            warn.append(f"스윕 {snap.n_sweeps}개만 쌓였습니다")
        if np.isfinite(snap.stillness) and snap.stillness < 0.85:
            warn.append("최근 스윕끼리 정합률이 낮습니다 — 보드나 차량이 움직였을 수 있습니다")
        if snap.clock == "arrival":
            warn.append("센서끼리 header 시계가 달라(PTP 미동기?) 이 PC 수신 시각으로 맞췄습니다")
        bad_sync = [self._short(n) for n, v in snap.sync_ms.items() if v > 200]
        if bad_sync:
            warn.append("LiDAR 와 시각 차이가 큰 카메라: " + ", ".join(bad_sync))
        saved = ""
        if self.rec_chk.isChecked() and st.session_dir is not None:
            out = st.session_dir / "capture"
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                import shutil

                shutil.rmtree(out)
            live.record_mcap(snap, str(out))
            st.snapshot_path = str(out)
            saved = f"<br>저장: {out}"
        sync = ", ".join(f"{self._short(n)} {v:.0f}" for n, v in snap.sync_ms.items())
        self.snap_info.setText(
            f"<b>캡처</b> {len(snap.cloud):,}점 (스윕 {snap.n_sweeps}) · 카메라 {len(slots)}대 · "
            f"시각차 ms: {sync} · 스윕 정합 {snap.stillness:.0%}"
            + (f"<br>{snap.note}" if snap.note else "")
            + (f"<br><span style='color:{Theme.BAD}'>" + " · ".join(warn) + "</span>" if warn else "") + saved)
        self.captured.emit()

    def _short(self, topic: str) -> str:
        """A camera's name for messages, rather than its whole topic."""
        return next((t.name.text() for t in self.tiles.values() if t.topic.name == topic), topic)

    def shutdown(self):
        self.stop()
        if self._probe is not None:
            for b in self._probe.backends.values():
                b.close()
            self._probe = None
