"""Live capture for the calibration room: buffers, one-moment snapshots, mcap.

Finding topics and receiving them from ROS 1, ROS 2 (rclpy) or plain DDS is
shared with the Camera Intrinsic tool and lives in shared.ros.live; its names
are re-exported here.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from shared.ros.live import (  # noqa: F401
    KINDS,
    TS1,
    TS2,
    Backend,
    LiveTopic,
    Probe,
    Received,
    Ros1Tcpros,
    Ros2Dds,
    Ros2Rclpy,
    dds_available,
    probe,
    rclpy_available,
    ros1_master_uri,
    ros1_name,
    ros2_name,
    socket_buffer_warning,
)


# ================================================================== session


@dataclass
class TopicStats:
    count: int = 0
    hz: float = 0.0
    age_s: float = float("inf")
    stamp_lag_ms: float = float("nan")  # arrival minus header stamp


class LiveSession:
    """Subscriptions plus ring buffers, and snapshots that look like a bag read.

    Any number of cameras is subscribed at once -- which of them go into the
    calibration is decided by looking at their pictures, since a topic name
    rarely says where a camera points. Clouds are decoded as they arrive (they
    are needed for every overlay and the decode is cheap next to the transfer);
    images are kept compressed and decoded when asked for, newest only.
    """

    def __init__(self, probe_result: Probe, max_sweeps: int = 20, max_frames: int = 10):
        self.probe = probe_result
        self.max_sweeps, self.max_frames = max_sweeps, max_frames
        self._buf: dict[str, deque] = {}
        self._arrivals: dict[str, deque] = {}
        self._lock = threading.Lock()
        self.lidar: list[LiveTopic] = []
        self.cameras: list[LiveTopic] = []
        self._img_cache: dict[str, tuple] = {}

    @property
    def camera(self) -> LiveTopic | None:
        return self.cameras[0] if self.cameras else None

    def start(self, lidar: list[LiveTopic], cameras):
        self.lidar = list(lidar)
        self.cameras = [cameras] if isinstance(cameras, LiveTopic) else list(cameras or [])
        for t in self.lidar + self.cameras:
            depth = self.max_sweeps if t.kind == "lidar" else self.max_frames
            self._buf[t.key] = deque(maxlen=depth)
            self._arrivals[t.key] = deque(maxlen=50)
            self.probe.backends[t.backend].subscribe(t, self._make_cb(t))

    def _make_cb(self, t: LiveTopic):
        from gui.core.decode import cloud_extra, cloud_to_struct

        def cb(r: Received):
            if t.kind == "lidar":
                rec = cloud_to_struct(r.msg)
                xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float32)
                ok = np.isfinite(xyz).all(axis=1)
                inten = cloud_extra(r.msg, "intensity")
                ring = cloud_extra(r.msg, "ring")
                item = (r, xyz[ok], None if inten is None else np.asarray(inten)[ok].astype(np.float32),
                        None if ring is None else np.asarray(ring)[ok].astype(np.int32))
            else:
                item = (r,)
            with self._lock:
                self._buf[t.key].append(item)
                self._arrivals[t.key].append(r.recv_ns)
        return cb

    def stop(self):
        for b in self.probe.backends.values():
            b.close()

    # ------------------------------------------------------------- reading

    def stats(self) -> dict[str, TopicStats]:
        now = time.time_ns()
        out = {}
        with self._lock:
            for key, arr in self._arrivals.items():
                st = TopicStats(count=len(self._buf[key]))
                if len(arr) >= 2:
                    st.hz = (len(arr) - 1) / max((arr[-1] - arr[0]) / 1e9, 1e-9)
                if arr:
                    st.age_s = (now - arr[-1]) / 1e9
                    r = self._buf[key][-1][0]
                    st.stamp_lag_ms = (r.recv_ns - r.stamp_ns) / 1e6
                out[key] = st
        return out

    def _camera_key(self, camera) -> str | None:
        """A camera given as LiveTopic, key or topic name -> its buffer key."""
        if camera is None:
            return self.camera.key if self.camera else None
        if isinstance(camera, LiveTopic):
            return camera.key
        for t in self.cameras:
            if camera in (t.key, t.name):
                return t.key
        return None

    def latest_image(self, camera=None):
        """(stamp_ns, BGR) of the newest frame of `camera` (default: the first), decoded once."""
        from gui.core.decode import image_to_bgr

        key = self._camera_key(camera)
        with self._lock:
            buf = self._buf.get(key) if key else None
            r = buf[-1][0] if buf else None
        if r is None:
            return None, None
        cached = self._img_cache.get(key)
        if cached is None or cached[0] != r.stamp_ns:
            cached = (r.stamp_ns, image_to_bgr(r.msg))
            self._img_cache[key] = cached
        return cached

    def latest_cloud(self, sweeps: int = 1):
        """Newest `sweeps` sweeps of every LiDAR topic, stacked, plus intensity."""
        xyz, inten = [], []
        with self._lock:
            for t in self.lidar:
                for item in list(self._buf[t.key])[-sweeps:]:
                    xyz.append(item[1])
                    inten.append(item[2])
        if not xyz:
            return np.empty((0, 3), np.float32), None
        it = np.concatenate(inten) if all(i is not None for i in inten) else None
        return np.vstack(xyz), it

    def snapshot(self, sweeps: int = 1, frames: int = 1, cameras=None) -> "Snapshot":
        """Freeze one moment into what `room.load_capture` returns, for every camera.

        The moment is the newest sweep that every LiDAR topic has delivered;
        each topic contributes the `sweeps` sweeps ending there, each camera
        (`cameras`: topics, keys or names; default all) the `frames` frames
        nearest to it. With the defaults that is one sweep and one picture per
        camera -- the room stands still, so more is only noise averaging.
        """
        from gui.core.decode import image_to_bgr

        keys = [self._camera_key(c) for c in cameras] if cameras is not None else [t.key for t in self.cameras]
        keys = [k for k in keys if k]
        with self._lock:
            per_all = {t.key: list(self._buf[t.key]) for t in self.lidar}
            cam_all = {k: list(self._buf[k]) for k in keys}
        # Time order, not arrival order, so that the "middle sweep" and the frame
        # order are the ones a bag reader picks from the recording of this same
        # snapshot -- otherwise replaying it gives a slightly different answer.
        per_all = {k: sorted(v, key=lambda i: i[0].stamp_ns) for k, v in per_all.items()}
        if not any(per_all.values()):
            raise RuntimeError("LiDAR 버퍼가 비어 있습니다 — 토픽이 들어오는지 확인하세요")
        empty = [k.split(":", 1)[1] for k, v in cam_all.items() if not v]
        if empty:
            raise RuntimeError("카메라 프레임이 아직 없습니다: " + ", ".join(empty))
        clock = _common_clock(list(per_all.values()) + list(cam_all.values()))
        tm = (lambda r: r.stamp_ns) if clock == "header" else (lambda r: r.recv_ns)
        ref = min(tm(v[-1][0]) for v in per_all.values() if v)
        per = {}
        for k, items in per_all.items():
            if not items:
                continue
            stamps = np.array([tm(i[0]) for i in items])
            end = int(np.argmin(np.abs(stamps - ref)))
            per[k] = items[max(0, end - sweeps + 1):end + 1]
        cams = {}
        for k, items in cam_all.items():
            near = sorted(items, key=lambda c: abs(tm(c[0]) - ref))[:frames]
            cams[k] = sorted(near, key=lambda c: tm(c[0]))

        clouds = [i for items in per.values() for i in items]
        cloud = np.vstack([i[1] for i in clouds])
        inten = np.concatenate([i[2] for i in clouds]) if all(i[2] is not None for i in clouds) else None
        single = np.vstack([items[len(items) // 2][1] for items in per.values() if items])
        rings = [i[3] for i in clouds]
        ring = None
        if len(per) == 1 and all(r is not None for r in rings):
            span = max(int(r.max()) + 1 for r in rings if len(r)) if any(len(r) for r in rings) else 1
            ring = np.concatenate([r + k * span for k, r in enumerate(rings)])
        if len(per) > 1:
            # As `room.load_capture` does, so a replay of the recording agrees.
            from gui.core.room import dedupe_topics

            cloud, single, inten = dedupe_topics(cloud, single, inten)
        name = {t.key: t.name for t in self.cameras}
        images = {name[k]: [image_to_bgr(c[0].msg) for c in v] for k, v in cams.items()}
        n = min(len(v) for v in per.values())

        cloud_stamps = np.array([tm(i[0]) for i in clouds])
        sync = {name[k]: float(np.median([np.min(np.abs(cloud_stamps - tm(c[0]))) for c in v])) / 1e6
                for k, v in cams.items()}
        by_key = {t.key: t for t in self.lidar + self.cameras}
        raw = [(by_key[k], i[0]) for k, items in per.items() for i in items]
        raw += [(by_key[k], c[0]) for k, v in cams.items() for c in v]
        return Snapshot(cloud, single, ring, images, n, inten, raw=raw, sync_ms=sync,
                        # Over the whole buffer, not just the sweeps taken: with
                        # one sweep there is nothing to compare, but the room
                        # still has to have stood still for the picture to match.
                        stillness=_stillness({k: v[-10:] for k, v in per_all.items()}), stamp_ns=ref,
                        clock=clock)


def _common_clock(buffers: list) -> str:
    """"header" if the topics' header stamps share a clock, else "arrival".

    Not every rig syncs its sensors: on one here the LiDAR stamps its sweeps
    with seconds since it booted (3148 s) while the cameras stamp the date.
    Compared by header, every camera was 57 years from the LiDAR. Each topic's
    lag -- arrival minus header stamp -- is steady if its clock is; when the
    lags differ by more than a second between topics, the clocks are not the
    same, and this machine's arrival times (one clock, a few tens of ms of
    transport late) are the better yardstick for a room that stands still.
    """
    lags = []
    for items in buffers:
        if items:
            lags.append(float(np.median([i[0].recv_ns - i[0].stamp_ns for i in items[-10:]])))
    if len(lags) < 2:
        return "header"
    return "header" if max(lags) - min(lags) < 1e9 else "arrival"


def _stillness(per: dict) -> float:
    """Share of the newest sweep's points within 5 cm of the oldest sweep's.

    The room has to stand still while the snapshot accumulates. A board nudged,
    or the vehicle rocking as someone gets in, shows up as this dropping.
    """
    from scipy.spatial import cKDTree

    vals = []
    for items in per.values():
        if len(items) < 2:
            continue
        a, b = items[0][1], items[-1][1]
        a = a[np.linalg.norm(a, axis=1) < 20]
        b = b[np.linalg.norm(b, axis=1) < 20]
        if len(a) < 100 or len(b) < 100:
            continue
        pick = np.random.default_rng(0).choice(len(b), min(len(b), 5000), replace=False)
        d, _ = cKDTree(a).query(b[pick], k=1)
        vals.append(float(np.mean(d < 0.05)))
    return float(np.mean(vals)) if vals else float("nan")


@dataclass
class Snapshot:
    cloud: np.ndarray
    single: np.ndarray
    ring: np.ndarray | None
    images: dict  # camera topic -> [BGR]
    n_sweeps: int
    intensity: np.ndarray | None
    raw: list = field(default_factory=list)  # (LiveTopic, Received)
    sync_ms: dict = field(default_factory=dict)  # camera topic -> median gap to the nearest sweep stamp
    stillness: float = float("nan")
    stamp_ns: int = 0  # the moment the snapshot was taken at (LiDAR stamp, in `clock`)
    clock: str = "header"  # "header" stamps, or "arrival" times when the sensors' clocks disagree


def to_ros2(msg, msgtype: str, ts=TS2):
    """A decoded ROS 1 message rebuilt as the ROS 2 type, field by field.

    The byte-level `ros1_to_cdr` cannot do this: the two Header types differ
    (ROS 1 carries a `seq`), so the layouts do not line up. Matching by field
    name drops what ROS 2 does not have, which is how rosbags' own converter
    treats it.
    """
    fields = ts.fielddefs[msgtype][1]
    values = []
    for name, (kind, spec) in fields:
        v = getattr(msg, name)
        if kind == 2:  # nested message
            v = to_ros2(v, spec, ts)
        elif kind in (3, 4) and spec[0] == 2:  # array / sequence of messages
            v = [to_ros2(x, spec[1], ts) for x in v]
        values.append(v)
    return ts.types[msgtype](*values)


def record_mcap(snapshot: Snapshot, path: str) -> str:
    """Write the snapshot's messages as a rosbag2 (mcap), so it replays offline.

    ROS 1 payloads are converted to CDR on the way, so the file reads the same
    in ROS 2 tools and back through this calibrator's own bag reader.
    """
    from rosbags.rosbag2 import StoragePlugin, Writer

    conns = {}
    # Version 8 metadata: Humble's `ros2 bag` cannot parse version 9 (it chokes
    # on the QoS list), while newer distributions still read 8.
    with Writer(path, version=8, storage_plugin=StoragePlugin.MCAP) as w:
        key = (lambda tr: tr[1].stamp_ns) if snapshot.clock == "header" else (lambda tr: tr[1].recv_ns)
        for topic, r in sorted(snapshot.raw, key=key):
            if topic.name not in conns:
                conns[topic.name] = w.add_connection(topic.name, topic.msgtype, typestore=TS2)
            data = r.raw if r.fmt == "cdr" else TS2.serialize_cdr(to_ros2(r.msg, topic.msgtype), topic.msgtype)
            # Logged on the clock the snapshot was cut on, so reading the
            # recording back picks the same sweep and frames.
            w.write(conns[topic.name], r.stamp_ns if snapshot.clock == "header" else r.recv_ns, data)
    return path
