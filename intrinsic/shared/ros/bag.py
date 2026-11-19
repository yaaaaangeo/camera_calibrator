"""
shared.ros.bag
==============

ROS bag(.bag / rosbag2 폴더(.db3·.mcap)) 읽기 - ROS 설치 없이 ``rosbags`` 로.

Camera Intrinsic(영상 추출)과 CAMERA–LiDAR(스윕·프레임 찾기)가 같은 방식으로 연다.
타입 정의가 bag 안에 없는 오래된 rosbag2 녹화도 열리도록 ROS 2 표준 typestore 를
기본값으로 준다 - 정의가 들어 있는 bag(ROS 1 전부, 최근 rosbag2)은 자기 정의를 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# Message types we can currently turn into points or images. Anything else is
# listed for the user but not offered as a source.
LIDAR_TYPES = ("sensor_msgs/msg/PointCloud2",)
IMAGE_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")


_TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


def bag_path(path: str | Path) -> Path:
    """rosbag2 녹화 폴더 안의 .db3/.mcap 파일을 고른 경우 폴더로 바꾼다 (분할 파일 전부 읽게)."""
    p = Path(path)
    if p.is_file() and p.suffix in (".db3", ".mcap") and (p.parent / "metadata.yaml").is_file():
        return p.parent
    return p


def open_reader(paths) -> AnyReader:
    """AnyReader (열기 전). `paths` 는 경로 하나 또는 여러 개."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return AnyReader([bag_path(p) for p in paths], default_typestore=_TYPESTORE)




@dataclass
class TopicInfo:
    name: str
    msgtype: str
    count: int
    hz: float

    @property
    def kind(self) -> str:
        if self.msgtype in LIDAR_TYPES:
            return "lidar"
        if self.msgtype in IMAGE_TYPES:
            return "image"
        return "-"


@dataclass
class BagInfo:
    path: Path
    duration: float  # seconds
    start: float  # unix seconds
    end: float
    message_count: int
    topics: list[TopicInfo]

    def by_kind(self, kind: str) -> list[TopicInfo]:
        return [t for t in self.topics if t.kind == kind]


class BagSource:
    """A bag held open for repeated reads.

    Opening a large bag costs about a second, so the scrubber keeps one of
    these around rather than reopening per frame. Seeks go through the bag
    index, so jumping to an arbitrary time stays cheap.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._reader: AnyReader | None = None
        self.start_ns = 0
        self.end_ns = 0
        self._stamps: dict[str, list[int]] = {}

    def __enter__(self) -> "BagSource":
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    def open(self):
        if self._reader is not None:
            return
        self._reader = open_reader(self.path)
        self._reader.open()
        self.start_ns, self.end_ns = self._reader.start_time, self._reader.end_time

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self._stamps.clear()

    @property
    def duration(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9

    def at_fraction(self, f: float) -> int:
        """Bag timestamp `f` of the way through the recording (0..1)."""
        return self.start_ns + int((self.end_ns - self.start_ns) * min(max(f, 0.0), 1.0))

    def _connections(self, topic: str):
        conns = [c for c in self._reader.connections if c.topic == topic]
        if not conns:
            raise KeyError(f"topic not in bag: {topic}")
        return conns

    def message_count(self, topic: str) -> int:
        """How many messages a topic holds, from the index rather than by reading."""
        self.open()
        info = self._reader.topics.get(topic)
        return int(getattr(info, "msgcount", 0) or 0) if info else 0

    def timestamps(self, topic: str, progress=None) -> list[int]:
        """Every message time on a topic, read once and kept.

        This walks the whole bag, so its cost scales with the recording rather
        than with what is being asked for: 17 seconds on a 36 GB file. Every scene
        switch used to call it again for the same topic and get the same answer,
        which made opening a scene take 43 seconds and read as a hung window.
        Held per source, so it is dropped when the bag is closed.

        `progress(done, total)` is called every few hundred messages when given.
        The first read is slow enough that silence is indistinguishable from a
        crash -- which is exactly how it was reported.
        """
        cached = self._stamps.get(topic)
        if cached is not None:
            return cached
        self.open()
        total = self.message_count(topic)
        stamps = []
        # Report often enough to look alive, rarely enough not to pay for it: a
        # signal per message would cost more than the read itself.
        step = max(total // 100, 200)
        for _, stamp, _ in self._reader.messages(connections=self._connections(topic)):
            stamps.append(stamp)
            if progress is not None and len(stamps) % step == 0:
                progress(len(stamps), total)
        if progress is not None:
            progress(len(stamps), total)
        self._stamps[topic] = stamps
        return stamps

    def log_lag_ns(self, topic: str, samples: int = 15) -> int:
        """Median (bag time - header stamp) on a topic, in nanoseconds.

        Both sensors are timestamped when the message is written, and the two
        are not written the same distance behind the moment they describe: on
        these recordings the camera lands 45-59 ms after its header and the
        LiDAR 140-182 ms after the start of its sweep. Pairing on bag time
        therefore hands a scene a sweep about a tenth of a second older than the
        image -- 25 mm of board at the speed these were carried at. Read from
        the messages rather than configured, because it is a property of the
        recording, and returns 0 for a topic whose messages carry no header.
        """
        cached = getattr(self, "_lag", None)
        if cached is None:
            cached = self._lag = {}
        if topic in cached:
            return cached[topic]
        self.open()
        out = []
        for conn, stamp, raw in self._messages(topic, self.start_ns):
            msg = self._reader.deserialize(raw, conn.msgtype)
            header = getattr(msg, "header", None)
            if header is None:
                break
            out.append(stamp - (header.stamp.sec * 10**9 + header.stamp.nanosec))
            if len(out) >= samples:
                break
        cached[topic] = int(np.median(out)) if out else 0
        return cached[topic]

    def _messages(self, topic: str, start_ns: int, stop_ns: int | None = None):
        """Raw messages with start_ns <= stamp < stop_ns, whatever the reader thinks.

        The bound is not the same across storage backends: on the MCAP bags here
        `start` is exclusive, so asking from a message's own stamp skips it. That
        silently gave scenes the camera frame *after* the one captured -- a sweep
        interval late -- and left a one-sweep window with no sweep at all. Asking
        from a nanosecond earlier and filtering here is right either way.
        """
        self.open()
        kwargs = {"start": start_ns - 1}
        if stop_ns is not None:
            kwargs["stop"] = stop_ns
        for conn, stamp, raw in self._reader.messages(connections=self._connections(topic), **kwargs):
            if stamp < start_ns:
                continue
            if stop_ns is not None and stamp >= stop_ns:
                continue
            yield conn, stamp, raw

    def first_after(self, topic: str, t_ns: int):
        """(timestamp, deserialised message) for the first message at or after t_ns."""
        for conn, stamp, raw in self._messages(topic, t_ns):
            return stamp, self._reader.deserialize(raw, conn.msgtype)
        return None, None

    def nearest(self, topic: str, t_ns: int, window_ns: int = 200_000_000):
        """The message on `topic` closest to t_ns, searching a window either side."""
        best = (None, None)
        best_dt = None
        for conn, stamp, raw in self._messages(topic, t_ns - window_ns, t_ns + window_ns):
            dt = abs(stamp - t_ns)
            if best_dt is None or dt < best_dt:
                best_dt, best = dt, (stamp, self._reader.deserialize(raw, conn.msgtype))
        return best

    def range(self, topic: str, start_ns: int, stop_ns: int):
        """Yield (timestamp, message) over a time window -- used for accumulation."""
        for conn, stamp, raw in self._messages(topic, start_ns, stop_ns):
            yield stamp, self._reader.deserialize(raw, conn.msgtype)




def inspect(path: str | Path) -> BagInfo:
    """Read a bag's index and report what is inside. Does not decode messages."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with open_reader(path) as reader:
        start_ns, end_ns = reader.start_time, reader.end_time
        duration = (end_ns - start_ns) / 1e9

        topics = []
        for name, info in reader.topics.items():
            count = getattr(info, "msgcount", 0) or 0
            topics.append(
                TopicInfo(
                    name=name,
                    msgtype=getattr(info, "msgtype", "?"),
                    count=count,
                    hz=count / duration if duration > 0 else 0.0,
                )
            )

        return BagInfo(
            path=path,
            duration=duration,
            start=start_ns / 1e9,
            end=end_ns / 1e9,
            message_count=getattr(reader, "message_count", sum(t.count for t in topics)),
            topics=sorted(topics, key=lambda t: t.name),
        )
