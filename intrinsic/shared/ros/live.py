"""
shared.ros.live
===============

Live topics, from whichever ROS is on the network -- or none installed here.
Camera Intrinsic 의 실시간 캡처와 CAMERA–LiDAR 의 실시간 캡처가 같이 쓴다.
Live topics, from whichever ROS is on the network -- or none installed here.

The calibrator only needs three message types (PointCloud2, Image,
CompressedImage), and all three have the same layout in every ROS from Noetic
to Jazzy. So it does not have to care which ROS publishes them, only how to get
the bytes:

    ROS 1   "ros1"   a master answers on ROS_MASTER_URI. Spoken directly: XML-RPC
                     to the master, then TCPROS to each publisher. No rospy.
    ROS 2   "rclpy"  rclpy is importable (a ROS 2 install is sourced). Most
                     faithful: same middleware, QoS and discovery as the robot.
            "dds"    no ROS 2 here. A ROS 2 topic is plain DDS -- `rt/` name,
                     `pkg::msg::dds_::Name_` type, CDR payload -- so cyclonedds
                     reads it given the struct layout (shared/ros/dds_types.py).

Every backend hands on the same thing: a `rosbags` message object, exactly what
reading a bag gives, plus the serialized bytes so a snapshot can be written to
mcap and replayed offline. Nothing downstream knows the data was live.

What can and cannot be told about the publisher's ROS: ROS 1 versus ROS 2 is
certain. Within ROS 2, Iron and later (Jazzy included) advertise a type hash
in discovery and Humble and earlier do not, so "Iron or later" is decidable and
the exact distribution is not. For these message types it does not matter.

Large clouds over DDS: a 128-channel sweep is several megabytes, sent as many
UDP fragments, and the kernel drops what does not fit its receive buffer. The
stock net.core.rmem_max (208 KiB) is far too small for that between machines.
`socket_buffer_warning` says so rather than leaving it as silently missing sweeps.
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time
import xmlrpc.client
from dataclasses import dataclass, field
from xmlrpc.server import SimpleXMLRPCServer

import numpy as np
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

TS2 = get_typestore(Stores.ROS2_HUMBLE)  # CDR, ROS 2 names
TS1 = get_typestore(Stores.ROS1_NOETIC)  # ROS 1 wire format

KINDS = {
    "sensor_msgs/msg/PointCloud2": "lidar",
    "sensor_msgs/msg/Image": "image",
    "sensor_msgs/msg/CompressedImage": "image",
}


def ros1_name(msgtype: str) -> str:
    """'sensor_msgs/msg/PointCloud2' -> 'sensor_msgs/PointCloud2'."""
    return msgtype.replace("/msg/", "/")


def ros2_name(msgtype: str) -> str:
    """'sensor_msgs/PointCloud2' or 'sensor_msgs::msg::dds_::PointCloud2_' -> 'sensor_msgs/msg/PointCloud2'."""
    if "::" in msgtype:
        parts = msgtype.split("::")
        return f"{parts[0]}/msg/{parts[-1].rstrip('_')}"
    if "/msg/" in msgtype:
        return msgtype
    pkg, _, name = msgtype.partition("/")
    return f"{pkg}/msg/{name}"


@dataclass
class LiveTopic:
    name: str
    msgtype: str  # ROS 2 spelling, as rosbags uses
    backend: str  # "ros1" | "rclpy" | "dds"
    flavor: str  # what can honestly be said about the publisher's ROS
    publishers: int = 0
    # Every publisher offers reliable delivery. Then the subscription asks for it
    # too: a multi-megabyte cloud arrives as hundreds of UDP fragments, and with
    # a small kernel receive buffer some are always dropped -- best effort throws
    # the whole sweep away, reliable has the missing fragments sent again.
    # Measured here: 0 clouds received best-effort, all of them reliable.
    reliable: bool = False

    @property
    def kind(self) -> str:
        return KINDS.get(self.msgtype, "")

    @property
    def key(self) -> str:
        return f"{self.backend}:{self.name}"


@dataclass
class Received:
    """One message: decoded, with its serialized form kept for recording."""

    stamp_ns: int  # header stamp
    recv_ns: int  # wall clock on arrival
    msg: object  # rosbags message object
    raw: bytes  # serialized
    fmt: str  # "cdr" | "ros1"


def _stamp_ns(msg) -> int:
    h = getattr(msg, "header", None)
    if h is None:
        return time.time_ns()
    return int(h.stamp.sec) * 1_000_000_000 + int(getattr(h.stamp, "nanosec", getattr(h.stamp, "nsec", 0)))


# ================================================================== backends


class Backend:
    name = ""

    def discover(self, timeout: float) -> list[LiveTopic]:
        raise NotImplementedError

    def subscribe(self, topic: LiveTopic, callback, on_error=None) -> None:
        """`callback(Received)` from a background thread.

        `on_error(text)`, if given, hears why a subscribed topic is not
        delivering (a publisher that cannot be reached, say) -- otherwise that
        looks the same as a topic nobody publishes on.
        """
        raise NotImplementedError

    def problems(self) -> dict[str, str]:
        """Topic name -> why it is not delivering right now. Empty when all is well."""
        return {}

    def close(self) -> None:
        pass


# ------------------------------------------------------------------ ROS 2: rclpy


def rclpy_available() -> bool:
    try:
        import rclpy  # noqa: F401
        from rosidl_runtime_py.utilities import get_message  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


class Ros2Rclpy(Backend):
    name = "rclpy"

    def __init__(self, domain_id: int | None = None):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor

        if domain_id is not None:
            os.environ["ROS_DOMAIN_ID"] = str(domain_id)
        self._ctx = Context()
        rclpy.init(context=self._ctx)
        self._node = rclpy.create_node(f"room_calib_live_{os.getpid()}", context=self._ctx)
        self._exec = SingleThreadedExecutor(context=self._ctx)
        self._exec.add_node(self._node)
        self._run = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        self._subs = []

    def _spin(self):
        while self._run and self._ctx.ok():
            try:
                self._exec.spin_once(timeout_sec=0.1)
            except Exception:  # noqa: BLE001 - shutting down
                break

    def discover(self, timeout: float) -> list[LiveTopic]:
        time.sleep(timeout)
        out = []
        for name, types in self._node.get_topic_names_and_types():
            for t in types:
                infos = self._node.get_publishers_info_by_topic(name)
                if infos:
                    from rclpy.qos import ReliabilityPolicy

                    rel = all(i.qos_profile.reliability == ReliabilityPolicy.RELIABLE for i in infos)
                    out.append(LiveTopic(name, ros2_name(t), self.name, "ROS 2", len(infos), rel))
        return out

    def subscribe(self, topic: LiveTopic, callback, on_error=None) -> None:
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from rosidl_runtime_py.utilities import get_message

        cls = get_message(topic.msgtype)
        mt = topic.msgtype

        def on_raw(raw: bytes):
            msg = TS2.deserialize_cdr(raw, mt)
            callback(Received(_stamp_ns(msg), time.time_ns(), msg, bytes(raw), "cdr"))

        qos = (QoSProfile(depth=4, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE)
               if topic.reliable else qos_profile_sensor_data)
        self._subs.append(self._node.create_subscription(cls, topic.name, on_raw, qos, raw=True))

    def close(self) -> None:
        import rclpy

        self._run = False
        try:
            for s in self._subs:
                self._node.destroy_subscription(s)
            self._node.destroy_node()
            rclpy.shutdown(context=self._ctx)
        except Exception:  # noqa: BLE001
            pass
        self._thread.join(timeout=2)


# ------------------------------------------------------------- ROS 2: plain DDS


def dds_available() -> bool:
    try:
        import cyclonedds.domain  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


class Ros2Dds(Backend):
    """ROS 2 topics with no ROS 2 installed: cyclonedds and the struct layouts."""

    name = "dds"

    def __init__(self, domain_id: int | None = None):
        from cyclonedds.domain import DomainParticipant

        if domain_id is None:
            domain_id = int(os.environ.get("ROS_DOMAIN_ID", "0") or 0)
        self.domain_id = domain_id
        self._dp = DomainParticipant(domain_id)
        self._run = True
        self._threads = []

    def discover(self, timeout: float) -> list[LiveTopic]:
        from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication

        reader = BuiltinDataReader(self._dp, BuiltinTopicDcpsPublication)
        time.sleep(timeout)
        seen: dict[tuple[str, str], list] = {}
        for s in reader.take(N=10_000):
            if not s.topic_name.startswith("rt/"):
                continue
            key = (s.topic_name[2:], ros2_name(s.type_name))
            seen.setdefault(key, []).append(s)
        out = []
        for (name, mt), samples in seen.items():
            hashed = any(_has_type_hash(s) for s in samples)
            flavor = "ROS 2 (Iron 이상, type hash 있음)" if hashed else "ROS 2 (Humble 이하)"
            rel = all(_is_reliable(s) for s in samples)
            out.append(LiveTopic(name, mt, self.name, flavor, len(samples), rel))
        return out

    def subscribe(self, topic: LiveTopic, callback, on_error=None) -> None:
        from cyclonedds.core import Qos, Policy
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic
        from cyclonedds.util import duration

        from shared.ros.dds_types import TYPES, to_cdr

        if topic.msgtype not in TYPES:
            raise ValueError(f"DDS 백엔드가 모르는 타입: {topic.msgtype}")
        cls, fast = TYPES[topic.msgtype]
        rel = (Policy.Reliability.Reliable(duration(seconds=1)) if topic.reliable
               else Policy.Reliability.BestEffort)
        qos = Qos(rel, Policy.History.KeepLast(4))
        reader = DataReader(self._dp, Topic(self._dp, "rt" + topic.name, cls), qos=qos)
        mt = topic.msgtype
        take_raw = _raw_taker(reader)

        def decoded(sample):
            raw = to_cdr(sample, fast)
            return raw, TS2.deserialize_cdr(raw, mt)

        def loop_raw():
            from cyclonedds.core import InstanceState, ReadCondition, SampleState, ViewState, WaitSet

            waitset = WaitSet(self._dp)
            waitset.attach(ReadCondition(reader, ViewState.Any | InstanceState.Any | SampleState.Any))
            while self._run:
                raws = take_raw()
                if not raws:
                    waitset.wait(duration(milliseconds=200))
                    continue
                for raw in raws:
                    if not self._run:
                        break
                    if raw[:2] in _PLAIN_CDR:
                        msg = TS2.deserialize_cdr(raw, mt)
                    else:  # XCDR2 or unknown: let cyclonedds decode it, then re-encode
                        raw, msg = decoded(cls.deserialize(raw))
                    callback(Received(_stamp_ns(msg), time.time_ns(), msg, raw, "cdr"))

        def loop():
            if take_raw is not None:
                try:
                    return loop_raw()
                except (TypeError, AttributeError):  # the private call changed shape: slow path
                    pass
            while self._run:
                for sample in reader.take_iter(timeout=duration(milliseconds=200)):
                    if not self._run:
                        break
                    raw, msg = decoded(sample)
                    callback(Received(_stamp_ns(msg), time.time_ns(), msg, raw, "cdr"))

        th = threading.Thread(target=loop, daemon=True)
        th.start()
        self._threads.append(th)

    def close(self) -> None:
        self._run = False
        for th in self._threads:
            th.join(timeout=1)


# Encapsulation ids of classic (XCDR1) CDR, big and little endian -- what ROS 2
# writers send and what `rosbags` reads.
_PLAIN_CDR = (b"\x00\x00", b"\x00\x01")


def _raw_taker(reader):
    """A function returning the reader's waiting samples as serialized CDR, or None.

    `DataReader.take` would hand each sample through cyclonedds' Python
    deserializer, which turns `data` -- megabytes of a cloud or an image -- into
    a list of Python ints, one per byte. At sensor rates that holds the GIL so
    much that nothing else in the process gets to run (a LiDAR board search took
    minutes instead of seconds while subscribed). The serialized bytes are what
    `rosbags` decodes anyway, so take them as they are.

    `ddspy_take` is the C call `take` itself is built on, not a public API, so
    anything unexpected falls back to the slow path.
    """
    try:
        from cyclonedds._clayer import ddspy_take
        from cyclonedds.core import DDSException, InstanceState, SampleState, ViewState
    except ImportError:
        return None
    mask = SampleState.Any | ViewState.Any | InstanceState.Any

    def take(n: int = 8) -> list[bytes]:
        ret = ddspy_take(reader._ref, mask, n)
        if isinstance(ret, int):
            raise DDSException(ret, "take")
        return [bytes(data) for data, info in ret if info.valid_data]

    return take


def _is_reliable(sample) -> bool:
    try:
        from cyclonedds.core import Policy

        return isinstance(sample.qos[Policy.Reliability], Policy.Reliability.Reliable)
    except Exception:  # noqa: BLE001
        return False


def _has_type_hash(sample) -> bool:
    """Iron and later put `typehash=RIHS01_...` in each endpoint's user data."""
    try:
        from cyclonedds.core import Policy

        ud = sample.qos[Policy.Userdata]
        return ud is not None and b"typehash=" in bytes(ud.data)
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------- ROS 1: TCPROS


class _TimeoutTransport(xmlrpc.client.Transport):
    def __init__(self, timeout: float):
        super().__init__()
        self._timeout = timeout

    def make_connection(self, host):
        conn = super().make_connection(host)
        conn.timeout = self._timeout
        return conn


def _proxy(uri: str, timeout: float = 2.0):
    return xmlrpc.client.ServerProxy(uri, transport=_TimeoutTransport(timeout), allow_none=True)


def _ros1_header(fields: dict[str, str]) -> bytes:
    body = b"".join(struct.pack("<I", len(b)) + b for b in (f"{k}={v}".encode() for k, v in fields.items()))
    return struct.pack("<I", len(body)) + body


def _read_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("publisher closed the connection")
        buf += chunk
    return bytes(buf)


def _read_header(sock) -> dict[str, str]:
    (n,) = struct.unpack("<I", _read_exact(sock, 4))
    body = _read_exact(sock, n)
    out, i = {}, 0
    while i < len(body):
        (m,) = struct.unpack("<I", body[i:i + 4])
        k, _, v = body[i + 4:i + 4 + m].decode(errors="replace").partition("=")
        out[k] = v
        i += 4 + m
    return out


def ros1_master_uri() -> str:
    return os.environ.get("ROS_MASTER_URI", "http://localhost:11311")


def _callback_host(master_uri: str) -> str:
    """The address a ROS 1 master and publishers can reach this machine at.

    ROS_IP / ROS_HOSTNAME win, as in rospy. Otherwise rospy would use the
    hostname, which another machine often cannot resolve (no DNS on a vehicle
    network, no /etc/hosts entry); then publisherUpdate never arrives and a
    publisher that starts after the subscription is never heard. The address
    of the interface that routes to the master is reachable by construction.
    """
    host = os.environ.get("ROS_IP") or os.environ.get("ROS_HOSTNAME")
    if host:
        return host
    from urllib.parse import urlparse

    master_host = urlparse(master_uri).hostname or "localhost"
    if master_host in ("localhost", "127.0.0.1", "::1"):
        return "localhost" if master_host == "localhost" else master_host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((master_host, 11311))  # UDP: no packet sent, only routing
            return s.getsockname()[0]
    except OSError:
        return socket.gethostname()


def _explain(exc: Exception, pub_uri: str) -> str:
    """One line on why a ROS 1 publisher could not be read, with the usual cause."""
    from urllib.parse import urlparse

    host = getattr(exc, "ros_host", None) or urlparse(pub_uri).hostname or pub_uri
    if isinstance(exc, socket.gaierror):
        return (f"퍼블리셔 주소 {host} 를 이 PC 가 찾지 못합니다 — 로봇 PC 에서 ROS_IP 를 설정하거나 "
                f"이 PC 의 /etc/hosts 에 '{host}' 를 추가하세요")
    if isinstance(exc, (ConnectionRefusedError, TimeoutError, socket.timeout)) or (
            isinstance(exc, OSError) and exc.errno in (101, 110, 111, 113)):
        return f"퍼블리셔 {host} 에 접속하지 못했습니다 ({type(exc).__name__}) — 방화벽·네트워크·ROS_IP 확인"
    return f"퍼블리셔 {pub_uri}: {type(exc).__name__}: {exc}"


class Ros1Tcpros(Backend):
    """A ROS 1 subscriber written against the protocol, so no rospy is needed.

    Subscribes with md5sum `*`, which a publisher accepts from any subscriber,
    and takes the message definition from the publisher's own connection
    header -- so a type this machine has never seen still decodes.

    Checked against rospy/rosmaster 1.15 (Noetic): discovery, Image,
    CompressedImage and PointCloud2, publishers starting late, and recording.
    """

    name = "ros1"
    RETRY_S = 2.0

    def __init__(self, master_uri: str | None = None):
        self.master_uri = master_uri or ros1_master_uri()
        self.caller_id = f"/room_calib_live_{os.getpid()}_{id(self) & 0xFFFF:x}"
        self._server = SimpleXMLRPCServer(("0.0.0.0", 0), logRequests=False, allow_none=True)
        self._server.register_function(self._publisher_update, "publisherUpdate")
        self._server.register_function(lambda caller: [1, "", os.getpid()], "getPid")
        self._server.register_function(lambda caller, msg="": [1, "", 0], "shutdown")
        self._server.register_function(lambda caller: [1, "", self.master_uri], "getMasterUri")
        self._server.register_function(lambda caller: [1, "", [[n, ros1_name(t[0].msgtype)]
                                                               for n, t in self._subs.items()]],
                                       "getSubscriptions")
        self._server.register_function(lambda caller: [1, "", []], "getPublications")
        self.uri = f"http://{_callback_host(self.master_uri)}:{self._server.server_address[1]}/"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self._lock = threading.Lock()
        self._subs: dict[str, tuple[LiveTopic, object, object]] = {}
        self._pubs: dict[str, set[str]] = {}  # topic -> publisher URIs the master lists now
        self._connected: set[tuple[str, str]] = set()
        self._problems: dict[str, str] = {}
        self._socks = []
        self._run = True

    def master(self, timeout: float = 2.0):
        return _proxy(self.master_uri, timeout)

    def discover(self, timeout: float) -> list[LiveTopic]:
        m = self.master(max(timeout, 0.5))
        code, msg, state = m.getSystemState(self.caller_id)
        if code != 1:
            raise RuntimeError(f"master getSystemState: {msg}")
        _, _, types = m.getTopicTypes(self.caller_id)
        tmap = dict(types)
        pubs = {t: len(nodes) for t, nodes in state[0]}
        return [LiveTopic(t, ros2_name(tmap.get(t, "")), self.name, "ROS 1", n)
                for t, n in pubs.items() if n and t in tmap]

    def subscribe(self, topic: LiveTopic, callback, on_error=None) -> None:
        code, msg, pubs = self.master().registerSubscriber(
            self.caller_id, topic.name, ros1_name(topic.msgtype), self.uri)
        if code != 1:
            raise RuntimeError(f"master 가 구독을 거부했습니다: {msg}")
        with self._lock:
            self._subs[topic.name] = (topic, callback, on_error)
        self._set_publishers(topic.name, pubs)

    def problems(self) -> dict[str, str]:
        with self._lock:
            return dict(self._problems)

    def _publisher_update(self, caller_id, topic, publishers):
        self._set_publishers(topic, publishers)
        return [1, "", 0]

    def _set_publishers(self, topic_name: str, publishers):
        with self._lock:
            if topic_name not in self._subs:
                return
            self._pubs[topic_name] = set(publishers)
            if not publishers:
                self._problems[topic_name] = "퍼블리셔가 없습니다 (노드가 종료됨?)"
        for p in publishers:
            self._connect(topic_name, p)

    def _connect(self, topic_name: str, pub_uri: str):
        key = (topic_name, pub_uri)
        with self._lock:
            if key in self._connected or topic_name not in self._subs:
                return
            self._connected.add(key)
        threading.Thread(target=self._stream, args=(topic_name, pub_uri), daemon=True).start()

    def _report(self, topic_name: str, text: str):
        with self._lock:
            self._problems[topic_name] = text
            on_error = self._subs.get(topic_name, (None, None, None))[2]
        if on_error is not None:
            try:
                on_error(text)
            except Exception:  # noqa: BLE001
                pass

    def _stream(self, topic_name: str, pub_uri: str):
        """Read one publisher until it goes away; while the master still lists it, retry."""
        topic, callback, _ = self._subs[topic_name]
        try:
            while self._run:
                sock = None
                try:
                    sock = self._open(topic, pub_uri)
                    hdr = _read_header(sock)
                    if "error" in hdr:
                        raise RuntimeError(f"퍼블리셔가 거부: {hdr['error']}")
                    ts, mt = _ros1_typestore(hdr, topic.msgtype)
                    first = True
                    while self._run:
                        (n,) = struct.unpack("<I", _read_exact(sock, 4))
                        raw = _read_exact(sock, n)
                        msg = ts.deserialize_ros1(raw, mt)
                        if first:
                            first = False
                            with self._lock:
                                self._problems.pop(topic_name, None)
                        callback(Received(_stamp_ns(msg), time.time_ns(), msg, raw, "ros1"))
                except Exception as e:  # noqa: BLE001 - one publisher dropping must not end the others
                    if not self._run:
                        return
                    if not (isinstance(e, ConnectionError) and "closed" in str(e)):
                        self._report(topic_name, _explain(e, pub_uri))
                finally:
                    if sock is not None:
                        try:
                            sock.close()
                        except OSError:
                            pass
                        with self._lock:
                            if sock in self._socks:
                                self._socks.remove(sock)
                # Publisher gone for good (master no longer lists it): stop.
                # Otherwise it restarted, or was not reachable yet: try again.
                deadline = time.monotonic() + self.RETRY_S
                while self._run and time.monotonic() < deadline:
                    time.sleep(0.1)
                with self._lock:
                    if pub_uri not in self._pubs.get(topic_name, ()):
                        return
        finally:
            with self._lock:
                self._connected.discard((topic_name, pub_uri))

    def _open(self, topic: LiveTopic, pub_uri: str):
        code, msg, proto = _proxy(pub_uri).requestTopic(self.caller_id, topic.name, [["TCPROS"]])
        if code != 1 or not proto:
            raise RuntimeError(f"requestTopic 실패: {msg}")
        _, host, port = proto[:3]
        try:
            sock = socket.create_connection((host, int(port)), timeout=5)
        except OSError as e:
            e.ros_host = host  # the data address can differ from the XML-RPC one
            raise
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # A multi-megabyte sweep arrives in one go; a larger buffer keeps the
            # publisher from blocking on us while the previous one is decoded.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        except OSError:
            pass
        with self._lock:
            self._socks.append(sock)
        sock.sendall(_ros1_header({
            "callerid": self.caller_id, "topic": topic.name, "type": ros1_name(topic.msgtype),
            "md5sum": "*", "message_definition": "", "tcp_nodelay": "1",
        }))
        sock.settimeout(None)
        return sock

    def close(self) -> None:
        self._run = False
        for name in list(self._subs):
            try:
                self.master(1.0).unregisterSubscriber(self.caller_id, name, self.uri)
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            socks = list(self._socks)
        for s in socks:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass
        self._server.shutdown()
        self._server.server_close()


def _ros1_typestore(header: dict[str, str], msgtype: str):
    """The stock Noetic store, or one extended from the publisher's own definition."""
    pub_type = ros2_name(header.get("type", ros1_name(msgtype)))
    try:
        if TS1.generate_msgdef(pub_type)[1] == header.get("md5sum"):
            return TS1, pub_type
    except Exception:  # noqa: BLE001 - not in the stock store
        pass
    ts = get_typestore(Stores.EMPTY)
    ts.register(get_types_from_msg(header.get("message_definition", ""), pub_type))
    return ts, pub_type


# ============================================================ auto-detection


@dataclass
class Probe:
    topics: list[LiveTopic] = field(default_factory=list)
    backends: dict[str, Backend] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def probe(timeout: float = 2.0, ros1_uri: str | None = None, domain_id: int | None = None,
          ros2_backend: str = "auto") -> Probe:
    """Look for ROS 1 and ROS 2 at once and keep whichever answer open.

    `ros2_backend` is "auto" (rclpy if it imports, else plain DDS), "rclpy",
    "dds" or "off". When rclpy is used, a DDS participant still listens briefly
    if cyclonedds is present, only to label Humble-or-earlier versus Iron-or-later.
    """
    out = Probe()
    lock = threading.Lock()

    def ros1():
        b = None
        try:
            b = Ros1Tcpros(ros1_uri)
            topics = b.discover(timeout)
            with lock:
                out.backends["ros1"] = b
                out.topics += topics
                out.notes.append(f"ROS 1 master 응답: {b.master_uri} (토픽 {len(topics)}개, 이 PC 주소 {b.uri})")
        except (OSError, xmlrpc.client.Error, RuntimeError) as e:
            if b is not None:
                b.close()
            with lock:
                out.notes.append(f"ROS 1 없음 ({ros1_uri or ros1_master_uri()}: {type(e).__name__})")

    def ros2():
        kind = ros2_backend
        if kind == "off":
            return
        if kind == "auto":
            kind = "rclpy" if rclpy_available() else ("dds" if dds_available() else "")
        if not kind:
            with lock:
                out.notes.append("ROS 2 수단 없음 (rclpy 도 cyclonedds 도 import 불가)")
            return
        try:
            b = Ros2Rclpy(domain_id) if kind == "rclpy" else Ros2Dds(domain_id)
            topics = b.discover(timeout)
            labels = {}
            if kind == "rclpy" and dds_available():
                try:
                    d = Ros2Dds(domain_id)
                    labels = {t.name: t.flavor for t in d.discover(min(timeout, 1.5))}
                    d.close()
                except Exception:  # noqa: BLE001
                    pass
            for t in topics:
                t.flavor = labels.get(t.name, t.flavor)
            with lock:
                out.backends[kind] = b
                out.topics += topics
                out.notes.append(f"ROS 2 ({kind}, domain {os.environ.get('ROS_DOMAIN_ID', '0') if domain_id is None else domain_id}): 토픽 {len(topics)}개")
        except Exception as e:  # noqa: BLE001
            with lock:
                out.notes.append(f"ROS 2 ({kind}) 실패: {type(e).__name__}: {e}")

    ths = [threading.Thread(target=f, daemon=True) for f in (ros1, ros2)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(timeout + 5)
    warn = socket_buffer_warning()
    if warn and any(t.backend in ("rclpy", "dds") for t in out.topics):
        out.notes.append(warn)
    out.topics.sort(key=lambda t: (t.kind == "", t.kind, t.name))
    return out


def socket_buffer_warning(need: int = 8 << 20) -> str:
    try:
        with open("/proc/sys/net/core/rmem_max") as f:
            have = int(f.read().strip())
    except OSError:
        return ""
    if have >= need:
        return ""
    return (f"UDP 수신 버퍼 net.core.rmem_max = {have // 1024} KiB — 다른 PC 에서 오는 큰 클라우드는 "
            f"조각이 버려질 수 있습니다. 필요하면: sudo sysctl -w net.core.rmem_max={need}")
