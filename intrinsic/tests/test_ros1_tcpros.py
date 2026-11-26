"""ROS 1 실시간 수신(shared.ros.live.Ros1Tcpros) - rospy 없이 프로토콜만으로.

가짜 master(XML-RPC)와 가짜 퍼블리셔(XML-RPC requestTopic + TCPROS)를 띄워
탐색, 수신, 늦게 뜬 퍼블리셔(publisherUpdate), 접속 실패 사유 보고를 확인한다.
실제 rospy/rosmaster 1.15(Noetic) 와의 접속은 수동으로 확인했다.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer

import numpy as np
import pytest

from shared.ros import live


def _serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/"


class FakePublisher:
    """CompressedImage 를 TCPROS 로 내보내는 최소 ROS 1 노드."""

    def __init__(self, topic: str, advertise_host: str = "127.0.0.1"):
        self.topic = topic
        self.tcp = socket.socket()
        self.tcp.bind(("127.0.0.1", 0))
        self.tcp.listen(4)
        self.host = advertise_host
        self.rpc = SimpleXMLRPCServer(("127.0.0.1", 0), logRequests=False, allow_none=True)
        self.rpc.register_function(self._request_topic, "requestTopic")
        self.uri = _serve(self.rpc)
        self.headers = []
        threading.Thread(target=self._accept, daemon=True).start()

    def _request_topic(self, caller, topic, protocols):
        return [1, "", ["TCPROS", self.host, self.tcp.getsockname()[1]]]

    def _accept(self):
        while True:
            try:
                conn, _ = self.tcp.accept()
            except OSError:
                return
            threading.Thread(target=self._send, args=(conn,), daemon=True).start()

    def _send(self, conn):
        (n,) = struct.unpack("<I", live._read_exact(conn, 4))
        body = live._read_exact(conn, n)
        self.headers.append(body)
        msgdef, md5 = live.TS1.generate_msgdef("sensor_msgs/msg/CompressedImage")
        conn.sendall(live._ros1_header({"callerid": "/fake", "topic": self.topic, "md5sum": md5,
                                        "type": "sensor_msgs/CompressedImage", "message_definition": msgdef}))
        T = live.TS1.types
        for i in range(50):
            msg = T["sensor_msgs/msg/CompressedImage"](
                T["std_msgs/msg/Header"](i, T["builtin_interfaces/msg/Time"](100 + i, 0), "cam"),
                "jpeg", np.frombuffer(b"\xff\xd8 fake", np.uint8))
            raw = live.TS1.serialize_ros1(msg, "sensor_msgs/msg/CompressedImage")
            try:
                conn.sendall(struct.pack("<I", len(raw)) + bytes(raw))
            except OSError:
                return
            time.sleep(0.02)
        conn.close()

    def close(self):
        try:
            self.tcp.shutdown(socket.SHUT_RDWR)  # wakes the blocked accept()
        except OSError:
            pass
        self.tcp.close()
        self.rpc.shutdown()
        self.rpc.server_close()


class FakeMaster:
    def __init__(self):
        self.pubs: dict[str, list[str]] = {}
        self.subs: dict[str, list[str]] = {}
        s = SimpleXMLRPCServer(("127.0.0.1", 0), logRequests=False, allow_none=True)
        s.register_function(lambda c: [1, "", [[[t, ["/fake"]] for t in self.pubs], [], []]], "getSystemState")
        s.register_function(lambda c: [1, "", [[t, "sensor_msgs/CompressedImage"] for t in self.pubs]],
                            "getTopicTypes")
        s.register_function(self._register, "registerSubscriber")
        s.register_function(lambda c, t, api: [1, "", 1], "unregisterSubscriber")
        self.server = s
        self.uri = _serve(s)

    def _register(self, caller, topic, typ, api):
        self.subs.setdefault(topic, []).append(api)
        return [1, "", list(self.pubs.get(topic, []))]

    def advertise(self, topic, pub_uri):
        self.pubs.setdefault(topic, []).append(pub_uri)
        for api in self.subs.get(topic, []):
            xmlrpc.client.ServerProxy(api).publisherUpdate("/master", topic, self.pubs[topic])

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def master():
    m = FakeMaster()
    yield m
    m.close()


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not pred():
        time.sleep(0.02)
    return pred()


def test_discover_and_receive(master):
    pub = FakePublisher("/cam/compressed")
    master.advertise("/cam/compressed", pub.uri)
    pr = live.probe(timeout=0.3, ros1_uri=master.uri, ros2_backend="off")
    b = pr.backends["ros1"]
    try:
        (t,) = [t for t in pr.topics if t.kind == "image"]
        assert t.msgtype == "sensor_msgs/msg/CompressedImage" and t.backend == "ros1"
        got = []
        b.subscribe(t, got.append)
        assert _wait(lambda: len(got) >= 5)
        r = got[0]
        assert r.fmt == "ros1" and r.msg.format == "jpeg" and r.stamp_ns >= 100 * 10**9
        assert b"md5sum=*" in pub.headers[0]
        assert b.problems() == {}
    finally:
        b.close()
        pub.close()


def test_publisher_started_after_subscribe(master):
    b = live.Ros1Tcpros(master.uri)
    pub = FakePublisher("/late")
    try:
        got = []
        b.subscribe(live.LiveTopic("/late", "sensor_msgs/msg/CompressedImage", "ros1", "ROS 1"), got.append)
        assert got == []
        master.advertise("/late", pub.uri)  # master -> publisherUpdate -> connect
        assert _wait(lambda: len(got) >= 3)
    finally:
        b.close()
        pub.close()


def test_unreachable_publisher_is_reported(master):
    pub = FakePublisher("/far", advertise_host="robot-pc.invalid")
    master.advertise("/far", pub.uri)
    b = live.Ros1Tcpros(master.uri)
    errors = []
    try:
        b.subscribe(live.LiveTopic("/far", "sensor_msgs/msg/CompressedImage", "ros1", "ROS 1"),
                    lambda r: None, on_error=errors.append)
        assert _wait(lambda: errors, timeout=8)
        assert "robot-pc.invalid" in errors[0] and "ROS_IP" in errors[0]
        assert "/far" in b.problems()
    finally:
        b.close()
        pub.close()


def test_failed_probe_does_not_leak_server():
    before = threading.active_count()
    for _ in range(3):
        pr = live.probe(timeout=0.2, ros1_uri="http://127.0.0.1:1", ros2_backend="off")
        assert "ros1" not in pr.backends
    assert _wait(lambda: threading.active_count() <= before, timeout=2)


def test_callback_host_prefers_env(monkeypatch):
    monkeypatch.setenv("ROS_IP", "10.1.2.3")
    assert live._callback_host("http://192.168.0.10:11311") == "10.1.2.3"
    monkeypatch.delenv("ROS_IP")
    monkeypatch.delenv("ROS_HOSTNAME", raising=False)
    assert live._callback_host("http://localhost:11311") == "localhost"
