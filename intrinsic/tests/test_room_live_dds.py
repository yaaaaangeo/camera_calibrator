"""
tests/test_room_live_dds.py
===============================

카메라·LiDAR 실시간 캡처의 DDS 백엔드(ROS 2 미설치 PC 용)가 큰 PointCloud2 를
cyclonedds 의 Python 역직렬화 없이(직렬화된 CDR 그대로) 받아 rosbags 로 푸는지 확인한다.

예전 경로는 수 MB 의 data 를 바이트마다 Python int 리스트로 풀어 GIL 을 독점했고,
구독 중에는 LiDAR 보드 탐색이 수 분씩 멈췄다. 같은 프로세스 안의 writer 로
보내므로 네트워크 없이 돈다.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cyclonedds.domain")

ROOM_ROOT = Path(__file__).resolve().parents[2] / "extrinsic"
if str(ROOM_ROOT) not in sys.path:
    sys.path.append(str(ROOM_ROOT))

from gui.core import live  # noqa: E402
from shared.ros.dds_types import Header_, PointCloud2_, PointField_, Time_  # noqa: E402


def _cloud(n: int) -> tuple[PointCloud2_, np.ndarray]:
    xyz = np.random.default_rng(0).normal(size=(n, 3)).astype("<f4")
    fields = [PointField_(name=k, offset=4 * i, datatype=7, count=1) for i, k in enumerate("xyz")]
    msg = PointCloud2_(header=Header_(stamp=Time_(sec=12, nanosec=34), frame_id="lidar"), height=1, width=n,
                       fields=fields, is_bigendian=False, point_step=12, row_step=12 * n,
                       data=list(xyz.tobytes()), is_dense=True)
    return msg, xyz


def test_dds_backend_receives_large_cloud_without_python_deserialization(monkeypatch):
    from cyclonedds.core import Policy, Qos
    from cyclonedds.pub import DataWriter
    from cyclonedds.topic import Topic
    from cyclonedds.util import duration

    domain = 100 + os.getpid() % 100
    backend = live.Ros2Dds(domain)
    got: list[live.Received] = []
    arrived = threading.Event()

    def on_msg(r):
        got.append(r)
        arrived.set()

    topic = live.LiveTopic("/test_points", "sensor_msgs/msg/PointCloud2", "dds", "", 1, reliable=True)
    try:
        msg, xyz = _cloud(200_000)  # 2.4 MB, like a 128-channel sweep
        # The slow path must not be needed: fail loudly if cyclonedds' deserializer runs.
        monkeypatch.setattr(PointCloud2_, "deserialize",
                            classmethod(lambda cls, *a, **k: pytest.fail("Python 역직렬화 경로를 탔습니다")))
        backend.subscribe(topic, on_msg)
        qos = Qos(Policy.Reliability.Reliable(duration(seconds=1)), Policy.History.KeepLast(4))
        writer = DataWriter(backend._dp, Topic(backend._dp, "rt/test_points", PointCloud2_), qos=qos)
        deadline = time.time() + 10
        while not arrived.is_set() and time.time() < deadline:
            writer.write(msg)
            arrived.wait(0.5)
    finally:
        backend.close()

    assert got, "DDS 로 보낸 PointCloud2 를 받지 못했습니다"
    r = got[0]
    assert r.fmt == "cdr"
    assert r.stamp_ns == 12 * 1_000_000_000 + 34
    assert r.msg.width == 200_000 and r.msg.header.frame_id == "lidar"
    np.testing.assert_array_equal(np.frombuffer(bytes(r.msg.data), "<f4").reshape(-1, 3), xyz)
