"""
tests/test_ros_live.py
======================

Camera Intrinsic 의 실시간 구독(calibration.ros_live)은 CAMERA–LiDAR 와 같은 공용 수신
백엔드(shared.ros.live) 위에 있다. 여기서는 ROS 없이 확인할 수 있는 것을 본다:
import·생성이 ROS 설치와 무관한지, 탐색 결과에서 영상 토픽만 고르는지, 받은 메시지를
BGR 로 풀어 넘기는지, 디코딩 실패 알림을 3초에 한 번으로 줄이는지.
(실제 ROS 1 / ROS 2 / DDS 수신은 tests/test_room_live_dds.py 와 실제 bag 재생으로 검증)
"""

from __future__ import annotations

import numpy as np
import pytest

import calibration.ros_live as ros_live
from shared.ros import live as shared_live


class _FakeBackend:
    def __init__(self):
        self.subscribed, self.closed = [], False

    def subscribe(self, topic, callback):
        self.subscribed.append((topic, callback))

    def close(self):
        self.closed = True


def _fake_probe(monkeypatch, topics):
    backend = _FakeBackend()
    probe = shared_live.Probe(topics=topics, backends={"dds": backend, "ros1": backend}, notes=["fake"])
    monkeypatch.setattr(shared_live, "probe", lambda **kw: probe)
    return backend


def _image_msg(h=4, w=6, encoding="bgr8"):
    class Header:
        class stamp:  # noqa: N801
            sec, nanosec = 12, 500_000_000
    class Msg:
        header = Header
    m = Msg()
    m.height, m.width, m.encoding, m.step, m.is_bigendian = h, w, encoding, w * 3, 0
    m.data = np.full(h * w * 3, 7, np.uint8)
    return m


def test_import_and_construction_never_need_ros():
    assert ros_live.ROS_LIVE_BACKEND == "auto"
    ros_live._require_backend()  # 호환용, 예외 없음
    sub = ros_live.LiveTopicSubscriber()
    assert sub.notes == []
    assert "ROS 1" in ros_live.backend_status_text()


def test_list_image_topics_keeps_only_images_with_their_source(monkeypatch):
    topics = [shared_live.LiveTopic("/cam/image_raw", "sensor_msgs/msg/Image", "dds", "ROS 2"),
              shared_live.LiveTopic("/points", "sensor_msgs/msg/PointCloud2", "dds", "ROS 2"),
              shared_live.LiveTopic("/cam2/compressed", "sensor_msgs/msg/CompressedImage", "ros1", "ROS 1")]
    _fake_probe(monkeypatch, topics)
    found = ros_live.LiveTopicSubscriber().list_image_topics()
    assert [(t.name, t.source) for t in found] == [("/cam/image_raw", "ROS 2 (DDS)"), ("/cam2/compressed", "ROS 1")]


def test_start_decodes_frames_and_stop_closes_backends(monkeypatch):
    topic = shared_live.LiveTopic("/cam/image_raw", "sensor_msgs/msg/Image", "dds", "ROS 2")
    backend = _fake_probe(monkeypatch, [topic])
    sub = ros_live.LiveTopicSubscriber()
    frames = []
    sub.start("/cam/image_raw", "sensor_msgs/Image", on_frame=lambda img, t: frames.append((img.shape, t)))
    (chosen, callback), = backend.subscribed
    assert chosen is topic
    callback(shared_live.Received(12_500_000_000, 0, _image_msg(), b"", "cdr"))
    assert frames == [((4, 6, 3), 12.5)]
    with pytest.raises(RuntimeError, match="이미 구독"):
        sub.start("/cam/image_raw", "", on_frame=lambda *a: None)
    sub.stop()
    assert backend.closed


def test_decode_errors_are_rate_limited(monkeypatch):
    topic = shared_live.LiveTopic("/cam", "sensor_msgs/msg/Image", "dds", "ROS 2")
    backend = _fake_probe(monkeypatch, [topic])
    errors = []
    sub = ros_live.LiveTopicSubscriber()
    sub.start("/cam", "sensor_msgs/msg/Image", on_frame=lambda *a: None, on_error=errors.append)
    (_t, callback), = backend.subscribed
    bad = _image_msg(encoding="weird16")
    for _ in range(5):
        callback(shared_live.Received(0, 0, bad, b"", "cdr"))
    assert len(errors) == 1 and "weird16" in errors[0]


def test_missing_topic_reports_what_was_found(monkeypatch):
    _fake_probe(monkeypatch, [])
    with pytest.raises(RuntimeError, match="토픽을 찾지 못했습니다"):
        ros_live.LiveTopicSubscriber().start("/nope", "sensor_msgs/msg/Image", on_frame=lambda *a: None)


def test_live_dialog_has_no_separate_automatic_calibration_button(tmp_path, pattern_config):
    # 실시간 자동 캡처 → 완료(불러오기) → 캘리브레이션 실행 한 가지 길만 둔다.
    # 명령줄 자동화(./cli.sh automation live)는 그대로다.
    from PySide6.QtWidgets import QApplication, QPushButton

    import ui.live_capture_dialog as live_dialog

    QApplication.instance() or QApplication([])
    original = live_dialog.ROS_LIVE_BACKEND
    live_dialog.ROS_LIVE_BACKEND = None
    try:
        dialog = live_dialog.LiveCaptureDialog(str(tmp_path), pattern_config)
        texts = [b.text() for b in dialog.findChildren(QPushButton)]
        assert not any("P2" in t or "자동 캘리브레이션" in t for t in texts)
        assert dialog.capture_button is not None and dialog.auto_capture_check.isChecked()
        assert "완료 (불러오기)" in texts
        dialog.close()
    finally:
        live_dialog.ROS_LIVE_BACKEND = original
