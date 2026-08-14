"""
tests/test_rosbag_and_live_fixes.py
========================================

실제 사용자가 보고한 두 가지 문제의 회귀 테스트:

1. rosbag 읽기 실패: "Bag contains no type definitions. Instantiate
   AnyReader with a default_typestore argument." - AnyReader에
   default_typestore를 안 넘겨서 발생.

2. 실시간 구독이 "계속 프레임 수신 대기 중"에서 멈춤 - 카메라가 지원 안 하는
   인코딩(yuv422 등)으로 발행하면 프레임이 도착해도 조용히 버려져서
   사용자는 "아무것도 안 온다"고 착각하게 된다.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np


def test_any_reader_called_with_default_typestore():
    """rosbag_reader.py의 핵심 수정 - AnyReader() 생성 시 default_typestore를
    반드시 넘겨야 한다.
    """
    import calibration.rosbag_reader as rr

    with patch.object(rr, "AnyReader") as mock_reader_cls, patch.object(rr, "get_typestore") as mock_get_ts:
        mock_reader_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(connections=[]))
        mock_reader_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_ts.return_value = "FAKE_TYPESTORE"

        rr.list_image_topics("/fake/path")

        assert mock_reader_cls.called
        _, kwargs = mock_reader_cls.call_args
        assert "default_typestore" in kwargs
        assert kwargs["default_typestore"] == "FAKE_TYPESTORE"


def test_any_reader_default_typestore_used_by_extract_too():
    import calibration.rosbag_reader as rr

    with patch.object(rr, "AnyReader") as mock_reader_cls, patch.object(rr, "get_typestore") as mock_get_ts:
        fake_reader = MagicMock()
        fake_reader.connections = [MagicMock(topic="/cam", msgtype="sensor_msgs/msg/Image")]
        fake_reader.messages.return_value = []
        mock_reader_cls.return_value.__enter__ = MagicMock(return_value=fake_reader)
        mock_reader_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_ts.return_value = "FAKE_TYPESTORE"

        rr.extract_images_from_bag("/fake/path", "/cam", "/tmp/out_fake")

        _, kwargs = mock_reader_cls.call_args
        assert kwargs.get("default_typestore") == "FAKE_TYPESTORE"


def test_real_ros1_bag_still_works_with_default_typestore(tmp_path):
    """default_typestore를 추가해도 원래 타입 정의가 내장된 정상 ROS1 bag은
    여전히 잘 열려야 한다 - 회귀 방지.
    """
    import pytest
    pytest.importorskip("rosbags")

    from rosbags.rosbag1 import Writer
    from rosbags.typesys import get_typestore, Stores
    from rosbags.typesys.stores.ros1_noetic import sensor_msgs__msg__Image as RbImage
    from rosbags.typesys.stores.ros1_noetic import std_msgs__msg__Header as RbHeader
    from rosbags.typesys.stores.ros1_noetic import builtin_interfaces__msg__Time as RbTime

    ts = get_typestore(Stores.ROS1_NOETIC)
    bag_path = str(tmp_path / "regression.bag")
    with Writer(bag_path) as writer:
        conn = writer.add_connection("/cam", RbImage.__msgtype__, typestore=ts)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        msg = RbImage(
            header=RbHeader(seq=0, stamp=RbTime(sec=0, nanosec=0), frame_id="c"),
            height=10, width=10, encoding="bgr8", is_bigendian=0, step=30, data=img.reshape(-1),
        )
        writer.write(conn, 0, ts.serialize_ros1(msg, RbImage.__msgtype__))

    from calibration.rosbag_reader import list_image_topics
    topics = list_image_topics(bag_path)
    assert len(topics) == 1
    assert topics[0].name == "/cam"


def test_real_ros2_bag_with_embedded_typedefs_still_works(tmp_path):
    """default_typestore를 넘겨도, bag 안에 이미 타입 정의가 있으면 그게
    우선 사용되고 정상 동작해야 한다.
    """
    import pytest
    pytest.importorskip("rosbags")
    try:
        from rosbags.rosbag2 import Writer as Writer2
    except ImportError:
        pytest.skip("이 rosbags 버전엔 rosbag2 writer가 없음")

    from rosbags.typesys import get_typestore, Stores
    from rosbags.typesys.stores.ros2_humble import sensor_msgs__msg__Image as RbImage2
    from rosbags.typesys.stores.ros2_humble import std_msgs__msg__Header as RbHeader2
    from rosbags.typesys.stores.ros2_humble import builtin_interfaces__msg__Time as RbTime2

    ts2 = get_typestore(Stores.ROS2_HUMBLE)
    bag_dir = str(tmp_path / "regression_ros2_bag")
    with Writer2(bag_dir, version=8) as writer:
        conn = writer.add_connection("/camera/image_raw", RbImage2.__msgtype__, typestore=ts2)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        msg = RbImage2(
            header=RbHeader2(stamp=RbTime2(sec=0, nanosec=0), frame_id="c"),
            height=10, width=10, encoding="bgr8", is_bigendian=0, step=30, data=img.reshape(-1),
        )
        writer.write(conn, 0, ts2.serialize_cdr(msg, RbImage2.__msgtype__))

    from calibration.rosbag_reader import list_image_topics
    topics = list_image_topics(bag_dir)
    assert len(topics) == 1
    assert topics[0].name == "/camera/image_raw"


# ---------------------------------------------------------------------------
# 실시간 구독 - 디코딩 실패 시 조용히 안 사라지고 알려주는지
# ---------------------------------------------------------------------------

def test_unsupported_encoding_returns_none_reproduces_original_symptom():
    """지원 안 하는 인코딩(nv12 등)은 decode_image_message가 None을
    반환한다 - 예전 코드는 이 경우를 조용히 버려서 "대기 중" 상태로
    영원히 멈춘 것처럼 보이는 버그의 근본 원인이었다.

    (yuv422는 이후 지원 목록에 추가돼 더 이상 이 테스트에 적합하지 않다 -
    nv12처럼 아직 지원 안 하는 인코딩으로 검증한다.)
    """
    from calibration.ros_image_codec import decode_image_message

    class FakeMsg:
        encoding = "nv12"
        data = np.zeros(100, dtype=np.uint8)

    result = decode_image_message(FakeMsg(), "sensor_msgs/Image")
    assert result is None


def test_report_decode_error_calls_on_error_callback():
    import calibration.ros_live as rl

    sub = object.__new__(rl.LiveTopicSubscriber)
    sub._on_error = None
    sub._last_error_report_t = 0.0

    calls = []
    sub._on_error = lambda detail: calls.append(detail)

    sub._report_decode_error("encoding=yuv422 지원 안 함")
    assert len(calls) == 1
    assert "yuv422" in calls[0]


def test_report_decode_error_is_rate_limited():
    """매 프레임마다 알림이 뜨면 스팸이 된다 - 3초 이내 재호출은 무시돼야 한다."""
    import calibration.ros_live as rl

    sub = object.__new__(rl.LiveTopicSubscriber)
    sub._last_error_report_t = 0.0

    calls = []
    sub._on_error = lambda detail: calls.append(detail)

    sub._report_decode_error("first")
    sub._report_decode_error("second")
    sub._report_decode_error("third")

    assert len(calls) == 1


def test_report_decode_error_noop_when_no_callback_registered():
    """on_error를 안 넘긴 경우 크래시하면 안 된다."""
    import calibration.ros_live as rl

    sub = object.__new__(rl.LiveTopicSubscriber)
    sub._on_error = None
    sub._last_error_report_t = 0.0

    sub._report_decode_error("아무도 안 듣고 있음")


def test_extract_error_message_includes_actual_encoding_name(tmp_path):
    """"지원하지 않는 인코딩" 에러가 나면 실제로 어떤 인코딩이었는지 이름이
    메시지에 담겨야 한다 - 안 그러면 사용자가 원인을 알 방법이 없다
    (실제로 사용자가 이 문제를 겪었을 때, 어떤 인코딩인지 몰라 재현/진단이
    오래 걸렸다).
    """
    import pytest
    pytest.importorskip("rosbags")

    from rosbags.rosbag1 import Writer
    from rosbags.typesys import get_typestore, Stores
    from rosbags.typesys.stores.ros1_noetic import sensor_msgs__msg__Image as RbImage
    from rosbags.typesys.stores.ros1_noetic import std_msgs__msg__Header as RbHeader
    from rosbags.typesys.stores.ros1_noetic import builtin_interfaces__msg__Time as RbTime

    ts = get_typestore(Stores.ROS1_NOETIC)
    bag_path = str(tmp_path / "unsupported_enc.bag")

    with Writer(bag_path) as writer:
        conn = writer.add_connection("/camera1/image_raw", RbImage.__msgtype__, typestore=ts)
        img = np.zeros((10, 10), dtype=np.uint16)
        msg = RbImage(
            header=RbHeader(seq=0, stamp=RbTime(sec=0, nanosec=0), frame_id="c"),
            height=10, width=10, encoding="nv12", is_bigendian=0, step=10,
            data=img.reshape(-1).view(np.uint8),
        )
        writer.write(conn, 0, ts.serialize_ros1(msg, RbImage.__msgtype__))

    from calibration.rosbag_reader import extract_images_from_bag
    import pytest as _pytest
    with _pytest.raises(ValueError, match="nv12"):
        extract_images_from_bag(bag_path, "/camera1/image_raw", str(tmp_path / "out"))
