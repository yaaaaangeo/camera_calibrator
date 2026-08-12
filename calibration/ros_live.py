"""
camera_calibrator.calibration.ros_live
==========================================

실시간 ROS 카메라 토픽 구독 - ROS1(rospy) / ROS2(rclpy) 자동 감지.

이 모듈은 rosbag_reader.py와 다르다: rosbag_reader.py는 순수 Python
라이브러리(rosbags)로 ROS 설치 없이 동작하지만, **실시간 토픽 구독은
그럴 방법이 없다** - rospy/rclpy는 pip로 설치되는 게 아니라 실제 ROS1/ROS2가
컴퓨터에 설치돼 있어야만 import된다. 그래서 이 기능은:

  - ROS가 설치 안 된 환경에서는 import 자체가 실패하지 않고(지연 임포트),
    ROS_LIVE_BACKEND == None으로 남아서 UI가 "ROS를 찾을 수 없습니다"를
    보여줄 수 있게 한다.
  - rospy가 있으면 ROS1으로, 없고 rclpy가 있으면 ROS2로 자동 판단한다.
  - 두 백엔드의 차이(콜백이 자동으로 도는 rospy vs 직접 spin해야 하는 rclpy,
    토픽 목록 조회 API 차이)를 LiveTopicSubscriber 하나의 인터페이스 뒤로
    숨긴다 - UI 코드는 백엔드가 뭔지 몰라도 된다.

이미지 디코딩은 ros_image_codec.py를 공유한다 (rosbag_reader.py와 동일).

주의: 이 모듈의 rospy/rclpy 경로는 실제 ROS 런타임(roscore 또는 ROS2 데몬)이
있어야만 테스트 가능해서, 개발 환경(샌드박스)에서 end-to-end로 검증하지
못했다 - 잘 알려진 표준 API를 기준으로 작성했지만, 실제 ROS 환경에서
동작 확인이 필요하다.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from calibration.ros_image_codec import decode_image_message

_IMAGE_MSG_TYPES = {"sensor_msgs/Image", "sensor_msgs/CompressedImage"}
_IMAGE_MSG_TYPES_ROS2 = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}

# ---------------------------------------------------------------------------
# 백엔드 감지 (지연 임포트 - ROS 미설치 환경에서 이 모듈 자체의 import는 항상 성공해야 함)
# ---------------------------------------------------------------------------

ROS_LIVE_BACKEND: str | None = None
_rospy = None
_rclpy = None

try:
    import rospy as _rospy  # type: ignore
    from sensor_msgs.msg import Image as _RosImage1, CompressedImage as _RosCompressedImage1  # type: ignore
    ROS_LIVE_BACKEND = "ros1"
except ImportError:
    try:
        import rclpy as _rclpy  # type: ignore
        from rclpy.node import Node as _Ros2Node  # type: ignore
        from sensor_msgs.msg import Image as _RosImage2, CompressedImage as _RosCompressedImage2  # type: ignore
        ROS_LIVE_BACKEND = "ros2"
    except ImportError:
        ROS_LIVE_BACKEND = None


def _require_backend() -> None:
    if ROS_LIVE_BACKEND is None:
        raise ImportError(
            "실시간 ROS 구독을 쓰려면 이 컴퓨터에 ROS1 또는 ROS2가 설치되어 있고,\n"
            "환경이 source 되어 있어야 합니다 (예: 'source /opt/ros/noetic/setup.bash'\n"
            "또는 'source /opt/ros/humble/setup.bash'). rospy/rclpy는 pip로 설치되지 않습니다.\n"
            "(오프라인으로 이미 녹화된 bag 파일만 쓰신다면 '[rosbag에서 불러오기]'를 대신 쓰세요"
            " - 그건 ROS 설치가 필요 없습니다.)"
        )


@dataclass
class LiveTopic:
    name: str
    msg_type: str


# ---------------------------------------------------------------------------
# 공통 인터페이스
# ---------------------------------------------------------------------------

class LiveTopicSubscriber:
    """ROS1/ROS2 차이를 감춘 실시간 이미지 토픽 구독기.

    사용법:
        sub = LiveTopicSubscriber()
        topics = sub.list_image_topics()
        sub.start(topics[0].name, topics[0].msg_type, on_frame=my_callback)
        ...
        sub.stop()

    on_frame(img_bgr: np.ndarray, timestamp_sec: float)은 ROS 콜백 스레드(백그라운드
    스레드)에서 호출된다 - Qt 위젯을 직접 건드리면 안 되고, Qt Signal.emit()으로
    GUI 스레드에 넘겨야 한다 (PySide6는 스레드 간 emit을 자동으로 큐잉해서 안전하게
    처리해준다).
    """

    def __init__(self) -> None:
        _require_backend()
        self._backend = ROS_LIVE_BACKEND
        self._on_frame: Callable[[np.ndarray, float], None] | None = None
        self._running = False

        # ROS1 전용
        self._rospy_subscriber = None

        # ROS2 전용
        self._ros2_node = None
        self._ros2_spin_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # 토픽 목록
    # ------------------------------------------------------------------

    def list_image_topics(self) -> list[LiveTopic]:
        if self._backend == "ros1":
            return self._list_topics_ros1()
        return self._list_topics_ros2()

    def _list_topics_ros1(self) -> list[LiveTopic]:
        self._ensure_ros1_node()
        # rospy.get_published_topics()는 (토픽명, 타입문자열) 쌍의 리스트를 마스터에서 조회한다.
        published = _rospy.get_published_topics()
        return [LiveTopic(name=name, msg_type=t) for name, t in published if t in _IMAGE_MSG_TYPES]

    def _list_topics_ros2(self) -> list[LiveTopic]:
        node = self._ensure_ros2_node()
        # get_topic_names_and_types() -> [(name, [type_strings]), ...]
        topics = node.get_topic_names_and_types()
        result = []
        for name, types in topics:
            for t in types:
                if t in _IMAGE_MSG_TYPES_ROS2:
                    result.append(LiveTopic(name=name, msg_type=t))
                    break
        return result

    # ------------------------------------------------------------------
    # 노드 준비 (idempotent - 이미 떠 있으면 재사용)
    # ------------------------------------------------------------------

    def _ensure_ros1_node(self) -> None:
        if not _rospy.core.is_initialized():
            # disable_signals=True: PySide6가 이미 메인 스레드의 시그널 핸들러를 쓰므로
            # rospy가 SIGINT 등을 가로채지 않게 한다 (Qt 앱 안에서 안전하게 공존하려면 필수).
            _rospy.init_node("camera_calibrator_live", anonymous=True, disable_signals=True)

    def _ensure_ros2_node(self):
        if _rclpy is not None and not _rclpy.ok():
            _rclpy.init()
        if self._ros2_node is None:
            self._ros2_node = _Ros2Node("camera_calibrator_live")
        return self._ros2_node

    # ------------------------------------------------------------------
    # 구독 시작/종료
    # ------------------------------------------------------------------

    def start(self, topic: str, msg_type: str, on_frame: Callable[[np.ndarray, float], None]) -> None:
        if self._running:
            raise RuntimeError("이미 구독 중입니다. 먼저 stop()을 호출하세요.")
        self._on_frame = on_frame
        self._running = True

        if self._backend == "ros1":
            self._start_ros1(topic, msg_type)
        else:
            self._start_ros2(topic, msg_type)

    def _start_ros1(self, topic: str, msg_type: str) -> None:
        self._ensure_ros1_node()
        msg_class = _RosImage1 if msg_type == "sensor_msgs/Image" else _RosCompressedImage1

        def _callback(msg) -> None:
            if not self._running or self._on_frame is None:
                return
            img = decode_image_message(msg, msg_type)
            if img is None:
                return
            t_sec = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
            self._on_frame(img, t_sec)

        # rospy.Subscriber는 등록 즉시 백그라운드 스레드에서 콜백을 돌리기 시작한다 -
        # rospy.spin()을 따로 부를 필요 없다 (spin()은 메인 스레드를 막아둘 때만 필요).
        self._rospy_subscriber = _rospy.Subscriber(topic, msg_class, _callback, queue_size=1)

    def _start_ros2(self, topic: str, msg_type: str) -> None:
        from rclpy.qos import qos_profile_sensor_data

        node = self._ensure_ros2_node()
        msg_class = _RosImage2 if msg_type == "sensor_msgs/msg/Image" else _RosCompressedImage2

        def _callback(msg) -> None:
            if not self._running or self._on_frame is None:
                return
            img = decode_image_message(msg, msg_type)
            if img is None:
                return
            stamp = msg.header.stamp
            t_sec = stamp.sec + stamp.nanosec / 1e9 if stamp else time.time()
            self._on_frame(img, t_sec)

        node.create_subscription(msg_class, topic, _callback, qos_profile_sensor_data)

        # rclpy는 rospy와 달리 콜백이 저절로 안 돈다 - 명시적으로 spin을 계속 돌려야 한다.
        # Qt 이벤트 루프를 막지 않도록 별도 스레드에서 spin_once를 반복한다.
        def _spin_loop() -> None:
            while self._running and _rclpy.ok():
                _rclpy.spin_once(node, timeout_sec=0.1)

        self._ros2_spin_thread = threading.Thread(target=_spin_loop, daemon=True)
        self._ros2_spin_thread.start()

    def stop(self) -> None:
        self._running = False
        self._on_frame = None

        if self._backend == "ros1" and self._rospy_subscriber is not None:
            self._rospy_subscriber.unregister()
            self._rospy_subscriber = None

        if self._backend == "ros2":
            if self._ros2_spin_thread is not None:
                self._ros2_spin_thread.join(timeout=2.0)
                self._ros2_spin_thread = None
            if self._ros2_node is not None:
                self._ros2_node.destroy_node()
                self._ros2_node = None
