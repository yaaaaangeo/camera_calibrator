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

import logging
import threading
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Callable

import numpy as np

from calibration.ros_image_codec import decode_image_message

logger = logging.getLogger(__name__)

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
    logger.info("ROS1(rospy) 백엔드를 사용합니다.")
except ImportError:
    try:
        import rclpy as _rclpy  # type: ignore
        from rclpy.node import Node as _Ros2Node  # type: ignore
        from sensor_msgs.msg import Image as _RosImage2, CompressedImage as _RosCompressedImage2  # type: ignore
        ROS_LIVE_BACKEND = "ros2"
        logger.info("ROS2(rclpy) 백엔드를 사용합니다.")
    except ImportError:
        ROS_LIVE_BACKEND = None
        logger.info("rospy/rclpy 둘 다 찾지 못했습니다. 실시간 ROS 구독은 비활성화됩니다.")


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


@dataclass
class StereoLivePair:
    image_cam1: np.ndarray
    image_cam2: np.ndarray
    timestamp_cam1: float
    timestamp_cam2: float

    @property
    def sync_delta_ms(self) -> float:
        return abs(self.timestamp_cam1 - self.timestamp_cam2) * 1000.0


@dataclass
class LiveDualCaptureQAReport:
    backend: str | None
    topic_count: int
    selected_topic1: str | None
    selected_topic2: str | None
    max_sync_delta_ms: float
    output_dir: str
    status: str
    subscribed: bool = False
    captured_pair_count: int = 0
    last_sync_delta_ms: float | None = None
    subscribe_elapsed_sec: float | None = None
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def format(self) -> str:
        lines = [
            f"Backend: {self.backend or 'not found'}",
            f"Image topics: {self.topic_count}",
            f"Camera 1 topic: {self.selected_topic1 or 'not selected'}",
            f"Camera 2 topic: {self.selected_topic2 or 'not selected'}",
            f"Sync threshold: {self.max_sync_delta_ms:.1f} ms",
            f"Output: {self.output_dir}",
            f"Subscribed: {'yes' if self.subscribed else 'no'}",
            f"Captured pairs: {self.captured_pair_count}",
            f"Last sync delta: {'N/A' if self.last_sync_delta_ms is None else f'{self.last_sync_delta_ms:.1f} ms'}",
            f"Subscribe elapsed: {'N/A' if self.subscribe_elapsed_sec is None else f'{self.subscribe_elapsed_sec:.1f} sec'}",
            f"Status: {self.status}",
        ]
        if self.checks:
            lines.append("Checks:")
            lines.extend(f"- {item}" for item in self.checks)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {item}" for item in self.warnings)
        return "\n".join(lines)


def build_live_dual_capture_qa_report(
    *,
    topics: list[LiveTopic],
    selected_topic1: LiveTopic | None,
    selected_topic2: LiveTopic | None,
    output_dir: str,
    max_sync_delta_ms: float = 30.0,
    subscribed: bool = False,
    captured_pair_count: int = 0,
    last_sync_delta_ms: float | None = None,
    subscribe_elapsed_sec: float | None = None,
) -> LiveDualCaptureQAReport:
    checks: list[str] = []
    warnings: list[str] = []
    if ROS_LIVE_BACKEND is None:
        warnings.append("ROS1/ROS2 Python runtime was not found. Source the ROS environment before launching the app.")
    else:
        checks.append(f"ROS backend detected: {ROS_LIVE_BACKEND}")
    if len(topics) >= 2:
        checks.append("At least two image topics are visible.")
    else:
        warnings.append("Fewer than two image topics are visible.")
    if selected_topic1 is None or selected_topic2 is None:
        warnings.append("Camera 1/2 topics are not both selected.")
    elif selected_topic1.name == selected_topic2.name:
        warnings.append("Camera 1 and Camera 2 topics must be different.")
    else:
        checks.append("Camera 1/2 topics are distinct.")
    out = Path(output_dir)
    if out.exists():
        checks.append("Output directory exists.")
    else:
        warnings.append("Output directory does not exist yet.")
    if subscribed:
        checks.append("Dual topic subscription is running.")
        if last_sync_delta_ms is None:
            warnings.append("No synchronized stereo pair has been received yet.")
        elif last_sync_delta_ms <= max_sync_delta_ms:
            checks.append(f"Last synchronized pair is within threshold ({last_sync_delta_ms:.1f} ms).")
        else:
            warnings.append(f"Last synchronized pair exceeds threshold ({last_sync_delta_ms:.1f} ms).")
    else:
        warnings.append("Dual topic subscription is not running.")
    if captured_pair_count >= 50:
        checks.append("Captured pair count meets the 50-pair recommendation.")
    elif captured_pair_count > 0:
        warnings.append(f"Only {captured_pair_count} stereo pairs captured; 50+ pairs are recommended.")
    status = "ready" if not warnings else "needs_attention"
    return LiveDualCaptureQAReport(
        backend=ROS_LIVE_BACKEND,
        topic_count=len(topics),
        selected_topic1=selected_topic1.name if selected_topic1 else None,
        selected_topic2=selected_topic2.name if selected_topic2 else None,
        max_sync_delta_ms=float(max_sync_delta_ms),
        output_dir=str(out),
        status=status,
        subscribed=bool(subscribed),
        captured_pair_count=int(captured_pair_count),
        last_sync_delta_ms=last_sync_delta_ms,
        subscribe_elapsed_sec=subscribe_elapsed_sec,
        checks=checks,
        warnings=warnings,
    )


class StereoFrameSynchronizer:
    """Keep the latest frame from each camera and emit near-synchronous pairs."""

    def __init__(self, max_sync_delta_ms: float = 30.0) -> None:
        self.max_sync_delta_ms = float(max_sync_delta_ms)
        self._latest_cam1: tuple[np.ndarray, float] | None = None
        self._latest_cam2: tuple[np.ndarray, float] | None = None

    def submit(self, camera_index: int, image: np.ndarray, timestamp_sec: float) -> StereoLivePair | None:
        item = (image, float(timestamp_sec))
        if camera_index == 1:
            self._latest_cam1 = item
        elif camera_index == 2:
            self._latest_cam2 = item
        else:
            raise ValueError("camera_index must be 1 or 2")
        if self._latest_cam1 is None or self._latest_cam2 is None:
            return None

        img1, t1 = self._latest_cam1
        img2, t2 = self._latest_cam2
        delta_ms = abs(t1 - t2) * 1000.0
        if delta_ms <= self.max_sync_delta_ms:
            self._latest_cam1 = None
            self._latest_cam2 = None
            return StereoLivePair(img1, img2, t1, t2)

        if t1 < t2:
            self._latest_cam1 = None
        else:
            self._latest_cam2 = None
        return None


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
        self._on_error: Callable[[str], None] | None = None
        self._last_error_report_t: float = 0.0
        self._running = False

        # ROS1 전용
        self._rospy_subscriber = None

        # ROS2 전용
        self._ros2_node = None
        self._ros2_spin_thread: threading.Thread | None = None

        logger.debug("LiveTopicSubscriber 생성 (backend=%s)", self._backend)

    # ------------------------------------------------------------------
    # 토픽 목록
    # ------------------------------------------------------------------

    def list_image_topics(self) -> list[LiveTopic]:
        logger.debug("토픽 목록 조회 시작 (backend=%s)", self._backend)
        if self._backend == "ros1":
            topics = self._list_topics_ros1()
        else:
            topics = self._list_topics_ros2()
        logger.info("이미지 토픽 %d개 발견: %s", len(topics), [t.name for t in topics])
        return topics

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
            logger.debug("rospy 노드 초기화 (camera_calibrator_live)")
            _rospy.init_node("camera_calibrator_live", anonymous=True, disable_signals=True)

    def _ensure_ros2_node(self):
        if _rclpy is not None and not _rclpy.ok():
            logger.debug("rclpy.init() 호출")
            _rclpy.init()
        if self._ros2_node is None:
            logger.debug("rclpy 노드 생성 (camera_calibrator_live)")
            self._ros2_node = _Ros2Node("camera_calibrator_live")
        return self._ros2_node

    # ------------------------------------------------------------------
    # 구독 시작/종료
    # ------------------------------------------------------------------

    def start(
        self,
        topic: str,
        msg_type: str,
        on_frame: Callable[[np.ndarray, float], None],
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        """on_error: 프레임은 도착했지만 디코딩에 실패했을 때 호출된다
        (예: 지원하지 않는 encoding). 이게 없으면 "토픽은 맞게 골랐는데
        영원히 대기 중"처럼 보여서 사용자가 원인을 알 방법이 없다 - 최소
        3초 간격으로 rate-limit해서 콜백을 스팸하지 않는다.
        """
        if self._running:
            raise RuntimeError("이미 구독 중입니다. 먼저 stop()을 호출하세요.")
        self._on_frame = on_frame
        self._on_error = on_error
        self._running = True
        logger.info("토픽 구독 시작: topic=%s, msg_type=%s, backend=%s", topic, msg_type, self._backend)

        if self._backend == "ros1":
            self._start_ros1(topic, msg_type)
        else:
            self._start_ros2(topic, msg_type)

    def _report_decode_error(self, detail: str) -> None:
        # 프레임마다(예: 30fps) 호출될 수 있어 DEBUG로 매번 남기고, 실제로 화면에
        # 표시되는(=rate-limit을 통과한) 경우만 WARNING으로 승격한다 - 안 그러면
        # 로그 파일이 디코딩 실패 한 종류로 초 단위로 도배된다.
        logger.debug("프레임 디코딩 실패: %s", detail)
        if self._on_error is None:
            return
        now = time.monotonic()
        if now - self._last_error_report_t < 3.0:
            return  # 매 프레임마다 알림이 뜨면 스팸이 되므로 3초에 한 번만
        self._last_error_report_t = now
        logger.warning("프레임 디코딩 실패 (사용자에게 표시됨): %s", detail)
        self._on_error(detail)

    def _start_ros1(self, topic: str, msg_type: str) -> None:
        self._ensure_ros1_node()
        msg_class = _RosImage1 if msg_type == "sensor_msgs/Image" else _RosCompressedImage1

        def _callback(msg) -> None:
            if not self._running or self._on_frame is None:
                return
            img = decode_image_message(msg, msg_type)
            if img is None:
                detail = getattr(msg, "encoding", None) or getattr(msg, "format", "알 수 없음")
                self._report_decode_error(
                    f"프레임은 도착했지만 디코딩에 실패했습니다 (encoding/format='{detail}'). "
                    f"지원하지 않는 이미지 인코딩일 수 있습니다."
                )
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
                detail = getattr(msg, "encoding", None) or getattr(msg, "format", "알 수 없음")
                self._report_decode_error(
                    f"프레임은 도착했지만 디코딩에 실패했습니다 (encoding/format='{detail}'). "
                    f"지원하지 않는 이미지 인코딩일 수 있습니다."
                )
                return
            stamp = msg.header.stamp
            t_sec = stamp.sec + stamp.nanosec / 1e9 if stamp else time.time()
            self._on_frame(img, t_sec)

        node.create_subscription(msg_class, topic, _callback, qos_profile_sensor_data)

        # rclpy는 rospy와 달리 콜백이 저절로 안 돈다 - 명시적으로 spin을 계속 돌려야 한다.
        # Qt 이벤트 루프를 막지 않도록 별도 스레드에서 spin_once를 반복한다.
        def _spin_loop() -> None:
            logger.debug("ROS2 spin 스레드 시작")
            try:
                while self._running and _rclpy.ok():
                    _rclpy.spin_once(node, timeout_sec=0.1)
            except Exception:
                # 이 스레드는 daemon이라 예외가 나면 조용히 죽어서 "이유 없이 멈춤"처럼
                # 보인다 - 로그에 스택트레이스를 남겨서 추적 가능하게 한다.
                logger.exception("ROS2 spin 스레드가 예외로 종료됨")
            finally:
                logger.debug("ROS2 spin 스레드 종료")

        self._ros2_spin_thread = threading.Thread(target=_spin_loop, daemon=True)
        self._ros2_spin_thread.start()

    def stop(self) -> None:
        logger.info("토픽 구독 종료 (backend=%s)", self._backend)
        self._running = False
        self._on_frame = None
        self._on_error = None

        if self._backend == "ros1" and self._rospy_subscriber is not None:
            self._rospy_subscriber.unregister()
            self._rospy_subscriber = None

        if self._backend == "ros2":
            if self._ros2_spin_thread is not None:
                self._ros2_spin_thread.join(timeout=2.0)
                if self._ros2_spin_thread.is_alive():
                    logger.warning("ROS2 spin 스레드가 2초 안에 종료되지 않았습니다.")
                self._ros2_spin_thread = None
            if self._ros2_node is not None:
                self._ros2_node.destroy_node()
                self._ros2_node = None


class DualLiveTopicSubscriber:
    """Subscribe to two ROS image topics and emit synchronized stereo frames."""

    def __init__(self, *, max_sync_delta_ms: float = 30.0) -> None:
        _require_backend()
        self._sync = StereoFrameSynchronizer(max_sync_delta_ms=max_sync_delta_ms)
        self._lock = threading.Lock()
        self._sub1: LiveTopicSubscriber | None = None
        self._sub2: LiveTopicSubscriber | None = None
        self._on_pair: Callable[[StereoLivePair], None] | None = None

    def list_image_topics(self) -> list[LiveTopic]:
        sub = LiveTopicSubscriber()
        try:
            return sub.list_image_topics()
        finally:
            sub.stop()

    def start(
        self,
        topic1: str,
        msg_type1: str,
        topic2: str,
        msg_type2: str,
        on_pair: Callable[[StereoLivePair], None],
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        if self._sub1 is not None or self._sub2 is not None:
            raise RuntimeError("이미 dual topic 구독 중입니다. 먼저 stop()을 호출하세요.")
        self._on_pair = on_pair
        self._sub1 = LiveTopicSubscriber()
        self._sub2 = LiveTopicSubscriber()

        def handle(camera_index: int, image: np.ndarray, timestamp: float) -> None:
            with self._lock:
                pair = self._sync.submit(camera_index, image, timestamp)
            if pair is not None and self._on_pair is not None:
                self._on_pair(pair)

        self._sub1.start(topic1, msg_type1, lambda img, ts: handle(1, img, ts), on_error)
        try:
            self._sub2.start(topic2, msg_type2, lambda img, ts: handle(2, img, ts), on_error)
        except Exception:
            self._sub1.stop()
            self._sub1 = None
            self._sub2 = None
            raise

    def stop(self) -> None:
        for sub in (self._sub1, self._sub2):
            if sub is not None:
                sub.stop()
        self._sub1 = None
        self._sub2 = None
        self._on_pair = None
