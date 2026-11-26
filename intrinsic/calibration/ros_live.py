"""
calibration.ros_live
====================

실시간 ROS 카메라 토픽 구독 - Camera Intrinsic 용 얇은 인터페이스.

토픽 탐색과 수신은 CAMERA–LiDAR 와 같은 공용 백엔드(``shared.ros.live``)가 한다.
그래서 ROS 가 설치되지 않은 PC 에서도 동작한다:

    ROS 1   마스터(ROS_MASTER_URI)에 XML-RPC/TCPROS 로 직접 접속 - rospy 불필요
    ROS 2   ROS 2 가 source 돼 있으면 rclpy, 아니면 cyclonedds 로 DDS 직접 수신

이 모듈은 그 위에서 영상 토픽만 골라 보여주고, 받은 메시지를 BGR 이미지로 풀어
``on_frame(img_bgr, stamp_sec)`` 로 넘긴다. 디코딩 실패는 3초에 한 번만 알린다.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from shared.ros import live as shared_live
from shared.ros.decode import image_to_bgr

logger = logging.getLogger(__name__)

# 예전에는 rospy/rclpy 가 import 되는지로 "ros1"/"ros2"/None 을 정했다. 이제 수신은
# ROS 설치와 무관하므로(ROS 1 은 직접 접속, ROS 2 는 rclpy 또는 DDS) 항상 쓸 수 있다.
ROS_LIVE_BACKEND: str | None = "auto"


def available_ros2() -> str:
    """이 PC 에서 ROS 2 를 받는 수단: 'rclpy', 'dds' 또는 '' (없음)."""
    if shared_live.rclpy_available():
        return "rclpy"
    return "dds" if shared_live.dds_available() else ""


def backend_status_text() -> str:
    ros2 = available_ros2()
    ros2_text = {"rclpy": "ROS 2: rclpy (source 된 ROS 2 사용)",
                 "dds": "ROS 2: DDS 직접 수신 (ROS 설치 불필요)"}.get(ros2, "ROS 2: 받을 수단 없음 (cyclonedds 미설치)")
    return f"ROS 1: master 에 직접 접속 (rospy 불필요) · {ros2_text}"


def _require_backend() -> None:
    """호환용 - 수신은 항상 가능하다 (ROS 1 직접 접속)."""


@dataclass
class LiveTopic:
    name: str
    msg_type: str  # ROS 2 표기 (sensor_msgs/msg/Image)
    source: str = ""  # 어디서 받는지 - "ROS 1", "ROS 2 (rclpy)", "ROS 2 (DDS)"


_SOURCE_LABEL = {"ros1": "ROS 1", "rclpy": "ROS 2 (rclpy)", "dds": "ROS 2 (DDS)"}


class LiveTopicSubscriber:
    """실시간 영상 토픽 하나를 받는다.

        sub = LiveTopicSubscriber()
        topics = sub.list_image_topics()
        sub.start(topics[0].name, topics[0].msg_type, on_frame=callback)
        ...
        sub.stop()

    ``on_frame(img_bgr, stamp_sec)`` 은 수신 스레드에서 불린다 - Qt 위젯을 직접 건드리지
    말고 Signal 이나 최신 프레임 버퍼로 GUI 스레드에 넘긴다.
    """

    def __init__(self, ros1_uri: str | None = None, domain_id: int | None = None,
                 ros2_backend: str = "auto", probe_timeout: float = 2.0) -> None:
        self._ros1_uri = ros1_uri
        self._domain_id = domain_id
        self._ros2_backend = ros2_backend
        self._timeout = probe_timeout
        self._probe: shared_live.Probe | None = None
        self._on_frame: Callable[[np.ndarray, float], None] | None = None
        self._on_error: Callable[[str], None] | None = None
        self._last_error_report_t = 0.0
        self._running = False
        self._backend = "auto"  # 호환용 속성

    # ------------------------------------------------------------- 토픽 목록

    def _close_probe(self) -> None:
        if self._probe is not None:
            for backend in self._probe.backends.values():
                try:
                    backend.close()
                except Exception:  # noqa: BLE001 - 닫기 실패가 다음 탐색을 막으면 안 된다
                    logger.debug("backend close failed", exc_info=True)
            self._probe = None

    def _ensure_probe(self) -> shared_live.Probe:
        if self._probe is None:
            ros1 = self._ros1_uri if self._ros1_uri is not None else os.environ.get("ROS_MASTER_URI")
            self._probe = shared_live.probe(timeout=self._timeout, ros1_uri=ros1, domain_id=self._domain_id,
                                            ros2_backend=self._ros2_backend)
            for note in self._probe.notes:
                logger.info("토픽 탐색: %s", note)
        return self._probe

    def list_image_topics(self) -> list[LiveTopic]:
        """다시 탐색해서 영상 토픽(Image / CompressedImage)만 돌려준다."""
        if self._running:
            raise RuntimeError("구독 중에는 다시 탐색할 수 없습니다. 먼저 stop()을 호출하세요.")
        self._close_probe()
        probe = self._ensure_probe()
        topics = [LiveTopic(t.name, t.msgtype, _SOURCE_LABEL.get(t.backend, t.backend))
                  for t in probe.topics if t.kind == "image"]
        logger.info("이미지 토픽 %d개 발견: %s", len(topics), [t.name for t in topics])
        return topics

    @property
    def notes(self) -> list[str]:
        return list(self._probe.notes) if self._probe is not None else []

    # ------------------------------------------------------------- 구독

    def start(self, topic: str, msg_type: str, on_frame: Callable[[np.ndarray, float], None],
              on_error: Callable[[str], None] | None = None) -> None:
        """`topic` 을 구독한다. 탐색 전이면 먼저 탐색한다.

        on_error: 프레임은 왔는데 풀지 못했을 때 (예: 지원하지 않는 encoding). 없으면
        "토픽은 맞는데 영원히 대기 중"처럼 보여 원인을 알 수 없다 - 3초에 한 번만 부른다.
        """
        if self._running:
            raise RuntimeError("이미 구독 중입니다. 먼저 stop()을 호출하세요.")
        probe = self._ensure_probe()
        want = shared_live.ros2_name(msg_type) if msg_type else ""
        candidates = [t for t in probe.topics if t.name == topic and t.kind == "image"]
        if not candidates:
            # 탐색 뒤에 발행이 시작됐을 수 있다 - 한 번 더 찾아본다.
            self._close_probe()
            probe = self._ensure_probe()
            candidates = [t for t in probe.topics if t.name == topic and t.kind == "image"]
        if not candidates:
            raise RuntimeError(f"토픽을 찾지 못했습니다: {topic}\n" + "\n".join(probe.notes))
        chosen = next((t for t in candidates if t.msgtype == want), candidates[0])
        self._on_frame, self._on_error = on_frame, on_error
        self._running = True
        logger.info("토픽 구독 시작: %s (%s, %s)", topic, chosen.msgtype, chosen.backend)
        probe.backends[chosen.backend].subscribe(chosen, self._received, on_error=self._report_receive_error)

    def _received(self, r: shared_live.Received) -> None:
        if not self._running:
            return
        try:
            img = image_to_bgr(r.msg)
        except Exception as exc:  # noqa: BLE001 - 프레임 하나 실패가 구독을 끝내면 안 된다
            self._report_decode_error(str(exc))
            return
        callback = self._on_frame
        if callback is not None:
            callback(img, r.stamp_ns / 1e9)

    def _report_receive_error(self, detail: str) -> None:
        """토픽은 보이는데 받지 못할 때 (예: ROS 1 퍼블리셔 주소를 이 PC 가 못 찾음)."""
        logger.warning("토픽 수신 실패: %s", detail)
        if self._on_error is not None:
            self._last_error_report_t = time.monotonic()
            self._on_error(detail)

    @property
    def ros1_uri(self) -> str | None:
        return self._ros1_uri

    @ros1_uri.setter
    def ros1_uri(self, uri: str | None) -> None:
        """ROS 1 master 주소를 바꾼다 - 다음 탐색부터 쓴다."""
        if self._running:
            raise RuntimeError("구독 중에는 master 주소를 바꿀 수 없습니다.")
        uri = (uri or "").strip() or None
        if uri != self._ros1_uri:
            self._ros1_uri = uri
            self._close_probe()

    def _report_decode_error(self, detail: str) -> None:
        # 프레임마다(예: 30fps) 불릴 수 있어 DEBUG 로 남기고, 화면에 알리는 것만 WARNING.
        logger.debug("프레임 디코딩 실패: %s", detail)
        if self._on_error is None:
            return
        now = time.monotonic()
        if now - self._last_error_report_t < 3.0:
            return
        self._last_error_report_t = now
        logger.warning("프레임 디코딩 실패 (사용자에게 표시됨): %s", detail)
        self._on_error(detail)

    def stop(self) -> None:
        logger.info("토픽 구독 종료")
        self._running = False
        self._on_frame = None
        self._on_error = None
        self._close_probe()
