"""
camera_calibrator.calibration.rosbag_reader
================================================

설계 문서 11번의 "Calibration Tool <-> ROS" 연동을 한 단계 더 확장: ROS1(.bag) /
ROS2(.db3, .mcap) 로그 파일에서 이미지를 직접 뽑아 캘리브레이션 파이프라인이
바로 쓸 수 있는 정적 이미지 파일(.jpg)로 저장한다.

rospy/rclpy(ROS 정식 설치)에 의존하지 않고 순수 Python 라이브러리인
`rosbags`를 쓴다 - README 1번 "Windows/macOS/Linux 모두 가능" 원칙을 지키기
위해서다. ROS 정식 설치는 사실상 Linux 전용이라, 여기 의존했다면 이 툴의
크로스플랫폼성이 깨졌을 것이다. `rosbags`는 ROS1/ROS2 포맷 차이를
AnyReader가 흡수해주므로, 이 모듈은 두 포맷을 구분하는 코드를 따로 두지 않는다.

핵심 설계: 이 모듈의 역할은 "bag -> 이미지 파일 경로 리스트"로 끝난다.
추출된 파일 경로들은 detect_dataset()에 그대로 넘기면 되므로, 이후 파이프라인
(검출/품질/3모델/검증/export) 전부는 이미지가 폴더에서 왔는지 bag에서
왔는지 전혀 몰라도 된다 - 기존 코드를 한 줄도 건드리지 않는다.

이미지 디코딩(Image/CompressedImage -> BGR ndarray)은 ros_image_codec.py를
공유한다 - ros_live.py(실시간 구독)도 같은 디코딩 로직을 쓴다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from calibration.ros_image_codec import decode_image_message

logger = logging.getLogger(__name__)

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    ROSBAGS_AVAILABLE = True
except ImportError:  # rosbags는 선택적 의존성 - 설치 안 해도 나머지 툴은 그대로 동작해야 함
    AnyReader = None  # type: ignore
    Stores = None  # type: ignore
    get_typestore = None  # type: ignore
    ROSBAGS_AVAILABLE = False

_IMAGE_MSG_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}


@dataclass
class BagImageTopic:
    """UI에서 사용자가 고를 이미지 토픽 하나."""
    name: str
    msg_type: str
    count: int


def _require_rosbags() -> None:
    if not ROSBAGS_AVAILABLE:
        raise ImportError(
            "rosbag 읽기 기능을 쓰려면 'rosbags' 패키지가 필요합니다.\n"
            "    pip install rosbags\n"
            "(ROS 정식 설치는 필요 없습니다 - 순수 Python 라이브러리입니다.)"
        )


def _open_reader(bag_path: str) -> AnyReader:
    """AnyReader에 default_typestore를 넘겨서 연다.

    실제로 발생한 사용자 버그: 'Bag contains no type definitions.
    Instantiate AnyReader with a default_typestore argument.'

    원인: ROS2 bag(metadata.yaml + .db3/.mcap)은 녹화 방식/rosbag2 버전에
    따라 메시지 타입 정의(IDL)가 bag 안에 통째로 안 담기는 경우가 흔하다.
    이 경우 rosbags 라이브러리는 대체 typestore가 없으면 이 에러를 던진다.
    sensor_msgs/Image, CompressedImage는 ROS2 배포판이 달라도 정의가
    사실상 동일하므로, Stores.LATEST를 기본 typestore로 넘겨주면
    (ROS1 bag이나 타입 정의가 이미 내장된 ROS2 bag에는 영향 없음 - 그 경우엔
    bag 안의 정의가 우선 사용됨) 이 문제를 해결할 수 있다.
    """
    logger.debug("AnyReader 열기 (default_typestore=Stores.LATEST): %s", bag_path)
    return AnyReader([Path(bag_path)], default_typestore=get_typestore(Stores.LATEST))


def list_image_topics(bag_path: str) -> list[BagImageTopic]:
    """bag 안에서 이미지 메시지 타입(Image/CompressedImage)인 토픽만 골라 반환.

    같은 토픽이 멀티 커넥션(예: 여러 노드가 같은 토픽을 발행)으로 잡힐 수 있어
    토픽 이름 기준으로 메시지 수를 합산하고 중복 제거한다.
    """
    _require_rosbags()
    logger.info("bag 열기 시도: %s", bag_path)
    counts: dict[str, int] = {}
    types: dict[str, str] = {}

    with _open_reader(bag_path) as reader:
        for conn in reader.connections:
            if conn.msgtype not in _IMAGE_MSG_TYPES:
                continue
            counts[conn.topic] = counts.get(conn.topic, 0) + conn.msgcount
            types[conn.topic] = conn.msgtype

    topics = [
        BagImageTopic(name=topic, msg_type=types[topic], count=counts[topic])
        for topic in sorted(counts)
    ]
    logger.info("bag에서 이미지 토픽 %d개 발견: %s", len(topics), [t.name for t in topics])
    return topics


def extract_images_from_bag(
    bag_path: str,
    topic: str,
    output_dir: str,
    min_interval_sec: float = 0.5,
    max_images: int | None = None,
) -> list[str]:
    """선택한 토픽의 이미지를 min_interval_sec 이상 간격으로 샘플링해서
    output_dir에 .jpg로 저장하고, 저장된 파일 경로 리스트를 반환한다.

    시간 기반 샘플링을 쓰는 이유: bag의 프레임레이트가 15/30/60fps로 제각각이라
    "매 N번째 메시지"보다 "최소 몇 초 간격"이 실제 캘리브레이션 데이터셋 품질
    관점(자세 다양성, 설계 문서 7번)에서 훨씬 예측 가능하다 - 거의 똑같은
    프레임을 수십 장 추출하는 걸 방지한다.

    반환된 경로들은 detect_dataset()에 그대로 넘기면 된다.
    """
    _require_rosbags()
    logger.info(
        "bag에서 이미지 추출 시작: bag=%s, topic=%s, min_interval_sec=%s, max_images=%s",
        bag_path, topic, min_interval_sec, max_images,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    last_saved_t: float | None = None
    skipped_unsupported = 0
    seen_encodings: set[str] = set()  # 실패 이유를 구체적으로 알려주기 위해 수집

    with _open_reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            raise ValueError(f"토픽 '{topic}'을 bag에서 찾을 수 없습니다.")
        msg_type = connections[0].msgtype
        if msg_type not in _IMAGE_MSG_TYPES:
            raise ValueError(f"토픽 '{topic}'은 이미지 타입이 아닙니다 ({msg_type}).")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if max_images is not None and len(saved_paths) >= max_images:
                break

            t_sec = timestamp / 1e9  # rosbags 타임스탬프는 나노초 단위
            if last_saved_t is not None and (t_sec - last_saved_t) < min_interval_sec:
                continue

            msg = reader.deserialize(rawdata, connection.msgtype)
            img = decode_image_message(msg, msg_type)

            if img is None:
                skipped_unsupported += 1
                detail = getattr(msg, "encoding", None) or getattr(msg, "format", None)
                if detail:
                    seen_encodings.add(detail)
                logger.debug("디코딩 실패로 건너뜀 (t=%.3f, encoding/format=%s)", t_sec, detail)
                continue

            idx = len(saved_paths)
            filename = out_dir / f"bag_{idx:04d}_t{t_sec:.3f}.jpg"
            cv2.imwrite(str(filename), img)
            saved_paths.append(str(filename))
            last_saved_t = t_sec

    logger.info(
        "bag 추출 완료: %d장 저장, %d개 디코딩 실패로 건너뜀 (topic=%s)",
        len(saved_paths), skipped_unsupported, topic,
    )

    if not saved_paths and skipped_unsupported > 0:
        encodings_str = ", ".join(sorted(seen_encodings)) or "알 수 없음"
        logger.warning(
            "저장된 이미지가 0장입니다 - 발견된 인코딩이 전부 미지원: %s", encodings_str
        )
        raise ValueError(
            f"토픽 '{topic}'에서 지원하지 않는 인코딩만 발견돼 추출된 이미지가 없습니다 "
            f"({skipped_unsupported}개 건너뜀). 발견된 인코딩: {encodings_str}"
        )

    return saved_paths
