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
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from calibration.ros_image_codec import decode_image_message
from calibration.ros_pointcloud_codec import decode_pointcloud2_message

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
_POINTCLOUD_MSG_TYPES = {"sensor_msgs/msg/PointCloud2"}


def _portable_output_path(path: str) -> Path:
    if os.name == "nt" and path.replace("\\", "/").startswith("/tmp/"):
        return Path(tempfile.gettempdir()) / path.replace("\\", "/").removeprefix("/tmp/")
    return Path(path)


@dataclass
class BagImageTopic:
    """UI에서 사용자가 고를 이미지 토픽 하나."""
    name: str
    msg_type: str
    count: int


@dataclass
class TopicInfo:
    """일반화된 bag 토픽 메타데이터(이름 + 메시지 타입 + 메시지 수).

    BagImageTopic과 사실상 같은 모양이지만, list_image_topics의 기존
    호출부(Camera Intrinsic의 "rosbag에서 불러오기")를 건드리지 않기 위해
    BagImageTopic은 그대로 두고, 카메라/LiDAR 어느 쪽이든 쓸 수 있는
    새 코드(list_pointcloud_topics 등)는 이 타입을 쓴다.
    """
    name: str
    msg_type: str
    count: int | None = None


def _require_rosbags() -> None:
    if AnyReader is None or get_typestore is None:
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
    latest_store = Stores.LATEST if Stores is not None else "LATEST"
    return AnyReader([Path(bag_path)], default_typestore=get_typestore(latest_store))


def _list_topics_by_type(bag_path: str, msg_types: set[str]) -> list[TopicInfo]:
    """주어진 메시지 타입 집합에 속하는 토픽만 골라 반환하는 공용 헬퍼.

    같은 토픽이 멀티 커넥션(예: 여러 노드가 같은 토픽을 발행)으로 잡힐 수 있어
    토픽 이름 기준으로 메시지 수를 합산하고 중복 제거한다.
    list_image_topics/list_pointcloud_topics가 이 헬퍼를 공유한다.
    """
    _require_rosbags()
    counts: dict[str, int] = {}
    types: dict[str, str] = {}

    with _open_reader(bag_path) as reader:
        for conn in reader.connections:
            if conn.msgtype not in msg_types:
                continue
            counts[conn.topic] = counts.get(conn.topic, 0) + conn.msgcount
            types[conn.topic] = conn.msgtype

    return [
        TopicInfo(name=topic, msg_type=types[topic], count=counts[topic])
        for topic in sorted(counts)
    ]


def list_image_topics(bag_path: str) -> list[BagImageTopic]:
    """bag 안에서 이미지 메시지 타입(Image/CompressedImage)인 토픽만 골라 반환."""
    logger.info("bag 열기 시도: %s", bag_path)
    topics = _list_topics_by_type(bag_path, _IMAGE_MSG_TYPES)
    result = [BagImageTopic(name=t.name, msg_type=t.msg_type, count=t.count or 0) for t in topics]
    logger.info("bag에서 이미지 토픽 %d개 발견: %s", len(result), [t.name for t in result])
    return result


def list_pointcloud_topics(bag_path: str) -> list[TopicInfo]:
    """bag 안에서 PointCloud2 메시지 타입인 토픽만 골라 반환 (list_image_topics의 LiDAR 버전)."""
    logger.info("bag에서 PointCloud2 토픽 검색: %s", bag_path)
    topics = _list_topics_by_type(bag_path, _POINTCLOUD_MSG_TYPES)
    logger.info("bag에서 PointCloud2 토픽 %d개 발견: %s", len(topics), [t.name for t in topics])
    return topics


def read_bag_duration(bag_path: str) -> float:
    """bag 전체 duration(초). Camera/LiDAR Topic 선택 UI의 timeline scrubbing에 쓰인다."""
    _require_rosbags()
    with _open_reader(bag_path) as reader:
        return max(0.0, reader.duration / 1e9)


def _find_message_near_timestamp(bag_path: str, topic: str, msg_types: set[str], t_sec: float):
    """`topic`의 메시지 중 (bag 시작 + t_sec)에 가장 가까운 timestamp를 가진
    메시지 하나를 찾아 (역직렬화된 msg, msg_type, timestamp_ns)를 반환한다.
    Timeline을 스크럽할 때 "그 시점 근처 프레임"을 보여주는 용도 -
    extract_images_from_bag(전체를 시간 간격으로 샘플링해 디스크에 저장)과는
    다른, 단발성 조회 함수다.
    """
    _require_rosbags()
    with _open_reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            raise ValueError(f"토픽 '{topic}'을 bag에서 찾을 수 없습니다.")
        msg_type = connections[0].msgtype
        if msg_type not in msg_types:
            raise ValueError(f"토픽 '{topic}'의 메시지 타입이 예상과 다릅니다 ({msg_type}).")

        target_ns = reader.start_time + int(t_sec * 1e9)
        best = None
        best_diff = None
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            diff = abs(timestamp - target_ns)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best = (connection, timestamp, rawdata)
            elif timestamp > target_ns:
                # 메시지는 시간순으로 온다고 가정 - 목표 시점을 지나쳤고 방금
                # 최소값을 갱신하지 못했다면 더 가까워질 수 없으므로 중단.
                break
        if best is None:
            raise ValueError(f"토픽 '{topic}'에 메시지가 없습니다.")

        connection, timestamp, rawdata = best
        msg = reader.deserialize(rawdata, connection.msgtype)
        return msg, connection.msgtype, timestamp


def extract_image_near_timestamp(bag_path: str, topic: str, t_sec: float) -> tuple[np.ndarray, float, str]:
    """`topic`에서 t_sec(bag 시작 기준 초) 근처 이미지 프레임 하나를 디코딩해
    (BGR ndarray, timestamp_sec, frame_id)로 반환한다."""
    msg, msg_type, timestamp_ns = _find_message_near_timestamp(bag_path, topic, _IMAGE_MSG_TYPES, t_sec)
    img = decode_image_message(msg, msg_type)
    if img is None:
        raise ValueError(f"토픽 '{topic}'의 t={t_sec:.3f}s 근처 프레임을 디코딩하지 못했습니다.")
    frame_id = getattr(getattr(msg, "header", None), "frame_id", "") or ""
    return img, timestamp_ns / 1e9, frame_id


def extract_pointcloud_near_timestamp(bag_path: str, topic: str, t_sec: float) -> tuple[np.ndarray, float, str]:
    """`topic`에서 t_sec(bag 시작 기준 초) 근처 PointCloud2 프레임 하나를
    디코딩해 ((N,3) 또는 intensity 있으면 (N,4) ndarray, timestamp_sec, frame_id)로 반환한다."""
    msg, _msg_type, timestamp_ns = _find_message_near_timestamp(bag_path, topic, _POINTCLOUD_MSG_TYPES, t_sec)
    points = decode_pointcloud2_message(msg)
    frame_id = getattr(getattr(msg, "header", None), "frame_id", "") or ""
    return points, timestamp_ns / 1e9, frame_id


def iterate_images(bag_path: str, topic: str) -> Iterator[tuple[np.ndarray, float, str]]:
    """`topic`의 모든 이미지 프레임을 하나씩 디코딩해서 (BGR ndarray,
    t_sec(bag 시작 기준 초), frame_id)로 yield하는 제너레이터.

    extract_images_from_bag(시간 간격 샘플링 + 디스크에 .jpg 저장)과 달리
    아무것도 건너뛰거나 저장하지 않는다 -- camera_lidar.scene_extraction의
    Stable Scene Segment 탐지(연속 프레임 사이의 marker ID/자세 전환을
    보는 것)는 전체 프레임이 필요하기 때문이다. t_sec는
    extract_image_near_timestamp가 받는 인자(bag 시작 기준 상대 초)와 같은
    기준이라, 여기서 얻은 timestamp를 그대로 extract_pointcloud_near_timestamp
    에 넘겨 LiDAR 프레임과 짝지을 수 있다.
    """
    _require_rosbags()
    with _open_reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            raise ValueError(f"토픽 '{topic}'을 bag에서 찾을 수 없습니다.")
        msg_type = connections[0].msgtype
        if msg_type not in _IMAGE_MSG_TYPES:
            raise ValueError(f"토픽 '{topic}'은 이미지 타입이 아닙니다 ({msg_type}).")
        start_time = reader.start_time

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            try:
                msg = reader.deserialize(rawdata, connection.msgtype)
                img = decode_image_message(msg, connection.msgtype)
            except Exception:  # noqa: BLE001 -- one malformed/corrupt message must not kill (or, worse, hang on) a whole-bag scan
                logger.warning("프레임 디코딩 중 예외 발생, 건너뜀 (t=%s)", timestamp, exc_info=True)
                continue
            if img is None:
                continue
            frame_id = getattr(getattr(msg, "header", None), "frame_id", "") or ""
            t_sec = (timestamp - start_time) / 1e9
            yield img, t_sec, frame_id


def extract_images_from_bag(
    bag_path: str,
    topic: str,
    output_dir: str,
    min_interval_sec: float = 0.5,
    max_images: int | None = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> list[str]:
    """선택한 토픽의 이미지를 min_interval_sec 이상 간격으로 샘플링해서
    output_dir에 .jpg로 저장하고, 저장된 파일 경로 리스트를 반환한다.

    시간 기반 샘플링을 쓰는 이유: bag의 프레임레이트가 15/30/60fps로 제각각이라
    "매 N번째 메시지"보다 "최소 몇 초 간격"이 실제 캘리브레이션 데이터셋 품질
    관점(자세 다양성, 설계 문서 7번)에서 훨씬 예측 가능하다 - 거의 똑같은
    프레임을 수십 장 추출하는 걸 방지한다.

    반환된 경로들은 detect_dataset()에 그대로 넘기면 된다.

    이 함수는 이미지 수백~수천 장짜리 bag에서는 수십 초가 걸릴 수 있다 -
    반드시 UI(main_window.py)에서는 QThread 워커(ui/worker.py의
    BagExtractionWorker)를 통해서만 호출해야 한다. GUI 스레드에서 직접
    부르면 그동안 이벤트 루프가 멈춰 OS가 "응답 없음"으로 표시한다
    (실제 사용자 버그: 큰 rosbag을 불러올 때 "python3 is not responding").

    Args:
        progress_callback: (처리한 메시지 수, bag 안의 전체 메시지 수,
            지금까지 저장된 이미지 수)를 주기적으로 알려준다. UI가 진행률/
            상태 텍스트를 갱신하는 용도이며, 계산 로직에는 영향을 주지 않는다.
        cancel_check: 호출할 때마다 True를 반환하면 그 시점까지 저장된
            이미지만 반환하고 즉시 중단한다 (사용자가 진행률 다이얼로그에서
            취소를 눌렀을 때 사용). 지금까지 뽑은 이미지는 버리지 않는다 -
            큰 bag 앞부분만으로도 캘리브레이션을 시작할 수 있게 하기 위함.
    """
    _require_rosbags()
    logger.info(
        "bag에서 이미지 추출 시작: bag=%s, topic=%s, min_interval_sec=%s, max_images=%s",
        bag_path, topic, min_interval_sec, max_images,
    )

    out_dir = _portable_output_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    last_saved_t: float | None = None
    skipped_unsupported = 0
    seen_encodings: set[str] = set()  # 실패 이유를 구체적으로 알려주기 위해 수집
    processed = 0

    # 진행률 표시용 - 이 토픽에 몇 개 메시지가 있는지 미리 알아둔다
    # (list_image_topics에서 이미 구한 값과 같지만, 이 함수만 단독으로
    # 호출해도 동작해야 하므로 다시 조회한다).
    with _open_reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            raise ValueError(f"토픽 '{topic}'을 bag에서 찾을 수 없습니다.")
        msg_type = connections[0].msgtype
        if msg_type not in _IMAGE_MSG_TYPES:
            raise ValueError(f"토픽 '{topic}'은 이미지 타입이 아닙니다 ({msg_type}).")
        total_msgs = sum(c.msgcount for c in connections)

        # 매 메시지마다 progress_callback을 호출하면 콜백 자체(Qt signal emit)의
        # 오버헤드가 수천~수만 번 쌓여 오히려 느려질 수 있어, 20개 메시지에
        # 한 번 또는 최소 0.2초에 한 번 정도로만 알린다.
        # total_msgs가 정상적인 int가 아닌 상황(예: 테스트에서 reader를
        # MagicMock으로 대체한 경우)에도 추출 자체는 그대로 동작해야 하므로
        # 여기서만 방어적으로 처리한다.
        try:
            report_every = max(1, total_msgs // 200) if total_msgs else 20
        except TypeError:
            total_msgs = 0
            report_every = 20

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if cancel_check is not None and cancel_check():
                logger.info(
                    "사용자 취소로 추출 중단 (처리 %d/%d 메시지, 저장 %d장)",
                    processed, total_msgs, len(saved_paths),
                )
                break

            processed += 1
            if progress_callback is not None and processed % report_every == 0:
                progress_callback(processed, total_msgs, len(saved_paths))

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

        if progress_callback is not None:
            progress_callback(processed, total_msgs, len(saved_paths))

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
