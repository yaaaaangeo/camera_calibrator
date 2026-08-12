"""
camera_calibrator.calibration.ros_image_codec
==================================================

sensor_msgs/Image, sensor_msgs/CompressedImage -> BGR np.ndarray 디코딩 로직.

rosbag_reader.py(오프라인, rosbags 라이브러리로 역직렬화된 메시지)와
ros_live.py(실시간, 진짜 rospy/rclpy 메시지) 둘 다 이 모듈을 쓴다 - 두 경로가
같은 sensor_msgs 필드 이름(height, width, encoding, step, data / format, data)을
쓰기 때문에 메시지 객체를 duck-typing으로 받아 하나의 구현만 유지한다.
"""

from __future__ import annotations

import cv2
import numpy as np

# sensor_msgs/Image encoding -> (numpy dtype, 채널 수, BGR로 바꾸는 cv2 변환 코드 또는 None)
_ENCODING_MAP: dict[str, tuple[type, int, int | None]] = {
    "mono8": (np.uint8, 1, None),
    "8UC1": (np.uint8, 1, None),
    "bgr8": (np.uint8, 3, None),
    "rgb8": (np.uint8, 3, cv2.COLOR_RGB2BGR),
    "bgra8": (np.uint8, 4, cv2.COLOR_BGRA2BGR),
    "rgba8": (np.uint8, 4, cv2.COLOR_RGBA2BGR),
    "bayer_rggb8": (np.uint8, 1, cv2.COLOR_BayerRG2BGR),
    "bayer_bggr8": (np.uint8, 1, cv2.COLOR_BayerBG2BGR),
    "bayer_gbrg8": (np.uint8, 1, cv2.COLOR_BayerGB2BGR),
    "bayer_grbg8": (np.uint8, 1, cv2.COLOR_BayerGR2BGR),
    "mono16": (np.uint16, 1, None),
    "16UC1": (np.uint16, 1, None),
}


def decode_raw_image(msg) -> np.ndarray | None:
    """sensor_msgs/Image (duck-typed: .height .width .encoding .step .data) ->
    BGR(또는 그레이스케일) np.ndarray. 지원 안 하는 인코딩이면 None을 반환해서
    호출부가 그 프레임만 건너뛰게 한다 (전체 추출/구독이 죽으면 안 됨).

    msg.data는 rosbags(역직렬화된 numpy 배열)와 rospy(bytes/bytearray) 둘 다
    올 수 있어 np.frombuffer로 통일한다.
    """
    spec = _ENCODING_MAP.get(msg.encoding)
    if spec is None:
        return None
    dtype, channels, cvt = spec

    itemsize = np.dtype(dtype).itemsize
    raw = np.frombuffer(bytes(msg.data), dtype=dtype)
    row_elems = msg.step // itemsize
    if raw.size < msg.height * row_elems:
        return None  # 데이터 길이가 헤더와 안 맞음 - 손상된 프레임, 건너뜀

    arr = raw.reshape(msg.height, row_elems)
    arr = arr[:, : msg.width * channels]
    arr = arr.reshape(msg.height, msg.width, channels) if channels > 1 else arr.reshape(msg.height, msg.width)

    img = arr
    if cvt is not None:
        img = cv2.cvtColor(img, cvt)
    if dtype == np.uint16:
        # ChArUco 코너 검출엔 8비트로 충분 - 16비트 깊이 카메라(depth-aligned mono16 등) 대응
        img = (img.astype(np.float32) / 256.0).astype(np.uint8)
    return img


def decode_compressed_image(msg) -> np.ndarray | None:
    """sensor_msgs/CompressedImage (duck-typed: .data) -> BGR np.ndarray."""
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img  # 디코딩 실패하면 cv2가 None을 반환 - 그대로 전달해서 건너뛰게 함


def decode_image_message(msg, msg_type: str) -> np.ndarray | None:
    """msg_type 문자열(.../Image 또는 .../CompressedImage)로 알아서 분기."""
    if msg_type.endswith("CompressedImage"):
        return decode_compressed_image(msg)
    return decode_raw_image(msg)
