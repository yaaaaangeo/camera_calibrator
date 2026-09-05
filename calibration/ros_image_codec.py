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
    # 16비트 Bayer(고해상도/저조도 산업용 카메라에서 흔함) - COLOR_Bayer*2BGR는
    # 8/16비트 입력 둘 다 받아준다.
    "bayer_rggb16": (np.uint16, 1, cv2.COLOR_BayerRG2BGR),
    "bayer_bggr16": (np.uint16, 1, cv2.COLOR_BayerBG2BGR),
    "bayer_gbrg16": (np.uint16, 1, cv2.COLOR_BayerGB2BGR),
    "bayer_grbg16": (np.uint16, 1, cv2.COLOR_BayerGR2BGR),
    "mono16": (np.uint16, 1, None),
    "16UC1": (np.uint16, 1, None),
}

# YUV422류(2바이트/픽셀, 크로마 서브샘플링) - 위 _ENCODING_MAP과 reshape 방식
# 자체가 달라서 별도 표로 분리한다 (decode_raw_image에서 분기 처리).
#
# "yuv422"는 ROS(sensor_msgs) 관례상 역사적으로 UYVY 바이트 순서를 의미한다
# (실제 표준 YUV422와 이름은 같지만 바이트 순서가 다를 수 있어 흔히 혼동되는
# 지점 - v4l2_camera/usb_cam 등 흔한 ROS 카메라 드라이버가 이 인코딩으로
# 발행하는 경우가 많다). "yuv422_yuy2"는 최신 sensor_msgs에 추가된, 진짜
# YUYV(YUY2) 순서의 인코딩. "yuyv"/"uyvy"/"yuy2"는 드라이버가 ROS 표준
# 이름 대신 더 직접적인 이름을 쓰는 경우 대비.
_YUV422_ENCODING_MAP: dict[str, int] = {
    "yuv422": cv2.COLOR_YUV2BGR_UYVY,
    "yuv422_yuy2": cv2.COLOR_YUV2BGR_YUY2,
    "uyvy": cv2.COLOR_YUV2BGR_UYVY,
    "yuyv": cv2.COLOR_YUV2BGR_YUY2,
    "yuy2": cv2.COLOR_YUV2BGR_YUY2,
}

# Jetson CSI/GStreamer 기반 드라이버에서 흔한 4:2:0 semi-planar 형식. ROS 표준
# 상수는 대문자지만 일부 드라이버가 소문자 문자열을 발행해 둘 다 받는다.
_YUV420SP_ENCODING_MAP: dict[str, int] = {
    "NV12": cv2.COLOR_YUV2BGR_NV12,
    "nv12": cv2.COLOR_YUV2BGR_NV12,
    "NV21": cv2.COLOR_YUV2BGR_NV21,
    "nv21": cv2.COLOR_YUV2BGR_NV21,
}


def _message_buffer(data):
    """Return a zero-copy buffer when the ROS sequence type supports it.

    rospy bytes, rclpy ``array('B')`` and rosbags numpy arrays all expose the
    buffer protocol.  A defensive bytes fallback keeps custom message objects
    working without forcing a full-frame copy on the normal live path.
    """
    try:
        view = memoryview(data)
        return view if view.contiguous else bytes(data)
    except TypeError:
        return bytes(data)


def _decode_yuv422(msg, cvt_code: int) -> np.ndarray | None:
    """YUV422류(픽셀당 2바이트, 크로마 서브샘플링 패킹)는 _ENCODING_MAP의
    "채널 N개가 픽셀마다 연속으로 붙어있다" 가정이 안 맞아 별도 경로로 뺐다.
    """
    raw = np.frombuffer(_message_buffer(msg.data), dtype=np.uint8)
    row_bytes = msg.step  # uint8이라 itemsize=1, step은 이미 바이트 단위
    if raw.size < msg.height * row_bytes:
        return None  # 데이터 길이가 헤더와 안 맞음 - 손상된 프레임, 건너뜀

    arr = raw.reshape(msg.height, row_bytes)
    arr = arr[:, : msg.width * 2]  # YUV422은 픽셀당 2바이트
    arr = arr.reshape(msg.height, msg.width, 2)
    try:
        return cv2.cvtColor(arr, cvt_code)
    except cv2.error:
        return None


def _decode_yuv420sp(msg, cvt_code: int) -> np.ndarray | None:
    """NV12/NV21 (Y plane + interleaved UV/VU plane) -> BGR."""
    if msg.height % 2 or msg.width % 2:
        return None
    raw = np.frombuffer(_message_buffer(msg.data), dtype=np.uint8)
    row_bytes = msg.step
    storage_rows = msg.height * 3 // 2
    if raw.size < storage_rows * row_bytes:
        return None
    arr = raw[: storage_rows * row_bytes].reshape(storage_rows, row_bytes)
    arr = np.ascontiguousarray(arr[:, : msg.width])
    try:
        return cv2.cvtColor(arr, cvt_code)
    except cv2.error:
        return None


def decode_raw_image(msg) -> np.ndarray | None:
    """sensor_msgs/Image (duck-typed: .height .width .encoding .step .data) ->
    BGR(또는 그레이스케일) np.ndarray. 지원 안 하는 인코딩이면 None을 반환해서
    호출부가 그 프레임만 건너뛰게 한다 (전체 추출/구독이 죽으면 안 됨).

    msg.data는 rosbags(역직렬화된 numpy 배열)와 rospy(bytes/bytearray) 둘 다
    올 수 있어 np.frombuffer로 통일한다.
    """
    if msg.encoding in _YUV420SP_ENCODING_MAP:
        return _decode_yuv420sp(msg, _YUV420SP_ENCODING_MAP[msg.encoding])
    if msg.encoding in _YUV422_ENCODING_MAP:
        return _decode_yuv422(msg, _YUV422_ENCODING_MAP[msg.encoding])

    spec = _ENCODING_MAP.get(msg.encoding)
    if spec is None:
        return None
    dtype, channels, cvt = spec

    itemsize = np.dtype(dtype).itemsize
    raw = np.frombuffer(_message_buffer(msg.data), dtype=dtype)
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
    buf = np.frombuffer(_message_buffer(msg.data), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img  # 디코딩 실패하면 cv2가 None을 반환 - 그대로 전달해서 건너뛰게 함


def decode_image_message(msg, msg_type: str) -> np.ndarray | None:
    """msg_type 문자열(.../Image 또는 .../CompressedImage)로 알아서 분기."""
    if msg_type.endswith("CompressedImage"):
        return decode_compressed_image(msg)
    return decode_raw_image(msg)
