"""
tests/test_ros_image_codec.py
==================================

sensor_msgs/Image, CompressedImage 디코딩 - 인코딩별 엣지 케이스 전부 커버.
rosbag_reader.py와 ros_live.py가 공유하는 핵심 로직이라 가장 꼼꼼하게
테스트한다. ROS가 설치되어 있지 않아도(순수 numpy/cv2만 쓰므로) 항상 실행됨.
"""

from __future__ import annotations

import numpy as np
import cv2

from calibration.ros_image_codec import decode_raw_image, decode_compressed_image, decode_image_message


class _FakeImageMsg:
    """sensor_msgs/Image와 같은 필드(height, width, encoding, step, data)를
    가진 가짜 메시지. rosbags든 rospy든 필드 이름이 같아서 이걸로 충분하다.
    """
    def __init__(self, height, width, encoding, step, data):
        self.height, self.width, self.encoding, self.step, self.data = height, width, encoding, step, data


def test_mono8_roundtrip():
    gray = (np.random.rand(10, 15) * 255).astype(np.uint8)
    msg = _FakeImageMsg(10, 15, "mono8", 15, gray.reshape(-1))
    out = decode_raw_image(msg)
    assert out.shape == (10, 15)
    assert np.array_equal(out, gray)


def test_bgra8_drops_alpha_to_bgr():
    bgra = np.random.randint(0, 255, (8, 12, 4), dtype=np.uint8)
    msg = _FakeImageMsg(8, 12, "bgra8", 12 * 4, bgra.reshape(-1))
    out = decode_raw_image(msg)
    assert out.shape == (8, 12, 3)


def test_bayer_rggb8_demosaics_to_3channel():
    bayer = (np.random.rand(20, 20) * 255).astype(np.uint8)
    msg = _FakeImageMsg(20, 20, "bayer_rggb8", 20, bayer.reshape(-1))
    out = decode_raw_image(msg)
    assert out.shape == (20, 20, 3)


def test_padded_step_row_alignment_handled_correctly():
    """일부 카메라 드라이버는 step(row stride)에 정렬 패딩을 넣는다 -
    width*channels보다 step이 큰 경우를 정확히 잘라내야 한다.
    """
    w, h = 13, 5
    real = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    padded_step = w * 3 + 3  # 3바이트 패딩
    padded = np.zeros((h, padded_step), dtype=np.uint8)
    padded[:, : w * 3] = real.reshape(h, w * 3)
    msg = _FakeImageMsg(h, w, "bgr8", padded_step, padded.reshape(-1))

    out = decode_raw_image(msg)
    assert out.shape == (h, w, 3)
    assert np.array_equal(out, real), "패딩된 step(row alignment) 처리가 잘못됨"


def test_mono16_downcasts_to_8bit():
    depth = (np.random.rand(10, 10) * 65535).astype(np.uint16)
    msg = _FakeImageMsg(10, 10, "mono16", 20, depth.reshape(-1))
    out = decode_raw_image(msg)
    assert out.shape == (10, 10)
    assert out.dtype == np.uint8


def test_unsupported_encoding_returns_none_not_crash():
    msg = _FakeImageMsg(5, 5, "unknown_weird_encoding", 5, np.zeros(25, dtype=np.uint8))
    assert decode_raw_image(msg) is None


def test_corrupted_data_length_returns_none_not_crash():
    """데이터 길이가 height*step과 안 맞으면(손상된 프레임) 크래시 대신 None."""
    msg = _FakeImageMsg(100, 100, "bgr8", 300, np.zeros(10, dtype=np.uint8))  # 데이터 턱없이 부족
    assert decode_raw_image(msg) is None


def test_compressed_image_png_roundtrip():
    img = np.random.randint(0, 255, (30, 40, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".png", img)
    msg = type("M", (), {"data": enc})()
    out = decode_compressed_image(msg)
    assert out.shape == (30, 40, 3)


def test_compressed_image_jpeg_roundtrip():
    img = np.full((20, 20, 3), 128, dtype=np.uint8)
    ok, enc = cv2.imencode(".jpg", img)
    msg = type("M", (), {"data": enc})()
    out = decode_compressed_image(msg)
    assert out.shape == (20, 20, 3)


def test_decode_image_message_dispatches_by_type():
    img = np.full((10, 10, 3), 50, dtype=np.uint8)
    ok, enc = cv2.imencode(".png", img)
    compressed_msg = type("M", (), {"data": enc})()
    out = decode_image_message(compressed_msg, "sensor_msgs/msg/CompressedImage")
    assert out.shape == (10, 10, 3)

    raw = np.full((5, 5, 3), 1, dtype=np.uint8)
    raw_msg = _FakeImageMsg(5, 5, "bgr8", 15, raw.reshape(-1))
    out2 = decode_image_message(raw_msg, "sensor_msgs/msg/Image")
    assert out2.shape == (5, 5, 3)
