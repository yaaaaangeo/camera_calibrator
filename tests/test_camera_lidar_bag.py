"""
tests/test_camera_lidar_bag.py
=================================

Bag topic discovery (image AND PointCloud2) + near-timestamp preview
extraction, exercised against a real (synthetic) ROS1 bag -- same
Writer-based fixture pattern as tests/test_rosbag_reader.py, extended
with a PointCloud2 topic alongside the Image topic.

rosbags 패키지가 없으면 전체 스킵 (선택적 의존성) -- test_rosbag_reader.py와 동일.
"""

from __future__ import annotations

import numpy as np
import pytest

rosbags = pytest.importorskip("rosbags", reason="rosbags가 설치되어 있지 않음 (선택적 의존성)")

from rosbags.rosbag1 import Writer
from rosbags.typesys import get_typestore, Stores
from rosbags.typesys.stores.ros1_noetic import sensor_msgs__msg__Image as RbImage
from rosbags.typesys.stores.ros1_noetic import sensor_msgs__msg__PointCloud2 as RbPointCloud2
from rosbags.typesys.stores.ros1_noetic import sensor_msgs__msg__PointField as RbPointField
from rosbags.typesys.stores.ros1_noetic import std_msgs__msg__Header as RbHeader
from rosbags.typesys.stores.ros1_noetic import builtin_interfaces__msg__Time as RbTime

from calibration.rosbag_reader import (
    TopicInfo,
    extract_image_near_timestamp,
    extract_pointcloud_near_timestamp,
    iterate_images,
    list_image_topics,
    list_pointcloud_topics,
    read_bag_duration,
)

_POINT_STEP = 16  # x,y,z,intensity, all float32
_POINT_FIELDS = [
    RbPointField(name="x", offset=0, datatype=7, count=1),
    RbPointField(name="y", offset=4, datatype=7, count=1),
    RbPointField(name="z", offset=8, datatype=7, count=1),
    RbPointField(name="intensity", offset=12, datatype=7, count=1),
]
_N_FRAMES = 10
_FRAME_INTERVAL_SEC = 0.5
_N_POINTS_PER_FRAME = 40
_BASE_T_NS = 1_700_000_000_000_000_000  # arbitrary but realistic epoch offset


@pytest.fixture
def synthetic_combo_bag_path(tmp_path):
    """A ROS1 bag with both an Image topic (/camera/image_raw) and a
    PointCloud2 topic (/lidar/points), 10 frames each at 0.5s intervals
    (5s total). Each PointCloud2 frame's intensity field is set to the
    frame index, so extraction correctness can be checked directly."""
    ts = get_typestore(Stores.ROS1_NOETIC)
    bag_path = str(tmp_path / "combo.bag")
    w, h = 8, 8

    with Writer(bag_path) as writer:
        img_conn = writer.add_connection("/camera/image_raw", RbImage.__msgtype__, typestore=ts)
        pc_conn = writer.add_connection("/lidar/points", RbPointCloud2.__msgtype__, typestore=ts)

        for i in range(_N_FRAMES):
            t_ns = _BASE_T_NS + int(i * _FRAME_INTERVAL_SEC * 1e9)
            header = RbHeader(seq=i, stamp=RbTime(sec=t_ns // 10**9, nanosec=t_ns % 10**9), frame_id="cam")

            img = np.full((h, w, 3), (i * 10) % 256, dtype=np.uint8)
            img_msg = RbImage(
                header=header, height=h, width=w, encoding="bgr8", is_bigendian=0,
                step=w * 3, data=img.reshape(-1),
            )
            writer.write(img_conn, t_ns, ts.serialize_ros1(img_msg, RbImage.__msgtype__))

            pts = np.random.default_rng(i).random((_N_POINTS_PER_FRAME, 4)).astype(np.float32)
            pts[:, 3] = float(i)  # intensity encodes the frame index for verification
            pc_msg = RbPointCloud2(
                header=RbHeader(seq=i, stamp=RbTime(sec=t_ns // 10**9, nanosec=t_ns % 10**9), frame_id="lidar"),
                height=1, width=_N_POINTS_PER_FRAME, fields=_POINT_FIELDS, is_bigendian=False,
                point_step=_POINT_STEP, row_step=_POINT_STEP * _N_POINTS_PER_FRAME,
                data=np.frombuffer(pts.tobytes(), dtype=np.uint8), is_dense=True,
            )
            writer.write(pc_conn, t_ns + 1_000_000, ts.serialize_ros1(pc_msg, RbPointCloud2.__msgtype__))

    return bag_path


def test_list_image_topics_unaffected_by_pointcloud_topic(synthetic_combo_bag_path):
    """Existing behavior must not change just because a PointCloud2 topic
    now also exists in the bag."""
    topics = list_image_topics(synthetic_combo_bag_path)
    assert len(topics) == 1
    assert topics[0].name == "/camera/image_raw"
    assert topics[0].count == _N_FRAMES


def test_list_pointcloud_topics(synthetic_combo_bag_path):
    topics = list_pointcloud_topics(synthetic_combo_bag_path)
    assert topics == [TopicInfo(name="/lidar/points", msg_type="sensor_msgs/msg/PointCloud2", count=_N_FRAMES)]


def test_read_bag_duration(synthetic_combo_bag_path):
    duration = read_bag_duration(synthetic_combo_bag_path)
    # 10 frames at 0.5s intervals span 4.5s; allow the small PointCloud2 timestamp offset slack.
    assert 4.5 <= duration <= 4.6


def test_extract_image_near_timestamp(synthetic_combo_bag_path):
    # t=2.3s is closest to frame index 5 (t=2.5s among 0,0.5,...,4.5).
    img, timestamp_sec, frame_id = extract_image_near_timestamp(
        synthetic_combo_bag_path, "/camera/image_raw", 2.3
    )
    assert img.shape == (8, 8, 3)
    assert int(img[0, 0, 0]) == (5 * 10) % 256
    assert frame_id == "cam"
    assert timestamp_sec == pytest.approx(_BASE_T_NS / 1e9 + 2.5, abs=1e-6)


def test_extract_pointcloud_near_timestamp(synthetic_combo_bag_path):
    points, timestamp_sec, frame_id = extract_pointcloud_near_timestamp(
        synthetic_combo_bag_path, "/lidar/points", 2.3
    )
    assert points.shape == (_N_POINTS_PER_FRAME, 4)
    assert np.all(points[:, 3] == 5.0)  # frame index 5's intensity marker
    assert frame_id == "lidar"


def test_extract_near_timestamp_at_bag_start_and_end(synthetic_combo_bag_path):
    img_start, t_start, _ = extract_image_near_timestamp(synthetic_combo_bag_path, "/camera/image_raw", 0.0)
    assert int(img_start[0, 0, 0]) == 0
    img_end, t_end, _ = extract_image_near_timestamp(synthetic_combo_bag_path, "/camera/image_raw", 100.0)
    assert int(img_end[0, 0, 0]) == (9 * 10) % 256  # clamps to the last available frame


def test_extract_near_timestamp_unknown_topic_raises(synthetic_combo_bag_path):
    with pytest.raises(ValueError, match="찾을 수 없습니다"):
        extract_image_near_timestamp(synthetic_combo_bag_path, "/nonexistent/topic", 1.0)


def test_iterate_images_yields_every_frame_in_order_with_bag_relative_timestamps(synthetic_combo_bag_path):
    frames = list(iterate_images(synthetic_combo_bag_path, "/camera/image_raw"))
    assert len(frames) == _N_FRAMES
    for i, (img, t_sec, frame_id) in enumerate(frames):
        assert img.shape == (8, 8, 3)
        assert int(img[0, 0, 0]) == (i * 10) % 256
        assert t_sec == pytest.approx(i * _FRAME_INTERVAL_SEC, abs=1e-6)
        assert frame_id == "cam"


def test_iterate_images_is_reusable_as_a_generator_function(synthetic_combo_bag_path):
    """build_scene_candidates calls frames_factory() twice (detection pass,
    then representative-image pass) -- iterate_images must support being
    called again from scratch, not just iterated once."""
    first_pass = [t for _img, t, _fid in iterate_images(synthetic_combo_bag_path, "/camera/image_raw")]
    second_pass = [t for _img, t, _fid in iterate_images(synthetic_combo_bag_path, "/camera/image_raw")]
    assert first_pass == second_pass
    assert len(first_pass) == _N_FRAMES


def test_iterate_images_unknown_topic_raises(synthetic_combo_bag_path):
    with pytest.raises(ValueError, match="찾을 수 없습니다"):
        list(iterate_images(synthetic_combo_bag_path, "/nonexistent/topic"))
