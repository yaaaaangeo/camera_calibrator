"""
tests/test_camera_lidar_bag_topic_scan.py
=============================================

Tests for calibration.rosbag_reader.scan_bag_directory_topics and its
default-topic/CameraInfo-recommendation heuristics -- the Targetless
Bootstrap "pick an Input Bag Directory and topics auto-populate" feature.

Real (synthetic) ROS1 bags via rosbags.rosbag1.Writer, same Writer-based
fixture pattern as tests/test_camera_lidar_bag.py. Classification is by ROS
MESSAGE TYPE, never a topic-name substring guess -- these fixtures
deliberately include a std_msgs/String topic (a stand-in for something
like /tf) to prove type filtering, not name filtering, drives
classification.
"""

from __future__ import annotations

import numpy as np
import pytest

rosbags = pytest.importorskip("rosbags", reason="rosbags가 설치되어 있지 않음 (선택적 의존성)")

from rosbags.rosbag1 import Writer
from rosbags.typesys import get_typestore, Stores
from rosbags.typesys.stores.ros1_noetic import sensor_msgs__msg__CameraInfo as RbCameraInfo
from rosbags.typesys.stores.ros1_noetic import sensor_msgs__msg__Image as RbImage
from rosbags.typesys.stores.ros1_noetic import sensor_msgs__msg__PointCloud2 as RbPointCloud2
from rosbags.typesys.stores.ros1_noetic import sensor_msgs__msg__PointField as RbPointField
from rosbags.typesys.stores.ros1_noetic import sensor_msgs__msg__RegionOfInterest as RbRoi
from rosbags.typesys.stores.ros1_noetic import std_msgs__msg__Header as RbHeader
from rosbags.typesys.stores.ros1_noetic import std_msgs__msg__String as RbString
from rosbags.typesys.stores.ros1_noetic import builtin_interfaces__msg__Time as RbTime

from calibration.rosbag_reader import (
    IMAGE_TOPIC_KEYWORD_PRIORITY,
    BagTopicCoverage,
    pick_default_topic,
    recommend_camera_info_topic,
    scan_bag_directory_topics,
)

_POINT_STEP = 16
_POINT_FIELDS = [
    RbPointField(name="x", offset=0, datatype=7, count=1),
    RbPointField(name="y", offset=4, datatype=7, count=1),
    RbPointField(name="z", offset=8, datatype=7, count=1),
    RbPointField(name="intensity", offset=12, datatype=7, count=1),
]
_BASE_T_NS = 1_700_000_000_000_000_000


def _header(i: int, frame_id: str) -> RbHeader:
    t_ns = _BASE_T_NS + i * 10**8
    return RbHeader(seq=i, stamp=RbTime(sec=t_ns // 10**9, nanosec=t_ns % 10**9), frame_id=frame_id)


def _write_image(writer, ts, conn, i: int, w=4, h=4) -> None:
    img = np.full((h, w, 3), i % 256, dtype=np.uint8)
    msg = RbImage(
        header=_header(i, "cam"), height=h, width=w, encoding="bgr8", is_bigendian=0,
        step=w * 3, data=img.reshape(-1),
    )
    writer.write(conn, _BASE_T_NS + i * 10**8, ts.serialize_ros1(msg, RbImage.__msgtype__))


def _write_pointcloud(writer, ts, conn, i: int, n_points=20) -> None:
    pts = np.random.default_rng(i).random((n_points, 4)).astype(np.float32)
    msg = RbPointCloud2(
        header=_header(i, "lidar"), height=1, width=n_points, fields=_POINT_FIELDS, is_bigendian=False,
        point_step=_POINT_STEP, row_step=_POINT_STEP * n_points,
        data=np.frombuffer(pts.tobytes(), dtype=np.uint8), is_dense=True,
    )
    writer.write(conn, _BASE_T_NS + i * 10**8 + 1_000_000, ts.serialize_ros1(msg, RbPointCloud2.__msgtype__))


def _write_camera_info(writer, ts, conn, i: int, w=4, h=4) -> None:
    msg = RbCameraInfo(
        header=_header(i, "cam"), height=h, width=w, distortion_model="plumb_bob",
        D=np.zeros(5), K=np.eye(3).reshape(-1), R=np.eye(3).reshape(-1), P=np.zeros((3, 4)).reshape(-1),
        binning_x=0, binning_y=0, roi=RbRoi(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False),
    )
    writer.write(conn, _BASE_T_NS + i * 10**8 + 2_000_000, ts.serialize_ros1(msg, RbCameraInfo.__msgtype__))


def _write_string(writer, ts, conn, i: int) -> None:
    msg = RbString(data=f"msg{i}")
    writer.write(conn, _BASE_T_NS + i * 10**8 + 3_000_000, ts.serialize_ros1(msg, RbString.__msgtype__))


@pytest.fixture
def calibration_bags_dir(tmp_path):
    """3 bags (scene01/02/03), each with /camera/image_raw (Image),
    /ouster/points (PointCloud2), /camera/camera_info (CameraInfo), and a
    /robot/status (std_msgs/String -- wrong type, must never surface in any
    of the 3 candidate lists, standing in for something like /tf). scene03
    additionally has /debug/image (Image) -- a partial-coverage (1/3) topic."""
    ts = get_typestore(Stores.ROS1_NOETIC)
    bags_dir = tmp_path / "calibration_bags"
    bags_dir.mkdir()

    for scene_idx in range(3):
        bag_path = str(bags_dir / f"scene{scene_idx + 1:02d}.bag")
        with Writer(bag_path) as writer:
            img_conn = writer.add_connection("/camera/image_raw", RbImage.__msgtype__, typestore=ts)
            pc_conn = writer.add_connection("/ouster/points", RbPointCloud2.__msgtype__, typestore=ts)
            ci_conn = writer.add_connection("/camera/camera_info", RbCameraInfo.__msgtype__, typestore=ts)
            str_conn = writer.add_connection("/robot/status", RbString.__msgtype__, typestore=ts)

            for i in range(3):
                _write_image(writer, ts, img_conn, i)
                _write_pointcloud(writer, ts, pc_conn, i)
                _write_camera_info(writer, ts, ci_conn, i)
                _write_string(writer, ts, str_conn, i)

            if scene_idx == 2:
                debug_conn = writer.add_connection("/debug/image", RbImage.__msgtype__, typestore=ts)
                for i in range(3):
                    _write_image(writer, ts, debug_conn, i)

    return str(bags_dir)


# ---------------------------------------------------------------------------
# Test 1 -- message type filtering (never a topic-name guess)
# ---------------------------------------------------------------------------

def test_message_type_filtering_excludes_wrong_types(calibration_bags_dir):
    result = scan_bag_directory_topics(calibration_bags_dir)

    image_names = {c.name for c in result.image_topics}
    pointcloud_names = {c.name for c in result.pointcloud_topics}
    camera_info_names = {c.name for c in result.camera_info_topics}

    assert "/camera/image_raw" in image_names
    assert "/ouster/points" in pointcloud_names
    assert "/camera/camera_info" in camera_info_names

    assert "/robot/status" not in image_names
    assert "/robot/status" not in pointcloud_names
    assert "/robot/status" not in camera_info_names
    assert "/ouster/points" not in image_names
    assert "/camera/camera_info" not in image_names
    assert "/camera/image_raw" not in pointcloud_names


# ---------------------------------------------------------------------------
# Test 2 -- common topic across all bags
# ---------------------------------------------------------------------------

def test_common_topic_has_full_coverage(calibration_bags_dir):
    result = scan_bag_directory_topics(calibration_bags_dir)
    assert result.bag_count == 3

    image_by_name = {c.name: c for c in result.image_topics}
    assert image_by_name["/camera/image_raw"].bag_count == 3
    assert image_by_name["/camera/image_raw"].total_bags == 3

    pc_by_name = {c.name: c for c in result.pointcloud_topics}
    assert pc_by_name["/ouster/points"].bag_count == 3


# ---------------------------------------------------------------------------
# Test 3 -- partial-coverage topic sorted after common topics
# ---------------------------------------------------------------------------

def test_partial_coverage_topic_sorted_after_common_topics(calibration_bags_dir):
    result = scan_bag_directory_topics(calibration_bags_dir)

    image_by_name = {c.name: c for c in result.image_topics}
    assert image_by_name["/debug/image"].bag_count == 1
    assert image_by_name["/debug/image"].total_bags == 3

    names_in_order = [c.name for c in result.image_topics]
    assert names_in_order.index("/camera/image_raw") < names_in_order.index("/debug/image")


# ---------------------------------------------------------------------------
# Test 4 -- Image/CameraInfo auto pair recommendation (namespace matching)
# ---------------------------------------------------------------------------

def test_recommend_camera_info_topic_matches_namespace():
    candidates = [
        BagTopicCoverage(name="/camera/front/camera_info", msg_type="sensor_msgs/msg/CameraInfo", bag_count=3, total_bags=3),
        BagTopicCoverage(name="/camera/rear/camera_info", msg_type="sensor_msgs/msg/CameraInfo", bag_count=3, total_bags=3),
    ]
    assert recommend_camera_info_topic("/camera/front/image_raw", candidates) == "/camera/front/camera_info"


def test_recommend_camera_info_topic_stereo_namespace():
    candidates = [
        BagTopicCoverage(name="/stereo/left/camera_info", msg_type="sensor_msgs/msg/CameraInfo", bag_count=1, total_bags=1),
    ]
    assert recommend_camera_info_topic("/stereo/left/image_raw", candidates) == "/stereo/left/camera_info"


def test_recommend_camera_info_topic_none_when_no_overlap():
    candidates = [
        BagTopicCoverage(name="/totally/unrelated/camera_info", msg_type="sensor_msgs/msg/CameraInfo", bag_count=1, total_bags=1),
    ]
    assert recommend_camera_info_topic("/camera/front/image_raw", candidates) is None


def test_recommend_camera_info_topic_none_when_no_candidates():
    assert recommend_camera_info_topic("/camera/front/image_raw", []) is None


# ---------------------------------------------------------------------------
# Test 5 -- changing Image topic updates the CameraInfo recommendation
# ---------------------------------------------------------------------------

def test_recommend_camera_info_topic_updates_when_image_changes():
    candidates = [
        BagTopicCoverage(name="/camera/front/camera_info", msg_type="sensor_msgs/msg/CameraInfo", bag_count=1, total_bags=1),
        BagTopicCoverage(name="/camera/rear/camera_info", msg_type="sensor_msgs/msg/CameraInfo", bag_count=1, total_bags=1),
    ]
    assert recommend_camera_info_topic("/camera/front/image_raw", candidates) == "/camera/front/camera_info"
    assert recommend_camera_info_topic("/camera/rear/image_raw", candidates) == "/camera/rear/camera_info"


# ---------------------------------------------------------------------------
# Test 6 -- single-candidate auto-selection; keyword priority is a
# tiebreaker only, never a filter
# ---------------------------------------------------------------------------

def test_pick_default_topic_single_candidate():
    candidates = [BagTopicCoverage(name="/only/topic", msg_type="sensor_msgs/msg/Image", bag_count=1, total_bags=1)]
    assert pick_default_topic(candidates) == "/only/topic"


def test_pick_default_topic_no_candidates():
    assert pick_default_topic([]) is None


def test_pick_default_topic_keyword_priority_never_filters():
    candidates = [
        BagTopicCoverage(name="/weird/topic/name", msg_type="sensor_msgs/msg/Image", bag_count=2, total_bags=3),
        BagTopicCoverage(name="/camera/image_raw", msg_type="sensor_msgs/msg/Image", bag_count=3, total_bags=3),
    ]
    default = pick_default_topic(candidates, IMAGE_TOPIC_KEYWORD_PRIORITY)
    assert default == "/camera/image_raw"  # matches "image_raw" keyword
    # Both candidates remain in the list -- pick_default_topic only chose a
    # DEFAULT, it never removes anything from consideration.
    assert len(candidates) == 2


# ---------------------------------------------------------------------------
# Test 7 -- no CameraInfo topics found, Bootstrap must still be usable
# ---------------------------------------------------------------------------

def test_scan_result_with_zero_camera_info_topics(tmp_path):
    ts = get_typestore(Stores.ROS1_NOETIC)
    bags_dir = tmp_path / "no_camera_info_bags"
    bags_dir.mkdir()
    bag_path = str(bags_dir / "scene01.bag")
    with Writer(bag_path) as writer:
        img_conn = writer.add_connection("/camera/image_raw", RbImage.__msgtype__, typestore=ts)
        pc_conn = writer.add_connection("/ouster/points", RbPointCloud2.__msgtype__, typestore=ts)
        _write_image(writer, ts, img_conn, 0)
        _write_pointcloud(writer, ts, pc_conn, 0)

    result = scan_bag_directory_topics(str(bags_dir))
    assert result.camera_info_topics == []
    assert len(result.image_topics) == 1
    assert len(result.pointcloud_topics) == 1


# ---------------------------------------------------------------------------
# Test 8 -- directory changed: scanning a different directory never reuses
# the previous directory's topics
# ---------------------------------------------------------------------------

def test_scanning_a_different_directory_does_not_reuse_previous_results(calibration_bags_dir, tmp_path):
    other_dir = tmp_path / "other_bags"
    other_dir.mkdir()
    ts = get_typestore(Stores.ROS1_NOETIC)
    with Writer(str(other_dir / "sceneA.bag")) as writer:
        conn = writer.add_connection("/completely/different/image", RbImage.__msgtype__, typestore=ts)
        _write_image(writer, ts, conn, 0)

    result_a = scan_bag_directory_topics(calibration_bags_dir)
    result_b = scan_bag_directory_topics(str(other_dir))

    names_a = {c.name for c in result_a.image_topics}
    names_b = {c.name for c in result_b.image_topics}
    assert names_a.isdisjoint(names_b)
    assert "/completely/different/image" in names_b
    assert "/completely/different/image" not in names_a


def test_scan_raises_when_directory_has_no_bags(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        scan_bag_directory_topics(str(empty_dir))
