"""
tests/test_rosbag_reader.py
================================

rosbag(.bag)에서 이미지를 실제로 추출해 기존 detect_dataset() 파이프라인에
바로 들어가는지 end-to-end로 검증한다.

rosbags 패키지가 설치 안 된 환경(선택적 의존성)에서는 전체 파일을 스킵한다 -
CI에서 rosbags를 설치 안 했다면 이 테스트들만 건너뛰고 나머지는 정상 실행된다.
"""

from __future__ import annotations

import numpy as np
import pytest

rosbags = pytest.importorskip("rosbags", reason="rosbags가 설치되어 있지 않음 (선택적 의존성)")

from rosbags.rosbag1 import Writer
from rosbags.typesys import get_typestore, Stores
from rosbags.typesys.stores.ros1_noetic import sensor_msgs__msg__Image as RbImage
from rosbags.typesys.stores.ros1_noetic import std_msgs__msg__Header as RbHeader
from rosbags.typesys.stores.ros1_noetic import builtin_interfaces__msg__Time as RbTime

from calibration.rosbag_reader import list_image_topics, extract_images_from_bag


@pytest.fixture
def synthetic_bag_path(tmp_path):
    """10fps, 3초 분량(30프레임)의 bgr8 Image 토픽을 담은 ROS1 bag."""
    ts = get_typestore(Stores.ROS1_NOETIC)
    bag_path = str(tmp_path / "synthetic.bag")
    w, h = 64, 48

    with Writer(bag_path) as writer:
        conn = writer.add_connection("/camera/image_raw", RbImage.__msgtype__, typestore=ts)
        for i in range(30):
            img = np.full((h, w, 3), i * 5 % 256, dtype=np.uint8)
            t_ns = int(i * 0.1 * 1e9)
            msg = RbImage(
                header=RbHeader(seq=i, stamp=RbTime(sec=t_ns // 10**9, nanosec=t_ns % 10**9), frame_id="c"),
                height=h, width=w, encoding="bgr8", is_bigendian=0, step=w * 3,
                data=img.reshape(-1),
            )
            writer.write(conn, t_ns, ts.serialize_ros1(msg, RbImage.__msgtype__))

    return bag_path


def test_list_image_topics(synthetic_bag_path):
    topics = list_image_topics(synthetic_bag_path)
    assert len(topics) == 1
    assert topics[0].name == "/camera/image_raw"
    assert topics[0].count == 30


def test_extract_with_time_based_sampling(synthetic_bag_path, tmp_path):
    """0.5초 간격 샘플링 -> 3초 분량이니 약 6~7장이 나와야 한다 (전부 다 뽑으면 30장)."""
    out_dir = str(tmp_path / "extracted")
    paths = extract_images_from_bag(synthetic_bag_path, "/camera/image_raw", out_dir, min_interval_sec=0.5)
    assert 5 <= len(paths) <= 8, f"샘플링 개수가 예상 범위를 벗어남: {len(paths)}"
    for p in paths:
        import os
        assert os.path.exists(p)


def test_extracted_images_feed_into_existing_pipeline(synthetic_bag_path, tmp_path, pattern_config):
    """bag에서 추출한 이미지가 detect_dataset()에 아무 수정 없이 그대로
    들어가서 검출까지 되는지 - bag 리더가 파이프라인과 실제로 통합되는지 확인.

    이 테스트는 실제 ChArUco 보드가 담긴 bag이 아니라 일반 합성 프레임이므로
    "검출 성공"까지는 검증하지 않고, "예외 없이 파이프라인에 흘러들어가는지"만
    확인한다 (ChArUco 검출 자체는 test_pipeline_integration.py에서 별도 검증).
    """
    from calibration.detector import detect_dataset

    out_dir = str(tmp_path / "extracted2")
    paths = extract_images_from_bag(synthetic_bag_path, "/camera/image_raw", out_dir, min_interval_sec=1.0)
    assert len(paths) > 0

    dataset = detect_dataset(paths, pattern_config)
    assert dataset.num_total == len(paths)


def test_nonexistent_topic_raises_clear_error(synthetic_bag_path, tmp_path):
    with pytest.raises(ValueError, match="찾을 수 없습니다"):
        extract_images_from_bag(synthetic_bag_path, "/nonexistent/topic", str(tmp_path / "out"))


def test_max_images_limit(synthetic_bag_path, tmp_path):
    out_dir = str(tmp_path / "limited")
    paths = extract_images_from_bag(
        synthetic_bag_path, "/camera/image_raw", out_dir, min_interval_sec=0.05, max_images=3
    )
    assert len(paths) == 3


# ---------------------------------------------------------------------------
# progress_callback / cancel_check - BagExtractionWorker(ui/worker.py)가
# GUI 스레드를 막지 않고 진행률을 보여주기 위해 필요한 훅.
# 실제 사용자 버그: 이 훅이 없어서 main_window.py가 extract_images_from_bag()을
# GUI 스레드에서 동기 호출했고, 큰 bag에서 "python3 is not responding"이 떴었다.
# ---------------------------------------------------------------------------


def test_progress_callback_reports_total_and_reaches_full_count(synthetic_bag_path, tmp_path):
    calls = []
    out_dir = str(tmp_path / "progress")
    extract_images_from_bag(
        synthetic_bag_path, "/camera/image_raw", out_dir, min_interval_sec=0.05,
        progress_callback=lambda done, total, saved: calls.append((done, total, saved)),
    )
    assert calls, "progress_callback이 한 번도 호출되지 않음"
    # bag 전체 메시지 수(30개)를 모든 콜백이 동일하게 보고해야 한다.
    assert all(total == 30 for _, total, _ in calls)
    # 마지막 콜백은 전체 메시지를 다 처리했다는 뜻이어야 한다 (진행률 100%).
    assert calls[-1][0] == 30


def test_cancel_check_stops_early_but_keeps_already_saved_images(synthetic_bag_path, tmp_path):
    """취소해도 그 시점까지 뽑힌 이미지는 버리지 않아야 한다 - 큰 bag의
    앞부분만으로도 캘리브레이션을 시작할 수 있게 하기 위함."""
    out_dir = str(tmp_path / "cancelled")
    seen = {"n": 0}

    def cancel_after_a_few():
        seen["n"] += 1
        return seen["n"] > 3

    paths = extract_images_from_bag(
        synthetic_bag_path, "/camera/image_raw", out_dir, min_interval_sec=0.05,
        cancel_check=cancel_after_a_few,
    )
    assert 0 < len(paths) < 30
    for p in paths:
        import os
        assert os.path.exists(p)


def test_no_callback_or_cancel_check_behaves_exactly_as_before(synthetic_bag_path, tmp_path):
    """새 옵션 인자들은 기본값(None)일 때 기존 동작을 한 글자도 안 바꿔야 한다."""
    out_dir = str(tmp_path / "unchanged")
    paths = extract_images_from_bag(synthetic_bag_path, "/camera/image_raw", out_dir, min_interval_sec=0.5)
    assert 5 <= len(paths) <= 8
