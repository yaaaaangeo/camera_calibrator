"""
tests/test_detector.py
===========================

설계 문서 17번 Step2 - ChArUco Detection. 검출 성공/실패 상태가 프레임에
정확히 기록되는지, 실패해도(설계 문서: "검출 실패 이미지도 바로 삭제하지
않고 status만 기록") 파이프라인 전체가 죽지 않는지 확인한다.
"""

from __future__ import annotations

import numpy as np
import cv2

from calibration.detector import detect_dataset, summarize_dataset
from calibration.types import FrameStatus


def test_detect_dataset_succeeds_on_real_charuco_images(synthetic_distorted_dataset_dir, pattern_config):
    import glob
    paths = sorted(glob.glob(f"{synthetic_distorted_dataset_dir}/*.jpg"))
    dataset = detect_dataset(paths, pattern_config)

    assert dataset.num_total == len(paths)
    assert dataset.num_detected >= 10
    for f in dataset.frames:
        if f.detection and f.detection.success:
            assert f.status == FrameStatus.DETECTED
            assert f.detection.num_corners > 0
            assert f.detection.corners is not None
            assert f.detection.ids is not None


def test_detect_dataset_handles_blank_image_without_crashing(tmp_path, pattern_config):
    """보드가 전혀 없는 흰 이미지는 검출 실패로 기록돼야지, 파이프라인
    전체가 죽으면 안 된다 (설계 문서: 실패 이미지도 삭제 없이 status만 기록).
    """
    blank = np.full((480, 640, 3), 255, dtype=np.uint8)
    path = str(tmp_path / "blank.jpg")
    cv2.imwrite(path, blank)

    dataset = detect_dataset([path], pattern_config)
    assert dataset.num_total == 1
    assert dataset.num_detected == 0
    frame = dataset.frames[0]
    assert frame.status == FrameStatus.DETECTION_FAILED
    assert frame.image_info.path == path  # 파일 자체는 그대로 참조되어야 함


def test_detect_dataset_mixed_success_and_failure(synthetic_distorted_dataset_dir, tmp_path, pattern_config):
    """일부는 성공, 일부는 실패인 혼합 데이터셋에서도 각 프레임 상태가
    독립적으로 정확히 기록돼야 한다.
    """
    import glob
    good_paths = sorted(glob.glob(f"{synthetic_distorted_dataset_dir}/*.jpg"))[:3]

    blank = np.full((480, 640, 3), 255, dtype=np.uint8)
    blank_path = str(tmp_path / "blank_mixed.jpg")
    cv2.imwrite(blank_path, blank)

    dataset = detect_dataset(good_paths + [blank_path], pattern_config)
    assert dataset.num_total == 4
    statuses = {f.image_info.path: f.status for f in dataset.frames}
    assert statuses[blank_path] == FrameStatus.DETECTION_FAILED
    assert sum(1 for s in statuses.values() if s == FrameStatus.DETECTED) == 3


def test_summarize_dataset_no_crash(synthetic_dataset):
    text = summarize_dataset(synthetic_dataset)
    assert len(text) > 0
    assert "검출" in text or "%" in text
