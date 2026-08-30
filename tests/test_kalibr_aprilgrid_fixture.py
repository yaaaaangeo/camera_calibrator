"""
tests/test_kalibr_aprilgrid_fixture.py
===========================================

P1-C: tests/assets/aprilgrid/kalibr_aprilgrid_6x6.png (Kalibr AprilGrid 규칙을
재현한 fixture - 출처/한계는 tests/assets/aprilgrid/README.md 참고)로 실제
검출을 검증한다.

이 테스트가 PASS해도 UI/README의 "Kalibr [Experimental]" 표시를 자동으로
"Kalibr Compatible"로 바꾸지 않는다 - fixture가 진짜 Kalibr 산출물이 아니라는
한계 때문에(README 참고), 그 승격은 사람이 실제 Kalibr 산출물로도 확인한 뒤
결정해야 한다. 검출 실패 시 threshold를 완화해 억지로 통과시키지 않는다 -
실패하면 원인(태그 계열/간격 해석/dictionary)을 그대로 드러낸다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from calibration.detector import (
    build_aprilgrid_detector,
    build_aprilgrid_dictionary,
    build_detect_fn,
)
from calibration.types import AprilGridVariant, PatternConfig, PatternType

FIXTURE_PATH = Path(__file__).resolve().parent / "assets" / "aprilgrid" / "kalibr_aprilgrid_6x6.png"

SQUARES_X = 6
SQUARES_Y = 6
TAG_SIZE_M = 0.05
TAG_SPACING_RATIO = 0.3
SQUARE_SIZE_M = TAG_SIZE_M * (1.0 + TAG_SPACING_RATIO)


def _kalibr_pattern_config() -> PatternConfig:
    return PatternConfig(
        type=PatternType.APRILGRID,
        squares_x=SQUARES_X,
        squares_y=SQUARES_Y,
        square_size=SQUARE_SIZE_M,
        marker_size=TAG_SIZE_M,
        dictionary="DICT_APRILTAG_36h11",
        aprilgrid_variant=AprilGridVariant.KALIBR,
    )


def _opencv_pattern_config() -> PatternConfig:
    cfg = _kalibr_pattern_config()
    cfg.aprilgrid_variant = AprilGridVariant.OPENCV_APRILTAG3
    return cfg


def _detect(image, pattern: PatternConfig, image_id: str):
    """build_detect_fn()을 통해 실제 파이프라인(calibration/detector.py::
    build_detect_fn)과 동일한 방식으로 detector를 만들어 검출한다 - detect_aprilgrid를
    detector 인자 없이 직접 호출하면 이 OpenCV 빌드에는 없는 레거시
    cv2.aruco.detectMarkers 자유함수 폴백 경로를 타서 실패한다(운영 코드가
    실제로 쓰는 경로가 아님 - build_detect_fn은 항상 ArucoDetector를 만들어 넘긴다).
    """
    return build_detect_fn(pattern)(image, image_id)


@pytest.fixture(scope="module")
def fixture_image():
    assert FIXTURE_PATH.exists(), (
        f"Kalibr AprilGrid fixture missing: {FIXTURE_PATH}. "
        "Run scripts/generate_kalibr_aprilgrid_fixture.py to (re)generate it."
    )
    image = cv2.imread(str(FIXTURE_PATH))
    assert image is not None, f"Failed to load fixture image: {FIXTURE_PATH}"
    return image


def test_kalibr_variant_detects_all_tags(fixture_image):
    pattern = _kalibr_pattern_config()
    result = _detect(fixture_image, pattern, "kalibr_6x6")

    assert result.success, result.failure_reason
    expected_tag_count = SQUARES_X * SQUARES_Y
    expected_corner_count = expected_tag_count * 4
    assert result.num_corners == expected_corner_count

    detected_ids = set(int(i) for i in result.ids.reshape(-1))
    expected_ids = set(range(expected_corner_count))
    assert detected_ids == expected_ids


def test_kalibr_variant_corner_ordering_is_row_major_per_tag(fixture_image):
    """각 태그의 4개 코너 id는 marker_id*4 + (0..3)로 연속이어야 한다
    (calibration/detector.py::detect_aprilgrid의 관례) - object_points와
    image corner 순서가 태그 단위로 정확히 대응하는지 확인.
    """
    pattern = _kalibr_pattern_config()
    result = _detect(fixture_image, pattern, "kalibr_6x6")
    assert result.success, result.failure_reason

    ids = result.ids.reshape(-1)
    for tag_start in range(0, len(ids), 4):
        block = ids[tag_start:tag_start + 4]
        marker_id = block[0] // 4
        assert list(block) == [marker_id * 4 + i for i in range(4)]

    obj = result.object_points.reshape(-1, 3)
    assert obj[:, 2].max() == 0.0  # 평면 타겟, z=0

    # 태그 0(왼쪽 위)은 검출 순서와 무관하게 자기 블록(ids 0~3)의 첫 코너가
    # object space 원점(row=0,col=0)이어야 한다 - detectMarkers는 ID 순서가
    # 아니라 검출된 순서로 반환하므로, ids==0인 블록을 직접 찾는다.
    tag0_start = int(np.where(ids == 0)[0][0])
    tag0_obj = obj[tag0_start:tag0_start + 4]
    assert tag0_obj[0, 0] == pytest.approx(0.0)
    assert tag0_obj[0, 1] == pytest.approx(0.0)

    # 마지막 태그(오른쪽 아래, id = SQUARES_X*SQUARES_Y - 1)는 row/col이 커질수록
    # object space x/y도 커져야 한다 (row-major 배치가 실제로 지켜졌는지 확인).
    last_id = SQUARES_X * SQUARES_Y - 1
    last_start = int(np.where(ids == last_id * 4)[0][0])
    last_obj = obj[last_start:last_start + 4]
    assert last_obj[0, 0] > tag0_obj[0, 0]
    assert last_obj[0, 1] > tag0_obj[0, 1]


def test_opencv_variant_also_detects_the_same_fixture(fixture_image):
    """오늘 시점에 Kalibr variant는 OpenCV variant와 동일한 검출 경로를 쓴다
    (calibration/detector.py::detect_aprilgrid, 로그 문구만 다름) - 이 테스트는
    그 사실을 명시적으로 확인해 둔다. 나중에 실제로 분기하는 로직이 추가되면
    이 assertion이 그 변경을 놓치지 않고 잡아준다.
    """
    kalibr_result = _detect(fixture_image, _kalibr_pattern_config(), "k")
    opencv_result = _detect(fixture_image, _opencv_pattern_config(), "o")

    assert kalibr_result.success and opencv_result.success
    assert kalibr_result.num_corners == opencv_result.num_corners
    assert np.array_equal(kalibr_result.ids, opencv_result.ids)


def test_dictionary_and_detector_build_without_error():
    pattern = _kalibr_pattern_config()
    aruco_dict = build_aprilgrid_dictionary(pattern)
    detector = build_aprilgrid_detector(aruco_dict)
    assert aruco_dict is not None
    # ArucoDetector may be None on very old OpenCV builds (detect_aprilgrid falls
    # back to cv2.aruco.detectMarkers in that case) - just assert it doesn't raise.
    assert detector is None or hasattr(detector, "detectMarkers")
