"""
tests/test_frame_quality.py
================================

설계 문서 6번 - Frame Quality Score. 대화 중 돌렸던 수동 검증 스크립트를
정식 pytest 테스트로 옮긴 것: 선명/적정노출/코너 많은 프레임이 흐리거나
극단 노출인 프레임보다 점수가 높은지, coverage 기여도가 실제로 "저커버리지
영역을 채우는 프레임"에 보너스를 주는지, 재투영 오차 반영 후 점수가
정확한 방향으로 바뀌는지를 확인한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.frame_quality import (
    compute_frame_quality_scores,
    format_frame_quality_table,
    max_possible_corners,
)
from calibration.types import (
    CoverageCell,
    Dataset,
    DetectionResult,
    Frame,
    FrameStatus,
    ImageInfo,
)

W, H = 1920, 1080


def _make_frame(image_id, num_corners, sharpness, brightness, area_ratio, center, tilt=0.0):
    corners = np.array(
        [[center[0] + dx, center[1] + dy] for dx, dy in [(-20, -20), (20, -20), (20, 20), (-20, 20)]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    info = ImageInfo(
        image_id=image_id, path=f"{image_id}.jpg", width=W, height=H,
        sharpness=sharpness, brightness=brightness,
    )
    det = DetectionResult(
        image_id=image_id, success=True, corners=corners,
        object_points=None, ids=None, num_corners=num_corners,
        board_area_ratio=area_ratio, board_center_px=center, board_tilt_deg=tilt,
    )
    return Frame(image_info=info, detection=det, status=FrameStatus.DETECTED)


@pytest.fixture
def quality_test_frames(pattern_config):
    """좋은 프레임 2장 + 나쁜 프레임 3장(흐림/극단노출/코너부족)으로 구성된
    작은 합성 데이터셋. 실제 이미지 없이 순수 값만으로 점수 로직을 검증한다.
    """
    return [
        _make_frame("good_01", num_corners=22, sharpness=400, brightness=130, area_ratio=0.25, center=(960, 540)),
        _make_frame("good_02", num_corners=20, sharpness=350, brightness=120, area_ratio=0.30, center=(300, 200)),
        _make_frame("blurry", num_corners=18, sharpness=15, brightness=125, area_ratio=0.20, center=(960, 540)),
        _make_frame("dark", num_corners=20, sharpness=300, brightness=15, area_ratio=0.22, center=(1700, 900)),
        _make_frame("few_corners", num_corners=6, sharpness=280, brightness=128, area_ratio=0.18, center=(960, 540)),
    ]


def test_max_possible_corners(pattern_config):
    # squares_x=7, squares_y=5 -> 내부 교차점 6*4=24개
    assert max_possible_corners(pattern_config) == 24


def test_quality_score_range(quality_test_frames, pattern_config):
    dataset = Dataset(frames=quality_test_frames)
    compute_frame_quality_scores(dataset, pattern_config, (W, H), use_reprojection=False)

    for f in quality_test_frames:
        assert f.quality is not None, f"{f.image_info.image_id}의 quality가 계산되지 않음"
        assert 0 <= f.quality.overall_score <= 100


def test_good_frame_scores_higher_than_bad(quality_test_frames, pattern_config):
    """선명/적정노출/코너 많은 프레임이 흐리거나 극단 노출인 프레임보다
    점수가 높아야 한다 - Frame Quality Score의 핵심 기대 동작.
    """
    dataset = Dataset(frames=quality_test_frames)
    compute_frame_quality_scores(dataset, pattern_config, (W, H), use_reprojection=False)

    by_id = {f.image_info.image_id: f.quality.overall_score for f in quality_test_frames}

    assert by_id["good_01"] > by_id["blurry"], "선명한 프레임이 흐린 프레임보다 점수가 높아야 함"
    assert by_id["good_01"] > by_id["dark"], "정상 노출이 극단 노출보다 점수가 높아야 함"
    assert by_id["good_01"] > by_id["few_corners"], "코너 많은 프레임이 적은 프레임보다 점수가 높아야 함"


def test_coverage_contribution_rewards_underrepresented_area(quality_test_frames, pattern_config):
    """이미 잘 덮인 중앙 영역만 찍은 프레임보다, 아직 부족한 구석 영역을 찍은
    프레임의 geometric_score가 더 높아야 한다 (중복도의 반대 개념).
    """
    frames = quality_test_frames
    dataset = Dataset(frames=frames)

    # few_corners 프레임이 구석(0,0 근처)을 찍도록 좌표 수정
    corner_frame = next(f for f in frames if f.image_info.image_id == "few_corners")
    corner_frame.detection.corners = np.array(
        [[10, 10], [40, 10], [40, 40], [10, 40]], dtype=np.float32
    ).reshape(-1, 1, 2)
    corner_frame.detection.board_center_px = (25, 25)

    # 4x4 grid에서 중앙 셀들은 이미 채워짐(coverage_score=1.0), 구석은 비어있음(0.0)
    cells = []
    for r in range(4):
        for c in range(4):
            is_center = r in (1, 2) and c in (1, 2)
            cells.append(
                CoverageCell(row=r, col=c, corner_count=10 if is_center else 0,
                             coverage_score=1.0 if is_center else 0.0)
            )
    dataset.coverage_grid = cells

    compute_frame_quality_scores(dataset, pattern_config, (W, H), use_reprojection=False)

    good1 = next(f for f in frames if f.image_info.image_id == "good_01")
    assert corner_frame.quality.geometric_score > good1.quality.geometric_score, (
        "저커버리지(구석) 영역을 찍은 프레임이 고커버리지(중앙)만 찍은 프레임보다 "
        "geometric_score가 높아야 함"
    )


def test_reprojection_error_lowers_score_on_second_pass(quality_test_frames, pattern_config):
    """2단계(재투영 오차 반영) 계산에서, 오차가 큰 프레임의 점수가
    1단계보다 더 낮아져야 한다.
    """
    frames = quality_test_frames
    dataset = Dataset(frames=frames)
    compute_frame_quality_scores(dataset, pattern_config, (W, H), use_reprojection=False)

    blurry = next(f for f in frames if f.image_info.image_id == "blurry")
    score_before = blurry.quality.overall_score

    errors = {"good_01": 0.2, "good_02": 0.3, "blurry": 3.5, "dark": 0.5, "few_corners": 0.4}
    for f in frames:
        f.reprojection_error = errors[f.image_info.image_id]

    compute_frame_quality_scores(dataset, pattern_config, (W, H), use_reprojection=True)
    score_after = blurry.quality.overall_score

    assert score_after < score_before, "재투영 오차가 크게 반영되면 점수가 더 낮아져야 함"


def test_format_frame_quality_table_no_crash(quality_test_frames, pattern_config):
    dataset = Dataset(frames=quality_test_frames)
    compute_frame_quality_scores(dataset, pattern_config, (W, H), use_reprojection=False)
    table = format_frame_quality_table(dataset)
    assert "Score" in table or "Excellent" in table or len(table) > 0
