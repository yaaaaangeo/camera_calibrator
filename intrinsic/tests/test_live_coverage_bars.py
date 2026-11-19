"""
tests/test_live_coverage_bars.py
====================================

실시간 캡처용 X/Y/Size/Skew 바(calibration.quality.compute_live_coverage_bars)
검증. 실제 이미지 없이 DetectionResult를 직접 구성해서 순수 로직만 테스트한다.
"""

from __future__ import annotations

from calibration.quality import compute_live_coverage_bars
from calibration.types import DetectionResult, Frame, ImageInfo


def _frame(image_id: str, cx: float, cy: float, area_ratio: float, tilt_deg: float) -> Frame:
    info = ImageInfo(image_id=image_id, path=f"/tmp/{image_id}.jpg", width=640, height=480)
    det = DetectionResult(
        image_id=image_id,
        success=True,
        num_corners=20,
        board_center_px=(cx, cy),
        board_area_ratio=area_ratio,
        board_tilt_deg=tilt_deg,
    )
    return Frame(image_info=info, detection=det)


def test_empty_frames_gives_zero_bars():
    bars = compute_live_coverage_bars([], image_size=(640, 480))
    assert bars.x_coverage == 0.0
    assert bars.y_coverage == 0.0
    assert bars.size_coverage == 0.0
    assert bars.skew_coverage == 0.0


def test_single_frame_gives_zero_bars():
    """표준편차는 표본이 2개 이상이어야 의미가 있다 - 1장으로는 전부 0."""
    frames = [_frame("f0", 320, 240, 0.3, 5.0)]
    bars = compute_live_coverage_bars(frames, image_size=(640, 480))
    assert bars.x_coverage == 0.0
    assert bars.y_coverage == 0.0


def test_board_always_centered_gives_low_x_y_coverage():
    """보드 중심이 계속 이미지 중앙 근처에서만 찍히면 X/Y 바는 거의 안 찬다.
    (Edge RMS가 N/A로 나오는 실제 버그 재현 상황과 동일한 시나리오)
    """
    frames = [
        _frame("f0", 315, 235, 0.3, 5.0),
        _frame("f1", 320, 240, 0.3, 6.0),
        _frame("f2", 325, 245, 0.3, 4.0),
    ]
    bars = compute_live_coverage_bars(frames, image_size=(640, 480))
    assert bars.x_coverage < 0.2
    assert bars.y_coverage < 0.2


def test_board_spread_across_frame_gives_high_x_y_coverage():
    """보드 중심을 화면 좌/우/상/하로 실제로 이동시키며 찍으면 바가 채워진다."""
    frames = [
        _frame("f0", 640 * 0.1, 240, 0.3, 0.0),   # 왼쪽
        _frame("f1", 640 * 0.9, 240, 0.3, 0.0),   # 오른쪽
        _frame("f2", 320, 480 * 0.1, 0.3, 0.0),   # 위
        _frame("f3", 320, 480 * 0.9, 0.3, 0.0),   # 아래
    ]
    bars = compute_live_coverage_bars(frames, image_size=(640, 480))
    assert bars.x_coverage > 0.5
    assert bars.y_coverage > 0.5


def test_varying_area_ratio_increases_size_coverage():
    close_frames = [_frame(f"c{i}", 320, 240, 0.8, 0.0) for i in range(3)]
    varied_frames = close_frames + [
        _frame("far0", 320, 240, 0.1, 0.0),
        _frame("far1", 320, 240, 0.15, 0.0),
    ]
    low = compute_live_coverage_bars(close_frames, image_size=(640, 480))
    high = compute_live_coverage_bars(varied_frames, image_size=(640, 480))
    assert high.size_coverage > low.size_coverage


def test_varying_tilt_increases_skew_coverage():
    flat_frames = [_frame(f"t{i}", 320, 240, 0.3, 2.0 + i * 0.1) for i in range(3)]
    tilted_frames = flat_frames + [
        _frame("tilt0", 320, 240, 0.3, 35.0),
        _frame("tilt1", 320, 240, 0.3, -30.0),
    ]
    low = compute_live_coverage_bars(flat_frames, image_size=(640, 480))
    high = compute_live_coverage_bars(tilted_frames, image_size=(640, 480))
    assert high.skew_coverage > low.skew_coverage


def test_failed_detections_are_ignored():
    ok = _frame("ok", 640 * 0.1, 480 * 0.1, 0.3, 0.0)
    failed_info = ImageInfo(image_id="bad", path="/tmp/bad.jpg", width=640, height=480)
    failed_det = DetectionResult(image_id="bad", success=False, failure_reason="no charuco corners found")
    failed = Frame(image_info=failed_info, detection=failed_det)

    bars_with_failed = compute_live_coverage_bars([ok, failed], image_size=(640, 480))
    bars_without_failed = compute_live_coverage_bars([ok], image_size=(640, 480))
    assert bars_with_failed.x_coverage == bars_without_failed.x_coverage
    assert bars_with_failed.y_coverage == bars_without_failed.y_coverage


def test_diagonal_only_sweep_does_not_fully_fill_x_y_coverage():
    """실사용자 버그: 보드를 좌상단->우하단 대각선으로만 이동시키며 찍으면
    X 범위와 Y 범위가 각각 독립적으로는 빨리 100%에 도달했지만, 우상단/
    좌하단 사분면은 한 번도 찍히지 않아 사후 Coverage Map엔 빈 칸이
    남았다. 사분면 방문 여부를 상한으로 걸었으니 순수 대각선만으로는
    바가 다 차면 안 된다.
    """
    frames = [
        _frame(f"d{i}", 640 * 0.1 + i * (640 * 0.6 / 39), 480 * 0.1 + i * (480 * 0.6 / 39), 0.3, 0.0)
        for i in range(40)
    ]
    bars = compute_live_coverage_bars(frames, image_size=(640, 480))
    assert bars.x_coverage <= 0.5
    assert bars.y_coverage <= 0.5


def test_all_four_quadrants_visited_gives_full_x_y_coverage():
    frames = [
        _frame("tl", 100, 100, 0.3, 0.0),
        _frame("tr", 540, 100, 0.3, 0.0),
        _frame("bl", 100, 380, 0.3, 0.0),
        _frame("br", 540, 380, 0.3, 0.0),
    ]
    bars = compute_live_coverage_bars(frames, image_size=(640, 480))
    assert bars.x_coverage == 1.0
    assert bars.y_coverage == 1.0


def test_live_coverage_never_decreases_when_repetitive_frame_is_added():
    """누적 progress는 같은 자세를 더 찍었다고 이미 확보한 범위를 잃지 않는다."""
    varied = [
        _frame("left", 80, 240, 0.15, 5.0),
        _frame("right", 560, 240, 0.55, 40.0),
    ]
    before = compute_live_coverage_bars(varied, image_size=(640, 480))
    after = compute_live_coverage_bars(
        varied + [_frame(f"repeat_{i}", 320, 240, 0.30, 20.0) for i in range(20)],
        image_size=(640, 480),
    )

    assert after.x_coverage >= before.x_coverage
    assert after.y_coverage >= before.y_coverage
    assert after.size_coverage >= before.size_coverage
    assert after.skew_coverage >= before.skew_coverage
