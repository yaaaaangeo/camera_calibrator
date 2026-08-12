"""
camera_calibrator.calibration.frame_quality
================================================

설계 문서 6번 - Frame Quality Score.

    Image 01    Score 94   ✓ Excellent
    Image 02    Score 91   ✓ Excellent
    Image 03    Score 48   ⚠ Poor
    Image 04    Score 12   ✕ Reject

두 단계로 나눠서 호출한다:

1. compute_frame_quality_scores(dataset, ..., use_reprojection=False)
   - detector.py 검출 직후 + quality.py의 Coverage Map 계산 직후 호출.
   - 아직 캘리브레이션 전이라 reprojection_error가 없으므로 그 항목은 제외.
2. compute_frame_quality_scores(dataset, ..., use_reprojection=True)
   - 3모델 계산 후(Frame.reprojection_error가 채워진 뒤) 다시 호출해 점수를
     갱신한다. 재투영 오차가 반영되면 더 정확한 "쓸 만한 프레임인가"
     판단이 가능해진다.

설계 원칙 (문서 3.1번의 반복 경고와 동일한 이유):
    "0.5px = 자율주행 사용 가능 같은 절대 기준을 하드코딩하는 건 위험하다."
그래서 선명도(sharpness)/재투영 오차처럼 카메라·렌즈·해상도마다 절대값의
의미가 달라지는 지표는, 데이터셋 안에서의 상대적 위치(min-max 정규화)로
점수를 매긴다. 코너 수(보드 스펙 대비 비율)나 노출(이상적 중간톤과의
거리)처럼 절대적으로 의미가 고정된 지표만 절대 기준을 쓴다.
"""

from __future__ import annotations

import numpy as np

from calibration.types import (
    CoverageCell,
    Dataset,
    Frame,
    FrameQuality,
    PatternConfig,
    QualityGrade,
)

# ---------------------------------------------------------------------------
# 가중치 (detection_score 내부, geometric_score 내부, 최종 합산)
# ---------------------------------------------------------------------------

_W_CORNER = 0.30
_W_SHARPNESS = 0.20
_W_EXPOSURE = 0.15
_W_AREA = 0.15
_W_REPROJECTION = 0.20  # use_reprojection=True일 때만 활성화되고, 나머지 가중치는 재정규화됨

_W_DETECTION_OVERALL = 0.7  # overall_score = 0.7*detection + 0.3*geometric
_W_GEOMETRIC_OVERALL = 0.3

# 보드 면적 비율(이미지 대비)의 "선호 구간". 너무 작으면(멀리서 찍음) 코너 정밀도가
# 떨어지고, 너무 크면(너무 가까이서 찍음) 보드가 화면 밖으로 잘리기 쉽다.
_AREA_SWEET_LOW = 0.10
_AREA_SWEET_HIGH = 0.55


# ---------------------------------------------------------------------------
# 보드 스펙 기준 최대 코너 수
# ---------------------------------------------------------------------------

def max_possible_corners(pattern: PatternConfig) -> int:
    """ChArUco 코너는 내부 교차점 기준이라 (squares_x-1)*(squares_y-1)개다.
    detector.py의 build_charuco_board 계산 방식과 동일하게 맞춘다.
    """
    return max(1, (pattern.squares_x - 1) * (pattern.squares_y - 1))


# ---------------------------------------------------------------------------
# 상대 정규화 (데이터셋 내부 min-max, 절대 임계값 하드코딩 방지)
# ---------------------------------------------------------------------------

def _relative_scores(values: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    """frame_id -> raw value 딕셔너리를 frame_id -> 0~1 점수로 변환.

    값이 전부 같거나(예: 프레임이 1장) 비교 대상이 없으면 모두 0.5(중립)를 준다 -
    "데이터가 부족해서 판단 불가"를 "가장 나쁨"으로 취급하지 않기 위함.
    """
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return {k: 0.5 for k in values}

    out: dict[str, float] = {}
    for k, v in values.items():
        norm = (v - lo) / (hi - lo)
        out[k] = norm if higher_is_better else (1.0 - norm)
    return out


def _area_preference_score(ratio: float) -> float:
    """보드 면적 비율이 선호 구간([_AREA_SWEET_LOW, _AREA_SWEET_HIGH])에 있으면 1.0,
    벗어날수록 선형으로 감소. 구간의 절반 폭만큼 벗어나면 0.0.
    """
    if _AREA_SWEET_LOW <= ratio <= _AREA_SWEET_HIGH:
        return 1.0
    span = _AREA_SWEET_HIGH - _AREA_SWEET_LOW
    if ratio < _AREA_SWEET_LOW:
        dist = _AREA_SWEET_LOW - ratio
    else:
        dist = ratio - _AREA_SWEET_HIGH
    return max(0.0, 1.0 - dist / span)


def _exposure_score(brightness: float, ideal: float = 127.0) -> float:
    """중간톤(127)에서 멀어질수록 감소. 완전히 검거나 흰 이미지(0 또는 255)면 0점."""
    return max(0.0, 1.0 - abs(brightness - ideal) / ideal)


# ---------------------------------------------------------------------------
# Coverage 기여도 (기존 데이터와의 중복 정도의 반대 개념)
# ---------------------------------------------------------------------------

def _coverage_contribution(
    frame: Frame,
    coverage_grid: list[CoverageCell],
    image_size: tuple[int, int],
    rows: int,
    cols: int,
) -> float | None:
    """이 프레임의 코너들이 걸쳐 있는 셀들의 평균 coverage_score를 구해서,
    "이미 잘 채워진 영역 위주(중복)"인지 "아직 부족한 영역 위주(기여)"인지를
    0~1로 반환한다. 1에 가까울수록 부족한 영역을 채워주는 valuable한 프레임.

    coverage_grid는 이 프레임 자신의 코너도 포함해서 계산된 것이라
    완벽한 leave-one-out은 아니지만(다른 프레임들과 비교한 근사치),
    "이 프레임이 주로 어디를 찍었는가"를 판단하는 데는 충분하다.
    """
    det = frame.detection
    if not det or not det.success or det.corners is None or not coverage_grid:
        return None

    w, h = image_size
    cell_w, cell_h = w / cols, h / rows
    grid = {(c.row, c.col): c for c in coverage_grid}

    touched_scores: list[float] = []
    for x, y in det.corners.reshape(-1, 2):
        col = min(int(x // cell_w), cols - 1)
        row = min(int(y // cell_h), rows - 1)
        cell = grid.get((row, col))
        if cell is not None:
            touched_scores.append(min(1.0, cell.coverage_score))

    if not touched_scores:
        return None
    avg_existing_coverage = float(np.mean(touched_scores))
    return max(0.0, 1.0 - avg_existing_coverage)


# ---------------------------------------------------------------------------
# 등급 매핑 (설계 문서 3.1 RMS 등급과 동일한 6단계 어휘를 재사용)
# ---------------------------------------------------------------------------

def _grade_from_score(score: float) -> QualityGrade:
    if score >= 85:
        return QualityGrade.EXCELLENT
    if score >= 70:
        return QualityGrade.VERY_GOOD
    if score >= 50:
        return QualityGrade.GOOD
    if score >= 25:
        return QualityGrade.WARNING
    if score >= 10:
        return QualityGrade.POOR
    return QualityGrade.REJECT


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------

def compute_frame_quality_scores(
    dataset: Dataset,
    pattern_config: PatternConfig,
    image_size: tuple[int, int],
    use_reprojection: bool = False,
    rows: int = 4,
    cols: int = 4,
) -> None:
    """검출 성공한(성공 여부와 무관하게 활성화된) 모든 프레임에 FrameQuality를 채운다.

    검출 실패(DETECTION_FAILED) 프레임은 애초에 비교할 지표가 없으므로 건너뛴다 -
    이미 상태값(FrameStatus.DETECTION_FAILED)으로 구분되고 있어 별도 점수가 필요 없다.

    dataset.coverage_grid가 비어 있으면(quality.analyze_dataset_quality를 먼저
    호출하지 않은 경우) geometric_score는 0.5(중립)으로 대체한다.
    """
    frames = [
        f for f in dataset.frames
        if f.detection and f.detection.success and f.status.value not in ("detection_failed",)
    ]
    if not frames:
        return

    max_corners = max_possible_corners(pattern_config)

    # --- 데이터셋 전체에 걸친 상대 정규화용 raw value 수집 ---
    sharpness_raw = {
        f.image_info.image_id: f.image_info.sharpness
        for f in frames
        if f.image_info.sharpness is not None
    }
    sharpness_scores = _relative_scores(sharpness_raw, higher_is_better=True)

    reprojection_scores: dict[str, float] = {}
    if use_reprojection:
        reproj_raw = {
            f.image_info.image_id: f.reprojection_error
            for f in frames
            if f.reprojection_error is not None
        }
        reprojection_scores = _relative_scores(reproj_raw, higher_is_better=False)

    for frame in frames:
        fid = frame.image_info.image_id
        det = frame.detection

        # --- Detection Quality 구성요소 ---
        corner_score = min(1.0, det.num_corners / max_corners)
        sharp_score = sharpness_scores.get(fid, 0.5)
        exposure_score = (
            _exposure_score(frame.image_info.brightness)
            if frame.image_info.brightness is not None
            else 0.5
        )
        area_score = (
            _area_preference_score(det.board_area_ratio)
            if det.board_area_ratio is not None
            else 0.5
        )

        components = [
            (corner_score, _W_CORNER),
            (sharp_score, _W_SHARPNESS),
            (exposure_score, _W_EXPOSURE),
            (area_score, _W_AREA),
        ]
        if use_reprojection and fid in reprojection_scores:
            components.append((reprojection_scores[fid], _W_REPROJECTION))

        total_w = sum(w for _, w in components)
        detection_score = sum(s * w for s, w in components) / total_w if total_w > 0 else 0.0

        # --- Geometric Quality (coverage 기여도) ---
        contribution = _coverage_contribution(frame, dataset.coverage_grid, image_size, rows, cols)
        geometric_score = contribution if contribution is not None else 0.5

        overall = 100.0 * (
            _W_DETECTION_OVERALL * detection_score + _W_GEOMETRIC_OVERALL * geometric_score
        )

        frame.quality = FrameQuality(
            detection_score=round(detection_score * 100.0, 1),
            geometric_score=round(geometric_score * 100.0, 1),
            overall_score=round(overall, 1),
            grade=_grade_from_score(overall),
        )


# ---------------------------------------------------------------------------
# 출력용 요약 (설계 문서 6번 형식 그대로)
# ---------------------------------------------------------------------------

_GRADE_SYMBOL = {
    QualityGrade.EXCELLENT: "✓ Excellent",
    QualityGrade.VERY_GOOD: "✓ Very Good",
    QualityGrade.GOOD: "✓ Good",
    QualityGrade.WARNING: "⚠ Warning",
    QualityGrade.POOR: "⚠ Poor",
    QualityGrade.REJECT: "✕ Reject",
}


def format_frame_quality_table(dataset: Dataset) -> str:
    """
        Image 01    Score 94   ✓ Excellent
        Image 02    Score 91   ✓ Excellent
        Image 03    Score 48   ⚠ Poor
        Image 04    Score 12   ✕ Reject
    """
    lines = []
    for frame in dataset.frames:
        if frame.quality is None:
            continue
        q = frame.quality
        label = _GRADE_SYMBOL.get(q.grade, q.grade.value)
        lines.append(f"{frame.image_info.image_id:<16} Score {q.overall_score:5.1f}   {label}")
    return "\n".join(lines) if lines else "품질 점수가 계산된 프레임이 없습니다."
