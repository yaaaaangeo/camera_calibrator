"""
camera_calibrator.calibration.quality
=========================================

설계 문서 5번(Coverage Map), 7번(자세 다양성) 구현.

이 모듈은 "계산을 잘 하는" 부분이 아니라 "계산하기 전에 데이터가
괜찮은지 판단하는" 부분이다 (설계 문서 15번 "Dataset Quality Gate").
그래서 카메라 모델과 무관하게, detector.py 결과만으로 동작한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from calibration.types import (
    CameraConfig, CoverageCell, Dataset, DistributionStat, DiversityScores, Frame,
    PoseDistributionStats,
)
from calibration.models.common import infer_image_size


# ---------------------------------------------------------------------------
# Coverage Map (설계 문서 5번)
# ---------------------------------------------------------------------------

def compute_coverage_grid(
    dataset: Dataset,
    camera_config: CameraConfig,
    rows: int = 4,
    cols: int = 4,
) -> list[CoverageCell]:
    """이미지를 rows x cols 그리드로 나누고, 전체 데이터셋에서 검출된
    ChArUco 코너들이 각 셀에 몇 개씩 찍혔는지 센다.

    coverage_score는 "가장 핫한 셀 대비 상대값"이 아니라
    "코너가 균등하게 퍼졌다면 이 셀이 가져야 할 몫(fair share) 대비 비율"로
    정의한다. 전자를 쓰면 중앙에 코너가 몰리는(흔한) 데이터셋에서
    중간 정도로 괜찮은 셀들까지 전부 '부족'으로 오판된다 - 실제로 이 프로젝트
    개발 중 42장 합성 데이터셋에서 그 문제가 재현되어 여기로 바꿨다.
    """
    w, h = infer_image_size(dataset, camera_config)
    cell_w, cell_h = w / cols, h / rows
    counts = np.zeros((rows, cols), dtype=int)

    for frame in dataset.enabled_frames:
        det = frame.detection
        if not det or not det.success or det.corners is None:
            continue
        pts = det.corners.reshape(-1, 2)
        for x, y in pts:
            col = min(int(x // cell_w), cols - 1)
            row = min(int(y // cell_h), rows - 1)
            if 0 <= row < rows and 0 <= col < cols:
                counts[row, col] += 1

    total_corners = int(counts.sum())
    fair_share = total_corners / (rows * cols) if total_corners > 0 else 1.0

    cells: list[CoverageCell] = []
    for r in range(rows):
        for c in range(cols):
            cnt = int(counts[r, c])
            score = min(1.0, cnt / fair_share) if fair_share > 0 else 0.0
            cells.append(CoverageCell(row=r, col=c, corner_count=cnt, coverage_score=score))
    return cells


def _region_label(row: int, col: int, rows: int, cols: int) -> str:
    """설계 문서 5번 예시 문구("우측 하단 영역")와 동일한 형태로 라벨링.
    common.classify_regions와 같은 3분할 사고방식을 그리드 좌표에 적용한 것.
    """
    v = row / max(rows - 1, 1)
    hz = col / max(cols - 1, 1)
    vlabel = "상단" if v < 1 / 3 else ("하단" if v > 2 / 3 else "중단")
    hlabel = "좌측" if hz < 1 / 3 else ("우측" if hz > 2 / 3 else "중앙")

    if vlabel == "중단" and hlabel == "중앙":
        return "중앙"
    if vlabel == "중단":
        return hlabel
    if hlabel == "중앙":
        return vlabel
    return f"{hlabel} {vlabel}"


def coverage_warnings(
    cells: list[CoverageCell],
    rows: int = 4,
    cols: int = 4,
    low_threshold: float = 0.3,
    avg_corners_per_frame: float | None = None,
) -> list[str]:
    """설계 문서 5번:
        "우측 하단 영역의 관측이 부족합니다. 해당 영역을 포함하는
         이미지를 3장 이상 추가하세요."
    형태의 경고 문구를 생성.

    같은 라벨(예: "중앙")에 셀이 여러 개 걸치는 경우가 흔해서, 셀 단위로
    각각 경고를 내면 같은 문구가 반복되어 스팸이 된다. 라벨 단위로 묶어서
    한 번만 경고하고, 부족분을 채우려면 몇 장이 더 필요한지도 추정한다
    (avg_corners_per_frame이 주어지면).
    """
    groups: dict[str, list[CoverageCell]] = {}
    for cell in cells:
        if cell.coverage_score < low_threshold:
            label = _region_label(cell.row, cell.col, rows, cols)
            groups.setdefault(label, []).append(cell)

    fair_share_per_cell = None
    if avg_corners_per_frame:
        # cell.coverage_score = count / fair_share 이므로 fair_share = count/score (score>0일 때)
        # score가 0인 셀도 있어 역산이 불가하니, 전체 평균으로 대략적인 fair_share를 추정.
        non_zero = [c for c in cells if c.coverage_score > 0]
        if non_zero:
            fair_share_per_cell = float(
                np.mean([c.corner_count / c.coverage_score for c in non_zero])
            )

    warnings: list[str] = []
    for label, group_cells in groups.items():
        total_count = sum(c.corner_count for c in group_cells)
        avg_score = sum(c.coverage_score for c in group_cells) / len(group_cells)
        msg = f"{label} 영역의 관측이 부족합니다 (평균 coverage {avg_score:.0%})."

        if fair_share_per_cell and avg_corners_per_frame:
            needed_corners = max(0.0, fair_share_per_cell * len(group_cells) - total_count)
            n_images = int(np.ceil(needed_corners / avg_corners_per_frame)) if needed_corners > 0 else 0
            if n_images > 0:
                msg += f" 해당 영역을 포함하는 이미지를 {n_images}장 이상 추가하세요."
        warnings.append(msg)

    return warnings


def coverage_percentage(cells: list[CoverageCell], low_threshold: float = 0.3) -> float:
    """충분히 관측된 셀의 비율 (0~100). FinalResult.dataset_coverage_pct에 그대로 들어간다."""
    if not cells:
        return 0.0
    sufficient = sum(1 for c in cells if c.coverage_score >= low_threshold)
    return sufficient / len(cells) * 100.0


def format_coverage_grid(cells: list[CoverageCell], rows: int = 4, cols: int = 4) -> str:
    """터미널에서 바로 확인할 수 있는 ASCII coverage map."""
    grid = {(c.row, c.col): c for c in cells}
    lines = ["Coverage Map (코너 수 / 상대 점수)"]
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            cell = grid.get((r, c))
            if cell is None:
                row_cells.append("[   N/A   ]")
            else:
                row_cells.append(f"[{cell.corner_count:4d} {cell.coverage_score:4.0%}]")
        lines.append(" ".join(row_cells))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 자세(Pose) 다양성 (설계 문서 7번)
# ---------------------------------------------------------------------------

def _normalized_spread(values: list[float], full_score_spread: float) -> float:
    """값들의 표준편차를 0~1 점수로 변환하는 공용 휴리스�틱.
    full_score_spread에 도달하면 만점(1.0), 그 이하는 비례해서 낮아진다.

    V1 근사치다 - '진짜' 다양성 측정(예: 포즈 공간에서의 커버리지 밀도)은
    더 정교한 방법이 있지만, 표준편차 기반 근사로도 "다 똑같은 사진만
    찍었는가"는 충분히 구분해낸다.
    """
    if len(values) < 2:
        return 0.0
    spread = float(np.std(values))
    return float(min(1.0, spread / full_score_spread)) if full_score_spread > 0 else 0.0


def _distance_diversity_from_area_ratios(area_ratios: list[float]) -> float:
    """거리 다양성: 보드 면적 비율(=촬영 거리의 대리 지표)의 상대 퍼짐(CV).
    compute_diversity_scores와 compute_live_coverage_bars(실시간 캡처)가
    동일한 기준을 쓰도록 공용화 - 사후 Coverage 탭과 실시간 바가 서로
    다른 숫자를 말하면 사용자가 혼란스러우므로 반드시 같은 함수를 재사용한다.
    """
    if len(area_ratios) < 2 or np.mean(area_ratios) <= 0:
        return 0.0
    cv = float(np.std(area_ratios) / np.mean(area_ratios))
    return float(min(1.0, cv / 0.5))  # CV 0.5 이상이면 만점


def _rotation_diversity_from_tilts(tilts: list[float]) -> float:
    """자세(기울기) 다양성. 위 함수와 같은 이유로 공용화."""
    return _normalized_spread(tilts, full_score_spread=30.0)  # 표준편차 30도 이상이면 만점


def compute_diversity_scores(
    dataset: Dataset,
    cells: list[CoverageCell],
    rows: int = 4,
    cols: int = 4,
    low_threshold: float = 0.3,
) -> DiversityScores:
    """설계 문서 7번 4개 지표.

    - position_coverage / edge_coverage: Coverage Map(그리드)을 재사용해서
      계산한다 (edge_coverage는 그리드 테두리 셀만 따로 봄).
    - distance_diversity / rotation_diversity: board_area_ratio / board_tilt_deg
      의 퍼짐(표준편차) 기반 근사치.
    """
    frames = [
        f for f in dataset.enabled_frames if f.detection and f.detection.success
    ]

    # position coverage: 그리드 전체에서 충분히 관측된 셀 비율
    position_coverage = (
        sum(1 for c in cells if c.coverage_score >= low_threshold) / len(cells)
        if cells
        else 0.0
    )

    # edge coverage: 그리드 테두리 셀만
    edge_cells = [c for c in cells if c.row in (0, rows - 1) or c.col in (0, cols - 1)]
    edge_coverage = (
        sum(1 for c in edge_cells if c.coverage_score >= low_threshold) / len(edge_cells)
        if edge_cells
        else 0.0
    )

    area_ratios = [f.detection.board_area_ratio for f in frames if f.detection.board_area_ratio is not None]
    distance_diversity = _distance_diversity_from_area_ratios(area_ratios)

    tilts = [f.detection.board_tilt_deg for f in frames if f.detection.board_tilt_deg is not None]
    rotation_diversity = _rotation_diversity_from_tilts(tilts)

    return DiversityScores(
        position_coverage=position_coverage,
        distance_diversity=distance_diversity,
        rotation_diversity=rotation_diversity,
        edge_coverage=edge_coverage,
    )


# ---------------------------------------------------------------------------
# 실시간 캡처 중 X/Y/Size/Skew 바 (ROS camera_calibration의 cameracalibrator.py
# 스타일 실시간 피드백을 이 프로젝트 지표로 재현)
# ---------------------------------------------------------------------------

@dataclass
class LiveCoverageBars:
    """실시간 캡처 다이얼로그에 표시할 4개 진행률 바 (0~1).

    ROS의 고전 cameracalibrator.py GUI가 보여주는 X/Y/Size/Skew 바와
    같은 역할을 하되, 계산 방식은 이 프로젝트가 이미 쓰고 있는 지표를
    그대로 재사용한다:
      - x_coverage / y_coverage: 지금까지 캡처된 프레임들의
        board_center_px가 가로/세로로 얼마나 퍼져서 찍혔는가
        (classify_regions가 쓰는 것과 같은 "프레임 중심 1점" 기준).
      - size_coverage: distance_diversity와 동일 (보드 면적비 퍼짐 = 거리 다양성).
      - skew_coverage: rotation_diversity와 동일 (기울기 퍼짐).
    """
    x_coverage: float = 0.0
    y_coverage: float = 0.0
    size_coverage: float = 0.0
    skew_coverage: float = 0.0


def compute_live_coverage_bars(
    frames: list[Frame],
    image_size: tuple[int, int],
    x_full_spread_ratio: float = 0.25,
    y_full_spread_ratio: float = 0.25,
) -> LiveCoverageBars:
    """지금까지 캡처된 프레임 리스트로 X/Y/Size/Skew 바를 계산.

    캡처 즉시(파일 저장 직후) 호출되는 걸 전제로 가벼운 함수로 유지한다 -
    Dataset 전체 파이프라인(Coverage Map, Outlier 등)을 돌릴 필요 없이
    detect_charuco() 결과만 있으면 계산 가능하다.

    x_full_spread_ratio / y_full_spread_ratio: 이미지 너비/높이의 이 비율을
    양쪽 방향으로 확보해 전체 범위가 2배에 도달하면 만점(1.0)으로 본다.
    누적 min/max 범위 기반이라 프레임을 추가했을 때 점수가 감소하지 않는다.

    실사용자 버그: X/Y를 각자 독립적으로만 보면 보드를 좌상단->우하단
    대각선 방향으로만 쭉 옮기며 찍어도 X 범위와 Y 범위가 각각 빨리
    100%에 도달해버린다 - 실제로는 우상단/좌하단 사분면을 한 번도
    안 찍었는데 바는 다 찬 것처럼 보였고, 그 상태로 계산한 Coverage
    Map(사후 그리드)에는 군데군데 빈 칸이 남았다. 이미지를 2x2
    사분면으로 나눠 실제로 몇 개의 사분면을 찍었는지(quadrant_coverage)
    를 X/Y coverage의 상한으로 같이 걸어서, 두 축을 함께 골고루 움직여야
    바가 다 차게 한다.
    """
    w, h = image_size
    successful = [
        f for f in frames
        if f.detection and f.detection.success and f.detection.board_center_px is not None
    ]

    xs = [f.detection.board_center_px[0] for f in successful]
    ys = [f.detection.board_center_px[1] for f in successful]
    # 실시간 progress bar는 이미 확보한 촬영 범위를 뜻하므로 새 샘플이
    # 추가됐을 때 절대로 줄어들면 안 된다. 표준편차는 같은 자세를 한 장
    # 추가하는 것만으로도 작아질 수 있어 누적 UI에는 맞지 않는다. 지금까지
    # 관측한 min~max 범위를 쓰면 잘못 찍은 프레임은 바를 올리지 않을 뿐,
    # 이미 확보한 범위를 깎지는 않는다.
    def _range_coverage(values: list[float], full_range: float) -> float:
        if len(values) < 2 or full_range <= 0:
            return 0.0
        return float(min(1.0, (max(values) - min(values)) / full_range))

    def _quadrant_coverage(centers: list[tuple[float, float]]) -> float:
        """중심점이 실제로 몇 개의 2x2 사분면을 찍었는지 (0~1).
        min/max 범위와 마찬가지로 사분면 "방문 집합"은 누적이라 절대
        줄어들지 않는다.
        """
        if not centers:
            return 0.0
        visited = {(0 if x < w / 2 else 1, 0 if y < h / 2 else 1) for x, y in centers}
        return len(visited) / 4.0

    quadrant_coverage = _quadrant_coverage(list(zip(xs, ys)))
    x_coverage = min(_range_coverage(xs, w * x_full_spread_ratio * 2.0), quadrant_coverage)
    y_coverage = min(_range_coverage(ys, h * y_full_spread_ratio * 2.0), quadrant_coverage)

    area_ratios = [f.detection.board_area_ratio for f in successful if f.detection.board_area_ratio is not None]
    if len(area_ratios) < 2 or max(area_ratios) <= 0:
        size_coverage = 0.0
    else:
        relative_range = (max(area_ratios) - min(area_ratios)) / max(area_ratios)
        size_coverage = float(min(1.0, relative_range / 0.5))

    tilts = [f.detection.board_tilt_deg for f in successful if f.detection.board_tilt_deg is not None]
    skew_coverage = _range_coverage(tilts, 60.0)

    return LiveCoverageBars(
        x_coverage=x_coverage,
        y_coverage=y_coverage,
        size_coverage=size_coverage,
        skew_coverage=skew_coverage,
    )


def _bar(score: float, width: int = 10) -> str:
    filled = int(round(score * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def format_diversity_bars(diversity: DiversityScores) -> str:
    """설계 문서 7번 출력 형식 그대로.

        Position Coverage       █████████░ 90%
        Distance Diversity      ████████░░ 80%
        Rotation Diversity      ██████████ 100%
        Edge Coverage           █████████░ 90%
        Overall Dataset Quality █████████░ 91%
    """
    rows = [
        ("Position Coverage", diversity.position_coverage),
        ("Distance Diversity", diversity.distance_diversity),
        ("Rotation Diversity", diversity.rotation_diversity),
        ("Edge Coverage", diversity.edge_coverage),
        ("Overall Dataset Quality", diversity.overall),
    ]
    lines = []
    for label, score in rows:
        lines.append(f"{label:<24} {_bar(score)} {score*100:.0f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 한 번에 실행 (Dataset에 결과를 채워 넣음)
# ---------------------------------------------------------------------------

def analyze_dataset_quality(
    dataset: Dataset,
    camera_config: CameraConfig,
    rows: int = 4,
    cols: int = 4,
    low_threshold: float = 0.3,
) -> list[str]:
    """Dataset.coverage_grid / Dataset.diversity를 채우고, 부족한 영역에 대한
    경고 문구 리스트를 반환한다. 설계 문서 15번 파이프라인의
    'Dataset Quality Gate' 단계에서 UI 없이도 이 함수 하나로 확인 가능해야 한다.
    """
    cells = compute_coverage_grid(dataset, camera_config, rows=rows, cols=cols)
    dataset.coverage_grid = cells
    dataset.diversity = compute_diversity_scores(
        dataset, cells, rows=rows, cols=cols, low_threshold=low_threshold
    )

    frames = [f for f in dataset.enabled_frames if f.detection and f.detection.success]
    avg_corners = (
        float(np.mean([f.detection.num_corners for f in frames])) if frames else None
    )
    return coverage_warnings(
        cells, rows=rows, cols=cols, low_threshold=low_threshold, avg_corners_per_frame=avg_corners
    )


# ---------------------------------------------------------------------------
# 설계 문서 5번(분포 확장) / 6번(Pose Diversity 평가 강화)
# ---------------------------------------------------------------------------
# 기존 DiversityScores(0~1 점수 4개)는 "다양한가/아닌가"만 보여줬다. 여기서는
# X/Y 위치, board 크기, yaw/pitch/roll, 거리를 각각 mean/std/variance +
# coverage 점수로 분해해서 "정확히 어느 축이 부족한지" 보여준다.

def _estimate_rough_pose(
    object_points: np.ndarray, image_points: np.ndarray, image_size: tuple[int, int]
) -> tuple[float, float, float, float] | None:
    """cv2.solvePnP로 (yaw, pitch, roll, distance)를 거칠게 추정한다.

    아직 실제 카메라 파라미터를 모르는 단계(캘리브레이션 전)이므로 K를
    "focal length = max(w,h)"라는 흔한 경험적 근사로 가정한다 - 절대값은
    부정확할 수 있지만, 데이터셋 안에서 "자세가 얼마나 다양했는가"를 상대
    비교하는 진단 목적에는 이 정도 근사로 충분하다 (실제 캘리브레이션
    결과와는 무관하고, 오직 이 함수만을 위한 임시 가정).

    회전 각도는 표준 R->Euler(XYZ) 분해를 쓴다: pitch=X축, yaw=Y축, roll=Z축
    회전. distance는 tvec의 노름(카메라 원점에서 보드 원점까지 거리, object
    points와 같은 단위=보통 미터)이다.
    """
    if object_points is None or image_points is None or len(object_points) < 4:
        return None

    w, h = image_size
    f_guess = float(max(w, h))
    K_guess = np.array(
        [[f_guess, 0, w / 2.0], [0, f_guess, h / 2.0], [0, 0, 1]], dtype=np.float64
    )
    obj = object_points.reshape(-1, 1, 3).astype(np.float64)
    img = image_points.reshape(-1, 1, 2).astype(np.float64)

    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K_guess, None)
    except cv2.error:
        return None
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        # 짐벌락 근접 - roll을 0으로 두고 pitch만으로 근사 (드문 극단 케이스)
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0

    distance = float(np.linalg.norm(tvec))
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll), distance


def _stat_absolute(values: list[float], full_spread: float) -> DistributionStat:
    """표준편차가 full_spread에 도달하면 만점(1.0)인 절대 기준 coverage 점수.
    (x/y 위치 - 픽셀, yaw/pitch/roll - 도(度) 처럼 스케일이 프로젝트마다
    크게 안 변하는 값에 적합.)
    """
    if len(values) < 2:
        return DistributionStat(
            mean=float(values[0]) if values else None, sample_count=len(values)
        )
    arr = np.array(values, dtype=float)
    std = float(arr.std())
    coverage = float(min(1.0, std / full_spread)) if full_spread > 0 else 0.0
    return DistributionStat(
        mean=float(arr.mean()), std=std, variance=float(std ** 2),
        coverage_score=coverage, sample_count=len(values),
    )


def _stat_relative(values: list[float], full_cv: float) -> DistributionStat:
    """변동계수(CV=표준편차/평균) 기준 coverage 점수. board 면적비/거리처럼
    "절대 스케일이 보드 크기·단위계에 따라 달라지는" 값에 적합
    (_distance_diversity_from_area_ratios와 동일한 철학).
    """
    if len(values) < 2:
        return DistributionStat(
            mean=float(values[0]) if values else None, sample_count=len(values)
        )
    arr = np.array(values, dtype=float)
    mean, std = float(arr.mean()), float(arr.std())
    cv = std / mean if mean > 1e-9 else 0.0
    coverage = float(min(1.0, cv / full_cv)) if full_cv > 0 else 0.0
    return DistributionStat(
        mean=mean, std=std, variance=float(std ** 2),
        coverage_score=coverage, sample_count=len(values),
    )


def compute_pose_distribution_stats(
    dataset: Dataset, camera_config: CameraConfig
) -> PoseDistributionStats:
    """설계 문서 6번 - 7개 축(X/Y위치, board 면적, yaw/pitch/roll, 거리) 각각의
    분포(mean/std/variance/coverage)를 계산한다. compute_diversity_scores()가
    주는 0~1 요약 점수 4개보다 더 세분화된 진단용 지표다.
    """
    image_size = infer_image_size(dataset, camera_config)
    w, h = image_size
    frames = [f for f in dataset.enabled_frames if f.detection and f.detection.success]

    xs = [f.detection.board_center_px[0] for f in frames if f.detection.board_center_px]
    ys = [f.detection.board_center_px[1] for f in frames if f.detection.board_center_px]
    areas = [f.detection.board_area_ratio for f in frames if f.detection.board_area_ratio is not None]

    yaws, pitches, rolls, distances = [], [], [], []
    for f in frames:
        pose = _estimate_rough_pose(f.detection.object_points, f.detection.corners, image_size)
        if pose is None:
            continue
        yaw, pitch, roll, distance = pose
        yaws.append(yaw)
        pitches.append(pitch)
        rolls.append(roll)
        distances.append(distance)

    return PoseDistributionStats(
        x_position=_stat_absolute(xs, full_spread=w * 0.25),
        y_position=_stat_absolute(ys, full_spread=h * 0.25),
        board_area=_stat_relative(areas, full_cv=0.5),
        yaw=_stat_absolute(yaws, full_spread=20.0),
        pitch=_stat_absolute(pitches, full_spread=20.0),
        roll=_stat_absolute(rolls, full_spread=20.0),
        distance=_stat_relative(distances, full_cv=0.5),
    )


def format_pose_distribution_stats(stats: PoseDistributionStats) -> str:
    rows = [
        ("X Position (px)", stats.x_position),
        ("Y Position (px)", stats.y_position),
        ("Board Area (ratio)", stats.board_area),
        ("Yaw (deg, 근사)", stats.yaw),
        ("Pitch (deg, 근사)", stats.pitch),
        ("Roll (deg, 근사)", stats.roll),
        ("Distance (m, 근사)", stats.distance),
    ]
    lines = ["Pose Distribution (mean ± std, coverage)"]
    for label, s in rows:
        if s.sample_count == 0 or s.mean is None:
            lines.append(f"{label:<20} N/A")
            continue
        std_str = f"{s.std:.2f}" if s.std is not None else "N/A"
        lines.append(
            f"{label:<20} {s.mean:8.2f} ± {std_str:<8} coverage={s.coverage_score*100:.0f}%  (n={s.sample_count})"
        )
    return "\n".join(lines)
