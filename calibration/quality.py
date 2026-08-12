"""
camera_calibrator.calibration.quality
=========================================

설계 문서 5번(Coverage Map), 7번(자세 다양성) 구현.

이 모듈은 "계산을 잘 하는" 부분이 아니라 "계산하기 전에 데이터가
괜찮은지 판단하는" 부분이다 (설계 문서 15번 "Dataset Quality Gate").
그래서 카메라 모델과 무관하게, detector.py 결과만으로 동작한다.
"""

from __future__ import annotations

import numpy as np

from calibration.types import CameraConfig, CoverageCell, Dataset, DiversityScores
from calibration.models.common import classify_regions, infer_image_size


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
    # 거리 다양성: 보드 면적 비율(=촬영 거리의 대리 지표)의 상대 퍼짐(CV)
    distance_diversity = 0.0
    if len(area_ratios) >= 2 and np.mean(area_ratios) > 0:
        cv = float(np.std(area_ratios) / np.mean(area_ratios))
        distance_diversity = float(min(1.0, cv / 0.5))  # CV 0.5 이상이면 만점

    tilts = [f.detection.board_tilt_deg for f in frames if f.detection.board_tilt_deg is not None]
    rotation_diversity = _normalized_spread(tilts, full_score_spread=30.0)  # 표준편차 30도 이상이면 만점

    return DiversityScores(
        position_coverage=position_coverage,
        distance_diversity=distance_diversity,
        rotation_diversity=rotation_diversity,
        edge_coverage=edge_coverage,
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
