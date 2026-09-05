"""
camera_calibrator.calibration.spatial_error_map
====================================================

설계 문서 13번 - Spatial Error Map 강화. 기존 Edge Error Map(regional_error,
models/common.py)과 Radial Error Profile(radial_profile.py)은 "재투영 오차의
크기"만 본다 - 이 모듈은 여기에 "방향"을 더한다.

    "체계적인 방향 패턴이 나타나면 camera model이 데이터를 충분히
     설명하지 못할 가능성이 있다."

예를 들어 그리드의 모든 칸에서 재투영 오차 벡터(dx, dy)가 하나같이
바깥쪽(이미지 중심에서 멀어지는 방향)을 향한다면, 방사 왜곡 계수가
부족하거나(더 높은 차수가 필요) 모델 자체가 안 맞는다는 신호다. 반대로
방향이 칸마다 무작위(딱히 패턴이 없음)라면 순수 노이즈에 가깝다는 뜻이라
안심할 수 있는 신호다.

radial_profile.py의 collect_per_point_vectors()가 이미 모든 코너의
(x, y, dx, dy)를 모아주므로, 이 모듈은 그 결과를 rows x cols 그리드로
버킷팅해서 칸별 RMS/P95/평균방향만 계산한다 - 투영 로직을 새로 만들지 않는다.
"""

from __future__ import annotations

import math

import numpy as np

from calibration.radial_profile import collect_per_point_vectors
from calibration.types import CameraModelType, Frame, SpatialErrorCell, SpatialErrorMap


def bin_spatial_errors(
    xs: np.ndarray,
    ys: np.ndarray,
    dxs: np.ndarray,
    dys: np.ndarray,
    image_size: tuple[int, int],
    rows: int = 4,
    cols: int = 4,
) -> SpatialErrorMap:
    """이미 계산된 (검출 위치, dx, dy) 포인트 배열을 rows x cols 그리드로
    버킷팅해서 칸별 RMS/P95/평균 방향(dx, dy, 각도)으로 요약한다 - 투영 방식과
    무관한 순수 집계 로직이다.

    compute_spatial_error_map()이 이 함수를 감싸는 얇은 wrapper다(central
    카메라 모델 투영으로 xs/ys/dxs/dys를 구해서 넘김). Windshield Spherical처럼
    투영 방식 자체가 다른 경우, 자기 방식으로 계산한 벡터 배열을 이 함수에
    직접 넘겨 같은 집계 로직을 재사용한다.

    quality.compute_coverage_grid()와 같은 그리드 좌표 규칙(행=위->아래,
    열=왼쪽->오른쪽)을 쓴다 - 두 그리드를 나란히 봤을 때 셀 위치가 헷갈리지
    않도록.
    """
    w, h = image_size
    cells: list[SpatialErrorCell] = []
    if xs.size == 0 or w <= 0 or h <= 0:
        # 계산할 포인트가 없어도 빈 셀 그리드 구조는 유지한다 - 호출부가
        # "그리드가 아예 없음"과 "그리드는 있는데 데이터가 없음"을 구분할 필요는
        # 없고, 어차피 빈 셀은 num_points=0으로 남으므로 이렇게만 해도 충분하다.
        for r in range(rows):
            for c in range(cols):
                cells.append(SpatialErrorCell(row=r, col=c))
        return SpatialErrorMap(cells=cells, rows=rows, cols=cols)

    cell_w, cell_h = w / cols, h / rows
    magnitudes = np.hypot(dxs, dys)

    col_idx = np.clip((xs // cell_w).astype(int), 0, cols - 1)
    row_idx = np.clip((ys // cell_h).astype(int), 0, rows - 1)

    for r in range(rows):
        for c in range(cols):
            mask = (row_idx == r) & (col_idx == c)
            count = int(mask.sum())
            if count == 0:
                cells.append(SpatialErrorCell(row=r, col=c, num_points=0))
                continue

            cell_dx, cell_dy, cell_mag = dxs[mask], dys[mask], magnitudes[mask]
            mean_dx, mean_dy = float(cell_dx.mean()), float(cell_dy.mean())
            direction_deg = math.degrees(math.atan2(mean_dy, mean_dx))

            cells.append(SpatialErrorCell(
                row=r, col=c, num_points=count,
                rms=float(np.sqrt(np.mean(cell_mag ** 2))),
                p95=float(np.percentile(cell_mag, 95)),
                mean_dx=mean_dx, mean_dy=mean_dy,
                direction_deg=direction_deg,
            ))

    return SpatialErrorMap(cells=cells, rows=rows, cols=cols)


def compute_spatial_error_map(
    frames: list[Frame],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    model: CameraModelType,
    rows: int = 4,
    cols: int = 4,
) -> SpatialErrorMap:
    """이미지를 rows x cols 그리드로 나누고, 각 칸에 찍힌 코너들의
    재투영 오차를 RMS/P95/평균 방향(dx, dy, 각도)으로 요약한다.
    """
    xs, ys, dxs, dys = collect_per_point_vectors(frames, rvecs, tvecs, camera_matrix, distortion, model)
    return bin_spatial_errors(xs, ys, dxs, dys, image_size, rows=rows, cols=cols)


# ---------------------------------------------------------------------------
# 방향 화살표 (터미널/문자 출력용)
# ---------------------------------------------------------------------------

# 8방향 화살표 - direction_deg(atan2 기준, 0=+x 오른쪽, 90=+y 아래쪽,
# 이미지 좌표계라 y가 아래로 증가함에 유의)를 가장 가까운 화살표로 근사한다.
_ARROWS = ["\u2192", "\u2198", "\u2193", "\u2199", "\u2190", "\u2196", "\u2191", "\u2197"]


def _direction_arrow(direction_deg: float) -> str:
    idx = int(round(direction_deg / 45.0)) % 8
    return _ARROWS[idx]


def format_spatial_error_map(smap: SpatialErrorMap) -> str:
    """터미널 확인용 ASCII 그리드.

        Spatial Error Map (RMS / P95 / 방향)
        [0.31→ 0.52] [0.28↘ 0.41] [0.30↓ 0.48] [0.35↙ 0.55]
        [0.22→ 0.35] [0.15  0.20] [0.14  0.19] [0.25← 0.38]
        ...

    화살표가 없는 칸(포인트 없음 또는 방향이 거의 0에 가까움)은 공백으로 둔다.
    """
    if not smap.cells:
        return "Spatial Error Map을 계산할 데이터가 없습니다."

    grid = {(c.row, c.col): c for c in smap.cells}
    lines = ["Spatial Error Map (RMS / P95 / 방향)"]
    for r in range(smap.rows):
        row_cells = []
        for c in range(smap.cols):
            cell = grid.get((r, c))
            if cell is None or cell.num_points == 0:
                row_cells.append("[   N/A    ]")
                continue
            arrow = _direction_arrow(cell.direction_deg) if cell.direction_deg is not None else " "
            row_cells.append(f"[{cell.rms:4.2f}{arrow}{cell.p95:4.2f}]")
        lines.append(" ".join(row_cells))

    lines.append("")
    lines.append(
        "화살표는 그 칸의 평균 재투영 오차 벡터(dx,dy) 방향입니다. "
        "여러 칸이 한 방향(특히 중심 기준 바깥쪽/안쪽)으로 몰려 있으면 "
        "카메라 모델이 왜곡을 충분히 설명하지 못하고 있다는 신호입니다."
    )
    return "\n".join(lines)


def has_systematic_direction_bias(smap: SpatialErrorMap, min_cells: int = 4) -> bool:
    """설계 문서 13번 - "체계적인 방향 패턴" 자동 감지의 최소 버전.

    포인트가 있는 셀들의 평균 방향 벡터(정규화 후 평균)의 크기가 크면(=여러
    칸이 비슷한 방향을 가리키면) True. 벡터가 서로 다른 방향으로 흩어져
    있으면 평균이 서로 상쇄돼 작아진다 - "무작위 노이즈"와 "일관된 편향"을
    가르는 간단하지만 합리적인 근사치다.
    """
    valid = [c for c in smap.cells if c.num_points > 0 and c.mean_dx is not None]
    if len(valid) < min_cells:
        return False

    unit_vectors = []
    for c in valid:
        mag = math.hypot(c.mean_dx, c.mean_dy)
        if mag > 1e-9:
            unit_vectors.append((c.mean_dx / mag, c.mean_dy / mag))
    if not unit_vectors:
        return False

    mean_ux = sum(v[0] for v in unit_vectors) / len(unit_vectors)
    mean_uy = sum(v[1] for v in unit_vectors) / len(unit_vectors)
    consistency = math.hypot(mean_ux, mean_uy)  # 0(방향 제각각) ~ 1(전부 같은 방향)
    return consistency > 0.6
