"""
calibration.windshield.ghost.spatial_model
==============================================================

STEP 8A - Spatial Ghost Map.

검출된 point-source ghost들을 rows x cols grid에 버켓팅해서 cell별
평균 offset/distance/strength/sample_count를 집계한다(사용자 스펙 12-14
번). Displacement(vector field)와 Strength(heatmap)는 UI에서 절대 하나의
map으로 합치지 않는다 - 이 모듈은 두 값을 각 cell에 나란히 저장할 뿐,
시각화 단계에서 분리되는 것은 UI의 책임이다.

Region 단위(Center/Top/Bottom/Left/Right/Corners) 집계도 함께 제공한다
(사용자 스펙 15번).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from calibration.windshield.ghost.config import DEFAULT_MAD_OUTLIER_K
from calibration.windshield.ghost.types import GhostPointDetection, GhostRegionMetrics, GhostSpatialCell


def build_spatial_map(
    detections: list[GhostPointDetection],
    *,
    image_width: float,
    image_height: float,
    rows: int,
    cols: int,
) -> list[GhostSpatialCell]:
    """검출 결과를 rows x cols grid로 버켓팅해 cell별 평균 통계를 만든다.

    Ghost가 검출되지 않은(main만 있는) detection은 spatial map 집계에서
    제외한다 - offset/strength가 정의되지 않기 때문이다.
    """
    buckets: dict[tuple[int, int], list[GhostPointDetection]] = {}
    for det in detections:
        if not det.detected or det.offset_x_px is None or det.offset_y_px is None:
            continue
        col = int(det.main_x / max(image_width, 1e-9) * cols)
        row = int(det.main_y / max(image_height, 1e-9) * rows)
        col = min(max(col, 0), cols - 1)
        row = min(max(row, 0), rows - 1)
        buckets.setdefault((row, col), []).append(det)

    cells: list[GhostSpatialCell] = []
    for row in range(rows):
        for col in range(cols):
            members = buckets.get((row, col), [])
            if not members:
                cells.append(GhostSpatialCell(row=row, col=col, sample_count=0))
                continue
            offsets_x = np.array([m.offset_x_px for m in members], dtype=np.float64)
            offsets_y = np.array([m.offset_y_px for m in members], dtype=np.float64)
            distances = np.array([m.distance_px for m in members], dtype=np.float64)
            strengths = np.array(
                [m.strength_ratio if m.strength_ratio is not None else 0.0 for m in members],
                dtype=np.float64,
            )
            cells.append(
                GhostSpatialCell(
                    row=row,
                    col=col,
                    mean_offset_x_px=float(np.mean(offsets_x)),
                    mean_offset_y_px=float(np.mean(offsets_y)),
                    mean_distance_px=float(np.mean(distances)),
                    mean_strength_ratio=float(np.mean(strengths)),
                    sample_count=len(members),
                )
            )
    return cells


def _bucket_detections(
    detections: list[GhostPointDetection], *, image_width: float, image_height: float, rows: int, cols: int,
) -> dict[tuple[int, int], list[GhostPointDetection]]:
    buckets: dict[tuple[int, int], list[GhostPointDetection]] = {}
    for det in detections:
        if not det.detected or det.offset_x_px is None or det.offset_y_px is None:
            continue
        col = int(det.main_x / max(image_width, 1e-9) * cols)
        row = int(det.main_y / max(image_height, 1e-9) * rows)
        col = min(max(col, 0), cols - 1)
        row = min(max(row, 0), rows - 1)
        buckets.setdefault((row, col), []).append(det)
    return buckets


def _robust_median_and_mad(values: np.ndarray, k: float) -> tuple[float, float, int]:
    """Median + MAD 기반 outlier 제거(STEP 8 stabilization 3-D/3-E번):
    `|x - median(x)| <= k * 1.4826 * MAD`를 벗어나는 값을 제외한 뒤 다시
    median을 계산한다.

    `MAD==0` edge case(사용자 스펙 3-E번) - 대부분 값이 완전히 동일해서
    "전형적인 편차"가 0으로 나오는 경우, `k*1.4826*MAD` 임계값도 0이 되어
    버린다("median과 정확히 같지 않으면 outlier"라는 원칙이 그 자체로는
    맞지만) 부동소수점 잡음까지 outlier로 잘못 걷어내지 않도록, MAD==0일
    때는 아주 작은 절대 허용오차(1e-6)만 적용한다 - 그보다 큰 편차(예:
    나머지가 4.0으로 딱 붙어 있는데 하나만 30.0인 경우)는 여전히 outlier로
    제외된다."""
    if values.size == 0:
        return 0.0, 0.0, 0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    threshold = 1e-6 if mad <= 1e-9 else k * 1.4826 * mad
    inlier_mask = np.abs(values - med) <= threshold
    num_outliers = int(np.sum(~inlier_mask))
    if not np.any(inlier_mask):
        return med, mad, 0
    refined_median = float(np.median(values[inlier_mask]))
    return refined_median, mad, num_outliers


def build_robust_spatial_map_from_detections(
    detections: list[GhostPointDetection],
    *,
    image_width: float,
    image_height: float,
    rows: int,
    cols: int,
    mad_k: float = DEFAULT_MAD_OUTLIER_K,
) -> list[GhostSpatialCell]:
    """`build_spatial_map`과 같은 grid 버켓팅을 쓰지만, 단일 프레임이 아니라
    (보통 여러 프레임을 풀링한) detection 리스트 전체에 대해 median/MAD
    기반 robust aggregation을 적용한다(STEP 8 stabilization 3-C/3-D번) -
    outlier frame 하나가 cell 통계를 크게 왜곡하지 않게 한다."""
    buckets = _bucket_detections(detections, image_width=image_width, image_height=image_height, rows=rows, cols=cols)

    cells: list[GhostSpatialCell] = []
    for row in range(rows):
        for col in range(cols):
            members = buckets.get((row, col), [])
            if not members:
                cells.append(GhostSpatialCell(row=row, col=col, sample_count=0))
                continue
            dx_vals = np.array([m.offset_x_px for m in members], dtype=np.float64)
            dy_vals = np.array([m.offset_y_px for m in members], dtype=np.float64)
            dist_vals = np.array([m.distance_px for m in members], dtype=np.float64)
            strength_vals = np.array(
                [m.strength_ratio if m.strength_ratio is not None else 0.0 for m in members], dtype=np.float64,
            )

            dx_med, dx_mad, dx_out = _robust_median_and_mad(dx_vals, mad_k)
            dy_med, dy_mad, dy_out = _robust_median_and_mad(dy_vals, mad_k)
            dist_med, _dist_mad, _dist_out = _robust_median_and_mad(dist_vals, mad_k)
            strength_med, strength_mad, strength_out = _robust_median_and_mad(strength_vals, mad_k)

            cells.append(
                GhostSpatialCell(
                    row=row,
                    col=col,
                    mean_offset_x_px=dx_med,
                    mean_offset_y_px=dy_med,
                    mean_distance_px=dist_med,
                    mean_strength_ratio=strength_med,
                    sample_count=len(members),
                    mad_offset_x_px=dx_mad,
                    mad_offset_y_px=dy_mad,
                    mad_strength=strength_mad,
                    outlier_rejected_count=max(dx_out, dy_out, strength_out),
                    is_filled=False,
                )
            )
    return cells


def fill_empty_spatial_cells(cells: list[GhostSpatialCell]) -> list[GhostSpatialCell]:
    """빈 cell(sample_count==0)을 우선순위대로 채운다(STEP 8 stabilization
    3-F번): (1) 이미 값이 있으면 그대로, (2) grid 상 가장 가까운 valid
    cell 값으로 보간, (3) valid cell이 하나도 없으면 채우지 않고 그대로
    둔다(전체 dataset에 detection이 하나도 없다는 뜻이라 채울 근거가
    없음). 과도하게 검증되지 않은 복잡한 extrapolation(예: 방향성 있는
    선형 외삽)은 쓰지 않는다."""
    valid = [c for c in cells if c.sample_count > 0]
    if not valid:
        return cells

    filled: list[GhostSpatialCell] = []
    for c in cells:
        if c.sample_count > 0:
            filled.append(c)
            continue
        nearest = min(valid, key=lambda o: abs(o.row - c.row) + abs(o.col - c.col))
        filled.append(
            dataclasses.replace(
                c,
                mean_offset_x_px=nearest.mean_offset_x_px,
                mean_offset_y_px=nearest.mean_offset_y_px,
                mean_distance_px=nearest.mean_distance_px,
                mean_strength_ratio=nearest.mean_strength_ratio,
                is_filled=True,
            )
        )
    return filled


def _region_masks(image_width: float, image_height: float, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    nx = x / max(image_width, 1e-9)
    ny = y / max(image_height, 1e-9)
    masks: dict[str, np.ndarray] = {
        "center": (nx >= 0.33) & (nx <= 0.67) & (ny >= 0.33) & (ny <= 0.67),
        "top": ny < 0.33,
        "bottom": ny > 0.67,
        "left": nx < 0.33,
        "right": nx > 0.67,
        "corners": ((nx < 0.2) | (nx > 0.8)) & ((ny < 0.2) | (ny > 0.8)),
    }
    return masks


def compute_region_metrics(
    detections: list[GhostPointDetection],
    *,
    image_width: float,
    image_height: float,
) -> dict[str, GhostRegionMetrics]:
    """Center/Top/Bottom/Left/Right/Corners 영역별 Ghost Offset/Strength
    metric을 계산한다."""
    if not detections:
        names = ["center", "top", "bottom", "left", "right", "corners"]
        return {name: GhostRegionMetrics() for name in names}

    xs = np.array([d.main_x for d in detections], dtype=np.float64)
    ys = np.array([d.main_y for d in detections], dtype=np.float64)
    masks = _region_masks(image_width, image_height, xs, ys)

    results: dict[str, GhostRegionMetrics] = {}
    for name, mask in masks.items():
        idxs = np.nonzero(mask)[0]
        if idxs.size == 0:
            results[name] = GhostRegionMetrics()
            continue
        region_dets = [detections[i] for i in idxs]
        detected = [d for d in region_dets if d.detected and d.distance_px is not None]
        detection_rate = len(detected) / len(region_dets)
        if detected:
            mean_dist = float(np.mean([d.distance_px for d in detected]))
            mean_strength = float(
                np.mean([d.strength_ratio if d.strength_ratio is not None else 0.0 for d in detected])
            )
        else:
            mean_dist = 0.0
            mean_strength = 0.0
        results[name] = GhostRegionMetrics(
            mean_distance_px=mean_dist,
            mean_strength_ratio=mean_strength,
            detection_rate=float(detection_rate),
        )
    return results
