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

import numpy as np

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
