"""
calibration.windshield.ghost.point_detector
==============================================================

STEP 8A - Point-Source Ghost Detection.

Image -> Luminance -> bright source detection -> blob(peak) 검출 -> Main
peak과 Ghost peak(secondary local maximum) 분리(사용자 스펙 6-10번).
2번째로 밝은 pixel을 그냥 고르는 방식은 절대 쓰지 않는다 - Gaussian
smoothing -> connected-component blob -> intensity-weighted subpixel
centroid + integrated blob energy를 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

from calibration.windshield.ghost.config import (
    DEFAULT_BRIGHT_SOURCE_THRESHOLD,
    DEFAULT_GAUSSIAN_SIGMA,
    DEFAULT_MAX_SEARCH_RADIUS_PX,
    DEFAULT_MIN_BLOB_AREA_PX,
    DEFAULT_MIN_PEAK_DISTANCE_PX,
)
from calibration.windshield.ghost.types import GhostPointDetection

_ENERGY_EPS = 1e-6


@dataclass
class BrightBlob:
    x: float
    y: float
    energy: float
    area_px: int


def to_luminance(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8/float 이미지를 float32 luminance(0-255 scale)로 변환한다."""
    if image_bgr.ndim == 2:
        lum = image_bgr.astype(np.float32)
    else:
        lum = cv2.cvtColor(image_bgr.astype(np.float32), cv2.COLOR_BGR2GRAY)
    if image_bgr.dtype == np.uint8:
        return lum
    # float 입력이 [0,1] 범위라고 가정되면 0-255로 스케일한다(threshold와 일관성 유지).
    if lum.max() <= 1.0 + 1e-6:
        return lum * 255.0
    return lum


def detect_bright_blobs(
    image_bgr: np.ndarray,
    *,
    threshold: float = DEFAULT_BRIGHT_SOURCE_THRESHOLD,
    min_area_px: int = DEFAULT_MIN_BLOB_AREA_PX,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
    min_peak_distance_px: float = DEFAULT_MIN_PEAK_DISTANCE_PX,
) -> list[BrightBlob]:
    """밝은 광원 후보 blob들을 검출한다.

    Gaussian smoothing(노이즈로 인한 가짜 local maxima 억제) -> threshold
    mask -> `cv2.connectedComponentsWithStats`(blob 단위 분리) -> **각
    connected component 안에서 local maxima를 찾아, main과 ghost가 서로
    가까워 하나의 component로 뭉친 경우에도 peak 단위로 분리**한다(2번째로
    밝은 pixel을 그냥 고르는 대신, local maxima -> peak별 nearest-neighbor
    영역에 대한 intensity-weighted subpixel centroid + integrated energy를
    사용 - 사용자 스펙 7-8번).
    """
    lum = to_luminance(image_bgr)
    if gaussian_sigma > 0:
        ksize = max(3, int(round(gaussian_sigma * 6)) | 1)
        smoothed = cv2.GaussianBlur(lum, (ksize, ksize), gaussian_sigma)
    else:
        smoothed = lum

    mask_bool = smoothed >= threshold
    mask = mask_bool.astype(np.uint8)
    if not np.any(mask):
        return []

    footprint_size = max(3, int(round(min_peak_distance_px * 2 + 1)))
    local_max_mask = (maximum_filter(smoothed, size=footprint_size) == smoothed) & mask_bool

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    blobs: list[BrightBlob] = []
    for label in range(1, num_labels):  # 0 = background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        region_mask = labels == label
        ys, xs = np.nonzero(region_mask)
        weights = smoothed[region_mask].astype(np.float64)

        peak_mask_in_region = local_max_mask & region_mask
        peak_ys, peak_xs = np.nonzero(peak_mask_in_region)

        if peak_ys.size <= 1:
            total_weight = float(weights.sum())
            if total_weight <= _ENERGY_EPS:
                continue
            cx = float(np.sum(xs * weights) / total_weight)
            cy = float(np.sum(ys * weights) / total_weight)
            blobs.append(BrightBlob(x=cx, y=cy, energy=total_weight, area_px=area))
            continue

        # 하나의 component 안에 여러 local maximum(=main+ghost가 겹쳐 하나로
        # 뭉친 경우) -> 각 pixel을 가장 가까운 peak에 배정해(Voronoi 분할)
        # peak별로 독립된 subpixel centroid/energy를 계산한다.
        peaks = np.stack([peak_xs.astype(np.float64), peak_ys.astype(np.float64)], axis=1)
        pixel_xy = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        dists = np.linalg.norm(pixel_xy[:, None, :] - peaks[None, :, :], axis=2)
        assignment = np.argmin(dists, axis=1)

        for peak_idx in range(peaks.shape[0]):
            sel = assignment == peak_idx
            if not np.any(sel):
                continue
            sub_weights = weights[sel]
            total_weight = float(sub_weights.sum())
            if total_weight <= _ENERGY_EPS:
                continue
            sub_xs = xs[sel]
            sub_ys = ys[sel]
            cx = float(np.sum(sub_xs * sub_weights) / total_weight)
            cy = float(np.sum(sub_ys * sub_weights) / total_weight)
            blobs.append(BrightBlob(x=cx, y=cy, energy=total_weight, area_px=int(sel.sum())))

    return blobs


def pair_main_and_ghost_blobs(
    blobs: list[BrightBlob],
    *,
    max_search_radius_px: float = DEFAULT_MAX_SEARCH_RADIUS_PX,
) -> list[GhostPointDetection]:
    """검출된 blob들을 밝기(energy) 내림차순으로 소비하며, 각 "Main" 후보에
    대해 아직 짝짓지 않은 blob 중 `max_search_radius_px` 안의 가장 가까운
    blob을 "Ghost"로 짝짓는다(greedy nearest-neighbor - 사용자 스펙 9번).

    반경 안에 후보가 없으면 그 Main은 ghost 없음(`detected=False`)으로
    남는다. 이미 다른 Main의 Ghost로 소비된 blob은 재사용하지 않는다.
    """
    order = sorted(range(len(blobs)), key=lambda i: blobs[i].energy, reverse=True)
    consumed = [False] * len(blobs)
    detections: list[GhostPointDetection] = []

    for i in order:
        if consumed[i]:
            continue
        consumed[i] = True
        main = blobs[i]

        best_j = -1
        best_dist = None
        for j in order:
            if j == i or consumed[j]:
                continue
            dx = blobs[j].x - main.x
            dy = blobs[j].y - main.y
            dist = float(np.hypot(dx, dy))
            if dist <= max_search_radius_px and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_j = j

        if best_j < 0:
            detections.append(GhostPointDetection(main_x=main.x, main_y=main.y, detected=False))
            continue

        consumed[best_j] = True
        ghost = blobs[best_j]
        offset_x = ghost.x - main.x
        offset_y = ghost.y - main.y
        distance = float(np.hypot(offset_x, offset_y))
        strength_ratio = float(ghost.energy / (main.energy + _ENERGY_EPS))
        detections.append(
            GhostPointDetection(
                main_x=main.x,
                main_y=main.y,
                ghost_x=ghost.x,
                ghost_y=ghost.y,
                offset_x_px=offset_x,
                offset_y_px=offset_y,
                distance_px=distance,
                strength_ratio=strength_ratio,
                detected=True,
            )
        )

    return detections
